# DETRをステップごとに実装する

ResNet-50、Encoder/Decoder、Object Queries、予測ヘッド、Hungarian Matching、
分類・L1・GIoU損失、COCO形式Datasetと学習・推論コマンドを実装しています。

## 準備

```bash
uv sync
uv run pytest -q
```

COCO形式の画像と正解JSONを用意します。データは自動ダウンロードしません。

まず動作確認する場合は、図形画像の小さなデータセットを生成できます。

```bash
uv run detr make-demo-data
```

`data/demo/train2017`に16枚、`data/demo/val2017`に4枚の画像と、
`data/demo/annotations/instances_train2017.json`・`instances_val2017.json`を作ります。
赤い四角と青い楕円の2クラスで、写真の検出精度を評価するデータではありません。
既存ファイルがある出力先には上書きしません。

ダウンロードなしで学習・推論を試す例（小さい構成、1エポックの動作確認）：

```bash
uv run detr train \
  --image-dir data/demo/train2017 \
  --annotations data/demo/annotations/instances_train2017.json \
  --output-dir output/demo --epochs 1 --batch-size 2 \
  --short-side 64 --max-size 96 --no-augmentation \
  --d-model 32 --nhead 4 --dim-feedforward 64 \
  --encoder-layers 1 --decoder-layers 2 --num-queries 10

uv run detr predict \
  --checkpoint output/demo/checkpoint.pt \
  --image data/demo/val2017/000001.png \
  --output output/demo/prediction.png --threshold 0.1
```

1エポックで正確に検出できるとは限りません。閾値を下げて描画されるボックスも、
動作確認用の予測です。

```text
data/coco/
  train2017/                # 画像
  annotations/
    instances_train2017.json
```

JSONは`images`、`annotations`、`categories`を持ち、ボックスはCOCOの
`[左上x, 左上y, 幅, 高さ]`です。カテゴリIDは昇順で0始まりに変換されます。
学習・検証に同じカテゴリ一覧を使ってください。

## 学習

```bash
uv run detr train \
  --image-dir data/coco/train2017 \
  --annotations data/coco/annotations/instances_train2017.json \
  --output-dir output/train \
  --epochs 10 --batch-size 2 --device cpu \
  --pretrained-backbone
```

`--pretrained-backbone`は初回にImageNet重みを取得します。Transformerと
予測ヘッドはランダム初期化なので、これだけで検出できるわけではありません。
省略するとBackboneもランダム初期化になります。`--epochs 10`は実行例であり、
十分な精度が得られる学習回数を保証するものではありません。
CUDA環境では`--device cuda`を指定できます。現在のマッチングではSciPyを使うため、
MacのMPS対応はこの実装では検証していません。

学習ではランダム左右反転とマルチスケールリサイズを使います。
小さい画像での動作確認には、例えば次のオプションを追加します。

```bash
--short-side 32 --max-size 64 --no-augmentation \
--d-model 8 --nhead 2 --dim-feedforward 16 \
--encoder-layers 1 --decoder-layers 2 --num-queries 10
```

これは動作確認用の小さい構成です。各画像の正解物体数は`--num-queries`以下にします。
拡張を有効にして小さく実行する場合は`--train-scales 32 48 64`などを指定します。
`--short-side`は拡張無効時と推論時の短辺です。

各エポックで次を保存します。

`--dropout 0`でDropoutを無効にできます（省略時0.1）。
`--save-every 50`を指定すると重みは50エポックごとと最終エポックに保存します。
損失ログは保存間隔に関係なく毎エポック記録します。

- `output/train/checkpoint.pt`：最新のモデル、構成、Optimizer状態、クラス対応、前処理設定
- `output/train/metrics.jsonl`：エポックごとの学習損失。検証精度・mAPではありません

現在は単一プロセスでの基本的な学習です。学習率スケジュール、
ランダムクロップは未実装で、論文の学習設定の完全再現ではありません。
同じ出力先で再実行すると既存のログとチェックポイントを置き換えます。

### 学習再開と定期検証

保存済みログからグラフだけを作る場合（再学習は行いません）：

```bash
uv run detr plot-training \
  --log-dirs output/demo-500-resumed output/demo-500-resumed-90 \
  --output output/demo-500-resumed-90/progress.png
```

`metrics.jsonl`、`validation.jsonl`、`query-probe.jsonl`を読み、エポック順に結合します。
重複するエポックは後に指定したフォルダを優先します。ないログのパネルは未記録と表示します。

```bash
uv run detr train \
  --resume output/demo-500-stable/checkpoint.pt \
  --image-dir data/demo-500/train2017 \
  --annotations data/demo-500/annotations/instances_train2017.json \
  --output-dir output/demo-500-resumed \
  --epochs 60 --batch-size 8 --train-scales 48 64 80 \
  --val-image-dir data/demo-500/val2017 \
  --val-annotations data/demo-500/annotations/instances_val2017.json \
  --eval-every 5 --save-every 5 --seed 42
```

`--epochs`は追加回数ではなく最終エポック番号です。30から60なら30回追加します。
再開時はモデル構成（Dropoutを含む）・前処理・Optimizer状態と学習率を復元するため、
モデル構成・学習率のCLIオプションは使いません。バッチサイズ・学習用拡張設定は
コマンドで指定してください。新しいチェックポイントにはCPUのtorch乱数状態も保存します。
古いチェックポイントには乱数状態がないため、途切れない実行との完全一致は保証しません。

`metrics.jsonl`に学習損失、`validation.jsonl`に再開前および一定間隔・最終回の検証損失とAPを
記録します。`best.pt`はこの出力先で観測した最高検証APの重み、`checkpoint.pt`は最新の重みです。
同じ出力先で再開する場合、学習ログ末尾とチェックポイントのエポックが一致する必要があります。
新しい出力先に再開すれば、元のログと重みを残せます。

各検証時には同じ検証画像（ID昇順の先頭）を学習更新なしで観測し、
`query-probe.jsonl`へ各候補の特徴・全クラス確率・正解とのマッチング先を保存します。
`trace-queries`と違い、診断のためにその1枚だけで追加学習することはありません。

## 推論と描画

```bash
uv run detr predict \
  --checkpoint output/train/checkpoint.pt \
  --image data/coco/train2017/example.jpg \
  --output output/prediction.png \
  --threshold 0.5 --device cpu
```

チェックポイントからモデル構成とクラス対応を復元します。推論時にBackboneの重みを
再ダウンロードする必要はありません。ボックス・クラス名・スコアを描いた画像と、
同名のJSONを保存します。JSONの`box_xyxy`は元画像上のピクセル座標で、画像外の値も
取り得ます。描画時だけ境界内に制限します。閾値以上の予測がなければ、画像はそのままで
JSONは空配列になります。NMSは行いません。

## 検証データでの評価

候補が似た予測を出す理由を調べるには、1枚用の注釈を使って追跡できます。

```bash
uv run detr trace-queries \
  --checkpoint output/overfit-one/checkpoint.pt \
  --image-dir data/demo/train2017 \
  --annotations data/demo/annotations/instances_one_image.json \
  --output-dir output/query-trace --steps 100
```

保存した重み・Optimizer状態から追加で100回更新します。元の保存先は変更しません。
過去の300エポックを再現する処理ではありません。`query-trace.jsonl`に各更新で実際に
損失へ使ったマッチング・全候補の確率・Decoder特徴・Object Queriesの勾配ノルムを記録し、
更新後には同じ画像をDropout無効でも観測します。`query-trace.png`に推移、
`summary.json`に集計、`checkpoint.pt`に追加学習後の重みを保存します。

同じ重みからDropoutだけを無効にして比較する場合は、`--dropout 0 --seed 42`と
別の`--output-dir output/query-trace-no-dropout`を指定します。
通常のDropoutとMultiheadAttention内部のDropoutの両方が変更され、変更後の設定も
新しいチェックポイントに保存されます。省略時は元の設定を維持します。

1枚の正解と予測を比較する場合は、推論に注釈JSONを渡せます。

```bash
uv run detr predict \
  --checkpoint output/overfit-one/checkpoint.pt \
  --image data/demo/train2017/000003.png \
  --annotations data/demo/annotations/instances_one_image.json \
  --output output/overfit-one/comparison.png \
  --threshold 0.1 --top-k 1 --scale 6
```

正解枠は緑、予測枠は赤で描きます。`--scale`は描画だけの拡大です。
`--top-k`は描画と通常の予測JSONだけを絞り込みます。
`comparison.diagnostics.json`には、閾値・top-kに関係なく全候補について、
元のquery番号、クラス確率、no-object確率、同クラスの正解との最大IoUを保存します。
最大IoUは候補ごとの比較で、一対一マッチングやAPを表すものではありません。
同クラスの正解がない場合はIoUがnullです。
注釈内に同名ファイルが複数ある場合は`--image-id`も指定してください。

```bash
uv run detr evaluate \
  --checkpoint output/demo/checkpoint.pt \
  --image-dir data/demo/val2017 \
  --annotations data/demo/annotations/instances_val2017.json \
  --output output/demo/evaluation.json
```

保存された前処理を使い、ランダム拡張・勾配計算・重み更新なしで検証損失と
pycocotoolsの標準bbox評価を計算します。学習時と同じカテゴリ一覧が必要です。
APはIoU 0.50〜0.95の平均、AP50/AP75はそれぞれIoU 0.50/0.75でのAP、
AR100は画像あたり最大100検出での平均再現率です。数値は0〜1で、高いほど良い指標です。
正解がなく定義できない指標はCOCO仕様の-1になります。
AP評価ではスコア閾値で予測を事前除去しません。合成データでの評価は、写真に対する
検出性能を示すものではありません。
