# DETRをステップごとに実装する

ResNet-50、Encoder/Decoder、Object Queries、予測ヘッド、Hungarian Matching、
分類・L1・GIoU損失、COCO形式Datasetと学習・推論コマンドを実装しています。

## 準備

```bash
uv sync
uv run pytest -q
```

COCO形式の画像と正解JSONを用意します。データは自動ダウンロードしません。

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

- `output/train/checkpoint.pt`：最新のモデル、構成、Optimizer状態、クラス対応、前処理設定
- `output/train/metrics.jsonl`：エポックごとの学習損失。検証精度・mAPではありません

現在は単一プロセスでの基本的な学習です。学習率スケジュール、学習再開CLI、
検証・COCO AP、ランダムクロップは未実装で、論文の学習設定の完全再現ではありません。
同じ出力先で再実行すると既存のログとチェックポイントを置き換えます。

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
