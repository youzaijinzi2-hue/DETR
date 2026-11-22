"""論文3.2節のクラス予測とボックス予測。"""

from torch import Tensor, nn


class PredictionHead(nn.Module):
    """全候補・全Decoder層で同じ重みを使って予測する。

    クラスIDは0〜num_classes-1。最後のID=num_classesがno-object。
    ボックスは画像サイズに対して正規化した(cx, cy, w, h)を出力する。
    """

    def __init__(self, num_classes: int, d_model: int = 256) -> None:
        super().__init__()
        if num_classes <= 0:
            raise ValueError("num_classes must be positive")
        self.class_embed = nn.Linear(d_model, num_classes + 1)
        self.bbox_embed = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, 4),
        )

    def forward(self, features: Tensor) -> dict[str, Tensor]:
        """候補特徴をクラスlogitsと正規化ボックスに変換する。

        featuresは(B, N, d)、または全層分の(L, B, N, d)。
        先行する次元は保持し、末尾だけをクラス数+1、または4に変える。
        pred_logitsはsoftmax前の値（学習時のcross_entropyに渡すため）。
        pred_boxesにはsigmoidを適用して各成分を0〜1に収める。
        ボックスの角が画像内に収まることまでは保証しない。
        """
        return {
            "pred_logits": self.class_embed(features),
            "pred_boxes": self.bbox_embed(features).sigmoid(),
        }
