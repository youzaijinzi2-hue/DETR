"""論文3.2節のCNN Backbone: ResNet-50の最後の特徴マップを使う。"""

from collections import OrderedDict

from torch import Tensor, nn
from torch.nn import functional as F
from torchvision.models import ResNet50_Weights, resnet50
from torchvision.ops import FrozenBatchNorm2d


class ResNet50Backbone(nn.Module):
    """画像バッチを2048チャネル・出力ストライド32の特徴に変換する。

    pretrained=TrueではImageNet学習済み重みを取得する（初回は通信が必要）。
    Falseはランダム初期化で、ネットワークの動作確認に使う。
    論文の学習設定に合わせ、BatchNormの統計量・係数は固定する。
    CNNの畳み込み層は学習可能。初期層の凍結などの学習方針は別途設定する。
    """

    num_channels = 2048

    def __init__(self, *, pretrained: bool = False) -> None:
        super().__init__()
        weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        resnet = resnet50(weights=weights, norm_layer=FrozenBatchNorm2d)
        # avgpoolとfcを除き、空間情報を持つlayer4の出力を取り出す。
        self.body = nn.Sequential(
            OrderedDict(
                (name, layer)
                for name, layer in resnet.named_children()
                if name not in {"avgpool", "fc"}
            )
        )

    def forward(self, pixels: Tensor, padding_mask: Tensor) -> tuple[Tensor, Tensor]:
        """batch_imagesの出力を受け取り、特徴と対応するマスクを返す。

        pixels: (B, 3, H, W) の浮動小数点画像。学習済み重みを使う際は、
            [0, 1]画像をImageNetのmean/stdで標準化してからバッチ化する。
            このクラス自体はリサイズや画素値の標準化を行わない。
        padding_mask: (B, H, W) のboolテンソル。Trueがパディング。
        戻り値: (B, 2048, ceil(H/32), ceil(W/32)) の特徴と、
            (B, ceil(H/32), ceil(W/32)) のboolマスク。
        """
        features = self.body(pixels)
        # 最近傍補間で、True/Falseの領域を特徴マップの解像度に合わせる。
        feature_mask = F.interpolate(
            padding_mask[:, None].float(),
            size=features.shape[-2:],
            mode="nearest",
        )[:, 0].bool()
        return features, feature_mask
