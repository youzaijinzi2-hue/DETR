"""論文3.2節: CNNの特徴をTransformerへ渡す系列に変換する。"""

from torch import Tensor, nn


class InputProjection(nn.Module):
    """1×1畳み込みでチャネルを減らし、空間方向を1列に並べる。

    論文では d×HW と表記される系列を、この実装では
    batch_first=TrueのAttentionで使う (B, HW, d) の順で返す。
    位置エンコーディングとEncoder本体は、次のステップで追加する。
    """

    def __init__(self, in_channels: int = 2048, d_model: int = 256) -> None:
        super().__init__()
        # 各位置の特徴ベクトルを独立に変換する。高さ・幅は変わらない。
        self.proj = nn.Conv2d(in_channels, d_model, kernel_size=1)

    def forward(self, features: Tensor, padding_mask: Tensor) -> tuple[Tensor, Tensor]:
        """Backboneの出力から、系列と対応するパディングマスクを作る。

        Args:
            features: (B, in_channels, H, W) の特徴マップ。
            padding_mask: (B, H, W) のboolテンソル。Trueは無視する領域。

        Returns:
            tokens: (B, H*W, d_model)。左上から行ごとに並べた特徴。
            token_mask: (B, H*W)。tokensと同じ位置順のマスク。
        """
        projected = self.proj(features)  # (B, d_model, H, W)
        tokens = projected.flatten(2).transpose(1, 2)  # (B, H*W, d_model)
        token_mask = padding_mask.flatten(1)  # (B, H*W)
        return tokens, token_mask
