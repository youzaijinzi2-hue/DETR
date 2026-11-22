"""固定の2次元sin/cos位置エンコーディング（論文3.2節・付録A.1）。"""

import math

import torch
from torch import Tensor, nn


class PositionEmbeddingSine(nn.Module):
    """各位置の縦・横座標を、学習不要なsin/cosベクトルに変換する。

    公式実装のnormalize=Trueに対応する。マスクから有効領域の座標を数え、
    各軸を0〜2πに正規化する詳細は、公式models/position_encoding.pyに従う。
    出力はInputProjectionと同じ (B, H*W, d_model) の位置順で返す。
    """

    def __init__(self, d_model: int = 256) -> None:
        super().__init__()
        if d_model <= 0 or d_model % 4 != 0:
            raise ValueError("d_model must be a positive multiple of 4")
        self.d_model = d_model

    def forward(self, padding_mask: Tensor) -> Tensor:
        """(B, H, W) のboolマスクからfloat32の位置ベクトルを作る。

        前半のチャネルが縦位置、後半が横位置。
        パディング位置にも値はできるため、Attentionには別途マスクを渡す。
        この段階では画像特徴への加算は行わない。
        """
        if padding_mask.ndim != 3 or padding_mask.dtype != torch.bool:
            raise ValueError("padding_mask must be a bool tensor of shape (B, H, W)")
        valid = ~padding_mask
        y = valid.cumsum(dim=1, dtype=torch.float32)
        x = valid.cumsum(dim=2, dtype=torch.float32)
        # ゼロ除算を避け、バッチ化で追加した領域を座標の分母に含めない。
        y = y / (y[:, -1:, :] + 1e-6) * (2 * math.pi)
        x = x / (x[:, :, -1:] + 1e-6) * (2 * math.pi)

        axis_dim = self.d_model // 2
        index = torch.arange(axis_dim, dtype=torch.float32, device=padding_mask.device)
        frequencies = 10000 ** (2 * (index // 2) / axis_dim)

        def encode(coordinates: Tensor) -> Tensor:
            angles = coordinates.unsqueeze(-1) / frequencies
            # sin, cos, sin, cos, ... の順で異なる周波数を並べる。
            return torch.stack(
                (angles[..., 0::2].sin(), angles[..., 1::2].cos()), dim=-1
            ).flatten(-2)

        positions = torch.cat((encode(y), encode(x)), dim=-1)  # (B, H, W, d)
        return positions.flatten(1, 2)  # (B, H*W, d)
