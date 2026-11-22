"""論文3.2節・付録A.1のTransformer Encoder層。"""

from torch import Tensor, nn
from torch.nn import functional as F


class TransformerEncoderLayer(nn.Module):
    """位置情報をQuery・Keyに加える、post-norm構成のEncoder 1層。

    各Attentionヘッド内のQ/K/Vへの線形変換はMultiheadAttentionが行う。
    入出力の系列は (B, L, d_model)。Lは特徴マップのH*W。
    """

    def __init__(
        self,
        d_model: int = 256,
        nhead: int = 8,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.ffn_dropout = nn.Dropout(dropout)
        self.attn_dropout = nn.Dropout(dropout)
        self.output_dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(
        self, tokens: Tensor, positions: Tensor, padding_mask: Tensor
    ) -> Tensor:
        """特徴・位置情報・マスクを受け取り、周囲を参照した特徴を返す。

        tokens, positions: (B, L, d_model)。
        padding_mask: (B, L) のbool配列。Trueの位置をKey/Valueから除外する。
        各画像には最低1つの有効位置が必要。マスクされた位置の出力を
        ゼロにする処理ではないため、次の層にも同じマスクを渡す。
        """
        query_and_key = tokens + positions
        attended, _ = self.self_attn(
            query=query_and_key,
            key=query_and_key,
            value=tokens,
            key_padding_mask=padding_mask,
            need_weights=False,
        )
        tokens = self.norm1(tokens + self.attn_dropout(attended))

        transformed = self.linear2(self.ffn_dropout(F.relu(self.linear1(tokens))))
        return self.norm2(tokens + self.output_dropout(transformed))
