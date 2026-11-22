"""論文3.2節のObject QueriesとTransformer Decoderの1層。"""

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class ObjectQueries(nn.Module):
    """物体候補ごとの学習可能な位置ベクトルと、初期特徴を用意する。

    Object Queriesは画像上の固定座標や物体クラスではない。
    同じ学習可能なベクトルをバッチ内の各画像で使う。
    """

    def __init__(self, num_queries: int = 100, d_model: int = 256) -> None:
        super().__init__()
        if num_queries <= 0:
            raise ValueError("num_queries must be positive")
        self.embedding = nn.Embedding(num_queries, d_model)

    def forward(self, batch_size: int) -> tuple[Tensor, Tensor]:
        """(初期特徴, query位置ベクトル)を、ともに(B, N, d)で返す。"""
        query_positions = self.embedding.weight.unsqueeze(0).expand(batch_size, -1, -1)
        # 学習するquery位置ベクトルと、ゼロで開始する候補の内容を分ける。
        targets = torch.zeros_like(query_positions)
        return targets, query_positions


class TransformerDecoderLayer(nn.Module):
    """Self-Attention → Cross-Attention → FFNのpost-norm Decoder層。"""

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
        self.cross_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.ffn_dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(
        self,
        targets: Tensor,
        query_positions: Tensor,
        memory: Tensor,
        positions: Tensor,
        padding_mask: Tensor,
    ) -> Tensor:
        """候補特徴(B, N, d)を、画像特徴memory(B, L, d)で更新する。

        query_positions: (B, N, d)の学習可能なObject Queries。
        positions: (B, L, d)の画像側の位置エンコーディング。
        padding_mask: (B, L)のbool。Trueの画像領域を参照から除外する。
        各画像には最低1つの有効位置が必要。
        全候補を並列に更新するので、未来位置を隠す因果マスクは使わない。
        """
        query_and_key = targets + query_positions
        attended, _ = self.self_attn(
            query=query_and_key,
            key=query_and_key,
            value=targets,
            need_weights=False,
        )
        targets = self.norm1(targets + self.dropout1(attended))

        attended, _ = self.cross_attn(
            query=targets + query_positions,
            key=memory + positions,
            value=memory,
            key_padding_mask=padding_mask,
            need_weights=False,
        )
        targets = self.norm2(targets + self.dropout2(attended))

        transformed = self.linear2(self.ffn_dropout(F.relu(self.linear1(targets))))
        return self.norm3(targets + self.dropout3(transformed))
