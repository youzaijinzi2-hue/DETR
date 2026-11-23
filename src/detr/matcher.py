"""論文3.1節の、予測と正解の一対一マッチング。"""

import math
from collections.abc import Mapping, Sequence

import torch
from scipy.optimize import linear_sum_assignment
from torch import Tensor, nn
from torchvision.ops import box_convert, generalized_box_iou


class HungarianMatcher(nn.Module):
    """全体のコスト合計が最小になる予測・正解の対応を求める。

    コストは -クラス確率 + L1距離 - GIoU の重み付き和。
    クラスの学習損失とは異なり、ここでは-log確率を使わない。
    既定の重みはクラス1、L1が5、GIoUが2。
    """

    def __init__(
        self, cost_class: float = 1, cost_bbox: float = 5, cost_giou: float = 2
    ) -> None:
        super().__init__()
        weights = (cost_class, cost_bbox, cost_giou)
        if any(not math.isfinite(w) or w < 0 for w in weights) or not any(weights):
            raise ValueError(
                "cost weights must be finite, non-negative, and not all zero"
            )
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou

    @torch.no_grad()
    def forward(
        self, outputs: Mapping[str, Tensor], targets: Sequence[Mapping[str, Tensor]]
    ) -> list[tuple[Tensor, Tensor]]:
        """画像ごとに(予測インデックス, 正解インデックス)を返す。

        outputsは最終層など1層分のpred_logits(B, N, C+1)、pred_boxes(B, N, 4)。
        targetsは画像ごとの辞書。labels(M,)は0〜C-1のint64、
        boxes(M, 4)は正規化cxcywh（幅・高さは正）を渡す。
        正解にno-objectは含めない。各画像の正解数は候補数以下とする。
        戻り値のインデックスはCPU上のint64。未選択の候補は後の損失計算で
        no-objectとして扱う。この処理自体には勾配を流さない。
        """
        logits, boxes = outputs["pred_logits"], outputs["pred_boxes"]
        if len(targets) != logits.shape[0]:
            raise ValueError("targets length must match batch size")
        probabilities = logits.softmax(dim=-1)
        matches = []
        # 学習しやすいよう画像ごとに処理する。画像間の対応は作らない。
        for index, target in enumerate(targets):
            labels = target["labels"].to(device=logits.device)
            target_boxes = target["boxes"].to(device=boxes.device, dtype=boxes.dtype)
            if labels.ndim != 1 or labels.dtype != torch.int64:
                raise ValueError("labels must be a one-dimensional int64 tensor")
            if target_boxes.shape != (len(labels), 4):
                raise ValueError("target boxes must have shape (number of labels, 4)")
            if len(labels) > logits.shape[1]:
                raise ValueError("number of targets must not exceed number of queries")
            if ((labels < 0) | (labels >= logits.shape[-1] - 1)).any():
                raise ValueError("target labels must exclude the no-object class")
            if len(labels) == 0:
                matches.append(
                    (
                        torch.empty(0, dtype=torch.int64),
                        torch.empty(0, dtype=torch.int64),
                    )
                )
                continue

            class_cost = -probabilities[index][:, labels]
            bbox_cost = torch.cdist(boxes[index], target_boxes, p=1)
            giou_cost = -generalized_box_iou(
                box_convert(boxes[index], in_fmt="cxcywh", out_fmt="xyxy"),
                box_convert(target_boxes, in_fmt="cxcywh", out_fmt="xyxy"),
            )
            cost = (
                self.cost_class * class_cost
                + self.cost_bbox * bbox_cost
                + self.cost_giou * giou_cost
            )
            prediction_indices, target_indices = linear_sum_assignment(
                cost.cpu().numpy()
            )
            matches.append(
                (
                    torch.as_tensor(prediction_indices, dtype=torch.int64),
                    torch.as_tensor(target_indices, dtype=torch.int64),
                )
            )
        return matches
