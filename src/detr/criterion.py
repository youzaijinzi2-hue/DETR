"""論文3.1節の集合予測損失と、3.2節の補助損失（単一プロセス用）。"""

import math
from collections.abc import Mapping, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torchvision.ops import box_convert, generalized_box_iou

from detr.matcher import HungarianMatcher
from detr.model import DETROutput, Prediction


class SetCriterion(nn.Module):
    """各層でマッチングし、分類・L1・GIoUと重み付き総損失を返す。

    分類は全候補を対象に重み付き平均。ボックスは対応した候補のみを
    対象に、バッチ内の正解物体数で割る。分散学習での集計は未対応。
    """

    def __init__(
        self,
        num_classes: int,
        matcher: HungarianMatcher | None = None,
        *,
        eos_coef: float = 0.1,
        loss_class: float = 1,
        loss_bbox: float = 5,
        loss_giou: float = 2,
    ) -> None:
        super().__init__()
        if num_classes <= 0:
            raise ValueError("num_classes must be positive")
        if not math.isfinite(eos_coef) or eos_coef <= 0:
            raise ValueError("eos_coef must be finite and positive")
        weights = (loss_class, loss_bbox, loss_giou)
        if any(not math.isfinite(w) or w < 0 for w in weights) or not any(weights):
            raise ValueError(
                "loss weights must be finite, non-negative, and not all zero"
            )
        self.num_classes = num_classes
        self.matcher = matcher if matcher is not None else HungarianMatcher()
        self.weight_dict = dict(zip(("loss_ce", "loss_bbox", "loss_giou"), weights))
        class_weights = torch.ones(num_classes + 1)
        class_weights[-1] = eos_coef
        # パラメータではないが、criterion.to(device)で一緒に移動させる。
        self.register_buffer("class_weights", class_weights)

    def _layer_losses(
        self,
        prediction: Prediction,
        targets: Sequence[Mapping[str, Tensor]],
        num_boxes: int,
    ) -> dict[str, Tensor]:
        logits, boxes = prediction["pred_logits"], prediction["pred_boxes"]
        if logits.shape[-1] != self.num_classes + 1:
            raise ValueError("prediction class count must match num_classes + 1")
        indices = self.matcher(prediction, targets)
        # 未対応の候補はすべてno-object。対応したところだけ正解クラスにする。
        target_classes = torch.full(
            logits.shape[:2], self.num_classes, dtype=torch.int64, device=logits.device
        )
        matched_predictions = []
        matched_targets = []
        for image, (pred_index, target_index) in enumerate(indices):
            if len(pred_index) == 0:
                continue
            pred_index = pred_index.to(device=boxes.device)
            labels = targets[image]["labels"].to(device=logits.device)
            target_boxes = targets[image]["boxes"].to(boxes)
            target_index = target_index.to(device=boxes.device)
            target_classes[image, pred_index] = labels[target_index]
            matched_predictions.append(boxes[image, pred_index])
            matched_targets.append(target_boxes[target_index])

        loss_ce = F.cross_entropy(
            logits.transpose(1, 2),
            target_classes,
            weight=self.class_weights.to(logits),
        )
        if not matched_predictions:
            # 空バッチでもゼロのボックス損失からbackwardできるようにする。
            loss_bbox = boxes.sum() * 0
            loss_giou = boxes.sum() * 0
        else:
            predicted = torch.cat(matched_predictions)
            expected = torch.cat(matched_targets)
            loss_bbox = F.l1_loss(predicted, expected, reduction="sum") / num_boxes
            giou = generalized_box_iou(
                box_convert(predicted, in_fmt="cxcywh", out_fmt="xyxy"),
                box_convert(expected, in_fmt="cxcywh", out_fmt="xyxy"),
            )
            # 対応済みなので対角だけを使う。他の正解との組は損失に含めない。
            loss_giou = (1 - giou.diag()).sum() / num_boxes
        return {"loss_ce": loss_ce, "loss_bbox": loss_bbox, "loss_giou": loss_giou}

    def forward(
        self, outputs: DETROutput, targets: Sequence[Mapping[str, Tensor]]
    ) -> dict[str, Tensor]:
        """loss_totalをbackwardする。ほかのキーは未加重の各損失。

        中間層のキーには_0、_1…を付ける。各層は独立にマッチングする。
        loss_totalは最終層と全中間層の重み付き損失を合計したもの。
        辞書の全値を足すと二重計上になるため、総損失にはloss_totalだけを使う。
        """
        num_boxes = max(sum(len(target["labels"]) for target in targets), 1)
        final_prediction: Prediction = {
            "pred_logits": outputs["pred_logits"],
            "pred_boxes": outputs["pred_boxes"],
        }
        losses = self._layer_losses(final_prediction, targets, num_boxes)
        total = sum(self.weight_dict[name] * value for name, value in losses.items())
        for index, prediction in enumerate(outputs.get("aux_outputs", [])):
            auxiliary = self._layer_losses(prediction, targets, num_boxes)
            total = total + sum(
                self.weight_dict[name] * value for name, value in auxiliary.items()
            )
            losses.update(
                {f"{name}_{index}": value for name, value in auxiliary.items()}
            )
        losses["loss_total"] = total
        return losses
