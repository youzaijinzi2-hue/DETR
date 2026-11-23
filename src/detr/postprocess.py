"""モデルの予測を、元画像上の座標とクラス確率に変換する。"""

import torch
from torch import Tensor
from torchvision.ops import box_convert

from detr.model import Prediction


@torch.no_grad()
def postprocess(
    outputs: Prediction, image_sizes: Tensor, threshold: float = 0.5
) -> list[dict[str, Tensor]]:
    """画像ごとのscores・labels・boxes(xmin,ymin,xmax,ymax)を返す。

    image_sizesはパディング・リサイズ前の(B, 2)の高さ・幅。
    softmaxの分母にはno-objectも含め、その後物体クラスだけで最大値を選ぶ。
    スコアで絞り込み、降順に並べる。DETRなのでNMSは行わない。
    ボックスはここではクリップしない（画像外の座標を取り得る）。
    """
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1")
    logits, normalized_boxes = outputs["pred_logits"], outputs["pred_boxes"]
    if image_sizes.shape != (logits.shape[0], 2) or (image_sizes <= 0).any():
        raise ValueError("image_sizes must have shape (B, 2) with positive sizes")
    scores, labels = logits.softmax(-1)[..., :-1].max(-1)
    boxes = box_convert(normalized_boxes, in_fmt="cxcywh", out_fmt="xyxy")
    height, width = image_sizes.to(boxes).unbind(-1)
    boxes = boxes * torch.stack((width, height, width, height), -1)[:, None]
    results = []
    for score, label, box in zip(scores, labels, boxes):
        indices = torch.where(score >= threshold)[0]
        indices = indices[score[indices].argsort(descending=True)]
        results.append(
            {
                "scores": score[indices],
                "labels": label[indices],
                "boxes": box[indices],
                "query_indices": indices,
            }
        )
    return results
