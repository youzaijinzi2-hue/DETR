"""正解とのIoUと、全Object Queryのクラス確率を確認する。"""

import json
from pathlib import Path

import torch
from torchvision.ops import box_iou

from detr.model import Prediction
from detr.postprocess import postprocess


def read_ground_truth(
    annotation_file: str | Path, image_path: str | Path, image_id: int | None = None
) -> list[dict]:
    """対象画像の非crowd・正の面積の正解をピクセルxyxyで返す。"""
    data = json.loads(Path(annotation_file).read_text(encoding="utf-8"))
    candidates = [
        record
        for record in data["images"]
        if (
            record["id"] == image_id
            if image_id is not None
            else Path(record["file_name"]).name == Path(image_path).name
        )
    ]
    if len(candidates) != 1:
        raise ValueError(
            "image must match exactly one annotation record; specify --image-id"
        )
    selected = candidates[0]["id"]
    names = {category["id"]: category["name"] for category in data["categories"]}
    targets = []
    for annotation in data["annotations"]:
        if annotation["image_id"] != selected or annotation.get("iscrowd", 0):
            continue
        x, y, width, height = annotation["bbox"]
        if width <= 0 or height <= 0:
            continue
        targets.append(
            {
                "category_id": annotation["category_id"],
                "class_name": names[annotation["category_id"]],
                "box_xyxy": [x, y, x + width, y + height],
            }
        )
    return targets


@torch.no_grad()
def prediction_diagnostics(
    outputs: Prediction,
    image_size: tuple[int, int],
    ground_truth: list[dict],
    class_names: list[str],
    label_to_category_id: dict[int, int],
) -> dict:
    """1枚分の全候補を調べる。IoUは同クラスの正解との最大値（非一対一）。"""
    result = postprocess(outputs, torch.tensor([image_size]), threshold=0)[0]
    probabilities = outputs["pred_logits"][0].softmax(-1).cpu()
    target_boxes = torch.tensor(
        [target["box_xyxy"] for target in ground_truth], dtype=torch.float32
    ).reshape(-1, 4)
    ious = box_iou(result["boxes"].cpu(), target_boxes)
    queries = []
    for row, query_index in enumerate(result["query_indices"].tolist()):
        label = result["labels"][row].item()
        category_id = label_to_category_id[label]
        candidates = [
            i
            for i, target in enumerate(ground_truth)
            if target["category_id"] == category_id
        ]
        best_target = (
            max(candidates, key=lambda i: ious[row, i].item()) if candidates else None
        )
        queries.append(
            {
                "query_index": query_index,
                "class_name": class_names[label],
                "score": result["scores"][row].item(),
                "no_object_probability": probabilities[query_index, -1].item(),
                "class_probabilities": probabilities[query_index, :-1].tolist(),
                "box_xyxy": result["boxes"][row].tolist(),
                "best_same_class_gt_index": best_target,
                "best_same_class_gt_iou": ious[row, best_target].item()
                if best_target is not None
                else None,
            }
        )
    return {
        "ground_truth": ground_truth,
        "class_names": class_names,
        "queries": queries,
    }
