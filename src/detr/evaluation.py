"""検証損失とpycocotoolsによるCOCO bbox APの評価。"""

import copy
import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torch import Tensor, nn

from detr.postprocess import postprocess


def coco_metrics(
    annotation_file: str | Path, detections: list[dict], image_ids: list[int]
) -> dict[str, float]:
    """元のcategory_idとCOCO xywh座標を使って標準APを計算する。

    値は0〜1。評価対象に正解がない等、定義できない指標はCOCO仕様の-1。
    空の予測リストも扱い、指定した画像（検出ゼロの画像も含む）を評価する。
    """
    with Path(annotation_file).open(encoding="utf-8") as file:
        data = json.load(file)
    for index, annotation in enumerate(data["annotations"], start=1):
        annotation.setdefault("id", index)
        annotation.setdefault("iscrowd", 0)
        annotation.setdefault("area", annotation["bbox"][2] * annotation["bbox"][3])
    ground_truth = COCO()
    ground_truth.dataset = data
    ground_truth.createIndex()
    if detections:
        predictions = ground_truth.loadRes(detections)
    else:
        # loadResは空リストを受け付けないため、注釈ゼロのCOCOを直接作る。
        predictions = COCO()
        predictions.dataset = {
            "images": copy.deepcopy(data["images"]),
            "categories": copy.deepcopy(data["categories"]),
            "annotations": [],
        }
        predictions.createIndex()
    evaluator = COCOeval(ground_truth, predictions, iouType="bbox")
    evaluator.params.imgIds = sorted(set(image_ids))
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    return {
        "AP": float(evaluator.stats[0]),
        "AP50": float(evaluator.stats[1]),
        "AP75": float(evaluator.stats[2]),
        "AR100": float(evaluator.stats[8]),
    }


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    criterion: nn.Module,
    data_loader: Iterable[tuple[Sequence[Tensor], Sequence[Mapping[str, Tensor]]]],
    *,
    annotation_file: str | Path,
    label_to_category_id: dict[int, int],
) -> dict[str, float]:
    """重みを更新せず、バッチ平均の損失とCOCO APを返す。

    AP用の予測はスコアで事前に絞り込まず、COCO側の上位100件設定に従う。
    評価用DataLoaderではランダム拡張とdrop_lastを無効にする。
    """
    device = next(model.parameters()).device
    model_was_training, criterion_was_training = model.training, criterion.training
    model.eval()
    criterion.to(device).eval()
    totals: dict[str, float] = {}
    detections = []
    image_ids = []
    num_batches = 0
    try:
        for images, targets in data_loader:
            images = [image.to(device) for image in images]
            targets = [
                {key: value.to(device) for key, value in target.items()}
                for target in targets
            ]
            outputs = model(images)
            losses = criterion(outputs, targets)
            for name, value in losses.items():
                totals[name] = totals.get(name, 0.0) + value.item()
            sizes = torch.stack([target["orig_size"] for target in targets])
            results = postprocess(outputs, sizes, threshold=0)
            for target, result in zip(targets, results):
                image_id = target["image_id"].item()
                image_ids.append(image_id)
                for box, score, label in zip(
                    result["boxes"].cpu(),
                    result["scores"].cpu(),
                    result["labels"].cpu(),
                ):
                    x1, y1, x2, y2 = box.tolist()
                    detections.append(
                        {
                            "image_id": image_id,
                            "category_id": label_to_category_id[label.item()],
                            "bbox": [x1, y1, x2 - x1, y2 - y1],
                            "score": score.item(),
                        }
                    )
            num_batches += 1
        if num_batches == 0:
            raise ValueError("evaluation data_loader must not be empty")
        metrics = {name: value / num_batches for name, value in totals.items()}
        metrics.update(coco_metrics(annotation_file, detections, image_ids))
        return metrics
    finally:
        model.train(model_was_training)
        criterion.train(criterion_was_training)
