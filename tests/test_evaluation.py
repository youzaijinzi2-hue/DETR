import json

import pytest
import torch
from torch.utils.data import DataLoader

from detr.criterion import SetCriterion
from detr.dataset import CocoDetectionDataset
from detr.demo_data import make_demo_data
from detr.engine import collate_fn
from detr.evaluation import coco_metrics, evaluate_model
from detr.model import DETR


@pytest.fixture
def ground_truth(tmp_path):
    path = tmp_path / "annotations.json"
    path.write_text(
        json.dumps(
            {
                "images": [
                    {"id": 1, "width": 100, "height": 100},
                    {"id": 2, "width": 100, "height": 100},
                ],
                "categories": [{"id": 42, "name": "box"}],
                "annotations": [
                    {
                        "id": 1,
                        "image_id": 1,
                        "category_id": 42,
                        "bbox": [10, 20, 30, 40],
                        "area": 1200,
                        "iscrowd": 0,
                    }
                ],
            }
        )
    )
    return path


def test_perfect_predictions_have_ap_one(ground_truth):
    detections = [
        {"image_id": 1, "category_id": 42, "bbox": [10, 20, 30, 40], "score": 0.9}
    ]
    metrics = coco_metrics(ground_truth, detections, [1, 2])
    for name in ("AP", "AP50", "AP75", "AR100"):
        assert metrics[name] == pytest.approx(1)


def test_missed_boxes_have_zero_ap_and_empty_predictions_work(ground_truth):
    wrong = [{"image_id": 1, "category_id": 42, "bbox": [80, 80, 10, 10], "score": 0.9}]
    for detections in (wrong, []):
        metrics = coco_metrics(ground_truth, detections, [1, 2])
        assert metrics["AP"] == 0
        assert metrics["AR100"] == 0


def test_false_positive_on_empty_image_counts_against_ap(ground_truth):
    detections = [
        {"image_id": 2, "category_id": 42, "bbox": [10, 20, 30, 40], "score": 0.99},
        {"image_id": 1, "category_id": 42, "bbox": [10, 20, 30, 40], "score": 0.9},
    ]
    metrics = coco_metrics(ground_truth, detections, [1, 2])
    assert metrics["AP"] == pytest.approx(0.5)


def test_real_evaluation_keeps_weights_and_training_mode(tmp_path):
    root = make_demo_data(tmp_path / "demo", train_count=1, val_count=2)
    annotations = root / "annotations" / "instances_val2017.json"
    dataset = CocoDetectionDataset(
        root / "val2017", annotations, short_side=32, max_size=64
    )
    loader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)
    model = DETR(
        num_classes=2,
        num_queries=3,
        d_model=8,
        nhead=2,
        dim_feedforward=16,
        num_encoder_layers=1,
        num_decoder_layers=1,
    )
    criterion = SetCriterion(2)
    before = {name: value.clone() for name, value in model.state_dict().items()}

    metrics = evaluate_model(
        model,
        criterion,
        loader,
        annotation_file=annotations,
        label_to_category_id=dataset.label_to_category_id,
    )

    assert model.training and criterion.training
    assert metrics["loss_total"] > 0
    assert 0 <= metrics["AP"] <= 1
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name])
    assert all(parameter.grad is None for parameter in model.parameters())
