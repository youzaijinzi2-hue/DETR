import json

import pytest
import torch

from detr.diagnostics import prediction_diagnostics, read_ground_truth


def test_diagnostics_preserves_query_ids_probabilities_and_same_class_iou():
    outputs = {
        "pred_logits": torch.tensor([[[0.7, 0.1, 0.2], [0.1, 0.8, 0.1]]]).log(),
        "pred_boxes": torch.tensor([[[0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 1.0, 1.0]]]),
    }
    ground_truth = [
        {"category_id": 42, "class_name": "car", "box_xyxy": [0.0, 0.0, 200.0, 100.0]}
    ]

    report = prediction_diagnostics(
        outputs, (100, 200), ground_truth, ["cat", "car"], {0: 17, 1: 42}
    )

    first, second = report["queries"]
    assert first["query_index"] == 1
    assert first["score"] == pytest.approx(0.8)
    assert first["no_object_probability"] == pytest.approx(0.1)
    assert first["best_same_class_gt_iou"] == pytest.approx(1)
    assert first["best_same_class_gt_index"] == 0
    assert second["query_index"] == 0
    assert second["no_object_probability"] == pytest.approx(0.2)
    assert second["class_probabilities"] == pytest.approx([0.7, 0.1])
    assert second["best_same_class_gt_iou"] is None


def test_empty_ground_truth_produces_no_iou_match():
    outputs = {
        "pred_logits": torch.zeros(1, 1, 2),
        "pred_boxes": torch.full((1, 1, 4), 0.5),
    }
    report = prediction_diagnostics(outputs, (100, 200), [], ["cat"], {0: 1})
    assert report["queries"][0]["best_same_class_gt_iou"] is None


def test_ground_truth_selects_image_and_converts_xywh(tmp_path):
    path = tmp_path / "annotations.json"
    path.write_text(
        json.dumps(
            {
                "images": [
                    {"id": 1, "file_name": "a/image.png"},
                    {"id": 2, "file_name": "b/image.png"},
                ],
                "categories": [{"id": 17, "name": "cat"}],
                "annotations": [
                    {"image_id": 2, "category_id": 17, "bbox": [10, 20, 30, 40]},
                    {
                        "image_id": 2,
                        "category_id": 17,
                        "bbox": [0, 0, 30, 40],
                        "iscrowd": 1,
                    },
                ],
            }
        )
    )
    with pytest.raises(ValueError, match="specify --image-id"):
        read_ground_truth(path, "image.png")
    targets = read_ground_truth(path, "image.png", image_id=2)
    assert targets == [
        {"category_id": 17, "class_name": "cat", "box_xyxy": [10, 20, 40, 60]}
    ]
