import json
import math

import pytest
import torch
from PIL import Image
from torch.utils.data import DataLoader

from detr.criterion import SetCriterion
from detr.dataset import CocoDetectionDataset
from detr.engine import collate_fn, train_one_epoch
from detr.model import DETR


@pytest.fixture
def coco_files(tmp_path):
    Image.new("RGB", (40, 20), color=(255, 255, 255)).save(tmp_path / "a.png")
    Image.new("RGB", (20, 40), color=(0, 0, 0)).save(tmp_path / "b.png")
    data = {
        "images": [{"id": 2, "file_name": "b.png"}, {"id": 1, "file_name": "a.png"}],
        "categories": [{"id": 17, "name": "cat"}, {"id": 3, "name": "car"}],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 17, "bbox": [10, 5, 20, 10]},
            {"id": 2, "image_id": 1, "category_id": 3, "bbox": [-5, -5, 15, 15]},
            {"id": 3, "image_id": 1, "category_id": 3, "bbox": [0, 0, 0, 10]},
            {"id": 4, "image_id": 1, "category_id": 3, "bbox": [50, 0, 10, 10]},
            {
                "id": 5,
                "image_id": 1,
                "category_id": 17,
                "bbox": [0, 0, 40, 20],
                "iscrowd": 1,
            },
        ],
    }
    annotation_file = tmp_path / "annotations.json"
    annotation_file.write_text(json.dumps(data), encoding="utf-8")
    return tmp_path, annotation_file


def test_converts_coco_boxes_and_labels_and_removes_invalid_annotations(coco_files):
    dataset = CocoDetectionDataset(*coco_files, short_side=32, max_size=64)

    image, target = dataset[0]

    assert len(dataset) == 2
    assert dataset.num_classes == 2
    assert dataset.class_names == ["car", "cat"]
    assert dataset.category_id_to_label == {3: 0, 17: 1}
    assert dataset.label_to_category_id == {0: 3, 1: 17}
    assert image.shape == (3, 32, 64)
    assert image.dtype == torch.float32
    torch.testing.assert_close(target["labels"], torch.tensor([1, 0]))
    torch.testing.assert_close(
        target["boxes"],
        torch.tensor(
            [
                [0.5, 0.5, 0.5, 0.5],
                [0.125, 0.25, 0.25, 0.5],
            ]
        ),
    )
    torch.testing.assert_close(target["image_id"], torch.tensor([1]))
    torch.testing.assert_close(target["orig_size"], torch.tensor([20, 40]))
    torch.testing.assert_close(target["size"], torch.tensor([32, 64]))
    # 白画像の各チャネルが、ImageNetのmean/stdで標準化される。
    torch.testing.assert_close(
        image[:, 0, 0],
        (torch.ones(3) - torch.tensor([0.485, 0.456, 0.406]))
        / torch.tensor([0.229, 0.224, 0.225]),
    )


def test_empty_image_keeps_correct_shapes_and_max_size_caps_resize(coco_files):
    dataset = CocoDetectionDataset(*coco_files, short_side=100, max_size=64)

    image, target = dataset[1]

    assert image.shape == (3, 64, 32)
    assert target["labels"].shape == (0,)
    assert target["labels"].dtype == torch.int64
    assert target["boxes"].shape == (0, 4)
    assert target["boxes"].dtype == torch.float32


def test_repeated_reads_do_not_mutate_annotations(coco_files):
    dataset = CocoDetectionDataset(*coco_files, short_side=32, max_size=64)
    _, first = dataset[0]
    _, second = dataset[0]
    torch.testing.assert_close(first["boxes"], second["boxes"])


def test_files_to_dataloader_to_real_training_step(coco_files):
    torch.manual_seed(18)
    dataset = CocoDetectionDataset(*coco_files, short_side=32, max_size=64)
    loader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)
    model = DETR(
        num_classes=dataset.num_classes,
        num_queries=3,
        d_model=8,
        nhead=2,
        dim_feedforward=16,
        dropout=0,
        num_encoder_layers=1,
        num_decoder_layers=1,
    )
    criterion = SetCriterion(num_classes=dataset.num_classes)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    before = model.prediction_head.class_embed.weight.detach().clone()

    metrics = train_one_epoch(model, criterion, loader, optimizer)

    assert all(math.isfinite(value) for value in metrics.values())
    assert not torch.equal(before, model.prediction_head.class_embed.weight)


@pytest.mark.parametrize("short_side,max_size", [(0, 64), (32, 0), (-1, 64)])
def test_rejects_non_positive_resize_dimensions(coco_files, short_side, max_size):
    with pytest.raises(ValueError, match="must be positive"):
        CocoDetectionDataset(*coco_files, short_side=short_side, max_size=max_size)
