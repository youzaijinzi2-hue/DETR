import json

import pytest
import torch
from PIL import Image

from detr.cli import main
from detr.dataset import CocoDetectionDataset
from detr.demo_data import make_demo_data


def test_generates_readable_coco_data_with_matching_class_mapping(tmp_path):
    root = make_demo_data(tmp_path / "demo", train_count=3, val_count=2)
    for split, count in (("train2017", 3), ("val2017", 2)):
        annotation_file = root / "annotations" / f"instances_{split}.json"
        data = json.loads(annotation_file.read_text())
        dataset = CocoDetectionDataset(
            root / split, annotation_file, short_side=64, max_size=96
        )
        assert len(dataset) == len(data["images"]) == count
        assert dataset.class_names == ["rectangle", "ellipse"]
        for index in range(count):
            image, target = dataset[index]
            assert image.shape == (3, 64, 96)
            assert 1 <= len(target["labels"]) <= 3
            assert ((target["boxes"] >= 0) & (target["boxes"] <= 1)).all()
            assert torch.isfinite(image).all()
        for annotation in data["annotations"]:
            x, y, width, height = annotation["bbox"]
            assert 0 <= x < x + width <= 96
            assert 0 <= y < y + height <= 64
            with Image.open(
                root / split / f"{annotation['image_id']:06d}.png"
            ) as image:
                pixel = image.getpixel((x + width // 2, y + height // 2))
                expected = (
                    (220, 40, 40) if annotation["category_id"] == 1 else (40, 80, 220)
                )
                assert pixel == expected


def test_generation_is_reproducible_and_does_not_overwrite(tmp_path):
    first = make_demo_data(tmp_path / "first", train_count=2, val_count=1, seed=123)
    second = make_demo_data(tmp_path / "second", train_count=2, val_count=1, seed=123)
    for path in first.rglob("*"):
        if path.is_file():
            assert path.read_bytes() == (second / path.relative_to(first)).read_bytes()
    with pytest.raises(FileExistsError):
        make_demo_data(first)


def test_cli_creates_requested_split_counts(tmp_path):
    main(
        [
            "make-demo-data",
            "--output-dir",
            str(tmp_path / "demo"),
            "--train-count",
            "2",
            "--val-count",
            "1",
        ]
    )
    assert len(list((tmp_path / "demo" / "train2017").glob("*.png"))) == 2
    assert len(list((tmp_path / "demo" / "val2017").glob("*.png"))) == 1
