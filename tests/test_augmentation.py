import json

import pytest
import torch
from PIL import Image

from detr.augmentation import horizontal_flip
from detr.dataset import CocoDetectionDataset


def test_horizontal_flip_updates_image_and_center_without_modifying_inputs():
    image = torch.arange(18).reshape(3, 2, 3)
    boxes = torch.tensor([[0.2, 0.3, 0.1, 0.2], [0.8, 0.7, 0.4, 0.6]])
    original_image, original_boxes = image.clone(), boxes.clone()

    flipped_image, flipped_boxes = horizontal_flip(image, boxes)

    torch.testing.assert_close(flipped_image, image[:, :, [2, 1, 0]])
    torch.testing.assert_close(
        flipped_boxes,
        torch.tensor(
            [
                [0.8, 0.3, 0.1, 0.2],
                [0.2, 0.7, 0.4, 0.6],
            ]
        ),
    )
    torch.testing.assert_close(image, original_image)
    torch.testing.assert_close(boxes, original_boxes)
    restored_image, restored_boxes = horizontal_flip(flipped_image, flipped_boxes)
    torch.testing.assert_close(restored_image, image)
    torch.testing.assert_close(restored_boxes, boxes)


def test_horizontal_flip_handles_no_objects():
    _, boxes = horizontal_flip(torch.zeros(3, 2, 3), torch.empty(0, 4))
    assert boxes.shape == (0, 4)


@pytest.fixture
def asymmetric_coco(tmp_path):
    image = Image.new("RGB", (40, 20), color=(0, 0, 0))
    image.paste((255, 255, 255), (0, 0, 20, 20))
    image.save(tmp_path / "image.png")
    annotations = tmp_path / "annotations.json"
    annotations.write_text(
        json.dumps(
            {
                "images": [{"id": 1, "file_name": "image.png"}],
                "categories": [{"id": 3, "name": "car"}],
                "annotations": [
                    {"image_id": 1, "category_id": 3, "bbox": [0, 0, 20, 20]}
                ],
            }
        )
    )
    return tmp_path, annotations


def test_training_flips_both_image_and_annotation(asymmetric_coco):
    validation = CocoDetectionDataset(*asymmetric_coco, short_side=20, max_size=64)
    training = CocoDetectionDataset(
        *asymmetric_coco,
        training=True,
        train_scales=(20,),
        max_size=64,
        flip_probability=1,
    )

    original, original_target = validation[0]
    flipped, flipped_target = training[0]

    torch.testing.assert_close(flipped, original.flip(-1))
    torch.testing.assert_close(
        flipped_target["boxes"], torch.tensor([[0.75, 0.5, 0.5, 1.0]])
    )
    torch.testing.assert_close(flipped_target["labels"], original_target["labels"])
    torch.testing.assert_close(flipped_target["orig_size"], torch.tensor([20, 40]))


def test_multiscale_changes_size_without_changing_relative_boxes(asymmetric_coco):
    dataset = CocoDetectionDataset(
        *asymmetric_coco,
        training=True,
        train_scales=(16, 32),
        max_size=64,
        flip_probability=0,
    )
    torch.manual_seed(19)
    sizes = set()
    for _ in range(16):
        image, target = dataset[0]
        sizes.add(tuple(image.shape[-2:]))
        torch.testing.assert_close(
            target["boxes"], torch.tensor([[0.25, 0.5, 0.5, 1.0]])
        )
        assert target["size"].tolist() == list(image.shape[-2:])
    assert sizes == {(16, 32), (32, 64)}


def test_validation_ignores_random_augmentation_settings(asymmetric_coco):
    dataset = CocoDetectionDataset(
        *asymmetric_coco,
        short_side=20,
        max_size=64,
        training=False,
        train_scales=(16, 32),
        flip_probability=1,
    )
    first_image, first_target = dataset[0]
    second_image, second_target = dataset[0]
    torch.testing.assert_close(first_image, second_image)
    torch.testing.assert_close(first_target["boxes"], second_target["boxes"])
    torch.testing.assert_close(
        first_target["boxes"], torch.tensor([[0.25, 0.5, 0.5, 1.0]])
    )
    assert first_image.shape == (3, 20, 40)


def test_seed_reproduces_augmentation(asymmetric_coco):
    dataset = CocoDetectionDataset(
        *asymmetric_coco, training=True, train_scales=(16, 32), max_size=64
    )
    torch.manual_seed(20)
    first_image, first_target = dataset[0]
    torch.manual_seed(20)
    second_image, second_target = dataset[0]
    torch.testing.assert_close(first_image, second_image)
    torch.testing.assert_close(first_target["boxes"], second_target["boxes"])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"train_scales": ()},
        {"train_scales": (0,)},
        {"flip_probability": -0.1},
        {"flip_probability": 1.1},
    ],
)
def test_rejects_invalid_augmentation_settings(asymmetric_coco, kwargs):
    with pytest.raises(ValueError):
        CocoDetectionDataset(*asymmetric_coco, **kwargs)
