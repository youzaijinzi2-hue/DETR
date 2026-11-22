"""DETRの正解ボックスを正規化する処理のテスト。"""

import pytest
import torch

from detr.boxes import normalize_boxes_xyxy


@pytest.mark.parametrize("dtype", [torch.int64, torch.float32, torch.float64])
def test_normalizes_boxes_on_non_square_image(dtype):
    # 高さ100・幅200。非対称の座標で、xyや幅・高さの取り違えも検出する。
    boxes = torch.tensor(
        [[20, 10, 100, 50], [0, 0, 200, 100], [150, 60, 190, 100]],
        dtype=dtype,
    )
    original = boxes.clone()

    result = normalize_boxes_xyxy(boxes, image_size=(100, 200))

    expected_dtype = dtype if dtype.is_floating_point else torch.float32
    expected = torch.tensor(
        [[0.3, 0.3, 0.4, 0.4], [0.5, 0.5, 1.0, 1.0], [0.85, 0.8, 0.2, 0.4]],
        dtype=expected_dtype,
    )
    torch.testing.assert_close(result, expected)
    torch.testing.assert_close(boxes, original)
    assert result.device == boxes.device


def test_preserves_fractional_coordinates():
    boxes = torch.tensor([[0.5, 1.5, 4.5, 5.5]])

    result = normalize_boxes_xyxy(boxes, image_size=(10, 20))

    torch.testing.assert_close(result, torch.tensor([[0.125, 0.35, 0.2, 0.4]]))


def test_same_relative_box_is_independent_of_image_size():
    boxes = torch.tensor([[20.0, 10.0, 100.0, 50.0]])

    original = normalize_boxes_xyxy(boxes, image_size=(100, 200))
    resized = normalize_boxes_xyxy(boxes * 2, image_size=(200, 400))

    torch.testing.assert_close(original, resized)


def test_image_without_objects_returns_empty_boxes():
    boxes = torch.empty((0, 4), dtype=torch.float64)

    result = normalize_boxes_xyxy(boxes, image_size=(100, 200))

    torch.testing.assert_close(result, torch.empty((0, 4), dtype=torch.float64))


@pytest.mark.parametrize("shape", [(), (4,), (2, 3), (2, 5), (1, 2, 4), (0,)])
def test_rejects_invalid_box_shape(shape):
    with pytest.raises(ValueError, match=r"boxes must have shape \(N, 4\)"):
        normalize_boxes_xyxy(torch.empty(shape), image_size=(100, 200))


@pytest.mark.parametrize("image_size", [(0, 200), (100, 0), (-1, 200), (100, -1)])
def test_rejects_non_positive_image_size(image_size):
    boxes = torch.tensor([[20, 10, 100, 50]])

    with pytest.raises(ValueError, match="image height and width must be positive"):
        normalize_boxes_xyxy(boxes, image_size=image_size)
