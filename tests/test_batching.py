import pytest
import torch

from detr.batching import batch_images


def test_pads_right_and_bottom_and_marks_only_padding():
    wide = torch.arange(18, dtype=torch.float64).reshape(3, 2, 3)
    tall = torch.full((3, 4, 1), 7.0, dtype=torch.float64)
    originals = [wide.clone(), tall.clone()]

    pixels, mask = batch_images([wide, tall])

    assert pixels.shape == (2, 3, 4, 3)
    assert pixels.dtype == wide.dtype
    assert pixels.device == mask.device == wide.device
    torch.testing.assert_close(pixels[0, :, :2, :3], wide)
    torch.testing.assert_close(pixels[1, :, :4, :1], tall)
    assert torch.count_nonzero(pixels[0, :, 2:, :]) == 0
    assert torch.count_nonzero(pixels[1, :, :, 1:]) == 0
    torch.testing.assert_close(
        mask,
        torch.tensor(
            [
                [
                    [False, False, False],
                    [False, False, False],
                    [True, True, True],
                    [True, True, True],
                ],
                [
                    [False, True, True],
                    [False, True, True],
                    [False, True, True],
                    [False, True, True],
                ],
            ]
        ),
    )
    for image, original in zip([wide, tall], originals):
        torch.testing.assert_close(image, original)


def test_equal_size_images_need_no_padding_even_for_zero_pixels():
    images = [torch.zeros(3, 2, 4), torch.ones(3, 2, 4)]

    pixels, mask = batch_images(images)

    torch.testing.assert_close(pixels, torch.stack(images))
    assert not mask.any()


def test_single_image():
    image = torch.ones(3, 2, 4)

    pixels, mask = batch_images([image])

    torch.testing.assert_close(pixels, image.unsqueeze(0))
    assert mask.shape == (1, 2, 4)
    assert not mask.any()


@pytest.mark.parametrize(
    ("images", "message"),
    [
        ([], "images must not be empty"),
        ([torch.empty(2, 4)], "each image must have shape"),
        ([torch.empty(1, 2, 4)], "each image must have shape"),
        ([torch.empty(3, 0, 4)], "image height and width must be positive"),
        ([torch.empty(3, 2, 0)], "image height and width must be positive"),
        (
            [torch.empty(3, 2, 4), torch.empty(3, 2, 4, dtype=torch.float64)],
            "images must have the same dtype and device",
        ),
        (
            [torch.empty(3, 2, 4), torch.empty(3, 2, 4, device="meta")],
            "images must have the same dtype and device",
        ),
    ],
)
def test_rejects_invalid_batch(images, message):
    with pytest.raises(ValueError, match=message):
        batch_images(images)
