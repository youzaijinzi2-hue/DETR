import pytest
import torch

from detr.postprocess import postprocess


def test_converts_to_original_pixels_filters_no_object_and_sorts_scores():
    outputs = {
        "pred_logits": torch.tensor(
            [
                [
                    [0.6, 0.1, 0.3],
                    [0.1, 0.8, 0.1],
                    [0.05, 0.05, 0.9],
                ]
            ]
        ).log(),
        "pred_boxes": torch.tensor(
            [
                [
                    [0.5, 0.5, 0.5, 0.5],
                    [0.25, 0.5, 0.5, 1.0],
                    [0.5, 0.5, 0.2, 0.2],
                ]
            ]
        ),
    }

    result = postprocess(outputs, torch.tensor([[100, 200]]), threshold=0.5)[0]

    torch.testing.assert_close(result["scores"], torch.tensor([0.8, 0.6]))
    torch.testing.assert_close(result["labels"], torch.tensor([1, 0]))
    torch.testing.assert_close(
        result["boxes"],
        torch.tensor(
            [
                [0.0, 0.0, 100.0, 100.0],
                [50.0, 25.0, 150.0, 75.0],
            ]
        ),
    )


def test_uses_each_images_own_size_and_handles_no_detections():
    outputs = {
        "pred_logits": torch.zeros(2, 1, 2),
        "pred_boxes": torch.tensor([[[0.5, 0.5, 1.0, 1.0]]] * 2),
    }
    sizes = torch.tensor([[100, 200], [400, 300]])

    results = postprocess(outputs, sizes, threshold=0.5)

    torch.testing.assert_close(
        results[0]["boxes"], torch.tensor([[0.0, 0.0, 200.0, 100.0]])
    )
    torch.testing.assert_close(
        results[1]["boxes"], torch.tensor([[0.0, 0.0, 300.0, 400.0]])
    )
    empty = postprocess(outputs, sizes, threshold=1)
    for result in empty:
        assert result["boxes"].shape == (0, 4)
        assert result["scores"].shape == result["labels"].shape == (0,)


@pytest.mark.parametrize("threshold", [-0.1, 1.1, float("nan")])
def test_rejects_invalid_threshold(threshold):
    with pytest.raises(ValueError, match="threshold"):
        postprocess({}, torch.ones(1, 2), threshold=threshold)
