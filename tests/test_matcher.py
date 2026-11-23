import pytest
import torch

from detr.matcher import HungarianMatcher


def test_assignment_minimizes_total_cost_instead_of_greedy_choice():
    probabilities = torch.tensor([[[0.60, 0.39, 0.01], [0.59, 0.01, 0.40]]])
    outputs = {
        "pred_logits": probabilities.log(),
        "pred_boxes": torch.full((1, 2, 4), 0.5),
    }
    targets = [{"labels": torch.tensor([0, 1]), "boxes": torch.full((2, 4), 0.5)}]

    predictions, labels = HungarianMatcher(1, 0, 0)(outputs, targets)[0]

    # 予測0をクラス0に先に割り当てると合計0.61。逆なら0.98で有利。
    torch.testing.assert_close(predictions, torch.tensor([0, 1]))
    torch.testing.assert_close(labels, torch.tensor([1, 0]))


def test_class_cost_uses_probability_rather_than_log_probability():
    outputs = {
        "pred_logits": torch.tensor([[[0.80, 0.19, 0.01], [0.40, 0.01, 0.59]]]).log(),
        "pred_boxes": torch.full((1, 2, 4), 0.5),
    }
    targets = [{"labels": torch.tensor([0, 1]), "boxes": torch.full((2, 4), 0.5)}]

    _, labels = HungarianMatcher(1, 0, 0)(outputs, targets)[0]

    # 確率の和では対角(0.81 > 0.59)、logの和なら逆の対応になる。
    torch.testing.assert_close(labels, torch.tensor([0, 1]))


@pytest.mark.parametrize("weights", [(0, 1, 0), (0, 0, 1), (1, 5, 2)])
def test_box_cost_selects_correct_queries_and_leaves_extra_query_unmatched(weights):
    a = [0.2, 0.2, 0.1, 0.1]
    b = [0.8, 0.8, 0.1, 0.1]
    outputs = {
        "pred_logits": torch.zeros(1, 3, 2, requires_grad=True),
        "pred_boxes": torch.tensor([[b, [0.5, 0.5, 0.1, 0.1], a]], requires_grad=True),
    }
    targets = [{"labels": torch.tensor([0, 0]), "boxes": torch.tensor([a, b])}]

    predictions, labels = HungarianMatcher(*weights)(outputs, targets)[0]

    torch.testing.assert_close(predictions, torch.tensor([0, 2]))
    torch.testing.assert_close(labels, torch.tensor([1, 0]))
    assert predictions.device.type == labels.device.type == "cpu"
    assert not predictions.requires_grad and not labels.requires_grad
    assert outputs["pred_boxes"].grad is None


def test_handles_empty_targets_and_keeps_images_separate():
    outputs = {
        "pred_logits": torch.tensor(
            [[[4.0, 0.0], [0.0, 4.0]], [[0.0, 4.0], [4.0, 0.0]]]
        ),
        "pred_boxes": torch.full((2, 2, 4), 0.5),
    }
    empty = {"labels": torch.empty(0, dtype=torch.int64), "boxes": torch.empty(0, 4)}
    targets = [empty, {"labels": torch.tensor([0]), "boxes": torch.full((1, 4), 0.5)}]

    matches = HungarianMatcher()(outputs, targets)

    assert all(
        index.shape == (0,) and index.dtype == torch.int64 for index in matches[0]
    )
    torch.testing.assert_close(matches[1][0], torch.tensor([1]))
    torch.testing.assert_close(matches[1][1], torch.tensor([0]))
    all_empty = HungarianMatcher()(outputs, [empty, empty])
    assert all(len(prediction) == len(target) == 0 for prediction, target in all_empty)


@pytest.mark.parametrize("weights", [(0, 0, 0), (-1, 1, 1), (float("nan"), 1, 1)])
def test_rejects_invalid_weights(weights):
    with pytest.raises(ValueError, match="cost weights"):
        HungarianMatcher(*weights)


def test_rejects_no_object_in_ground_truth():
    outputs = {
        "pred_logits": torch.zeros(1, 2, 2),
        "pred_boxes": torch.full((1, 2, 4), 0.5),
    }
    targets = [{"labels": torch.tensor([1]), "boxes": torch.full((1, 4), 0.5)}]
    with pytest.raises(ValueError, match="exclude the no-object class"):
        HungarianMatcher()(outputs, targets)
