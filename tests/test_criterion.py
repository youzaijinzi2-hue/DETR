import math

import torch

from detr.criterion import SetCriterion
from detr.model import DETR


def test_classification_weights_no_object_and_perfect_boxes_have_zero_loss():
    logits = torch.tensor([[[2.0, 0.0, -1.0], [0.0, 1.0, 2.0]]], requires_grad=True)
    boxes = torch.tensor(
        [[[0.5, 0.5, 0.2, 0.2], [0.1, 0.1, 0.1, 0.1]]], requires_grad=True
    )
    targets = [{"labels": torch.tensor([0]), "boxes": boxes.detach()[0, :1]}]

    losses = SetCriterion(num_classes=2)(
        {"pred_logits": logits, "pred_boxes": boxes}, targets
    )

    expected_ce = (
        (torch.logsumexp(logits[0, 0], 0) - 2)
        + 0.1 * (torch.logsumexp(logits[0, 1], 0) - 2)
    ) / 1.1
    torch.testing.assert_close(losses["loss_ce"], expected_ce)
    torch.testing.assert_close(losses["loss_bbox"], torch.tensor(0.0))
    torch.testing.assert_close(losses["loss_giou"], torch.tensor(0.0))
    torch.testing.assert_close(losses["loss_total"], expected_ce)


def test_boxes_are_normalized_by_total_object_count_and_ignore_unmatched_queries():
    boxes = torch.tensor(
        [
            [[0.35, 0.5, 0.2, 0.2], [0.9, 0.9, 0.1, 0.1]],
            [[0.25, 0.5, 0.2, 0.2], [0.9, 0.9, 0.1, 0.1]],
        ],
        requires_grad=True,
    )
    targets = [
        {"labels": torch.tensor([0]), "boxes": torch.tensor([[0.25, 0.5, 0.2, 0.2]])}
        for _ in range(2)
    ]

    losses = SetCriterion(num_classes=1)(
        {"pred_logits": torch.zeros(2, 2, 2, requires_grad=True), "pred_boxes": boxes},
        targets,
    )

    torch.testing.assert_close(losses["loss_bbox"], torch.tensor(0.05))
    torch.testing.assert_close(losses["loss_giou"], torch.tensor(1 / 3))
    torch.testing.assert_close(
        losses["loss_total"], losses["loss_ce"] + 5 * 0.05 + 2 / 3
    )
    losses["loss_total"].backward()
    torch.testing.assert_close(boxes.grad[:, 1], torch.zeros(2, 4))
    assert torch.count_nonzero(boxes.grad[:, 0]) > 0


def test_empty_ground_truth_has_only_no_object_classification_and_can_backward():
    logits = torch.zeros(2, 3, 3, requires_grad=True)
    boxes = torch.full((2, 3, 4), 0.5, requires_grad=True)
    targets = [
        {"labels": torch.empty(0, dtype=torch.int64), "boxes": torch.empty(0, 4)}
        for _ in range(2)
    ]

    losses = SetCriterion(num_classes=2)(
        {"pred_logits": logits, "pred_boxes": boxes}, targets
    )

    torch.testing.assert_close(losses["loss_total"], torch.tensor(math.log(3)))
    assert losses["loss_bbox"].item() == losses["loss_giou"].item() == 0
    losses["loss_total"].backward()
    assert torch.isfinite(logits.grad).all()
    assert torch.count_nonzero(logits.grad) > 0
    torch.testing.assert_close(boxes.grad, torch.zeros_like(boxes))


def test_auxiliary_layer_is_matched_independently_and_contributes_to_total():
    boxes = torch.tensor(
        [[[0.2, 0.2, 0.1, 0.1], [0.8, 0.8, 0.1, 0.1]]], requires_grad=True
    )
    aux_boxes = boxes.detach().flip(1).requires_grad_()
    logits = torch.zeros(1, 2, 2, requires_grad=True)
    aux_logits = torch.zeros_like(logits, requires_grad=True)
    targets = [{"labels": torch.tensor([0]), "boxes": boxes.detach()[0, :1]}]
    outputs = {
        "pred_logits": logits,
        "pred_boxes": boxes,
        "aux_outputs": [{"pred_logits": aux_logits, "pred_boxes": aux_boxes}],
    }

    losses = SetCriterion(num_classes=1)(outputs, targets)

    assert losses["loss_bbox"].item() == losses["loss_bbox_0"].item() == 0
    torch.testing.assert_close(losses["loss_total"], torch.tensor(2 * math.log(2)))
    losses["loss_total"].backward()
    assert torch.count_nonzero(logits.grad) > 0
    assert torch.count_nonzero(aux_logits.grad) > 0


def test_model_loss_and_backward_with_mixed_empty_and_nonempty_targets():
    torch.manual_seed(16)
    model = DETR(
        num_classes=2,
        num_queries=3,
        d_model=8,
        nhead=2,
        dim_feedforward=16,
        dropout=0,
        num_encoder_layers=1,
        num_decoder_layers=2,
    )
    targets = [
        {"labels": torch.tensor([1]), "boxes": torch.tensor([[0.5, 0.5, 0.2, 0.3]])},
        {"labels": torch.empty(0, dtype=torch.int64), "boxes": torch.empty(0, 4)},
    ]
    outputs = model([torch.randn(3, 32, 64), torch.randn(3, 64, 32)])

    losses = SetCriterion(num_classes=2)(outputs, targets)
    losses["loss_total"].backward()

    assert set(losses) == {
        "loss_ce",
        "loss_bbox",
        "loss_giou",
        "loss_ce_0",
        "loss_bbox_0",
        "loss_giou_0",
        "loss_total",
    }
    assert all(torch.isfinite(value) for value in losses.values())
    for parameter in [
        model.backbone.body.conv1.weight,
        model.object_queries.embedding.weight,
        model.prediction_head.class_embed.weight,
        model.prediction_head.bbox_embed[-1].weight,
    ]:
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert torch.count_nonzero(parameter.grad) > 0
