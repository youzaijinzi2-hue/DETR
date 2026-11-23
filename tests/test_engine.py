import math

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

from detr.criterion import SetCriterion
from detr.engine import collate_fn, train_one_epoch
from detr.model import DETR


class ScalarModel(nn.Module):
    """更新量を手計算できるモデルで、学習ループ自体を検証する。"""

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))

    def forward(self, images):
        return self.weight


class SquareLoss(nn.Module):
    def forward(self, output, targets):
        return {"loss_total": output.square(), "loss_ce": output.square()}


def test_zeroes_gradients_each_batch_and_backpropagates_total_only():
    model = ScalarModel().eval()
    criterion = SquareLoss().eval()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    batches = [([torch.zeros(1)], [{}])] * 2

    metrics = train_one_epoch(model, criterion, batches, optimizer, max_norm=0)

    # w: 1 → 0.8 → 0.64。loss_ceまで足すと更新量が倍になってしまう。
    torch.testing.assert_close(model.weight, torch.tensor(0.64))
    assert metrics["loss_total"] == pytest.approx((1 + 0.8**2) / 2)
    assert all(isinstance(value, float) for value in metrics.values())
    assert model.training and criterion.training


def test_clips_gradient_before_optimizer_step():
    model = ScalarModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    train_one_epoch(
        model, SquareLoss(), [([torch.zeros(1)], [{}])], optimizer, max_norm=0.1
    )

    # 勾配2が0.1まで制限されるため、重みは約0.01だけ減る。
    torch.testing.assert_close(model.weight, torch.tensor(0.99))


def test_nonfinite_loss_does_not_update_parameters():
    class NonfiniteLoss(nn.Module):
        def forward(self, output, targets):
            return {"loss_total": output * float("nan")}

    model = ScalarModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    with pytest.raises(FloatingPointError, match="loss_total is not finite"):
        train_one_epoch(model, NonfiniteLoss(), [([torch.zeros(1)], [{}])], optimizer)
    torch.testing.assert_close(model.weight, torch.tensor(1.0))


def test_real_detr_updates_parameters_with_variable_images_and_empty_targets():
    torch.manual_seed(17)
    samples = [
        (
            torch.randn(3, 32, 64),
            {
                "labels": torch.tensor([1]),
                "boxes": torch.tensor([[0.5, 0.5, 0.2, 0.3]]),
            },
        ),
        (
            torch.randn(3, 64, 32),
            {"labels": torch.empty(0, dtype=torch.int64), "boxes": torch.empty(0, 4)},
        ),
    ]
    loader = DataLoader(samples, batch_size=2, collate_fn=collate_fn)
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
    criterion = SetCriterion(num_classes=2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    tracked = [
        model.backbone.body.conv1.weight,
        model.object_queries.embedding.weight,
        model.prediction_head.class_embed.weight,
    ]
    originals = [parameter.detach().clone() for parameter in tracked]

    metrics = train_one_epoch(model, criterion, loader, optimizer)

    assert {
        "loss_total",
        "loss_ce",
        "loss_bbox",
        "loss_giou",
        "loss_ce_0",
    } <= metrics.keys()
    assert all(math.isfinite(value) for value in metrics.values())
    for parameter, original in zip(tracked, originals):
        assert not torch.equal(parameter.detach(), original)
    assert samples[0][0].shape == (3, 32, 64)
    assert samples[1][1]["boxes"].shape == (0, 4)


def test_rejects_empty_loader():
    model = ScalarModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    with pytest.raises(ValueError, match="at least one batch"):
        train_one_epoch(model, SquareLoss(), [], optimizer)


@pytest.mark.parametrize("max_norm", [-1, float("nan"), float("inf")])
def test_rejects_invalid_gradient_norm(max_norm):
    model = ScalarModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    with pytest.raises(ValueError, match="max_norm must be finite and non-negative"):
        train_one_epoch(model, SquareLoss(), [], optimizer, max_norm=max_norm)
