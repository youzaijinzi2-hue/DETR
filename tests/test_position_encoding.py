import math

import pytest
import torch

from detr.input_projection import InputProjection
from detr.position_encoding import PositionEmbeddingSine


def test_coordinates_frequencies_and_row_major_order():
    mask = torch.zeros(1, 2, 3, dtype=torch.bool)

    positions = PositionEmbeddingSine(d_model=8)(mask)

    # 各軸4チャネル: sin(p), cos(p), sin(p/100), cos(p/100)。
    expected = []
    for row in range(1, 3):
        for column in range(1, 4):
            y = row / (2 + 1e-6) * 2 * math.pi
            x = column / (3 + 1e-6) * 2 * math.pi
            expected.append(
                [
                    math.sin(y),
                    math.cos(y),
                    math.sin(y / 100),
                    math.cos(y / 100),
                    math.sin(x),
                    math.cos(x),
                    math.sin(x / 100),
                    math.cos(x / 100),
                ]
            )
    torch.testing.assert_close(
        positions, torch.tensor([expected]), atol=1e-6, rtol=1e-5
    )


def test_padding_does_not_change_valid_positions():
    encoding = PositionEmbeddingSine()
    small_mask = torch.zeros(1, 2, 3, dtype=torch.bool)
    batch_mask = torch.ones(2, 4, 5, dtype=torch.bool)
    batch_mask[0, :2, :3] = False
    batch_mask[1] = False

    small = encoding(small_mask).reshape(1, 2, 3, 256)
    batched = encoding(batch_mask).reshape(2, 4, 5, 256)

    torch.testing.assert_close(batched[0, :2, :3], small[0])


def test_all_padding_produces_finite_values():
    positions = PositionEmbeddingSine()(torch.ones(1, 2, 3, dtype=torch.bool))

    assert positions.shape == (1, 6, 256)
    assert torch.isfinite(positions).all()


def test_matches_projected_tokens_without_learnable_parameters():
    encoding = PositionEmbeddingSine()
    mask = torch.zeros(2, 2, 3, dtype=torch.bool)
    tokens, _ = InputProjection()(torch.ones(2, 2048, 2, 3), mask)

    positions = encoding(mask)

    assert positions.shape == tokens.shape == (2, 6, 256)
    assert positions.dtype == torch.float32
    assert positions.device == mask.device
    assert not list(encoding.parameters())
    assert not positions.requires_grad


@pytest.mark.parametrize("d_model", [0, -4, 2, 6])
def test_rejects_invalid_embedding_dimension(d_model):
    with pytest.raises(ValueError, match="positive multiple of 4"):
        PositionEmbeddingSine(d_model=d_model)


@pytest.mark.parametrize("mask", [torch.zeros(2, 3), torch.zeros(1, 2, 3)])
def test_rejects_invalid_mask(mask):
    with pytest.raises(ValueError, match="padding_mask must be a bool tensor"):
        PositionEmbeddingSine()(mask)
