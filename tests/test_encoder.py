import torch
from torch import nn

from detr.encoder import TransformerEncoderLayer
from detr.input_projection import InputProjection
from detr.position_encoding import PositionEmbeddingSine


def make_layer():
    return TransformerEncoderLayer(
        d_model=8, nhead=2, dim_feedforward=16, dropout=0
    ).eval()


def test_padding_values_and_positions_do_not_affect_valid_outputs():
    torch.manual_seed(0)
    layer = make_layer()
    tokens = torch.randn(2, 6, 8)
    positions = torch.randn_like(tokens)
    mask = torch.tensor(
        [
            [False, False, True, True, True, True],
            [False, False, False, False, False, True],
        ]
    )
    changed_tokens = tokens.clone()
    changed_positions = positions.clone()
    changed_tokens[mask] = 1000 * torch.randn_like(tokens[mask])
    changed_positions[mask] = 1000 * torch.randn_like(positions[mask])

    original = layer(tokens, positions, mask)
    changed = layer(changed_tokens, changed_positions, mask)

    torch.testing.assert_close(original[~mask], changed[~mask])


def test_zero_positions_matches_standard_pytorch_encoder():
    torch.manual_seed(1)
    layer = make_layer()
    reference = nn.TransformerEncoderLayer(
        d_model=8, nhead=2, dim_feedforward=16, dropout=0, batch_first=True
    ).eval()
    reference.load_state_dict(layer.state_dict(), strict=True)
    tokens = torch.randn(2, 6, 8)
    mask = torch.zeros(2, 6, dtype=torch.bool)
    mask[0, -2:] = True

    result = layer(tokens, torch.zeros_like(tokens), mask)
    expected = reference(tokens, src_key_padding_mask=mask)

    torch.testing.assert_close(result, expected)


def test_positions_affect_attention_but_are_not_added_to_values():
    torch.manual_seed(2)
    layer = make_layer()
    tokens = torch.randn(1, 6, 8)
    positions = torch.randn_like(tokens)
    mask = torch.zeros(1, 6, dtype=torch.bool)
    captured = {}

    def record_inputs(module, args, kwargs):
        captured.update(kwargs)

    hook = layer.self_attn.register_forward_pre_hook(record_inputs, with_kwargs=True)
    try:
        result = layer(tokens, positions, mask)
    finally:
        hook.remove()

    torch.testing.assert_close(captured["query"], tokens + positions)
    torch.testing.assert_close(captured["key"], tokens + positions)
    torch.testing.assert_close(captured["value"], tokens)
    without_positions = layer(tokens, torch.zeros_like(tokens), mask)
    assert not torch.allclose(result, without_positions)


def test_projection_positions_and_encoder_connect_and_backpropagate():
    torch.manual_seed(3)
    projection = InputProjection(in_channels=4, d_model=8)
    layer = make_layer()
    features = torch.randn(2, 4, 2, 3, requires_grad=True)
    mask = torch.zeros(2, 2, 3, dtype=torch.bool)
    mask[0, 1] = True
    tokens, token_mask = projection(features, mask)
    positions = PositionEmbeddingSine(d_model=8)(mask)

    result = layer(tokens, positions, token_mask)
    (result * torch.randn_like(result)).sum().backward()

    assert result.shape == (2, 6, 8)
    assert torch.isfinite(result).all()
    for gradient in [
        features.grad,
        projection.proj.weight.grad,
        layer.self_attn.in_proj_weight.grad,
        layer.linear1.weight.grad,
    ]:
        assert gradient is not None
        assert torch.isfinite(gradient).all()
        assert torch.count_nonzero(gradient) > 0
