import pytest
import torch
from torch import nn

from detr.encoder import TransformerEncoder
from detr.input_projection import InputProjection
from detr.position_encoding import PositionEmbeddingSine


def make_encoder():
    return TransformerEncoder(d_model=8, nhead=2, dim_feedforward=16, dropout=0).eval()


def test_default_six_layers_have_independent_parameters():
    encoder = make_encoder()
    assert len(encoder.layers) == 6
    for name, _ in encoder.layers[0].named_parameters():
        parameters = [dict(layer.named_parameters())[name] for layer in encoder.layers]
        assert len({p.data_ptr() for p in parameters}) == 6


def test_six_layers_match_standard_encoder_with_zero_positions():
    torch.manual_seed(4)
    encoder = make_encoder()
    reference = nn.TransformerEncoder(
        nn.TransformerEncoderLayer(
            d_model=8, nhead=2, dim_feedforward=16, dropout=0, batch_first=True
        ),
        num_layers=6,
        enable_nested_tensor=False,
    ).eval()
    reference.load_state_dict(encoder.state_dict(), strict=True)
    tokens = torch.randn(2, 6, 8)
    mask = torch.zeros(2, 6, dtype=torch.bool)
    mask[0, -2:] = True

    result = encoder(tokens, torch.zeros_like(tokens), mask)

    torch.testing.assert_close(result, reference(tokens, src_key_padding_mask=mask))


def test_padding_is_ignored_through_all_six_layers():
    torch.manual_seed(5)
    encoder = make_encoder()
    tokens = torch.randn(2, 6, 8)
    positions = torch.randn_like(tokens)
    mask = torch.zeros(2, 6, dtype=torch.bool)
    mask[0, -3:] = True
    mask[1, -1] = True
    changed_tokens = tokens.clone()
    changed_positions = positions.clone()
    changed_tokens[mask] = 1000 * torch.randn_like(tokens[mask])
    changed_positions[mask] = 1000 * torch.randn_like(positions[mask])

    original = encoder(tokens, positions, mask)
    changed = encoder(changed_tokens, changed_positions, mask)

    torch.testing.assert_close(original[~mask], changed[~mask])


def test_projected_features_backpropagate_through_every_layer():
    torch.manual_seed(6)
    projection = InputProjection(in_channels=4, d_model=8)
    encoder = make_encoder()
    features = torch.randn(2, 4, 2, 3, requires_grad=True)
    mask = torch.zeros(2, 2, 3, dtype=torch.bool)
    mask[0, 1] = True
    tokens, token_mask = projection(features, mask)
    positions = PositionEmbeddingSine(d_model=8)(mask)

    memory = encoder(tokens, positions, token_mask)
    (memory[~token_mask] * torch.randn_like(memory[~token_mask])).sum().backward()

    assert memory.shape == (2, 6, 8)
    assert torch.isfinite(memory).all()
    parameters = [projection.proj.weight]
    for layer in encoder.layers:
        parameters.extend([layer.self_attn.in_proj_weight, layer.linear1.weight])
    for gradient in [features.grad, *(parameter.grad for parameter in parameters)]:
        assert gradient is not None
        assert torch.isfinite(gradient).all()
        assert torch.count_nonzero(gradient) > 0


@pytest.mark.parametrize("num_layers", [0, -1])
def test_rejects_non_positive_layer_count(num_layers):
    with pytest.raises(ValueError, match="num_layers must be positive"):
        TransformerEncoder(num_layers=num_layers)
