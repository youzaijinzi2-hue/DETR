import pytest
import torch
from torch import nn

from detr.decoder import ObjectQueries, TransformerDecoder


def make_decoder():
    return TransformerDecoder(d_model=8, nhead=2, dim_feedforward=16, dropout=0).eval()


def test_default_six_layers_have_independent_parameters():
    decoder = make_decoder()
    assert len(decoder.layers) == 6
    for name, _ in decoder.layers[0].named_parameters():
        parameters = [dict(layer.named_parameters())[name] for layer in decoder.layers]
        assert len({p.data_ptr() for p in parameters}) == 6


def test_all_intermediate_outputs_match_standard_decoder_with_zero_positions():
    torch.manual_seed(10)
    decoder = make_decoder()
    reference = nn.TransformerDecoder(
        nn.TransformerDecoderLayer(
            d_model=8, nhead=2, dim_feedforward=16, dropout=0, batch_first=True
        ),
        num_layers=6,
        norm=nn.LayerNorm(8),
    ).eval()
    reference.load_state_dict(
        {
            name.replace("cross_attn.", "multihead_attn."): value
            for name, value in decoder.state_dict().items()
        },
        strict=True,
    )
    expected_intermediate = []

    def collect(module, args, output):
        expected_intermediate.append(reference.norm(output))

    hooks = [layer.register_forward_hook(collect) for layer in reference.layers]
    targets = torch.randn(2, 3, 8)
    memory = torch.randn(2, 6, 8)
    mask = torch.zeros(2, 6, dtype=torch.bool)
    mask[0, -2:] = True
    try:
        expected_final = reference(targets, memory, memory_key_padding_mask=mask)
    finally:
        for hook in hooks:
            hook.remove()

    result = decoder(
        targets, torch.zeros_like(targets), memory, torch.zeros_like(memory), mask
    )

    assert result.shape == (6, 2, 3, 8)
    torch.testing.assert_close(result, torch.stack(expected_intermediate))
    torch.testing.assert_close(result[-1], expected_final)


def test_padding_is_ignored_in_every_decoder_layer():
    torch.manual_seed(11)
    decoder = make_decoder()
    targets, queries = ObjectQueries(num_queries=3, d_model=8)(2)
    memory = torch.randn(2, 6, 8)
    positions = torch.randn_like(memory)
    mask = torch.zeros(2, 6, dtype=torch.bool)
    mask[0, -3:] = True
    mask[1, -1] = True
    changed_memory = memory.clone()
    changed_positions = positions.clone()
    changed_memory[mask] = 1000 * torch.randn_like(memory[mask])
    changed_positions[mask] = 1000 * torch.randn_like(positions[mask])

    original = decoder(targets, queries, memory, positions, mask)
    changed = decoder(targets, queries, changed_memory, changed_positions, mask)

    torch.testing.assert_close(original, changed)


def test_final_output_backpropagates_through_all_layers_and_queries():
    torch.manual_seed(12)
    decoder = make_decoder()
    queries = ObjectQueries(num_queries=3, d_model=8)
    targets, query_positions = queries(2)
    memory = torch.randn(2, 6, 8, requires_grad=True)
    positions = torch.randn_like(memory)
    mask = torch.zeros(2, 6, dtype=torch.bool)
    mask[0, -2:] = True

    result = decoder(targets, query_positions, memory, positions, mask)
    (result[-1] * torch.randn_like(result[-1])).sum().backward()

    assert torch.isfinite(result).all()
    parameters = [queries.embedding.weight, decoder.norm.weight]
    for layer in decoder.layers:
        parameters.extend([layer.cross_attn.in_proj_weight, layer.linear1.weight])
    for gradient in [memory.grad, *(parameter.grad for parameter in parameters)]:
        assert gradient is not None
        assert torch.isfinite(gradient).all()
        assert torch.count_nonzero(gradient) > 0
    # マスクした画像特徴は、全Decoder層で参照されない。
    torch.testing.assert_close(memory.grad[mask], torch.zeros_like(memory.grad[mask]))


@pytest.mark.parametrize("num_layers", [0, -1])
def test_rejects_non_positive_layer_count(num_layers):
    with pytest.raises(ValueError, match="num_layers must be positive"):
        TransformerDecoder(num_layers=num_layers)
