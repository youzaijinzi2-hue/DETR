import pytest
import torch
from torch import nn

from detr.decoder import ObjectQueries, TransformerDecoderLayer
from detr.encoder import TransformerEncoder
from detr.input_projection import InputProjection
from detr.position_encoding import PositionEmbeddingSine


def make_layer():
    return TransformerDecoderLayer(
        d_model=8, nhead=2, dim_feedforward=16, dropout=0
    ).eval()


def test_object_queries_share_embeddings_but_start_with_zero_contents():
    queries = ObjectQueries(num_queries=3, d_model=8)
    targets, positions = queries(batch_size=2)

    assert targets.shape == positions.shape == (2, 3, 8)
    assert torch.count_nonzero(targets) == 0
    assert not targets.requires_grad
    assert positions.requires_grad
    torch.testing.assert_close(positions[0], queries.embedding.weight)
    torch.testing.assert_close(positions[1], queries.embedding.weight)
    assert not torch.equal(positions[0, 0], positions[0, 1])


def test_zero_positions_matches_standard_pytorch_decoder():
    torch.manual_seed(7)
    layer = make_layer()
    reference = nn.TransformerDecoderLayer(
        d_model=8, nhead=2, dim_feedforward=16, dropout=0, batch_first=True
    ).eval()
    state = {
        name.replace("cross_attn.", "multihead_attn."): value
        for name, value in layer.state_dict().items()
    }
    reference.load_state_dict(state, strict=True)
    targets = torch.randn(2, 3, 8)
    memory = torch.randn(2, 6, 8)
    mask = torch.zeros(2, 6, dtype=torch.bool)
    mask[0, -2:] = True

    result = layer(
        targets, torch.zeros_like(targets), memory, torch.zeros_like(memory), mask
    )

    torch.testing.assert_close(
        result, reference(targets, memory, memory_key_padding_mask=mask)
    )


def test_padded_memory_and_positions_do_not_affect_candidates():
    torch.manual_seed(8)
    layer = make_layer()
    targets, query_positions = ObjectQueries(num_queries=3, d_model=8)(2)
    memory = torch.randn(2, 6, 8)
    positions = torch.randn_like(memory)
    mask = torch.zeros(2, 6, dtype=torch.bool)
    mask[0, -3:] = True
    mask[1, -1] = True
    changed_memory = memory.clone()
    changed_positions = positions.clone()
    changed_memory[mask] = 1000 * torch.randn_like(memory[mask])
    changed_positions[mask] = 1000 * torch.randn_like(positions[mask])

    original = layer(targets, query_positions, memory, positions, mask)
    changed = layer(targets, query_positions, changed_memory, changed_positions, mask)

    torch.testing.assert_close(original, changed)
    # 有効な画像特徴には反応する。
    different = layer(targets, query_positions, memory.flip(-1), positions, mask)
    assert not torch.allclose(original, different)


def test_encoder_to_decoder_backpropagates_into_object_queries():
    torch.manual_seed(9)
    projection = InputProjection(in_channels=4, d_model=8)
    encoder = TransformerEncoder(
        d_model=8, nhead=2, dim_feedforward=16, dropout=0, num_layers=2
    )
    queries = ObjectQueries(num_queries=3, d_model=8)
    decoder_layer = make_layer()
    features = torch.randn(2, 4, 2, 3, requires_grad=True)
    mask = torch.zeros(2, 2, 3, dtype=torch.bool)
    mask[0, 1] = True
    tokens, token_mask = projection(features, mask)
    positions = PositionEmbeddingSine(d_model=8)(mask)
    memory = encoder(tokens, positions, token_mask)
    targets, query_positions = queries(batch_size=2)

    result = decoder_layer(targets, query_positions, memory, positions, token_mask)
    (result * torch.randn_like(result)).sum().backward()

    assert result.shape == (2, 3, 8)
    assert torch.isfinite(result).all()
    for gradient in [
        features.grad,
        queries.embedding.weight.grad,
        encoder.layers[0].self_attn.in_proj_weight.grad,
        decoder_layer.cross_attn.in_proj_weight.grad,
        decoder_layer.linear1.weight.grad,
    ]:
        assert gradient is not None
        assert torch.isfinite(gradient).all()
        assert torch.count_nonzero(gradient) > 0


@pytest.mark.parametrize("num_queries", [0, -1])
def test_rejects_non_positive_query_count(num_queries):
    with pytest.raises(ValueError, match="num_queries must be positive"):
        ObjectQueries(num_queries=num_queries)
