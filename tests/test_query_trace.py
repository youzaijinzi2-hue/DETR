import json

import pytest
import torch

from detr.criterion import SetCriterion
from detr.model import DETR
from detr.query_trace import (
    feature_similarity,
    set_dropout,
    summarize_trace,
    trace_queries,
)


def test_disabling_dropout_preserves_weights_and_makes_training_deterministic():
    model = DETR(
        num_classes=2,
        num_queries=3,
        d_model=8,
        nhead=2,
        dim_feedforward=16,
        num_encoder_layers=1,
        num_decoder_layers=1,
    )
    original = {name: value.clone() for name, value in model.state_dict().items()}
    set_dropout(model, 0)
    model.train()
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            assert module.p == 0
        if isinstance(module, torch.nn.MultiheadAttention):
            assert module.dropout == 0
    images = [torch.randn(3, 32, 64)]
    with torch.no_grad():
        first, second = model(images), model(images)
    torch.testing.assert_close(first["pred_logits"], second["pred_logits"])
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, original[name])


@pytest.mark.parametrize("probability", [-0.1, 1.1, float("nan")])
def test_rejects_invalid_dropout(probability):
    with pytest.raises(ValueError, match="dropout must be between"):
        set_dropout(torch.nn.Linear(1, 1), probability)


def test_feature_similarity_distinguishes_identical_and_orthogonal_vectors():
    assert feature_similarity(torch.ones(3, 4))[
        "mean_pairwise_cosine"
    ] == pytest.approx(1)
    assert feature_similarity(torch.eye(3))["mean_pairwise_cosine"] == pytest.approx(0)
    assert feature_similarity(torch.ones(1, 4))["mean_pairwise_cosine"] is None


def test_trace_records_actual_training_and_eval_matches_and_gradients(tmp_path):
    torch.manual_seed(22)
    model = DETR(
        num_classes=2,
        num_queries=3,
        d_model=8,
        nhead=2,
        dim_feedforward=16,
        num_encoder_layers=1,
        num_decoder_layers=2,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = SetCriterion(2)
    image = torch.randn(3, 32, 64)
    target = {
        "labels": torch.tensor([0]),
        "boxes": torch.tensor([[0.5, 0.5, 0.2, 0.3]]),
    }
    path = tmp_path / "trace.jsonl"
    records = trace_queries(
        model, criterion, optimizer, image, target, steps=2, output_file=path
    )
    assert len(records) == 3
    assert len([json.loads(line) for line in path.read_text().splitlines()]) == 3
    for record in records:
        for mode in ("eval", "train"):
            if mode not in record:
                continue
            queries = record[mode]["queries"]
            assert len(queries) == 3
            assert sum(q["matched_target_index"] is not None for q in queries) == 1
            assert sum(q["expected_class"] == 2 for q in queries) == 2
            assert all(
                sum(q["class_probabilities"]) == pytest.approx(1) for q in queries
            )
            assert all(len(q["feature"]) == 8 for q in queries)
    assert any(g > 0 for g in records[1]["train"]["query_gradient_norms"])
    summary = summarize_trace(records)
    assert summary["additional_steps"] == 2
    assert len(summary["train_matched_queries"]) == 2
    # フックは全て取り外され、元のモジュールへ残らない。
    assert not model.encoder._forward_hooks
    assert not model.decoder._forward_hooks
    assert not criterion.matcher._forward_hooks
