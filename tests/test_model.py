import pytest
import torch

from detr.model import DETR


@pytest.fixture
def model():
    torch.manual_seed(15)
    return DETR(
        num_classes=2,
        num_queries=3,
        d_model=8,
        nhead=2,
        dim_feedforward=16,
        dropout=0,
        num_encoder_layers=2,
        num_decoder_layers=2,
    ).eval()


def test_different_image_sizes_produce_final_and_auxiliary_predictions(model):
    images = [torch.randn(3, 64, 96), torch.randn(3, 32, 64)]
    with torch.no_grad():
        outputs = model(images)

    assert outputs["pred_logits"].shape == (2, 3, 3)
    assert outputs["pred_boxes"].shape == (2, 3, 4)
    assert len(outputs["aux_outputs"]) == 1
    for prediction in [outputs, *outputs["aux_outputs"]]:
        assert prediction["pred_logits"].shape == (2, 3, 3)
        assert prediction["pred_boxes"].shape == (2, 3, 4)
        assert torch.isfinite(prediction["pred_logits"]).all()
        boxes = prediction["pred_boxes"]
        assert torch.isfinite(boxes).all()
        assert ((boxes >= 0) & (boxes <= 1)).all()


def test_disabling_auxiliary_predictions_preserves_final_predictions(model):
    images = [torch.randn(3, 32, 64)]
    with torch.no_grad():
        with_auxiliary = model(images)
        model.aux_loss = False
        without_auxiliary = model(images)

    assert set(without_auxiliary) == {"pred_logits", "pred_boxes"}
    for key in without_auxiliary:
        torch.testing.assert_close(with_auxiliary[key], without_auxiliary[key])


def test_image_order_only_changes_output_batch_order(model):
    images = [torch.randn(3, 64, 96), torch.randn(3, 32, 64)]
    with torch.no_grad():
        original = model(images)
        reversed_outputs = model(images[::-1])

    for key in ("pred_logits", "pred_boxes"):
        torch.testing.assert_close(original[key].flip(0), reversed_outputs[key])


def test_predictions_backpropagate_from_heads_to_image_and_backbone(model):
    image = torch.randn(3, 32, 64, requires_grad=True)
    outputs = model([image])
    # これは検出用損失ではなく、全経路の微分可能性を確かめる検査。
    loss = sum(
        (prediction[key] * torch.randn_like(prediction[key])).sum()
        for prediction in [outputs, *outputs["aux_outputs"]]
        for key in ("pred_logits", "pred_boxes")
    )
    loss.backward()

    parameters = [
        model.backbone.body.conv1.weight,
        model.input_projection.proj.weight,
        model.encoder.layers[0].self_attn.in_proj_weight,
        model.object_queries.embedding.weight,
        model.decoder.layers[0].cross_attn.in_proj_weight,
        model.prediction_head.class_embed.weight,
        model.prediction_head.bbox_embed[-1].weight,
    ]
    for gradient in [image.grad, *(parameter.grad for parameter in parameters)]:
        assert gradient is not None
        assert torch.isfinite(gradient).all()
        assert torch.count_nonzero(gradient) > 0


def test_rejects_empty_image_list(model):
    with pytest.raises(ValueError, match="images must not be empty"):
        model([])
