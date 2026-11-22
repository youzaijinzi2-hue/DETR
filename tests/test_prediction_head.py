import pytest
import torch

from detr.decoder import ObjectQueries, TransformerDecoder
from detr.prediction_head import PredictionHead


def test_class_head_returns_raw_logits_including_no_object():
    head = PredictionHead(num_classes=2, d_model=4)
    with torch.no_grad():
        head.class_embed.weight.zero_()
        head.class_embed.bias.copy_(torch.tensor([-2.0, 0.0, 3.0]))

    logits = head(torch.ones(2, 3, 4))["pred_logits"]

    torch.testing.assert_close(logits, torch.tensor([-2.0, 0.0, 3.0]).expand(2, 3, 3))
    # 最後のクラス（ID=2）はno-object。softmaxはまだ行っていない。
    assert (logits.argmax(-1) == 2).all()


def test_box_head_applies_relu_and_sigmoid_in_coordinate_order():
    head = PredictionHead(num_classes=2, d_model=4)
    with torch.no_grad():
        for layer in (head.bbox_embed[0], head.bbox_embed[2], head.bbox_embed[4]):
            layer.weight.copy_(torch.eye(4))
            layer.bias.zero_()
    features = torch.tensor([[[-2.0, 0.0, 1.0, 2.0]]])

    boxes = head(features)["pred_boxes"]

    torch.testing.assert_close(
        boxes, torch.tensor([[[0.5, 0.5, 0.7310586, 0.8807971]]])
    )
    assert ((boxes >= 0) & (boxes <= 1)).all()


def test_same_head_is_used_for_every_layer_image_and_query():
    torch.manual_seed(13)
    head = PredictionHead(num_classes=3, d_model=8)
    features = torch.randn(2, 2, 3, 8)

    all_outputs = head(features)

    assert all_outputs["pred_logits"].shape == (2, 2, 3, 4)
    assert all_outputs["pred_boxes"].shape == (2, 2, 3, 4)
    for layer in range(2):
        for image in range(2):
            for query in range(3):
                individual = head(features[layer, image, query])
                for key in all_outputs:
                    torch.testing.assert_close(
                        all_outputs[key][layer, image, query], individual[key]
                    )


def test_decoder_predictions_backpropagate_through_both_heads():
    torch.manual_seed(14)
    queries = ObjectQueries(num_queries=3, d_model=8)
    decoder = TransformerDecoder(d_model=8, nhead=2, dim_feedforward=16, dropout=0)
    head = PredictionHead(num_classes=2, d_model=8)
    targets, query_positions = queries(2)
    memory = torch.randn(2, 6, 8, requires_grad=True)
    positions = torch.randn_like(memory)
    mask = torch.zeros(2, 6, dtype=torch.bool)
    features = decoder(targets, query_positions, memory, positions, mask)

    outputs = head(features)
    assert outputs["pred_logits"].shape == (6, 2, 3, 3)
    assert outputs["pred_boxes"].shape == (6, 2, 3, 4)
    # 損失関数は別ステップ。ここでは接続と勾配だけを検証する。
    loss = sum((value * torch.randn_like(value)).sum() for value in outputs.values())
    loss.backward()

    parameters = [queries.embedding.weight, *head.parameters()]
    for gradient in [memory.grad, *(parameter.grad for parameter in parameters)]:
        assert gradient is not None
        assert torch.isfinite(gradient).all()
        assert torch.count_nonzero(gradient) > 0


@pytest.mark.parametrize("num_classes", [0, -1])
def test_rejects_non_positive_class_count(num_classes):
    with pytest.raises(ValueError, match="num_classes must be positive"):
        PredictionHead(num_classes=num_classes)
