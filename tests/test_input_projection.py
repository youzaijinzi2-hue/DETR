import torch

from detr.backbone import ResNet50Backbone
from detr.batching import batch_images
from detr.input_projection import InputProjection


def test_projection_mixes_channels_without_mixing_spatial_positions():
    projection = InputProjection(in_channels=2, d_model=2)
    with torch.no_grad():
        projection.proj.weight.copy_(
            torch.tensor([[1.0, 2.0], [-1.0, 1.0]]).reshape(2, 2, 1, 1)
        )
        projection.proj.bias.copy_(torch.tensor([0.5, -0.5]))
    features = torch.tensor([[[[1.0, 2.0, 3.0]], [[10.0, 20.0, 30.0]]]])
    mask = torch.tensor([[[False, True, False]]])

    tokens, token_mask = projection(features, mask)

    # 各位置で [a, b] → [a + 2b + 0.5, -a + b - 0.5]。
    torch.testing.assert_close(
        tokens, torch.tensor([[[21.5, 8.5], [42.5, 17.5], [63.5, 26.5]]])
    )
    torch.testing.assert_close(token_mask, torch.tensor([[False, True, False]]))


def test_tokens_and_mask_follow_the_same_row_major_order():
    projection = InputProjection(in_channels=1, d_model=1)
    with torch.no_grad():
        projection.proj.weight.fill_(1)
        projection.proj.bias.zero_()
    features = torch.arange(12.0).reshape(2, 1, 2, 3)
    mask = torch.tensor(
        [
            [[False, True, False], [True, False, True]],
            [[True, True, False], [False, False, True]],
        ]
    )

    tokens, token_mask = projection(features, mask)

    torch.testing.assert_close(
        tokens,
        torch.tensor(
            [
                [[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]],
                [[6.0], [7.0], [8.0], [9.0], [10.0], [11.0]],
            ]
        ),
    )
    torch.testing.assert_close(
        token_mask,
        torch.tensor(
            [
                [False, True, False, True, False, True],
                [True, True, False, False, False, True],
            ]
        ),
    )


def test_projection_and_input_receive_gradients():
    projection = InputProjection(in_channels=1, d_model=1)
    with torch.no_grad():
        projection.proj.weight.fill_(2)
        projection.proj.bias.zero_()
    features = torch.ones(1, 1, 2, 3, requires_grad=True)

    tokens, _ = projection(features, torch.zeros(1, 2, 3, dtype=torch.bool))
    tokens.sum().backward()

    torch.testing.assert_close(features.grad, torch.full_like(features, 2))
    torch.testing.assert_close(projection.proj.weight.grad, torch.tensor([[[[6.0]]]]))
    torch.testing.assert_close(projection.proj.bias.grad, torch.tensor([6.0]))


def test_connects_image_batch_and_backbone_to_transformer_input():
    pixels, mask = batch_images([torch.ones(3, 64, 96), torch.ones(3, 32, 64)])
    backbone = ResNet50Backbone().eval()
    projection = InputProjection(in_channels=backbone.num_channels)

    with torch.no_grad():
        features, feature_mask = backbone(pixels, mask)
        tokens, token_mask = projection(features, feature_mask)

    assert tokens.shape == (2, 6, 256)
    assert torch.isfinite(tokens).all()
    torch.testing.assert_close(
        token_mask,
        torch.tensor(
            [
                [False, False, False, False, False, False],
                [False, False, True, True, True, True],
            ]
        ),
    )
