import pytest
import torch
from torchvision.ops import FrozenBatchNorm2d

from detr.backbone import ResNet50Backbone
from detr.batching import batch_images


@pytest.fixture(scope="module")
def backbone():
    # weights=Noneなので、テスト時に学習済み重みをダウンロードしない。
    return ResNet50Backbone().eval()


def test_extracts_spatial_features_and_resizes_padding_mask(backbone):
    pixels, padding_mask = batch_images([torch.ones(3, 64, 96), torch.ones(3, 32, 64)])

    with torch.no_grad():
        features, mask = backbone(pixels, padding_mask)

    assert features.shape == (2, 2048, 2, 3)
    assert torch.isfinite(features).all()
    assert torch.count_nonzero(features) > 0
    assert mask.device == features.device
    torch.testing.assert_close(
        mask,
        torch.tensor(
            [
                [[False, False, False], [False, False, False]],
                [[False, False, True], [True, True, True]],
            ]
        ),
    )


def test_handles_image_size_not_divisible_by_32(backbone):
    pixels, padding_mask = batch_images([torch.ones(3, 65, 97)])

    with torch.no_grad():
        features, mask = backbone(pixels, padding_mask)

    assert features.shape == (1, 2048, 3, 4)
    assert mask.shape == (1, 3, 4)
    assert not mask.any()


def test_cnn_receives_gradients(backbone):
    pixels, padding_mask = batch_images([torch.ones(3, 32, 32)])
    backbone.zero_grad(set_to_none=True)

    features, _ = backbone(pixels, padding_mask)
    features.square().mean().backward()

    gradient = backbone.body.layer4[2].conv3.weight.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert torch.count_nonzero(gradient) > 0
    backbone.zero_grad(set_to_none=True)


def test_batch_norm_stays_fixed_in_training_mode(backbone):
    norms = [m for m in backbone.modules() if isinstance(m, FrozenBatchNorm2d)]
    assert norms
    assert not any(isinstance(m, torch.nn.BatchNorm2d) for m in backbone.modules())
    assert not any(list(m.parameters()) for m in norms)
    norm = norms[0]
    image_features = torch.randn(2, norm.weight.numel(), 3, 4)
    before = {name: value.clone() for name, value in norm.named_buffers()}

    backbone.train()
    try:
        training_output = norm(image_features)
        norm.eval()
        torch.testing.assert_close(training_output, norm(image_features))
        for name, value in norm.named_buffers():
            torch.testing.assert_close(value, before[name])
    finally:
        backbone.eval()
