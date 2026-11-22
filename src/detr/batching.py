"""論文3.2節・脚注2の、異なる画像サイズのゼロパディング。"""

from collections.abc import Sequence

import torch
from torch import Tensor


def batch_images(images: Sequence[Tensor]) -> tuple[Tensor, Tensor]:
    """RGB画像を左上に配置し、右と下をゼロで埋める。

    Args:
        images: 各要素が (3, H, W) の画像。同じdtype・deviceを使う。

    Returns:
        pixels: (B, 3, 最大H, 最大W)。入力のdtype・deviceを保持する。
        padding_mask: (B, 最大H, 最大W) のboolテンソル。
            実画像はFalse、パディングはTrue（後でAttentionから除外する）。

    リサイズは行わない。正解ボックスはパディング後のサイズではなく、
    各画像の元のサイズで正規化しておく。
    """
    if len(images) == 0:
        raise ValueError("images must not be empty")
    first = images[0]
    for image in images:
        if image.ndim != 3 or image.shape[0] != 3:
            raise ValueError("each image must have shape (3, H, W)")
        if image.shape[1] <= 0 or image.shape[2] <= 0:
            raise ValueError("image height and width must be positive")
        if image.dtype != first.dtype or image.device != first.device:
            raise ValueError("images must have the same dtype and device")

    max_height = max(image.shape[1] for image in images)
    max_width = max(image.shape[2] for image in images)
    pixels = first.new_zeros((len(images), 3, max_height, max_width))
    padding_mask = torch.ones(
        (len(images), max_height, max_width), dtype=torch.bool, device=first.device
    )
    for index, image in enumerate(images):
        height, width = image.shape[-2:]
        pixels[index, :, :height, :width] = image
        padding_mask[index, :height, :width] = False

    return pixels, padding_mask
