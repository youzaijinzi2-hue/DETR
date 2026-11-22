"""画像上の座標を、DETRが予測する正規化ボックスへ変換する。"""

import torch
from torch import Tensor


def normalize_boxes_xyxy(boxes: Tensor, image_size: tuple[int, int]) -> Tensor:
    """ピクセル座標のxyxyを、正規化したcxcywhへ変換する。

    Args:
        boxes: shape (N, 4) の (左端x, 上端y, 右端x, 下端y)。
            画像の範囲内にあり、幅と高さが正のボックスを渡す。
        image_size: 画像の (高さ, 幅)。

    Returns:
        shape (N, 4) の (中心x, 中心y, 幅, 高さ)。各値は0〜1。
        物体がない画像では、入力・出力ともに shape (0, 4)。
    """
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError("boxes must have shape (N, 4)")
    height, width = image_size
    if height <= 0 or width <= 0:
        raise ValueError("image height and width must be positive")

    # 整数座標で渡されても、小数を保持できるようにする。
    if not boxes.is_floating_point():
        boxes = boxes.float()
    x_min, y_min, x_max, y_max = boxes.unbind(dim=-1)
    center_x = (x_min + x_max) / 2
    center_y = (y_min + y_max) / 2
    box_width = x_max - x_min
    box_height = y_max - y_min

    boxes_cxcywh = torch.stack(
        (center_x, center_y, box_width, box_height), dim=-1
    )
    scale = boxes.new_tensor((width, height, width, height))
    return boxes_cxcywh / scale
