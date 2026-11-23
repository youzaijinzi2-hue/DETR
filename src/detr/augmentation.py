"""画像と正解ボックスを一緒に変換するデータ拡張。"""

from torch import Tensor


def horizontal_flip(image: Tensor, boxes: Tensor) -> tuple[Tensor, Tensor]:
    """画像と正規化cxcywhボックスを左右反転する（入力は変更しない）。

    左右反転後の中心xは1-cx。中心y・幅・高さは変わらない。
    boxesは(N, 4)で、物体がない(0, 4)の入力も扱う。
    """
    flipped_boxes = boxes.clone()
    flipped_boxes[:, 0] = 1 - boxes[:, 0]
    return image.flip(-1), flipped_boxes
