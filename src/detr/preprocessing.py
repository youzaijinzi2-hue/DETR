"""学習と推論で共有する、RGB画像のリサイズと標準化。"""

from torch import Tensor
from torchvision.transforms import functional as TF


def prepare_image(image: Tensor, short_side: int = 800, max_size: int = 1333) -> Tensor:
    """(3, H, W)のuint8画像を、標準化済みfloat32画像に変換する。"""
    if short_side <= 0 or max_size <= 0:
        raise ValueError("short_side and max_size must be positive")
    height, width = image.shape[-2:]
    scale = min(short_side / min(height, width), max_size / max(height, width))
    size = [max(1, round(height * scale)), max(1, round(width * scale))]
    image = TF.resize(image, size, antialias=True).float() / 255
    return TF.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
