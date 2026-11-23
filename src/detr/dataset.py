"""COCO形式の検出データを、DETRの画像・正解テンソルへ変換する。"""

import json
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

from detr.boxes import normalize_boxes_xyxy


class CocoDetectionDataset(Dataset):
    """COCOのimages/annotations/categoriesを持つJSONを読み込む。

    物体検出のみを対象に、iscrowdの注釈と面積ゼロのボックスを除外する。
    category_idは昇順で0始まりに詰め直す（公式コードのID保持とは異なる）。
    学習・検証のJSONには同じcategories一覧が必要。
    固定サイズへのリサイズと標準化を行う基本版。ランダム拡張は未実装。
    """

    def __init__(
        self,
        image_dir: str | Path,
        annotation_file: str | Path,
        *,
        short_side: int = 800,
        max_size: int = 1333,
    ) -> None:
        if short_side <= 0 or max_size <= 0:
            raise ValueError("short_side and max_size must be positive")
        self.image_dir = Path(image_dir)
        self.short_side = short_side
        self.max_size = max_size
        with Path(annotation_file).open(encoding="utf-8") as file:
            data = json.load(file)
        self.images = sorted(data["images"], key=lambda image: image["id"])
        categories = sorted(data["categories"], key=lambda category: category["id"])
        self.category_id_to_label = {
            category["id"]: label for label, category in enumerate(categories)
        }
        self.label_to_category_id = {
            label: category_id
            for category_id, label in self.category_id_to_label.items()
        }
        self.class_names = [category["name"] for category in categories]
        self.num_classes = len(categories)
        if self.num_classes == 0:
            raise ValueError("categories must not be empty")
        image_ids = {image["id"] for image in self.images}
        self.annotations = defaultdict(list)
        for annotation in data["annotations"]:
            if annotation["image_id"] not in image_ids:
                raise ValueError("annotation references an unknown image_id")
            if annotation["category_id"] not in self.category_id_to_label:
                raise ValueError("annotation references an unknown category_id")
            self.annotations[annotation["image_id"]].append(annotation)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> tuple[Tensor, dict[str, Tensor]]:
        record = self.images[index]
        with Image.open(self.image_dir / record["file_name"]) as source:
            image = TF.pil_to_tensor(source.convert("RGB"))
        height, width = image.shape[-2:]
        annotations = [
            annotation
            for annotation in self.annotations[record["id"]]
            if not annotation.get("iscrowd", 0)
        ]
        # COCOは左上x,y,幅,高さ。まず左上・右下のピクセル座標へ変える。
        boxes = torch.tensor(
            [annotation["bbox"] for annotation in annotations], dtype=torch.float32
        ).reshape(-1, 4)
        labels = torch.tensor(
            [
                self.category_id_to_label[annotation["category_id"]]
                for annotation in annotations
            ],
            dtype=torch.int64,
        )
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(0, width)
        boxes[:, 1::2].clamp_(0, height)
        valid = (boxes[:, 2:] > boxes[:, :2]).all(dim=1)
        boxes = normalize_boxes_xyxy(boxes[valid], (height, width))
        labels = labels[valid]

        scale = min(
            self.short_side / min(height, width), self.max_size / max(height, width)
        )
        new_height = max(1, round(height * scale))
        new_width = max(1, round(width * scale))
        image = TF.resize(image, [new_height, new_width], antialias=True)
        image = image.float() / 255
        image = TF.normalize(
            image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        # 画像全体のリサイズでは、正規化したボックスの値は変わらない。
        target = {
            "labels": labels,
            "boxes": boxes,
            "image_id": torch.tensor([record["id"]]),
            "orig_size": torch.tensor([height, width]),
            "size": torch.tensor([new_height, new_width]),
        }
        return image, target
