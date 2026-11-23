"""動作確認用に、図形画像とCOCO形式の注釈を生成する。"""

import json
import random
from pathlib import Path

from PIL import Image, ImageDraw


def make_demo_data(
    output_dir: str | Path = "data/demo",
    *,
    train_count: int = 16,
    val_count: int = 4,
    seed: int = 42,
) -> Path:
    """赤いrectangle・青いellipseの2クラスを生成する。既存データは上書きしない。

    96×64の各画像に1〜3個の図形を配置する。写真の検出精度を測るデータではなく、
    入出力と学習の動作確認用。ローカル乱数を使い、同じseedで再現できる。
    """
    if train_count <= 0 or val_count <= 0:
        raise ValueError("train_count and val_count must be positive")
    root = Path(output_dir)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"output directory is not empty: {root}")
    annotations_dir = root / "annotations"
    annotations_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    categories = [{"id": 1, "name": "rectangle"}, {"id": 2, "name": "ellipse"}]
    for split, count in (("train2017", train_count), ("val2017", val_count)):
        image_dir = root / split
        image_dir.mkdir()
        images, annotations = [], []
        for image_id in range(1, count + 1):
            image = Image.new(
                "RGB", (96, 64), tuple(rng.randint(235, 255) for _ in range(3))
            )
            drawing = ImageDraw.Draw(image)
            for slot in rng.sample(range(3), rng.randint(1, 3)):
                x, y = slot * 32 + rng.randint(0, 7), rng.randint(4, 32)
                width, height = rng.randint(12, 24), rng.randint(12, 24)
                category_id = len(annotations) % 2 + 1
                bounds = (x, y, x + width - 1, y + height - 1)
                if category_id == 1:
                    drawing.rectangle(bounds, fill=(220, 40, 40))
                else:
                    drawing.ellipse(bounds, fill=(40, 80, 220))
                annotations.append(
                    {
                        "id": len(annotations) + 1,
                        "image_id": image_id,
                        "category_id": category_id,
                        "bbox": [x, y, width, height],
                        "area": width * height,
                        "iscrowd": 0,
                    }
                )
            file_name = f"{image_id:06d}.png"
            image.save(image_dir / file_name)
            images.append(
                {"id": image_id, "file_name": file_name, "height": 64, "width": 96}
            )
        (annotations_dir / f"instances_{split}.json").write_text(
            json.dumps(
                {
                    "images": images,
                    "annotations": annotations,
                    "categories": categories,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    return root
