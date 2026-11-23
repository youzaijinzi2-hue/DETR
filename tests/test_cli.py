import json

import pytest
import torch
from PIL import Image, ImageDraw

from detr.checkpoint import load_model
from detr.cli import main


def test_training_saves_reloadable_checkpoint_and_predicts_annotated_image(tmp_path):
    torch.manual_seed(21)
    image_path = tmp_path / "image.png"
    image = Image.new("RGB", (64, 32), "white")
    ImageDraw.Draw(image).rectangle((16, 8, 48, 24), fill="blue")
    image.save(image_path)
    annotation_path = tmp_path / "annotations.json"
    annotation_path.write_text(
        json.dumps(
            {
                "images": [{"id": 1, "file_name": "image.png"}],
                "categories": [{"id": 42, "name": "rectangle"}],
                "annotations": [
                    {"image_id": 1, "category_id": 42, "bbox": [16, 8, 32, 16]}
                ],
            }
        )
    )
    output_dir = tmp_path / "train"

    main(
        [
            "train",
            "--image-dir",
            str(tmp_path),
            "--annotations",
            str(annotation_path),
            "--output-dir",
            str(output_dir),
            "--epochs",
            "1",
            "--batch-size",
            "1",
            "--short-side",
            "32",
            "--max-size",
            "64",
            "--no-augmentation",
            "--num-queries",
            "3",
            "--d-model",
            "8",
            "--nhead",
            "2",
            "--dim-feedforward",
            "16",
            "--encoder-layers",
            "1",
            "--decoder-layers",
            "2",
            "--dropout",
            "0",
            "--save-every",
            "50",
        ]
    )

    checkpoint_path = output_dir / "checkpoint.pt"
    model, checkpoint = load_model(checkpoint_path)
    assert not model.training
    assert checkpoint["epoch"] == 1
    assert checkpoint["model_config"]["dropout"] == 0
    assert checkpoint["class_names"] == ["rectangle"]
    assert checkpoint["label_to_category_id"] == {0: 42}
    assert checkpoint["optimizer"]["state"]
    assert not (output_dir / "checkpoint.pt.tmp").exists()
    for name, parameter in model.state_dict().items():
        torch.testing.assert_close(parameter, checkpoint["model"][name])
    metrics = json.loads((output_dir / "metrics.jsonl").read_text())
    assert metrics["epoch"] == 1
    assert metrics["loss_total"] > 0

    prediction_path = tmp_path / "predicted.png"
    main(
        [
            "predict",
            "--checkpoint",
            str(checkpoint_path),
            "--image",
            str(image_path),
            "--output",
            str(prediction_path),
            "--threshold",
            "0",
        ]
    )

    with Image.open(prediction_path) as result:
        assert result.width > 64
        assert result.height >= 32
        assert result.tobytes() != image.tobytes()
    records = json.loads(prediction_path.with_suffix(".json").read_text())
    assert len(records) == 3
    assert all(record["class_name"] == "rectangle" for record in records)
    assert all(record["category_id"] == 42 for record in records)
    assert all(len(record["box_xyxy"]) == 4 for record in records)

    comparison_path = tmp_path / "comparison.png"
    main(
        [
            "predict",
            "--checkpoint",
            str(checkpoint_path),
            "--image",
            str(image_path),
            "--output",
            str(comparison_path),
            "--threshold",
            "0",
            "--top-k",
            "1",
            "--scale",
            "3",
            "--annotations",
            str(annotation_path),
        ]
    )
    with Image.open(comparison_path) as comparison:
        assert comparison.width > 192
        assert comparison.height >= 96
        assert any(
            comparison.getpixel((x, y)) == (0, 130, 60)
            for x in range(comparison.width)
            for y in range(comparison.height)
        )
    assert len(json.loads(comparison_path.with_suffix(".json").read_text())) == 1
    report = json.loads(
        comparison_path.with_name("comparison.diagnostics.json").read_text()
    )
    assert len(report["ground_truth"]) == 1
    assert len(report["queries"]) == 3
    assert all(0 <= query["best_same_class_gt_iou"] <= 1 for query in report["queries"])

    resumed_dir = tmp_path / "resumed"
    main(
        [
            "train",
            "--resume",
            str(checkpoint_path),
            "--epochs",
            "2",
            "--image-dir",
            str(tmp_path),
            "--annotations",
            str(annotation_path),
            "--output-dir",
            str(resumed_dir),
            "--batch-size",
            "1",
            "--no-augmentation",
            "--val-image-dir",
            str(tmp_path),
            "--val-annotations",
            str(annotation_path),
            "--eval-every",
            "1",
            "--save-every",
            "50",
            "--lr",
            "0.1",
        ]
    )
    resumed = torch.load(resumed_dir / "checkpoint.pt", weights_only=True)
    assert resumed["epoch"] == 2
    assert resumed["model_config"] == checkpoint["model_config"]
    assert (
        resumed["optimizer"]["param_groups"][0]["lr"]
        == checkpoint["optimizer"]["param_groups"][0]["lr"]
    )
    for key, state in checkpoint["optimizer"]["state"].items():
        torch.testing.assert_close(
            resumed["optimizer"]["state"][key]["step"], state["step"] + 1
        )
    assert "rng_state" in resumed
    assert (resumed_dir / "best.pt").exists()
    validation_rows = [
        json.loads(line)
        for line in (resumed_dir / "validation.jsonl").read_text().splitlines()
    ]
    assert [row["epoch"] for row in validation_rows] == [1, 2]
    probes = [
        json.loads(line)
        for line in (resumed_dir / "query-probe.jsonl").read_text().splitlines()
    ]
    assert [row["epoch"] for row in probes] == [1, 2]
    assert all(len(row["queries"]) == 3 for row in probes)
    assert json.loads((resumed_dir / "metrics.jsonl").read_text())["epoch"] == 2
    assert torch.load(checkpoint_path, weights_only=True)["epoch"] == 1
    with pytest.raises(ValueError, match="epochs must exceed checkpoint"):
        main(
            [
                "train",
                "--resume",
                str(checkpoint_path),
                "--epochs",
                "1",
                "--image-dir",
                str(tmp_path),
                "--annotations",
                str(annotation_path),
            ]
        )
