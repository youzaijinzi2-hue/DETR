import json

import pytest
from PIL import Image

from detr.cli import main
from detr.plot_training import load_history, plot_training


def write_log(directory, filename, rows):
    directory.mkdir(exist_ok=True)
    (directory / filename).write_text("\n".join(json.dumps(row) for row in rows))


def test_merges_and_sorts_epochs_with_later_directory_taking_priority(tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    write_log(
        first,
        "metrics.jsonl",
        [{"epoch": 2, "loss_total": 4}, {"epoch": 1, "loss_total": 5}],
    )
    write_log(
        second,
        "metrics.jsonl",
        [{"epoch": 2, "loss_total": 3}, {"epoch": 3, "loss_total": 2}],
    )
    assert load_history([first, second], "metrics.jsonl") == [
        {"epoch": 1, "loss_total": 5},
        {"epoch": 2, "loss_total": 3},
        {"epoch": 3, "loss_total": 2},
    ]


def test_cli_creates_chart_without_model_or_dataset(tmp_path, monkeypatch):
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "cache"))
    logs = tmp_path / "logs"
    write_log(
        logs,
        "metrics.jsonl",
        [{"epoch": 1, "loss_total": 5}, {"epoch": 2, "loss_total": 3}],
    )
    write_log(
        logs,
        "validation.jsonl",
        [
            {"epoch": 1, "loss_total": 6, "AP": 0.1},
            {"epoch": 2, "loss_total": 4, "AP": 0.2},
        ],
    )
    write_log(
        logs,
        "query-probe.jsonl",
        [
            {
                "epoch": epoch,
                "image_id": 7,
                "queries": [
                    {"query_index": 1, "class_probabilities": [0.2, 0.8]},
                    {"query_index": 0, "class_probabilities": [0.9, 0.1]},
                ],
                "decoder_layers": [{"mean_pairwise_cosine": 0.5}],
            }
            for epoch in (1, 2)
        ],
    )
    output = tmp_path / "plot.png"
    main(["plot-training", "--log-dirs", str(logs), "--output", str(output)])
    with Image.open(output) as image:
        assert image.size == (2160, 1440)
        assert any(low != high for low, high in image.convert("RGB").getextrema())


def test_training_only_logs_are_supported(tmp_path, monkeypatch):
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "cache"))
    write_log(tmp_path, "metrics.jsonl", [{"epoch": 1, "loss_total": 2}])
    assert plot_training([tmp_path], tmp_path / "plot.png").exists()


def test_reports_missing_or_broken_logs(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_history([tmp_path / "missing"], "metrics.jsonl")
    with pytest.raises(ValueError, match="no training"):
        plot_training([tmp_path], tmp_path / "plot.png")
    (tmp_path / "metrics.jsonl").write_text('{"epoch":')
    with pytest.raises(ValueError, match="metrics.jsonl:1"):
        load_history([tmp_path], "metrics.jsonl")
