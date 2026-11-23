"""保存済みログから、学習・検証・候補特徴の推移を描画する。"""

import json
import os
from collections.abc import Sequence
from pathlib import Path


def load_history(log_dirs: Sequence[str | Path], filename: str) -> list[dict]:
    """エポック順に結合する。重複エポックは後に指定したフォルダを優先する。"""
    rows = {}
    for directory in log_dirs:
        directory = Path(directory)
        if not directory.is_dir():
            raise FileNotFoundError(f"log directory does not exist: {directory}")
        path = directory / filename
        if not path.exists():
            continue
        for line_number, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), 1
        ):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                rows[row["epoch"]] = row
            except (ValueError, KeyError, TypeError) as error:
                raise ValueError(f"invalid log at {path}:{line_number}") from error
    return [rows[epoch] for epoch in sorted(rows)]


def plot_training(log_dirs: Sequence[str | Path], output: str | Path) -> Path:
    """ログを読むだけで4分割グラフを保存する。モデルやデータセットは読み込まない。"""
    train = load_history(log_dirs, "metrics.jsonl")
    validation = load_history(log_dirs, "validation.jsonl")
    probes = load_history(log_dirs, "query-probe.jsonl")
    if not train and not validation and not probes:
        raise ValueError("no training, validation, or query-probe logs found")
    if len({row["image_id"] for row in probes}) > 1:
        raise ValueError(
            "query-probe logs refer to different images; plot them separately"
        )
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(output.parent.resolve() / ".mpl-cache"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), layout="constrained")
    axes[0, 0].set(title="Validation AP", ylabel="COCO AP")
    if validation:
        axes[0, 0].plot(
            [row["epoch"] for row in validation],
            [row["AP"] for row in validation],
            marker="o",
        )
    else:
        axes[0, 0].text(
            0.5, 0.5, "No validation data", transform=axes[0, 0].transAxes, ha="center"
        )
    axes[0, 1].set(title="Total loss", ylabel="Loss")
    for rows, label in ((train, "Train"), (validation, "Validation")):
        if rows:
            axes[0, 1].plot(
                [row["epoch"] for row in rows],
                [row["loss_total"] for row in rows],
                label=label,
            )
    if train or validation:
        axes[0, 1].legend()
    axes[1, 0].set(
        title="Fixed validation image: query object scores",
        ylabel="Best object probability",
    )
    axes[1, 1].set(
        title="Fixed validation image: feature differences", ylabel="1 - mean cosine"
    )
    if probes:
        # リスト上の位置ではなく、保存されたquery番号で各系列を対応づける。
        query_ids = sorted(
            {query["query_index"] for row in probes for query in row["queries"]}
        )
        for index in query_ids:
            points = [
                (row["epoch"], max(query["class_probabilities"][:-1]))
                for row in probes
                for query in row["queries"]
                if query["query_index"] == index
            ]
            axes[1, 0].plot(
                [p[0] for p in points],
                [p[1] for p in points],
                linewidth=1,
                label=f"Query {index}",
            )
        if len(query_ids) <= 10:
            axes[1, 0].legend(ncol=5, fontsize=8)
        for layer in range(len(probes[0]["decoder_layers"])):
            points = [
                (row["epoch"], row["decoder_layers"][layer]["mean_pairwise_cosine"])
                for row in probes
                if layer < len(row["decoder_layers"])
            ]
            points = [
                (epoch, 1 - cosine) for epoch, cosine in points if cosine is not None
            ]
            if points:
                axes[1, 1].plot(
                    [p[0] for p in points],
                    [p[1] for p in points],
                    marker="o",
                    label=f"Decoder layer {layer + 1}",
                )
        if axes[1, 1].lines:
            axes[1, 1].legend()
    else:
        for axis in axes[1]:
            axis.text(
                0.5, 0.5, "No query-probe data", transform=axis.transAxes, ha="center"
            )
    for axis in axes.flat:
        axis.set_xlabel("Epoch")
        axis.grid(alpha=0.2)
    try:
        fig.savefig(output, dpi=180)
    finally:
        plt.close(fig)
    return output


def run_plot_training(args) -> None:
    output = plot_training(args.log_dirs, args.output)
    print(f"Saved: {output}")
