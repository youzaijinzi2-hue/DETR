"""固定した画像1枚で、学習中の候補特徴・確率・実際の対応先を追跡する。"""

import json
import os
from itertools import pairwise
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from detr.checkpoint import load_model, save_checkpoint
from detr.criterion import SetCriterion
from detr.dataset import CocoDetectionDataset


def set_dropout(model: nn.Module, probability: float) -> None:
    """重みを変えず、通常のDropoutとAttention内のDropout率をそろえる。"""
    if not 0 <= probability <= 1:
        raise ValueError("dropout must be between 0 and 1")
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.p = probability
        elif isinstance(module, nn.MultiheadAttention):
            module.dropout = probability


def _observe(model, images):
    captured = {}

    def decoder_hook(module, args, output):
        captured["decoded"] = output.detach()

    def encoder_hook(module, args, output):
        captured["memory"] = output.detach()

    hooks = [
        model.decoder.register_forward_hook(decoder_hook),
        model.encoder.register_forward_hook(encoder_hook),
    ]
    try:
        outputs = model(images)
    finally:
        for hook in hooks:
            hook.remove()
    return outputs, captured


def feature_similarity(features):
    """(N,d)の異なる候補間の平均cos類似度と、候補方向の標準偏差。"""
    features = features.float()
    normalized = F.normalize(features, dim=-1)
    cosine = normalized @ normalized.T
    off_diagonal = ~torch.eye(len(features), dtype=torch.bool, device=features.device)
    return {
        "mean_pairwise_cosine": cosine[off_diagonal].mean().item()
        if len(features) > 1
        else None,
        "mean_feature_std": features.std(dim=0, correction=0).mean().item(),
    }


def describe_queries(outputs, captured, indices, targets):
    """元のquery番号順。expected_classは実際に採用されたマッチングから作る。"""
    probabilities = outputs["pred_logits"][0].detach().softmax(-1).cpu()
    features = captured["decoded"][-1, 0].cpu()
    predicted_indices, target_indices = indices[0]
    matches = dict(zip(predicted_indices.tolist(), target_indices.tolist()))
    queries = []
    for index in range(len(features)):
        target_index = matches.get(index)
        expected_class = (
            targets[0]["labels"][target_index].item()
            if target_index is not None
            else probabilities.shape[-1] - 1
        )
        queries.append(
            {
                "query_index": index,
                "matched_target_index": target_index,
                "expected_class": expected_class,
                "class_probabilities": probabilities[index].tolist(),
                "feature": features[index].tolist(),
            }
        )
    return {
        "queries": queries,
        "decoder_layers": [
            feature_similarity(layer[0]) for layer in captured["decoded"]
        ],
        "encoder_memory": feature_similarity(captured["memory"][0]),
    }


@torch.no_grad()
def query_snapshot(model, criterion, image, target):
    """固定画像を学習更新なしで観測し、候補特徴・確率・対応先を返す。"""
    was_training = model.training
    device = next(model.parameters()).device
    targets = [{key: value.to(device) for key, value in target.items()}]
    model.eval()
    try:
        outputs, captured = _observe(model, [image.to(device)])
        return describe_queries(
            outputs, captured, criterion.matcher(outputs, targets), targets
        )
    finally:
        model.train(was_training)


def trace_queries(
    model, criterion, optimizer, image, target, *, steps: int, output_file: str | Path
):
    """現在の重みからsteps回更新する。元のチェックポイントには書き込まない。

    train側はDropout有効で実際に損失へ使われた対応をフックで取得する。
    eval側は各更新後、同じ画像をDropout無効で観測する。過去の学習履歴の復元ではない。
    """
    if steps < 0:
        raise ValueError("steps must be non-negative")
    device = next(model.parameters()).device
    images = [image.to(device)]
    targets = [{key: value.to(device) for key, value in target.items()}]
    criterion.to(device)
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    records = []

    def evaluation_snapshot():
        model.eval()
        with torch.no_grad():
            outputs, captured = _observe(model, images)
            matches = criterion.matcher(outputs, targets)
            result = describe_queries(outputs, captured, matches, targets)
            result["losses"] = {
                key: value.item() for key, value in criterion(outputs, targets).items()
            }
        return result

    with output_file.open("w", encoding="utf-8") as stream:
        initial = {
            "step": 0,
            "dropout": model.decoder.layers[0].self_attn.dropout,
            "eval": evaluation_snapshot(),
            "object_query_similarity": feature_similarity(
                model.object_queries.embedding.weight.detach()
            ),
            "object_query_embeddings": model.object_queries.embedding.weight.detach()
            .cpu()
            .tolist(),
        }
        records.append(initial)
        stream.write(json.dumps(initial, allow_nan=False) + "\n")
        for step in range(1, steps + 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            outputs, captured = _observe(model, images)
            actual_matches = []

            def matcher_hook(module, args, result, destination=actual_matches):
                destination.append(result)

            hook = criterion.matcher.register_forward_hook(matcher_hook)
            try:
                losses = criterion(outputs, targets)
            finally:
                hook.remove()
            if not torch.isfinite(losses["loss_total"]).item():
                raise FloatingPointError("nonfinite trace loss")
            losses["loss_total"].backward()
            training = describe_queries(outputs, captured, actual_matches[0], targets)
            training["losses"] = {key: value.item() for key, value in losses.items()}
            training["query_gradient_norms"] = (
                model.object_queries.embedding.weight.grad.detach()
                .norm(dim=-1)
                .cpu()
                .tolist()
            )
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 0.1, error_if_nonfinite=True
            )
            optimizer.step()
            record = {"step": step, "train": training, "eval": evaluation_snapshot()}
            records.append(record)
            stream.write(json.dumps(record, allow_nan=False) + "\n")
            stream.flush()
    return records


def run_trace(args):
    if args.steps <= 0:
        raise ValueError("steps must be positive")
    torch.manual_seed(args.seed)
    model, checkpoint = load_model(args.checkpoint, args.device)
    model_config = dict(checkpoint["model_config"])
    if args.dropout is not None:
        set_dropout(model, args.dropout)
        model_config["dropout"] = args.dropout
    dataset = CocoDetectionDataset(
        args.image_dir, args.annotations, training=False, **checkpoint["preprocessing"]
    )
    if len(dataset) != 1:
        raise ValueError("trace-queries requires exactly one image")
    if dataset.label_to_category_id != checkpoint["label_to_category_id"]:
        raise ValueError("trace categories must match checkpoint")
    optimizer = torch.optim.AdamW(
        [
            {
                "params": [
                    p
                    for name, p in model.named_parameters()
                    if not name.startswith("backbone.")
                ]
            },
            {"params": model.backbone.parameters()},
        ]
    )
    optimizer.load_state_dict(checkpoint["optimizer"])
    output_dir = Path(args.output_dir)
    # チェックポイントを保存した元フォルダへの意図しない上書きを避ける。
    if output_dir.resolve() == Path(args.checkpoint).resolve().parent:
        raise ValueError("use a separate output directory for tracing")
    image, target = dataset[0]
    records = trace_queries(
        model,
        SetCriterion(dataset.num_classes),
        optimizer,
        image,
        target,
        steps=args.steps,
        output_file=output_dir / "query-trace.jsonl",
    )
    save_checkpoint(
        output_dir / "checkpoint.pt",
        model,
        optimizer,
        model_config=model_config,
        class_names=checkpoint["class_names"],
        label_to_category_id=checkpoint["label_to_category_id"],
        preprocessing=checkpoint["preprocessing"],
        epoch=checkpoint["epoch"] + args.steps,
    )
    summary = summarize_trace(records)
    summary["dropout"] = model_config.get("dropout", 0.1)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    plot_trace(records, output_dir / "query-trace.png")
    print(
        json.dumps(
            {
                key: value
                for key, value in summary.items()
                if key not in {"train_matched_queries", "eval_matched_queries"}
            },
            indent=2,
        )
    )


def summarize_trace(records):
    def matched(snapshot):
        return [
            query["query_index"]
            for query in snapshot["queries"]
            if query["matched_target_index"] is not None
        ]

    train_matches = [matched(record["train"]) for record in records[1:]]
    eval_matches = [matched(record["eval"]) for record in records]
    return {
        "additional_steps": len(records) - 1,
        "train_matched_queries": train_matches,
        "eval_matched_queries": eval_matches,
        "train_matching_switches": sum(a != b for a, b in pairwise(train_matches)),
        "eval_matching_switches": sum(a != b for a, b in pairwise(eval_matches)),
        "initial_eval_similarity": records[0]["eval"]["decoder_layers"],
        "final_eval_similarity": records[-1]["eval"]["decoder_layers"],
        "initial_eval_loss": records[0]["eval"]["losses"]["loss_total"],
        "final_eval_loss": records[-1]["eval"]["losses"]["loss_total"],
    }


def plot_trace(records, output_file):
    """候補の確率、対応先、特徴の類似度の推移をPNGにまとめる。"""
    output_file = Path(output_file)
    os.environ.setdefault("MPLCONFIGDIR", str(output_file.parent / ".mpl-cache"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = [record["step"] for record in records]
    fig, axes = plt.subplots(3, 1, figsize=(9, 9), layout="constrained")
    for query in records[0]["eval"]["queries"]:
        index = query["query_index"]
        values = [
            max(record["eval"]["queries"][index]["class_probabilities"][:-1])
            for record in records
        ]
        axes[0].plot(steps, values, linewidth=1, label=f"Query {index}")
    axes[0].set(
        ylabel="Best object probability",
        title="Fixed-image query tracing (continued training)",
    )
    axes[0].legend(ncol=5, fontsize=8)
    dropout = records[0].get("dropout", 0.1)
    for phase, marker, label in (
        ("train", "o", f"Training (dropout={dropout:g})"),
        ("eval", ".", "Evaluation (dropout off)"),
    ):
        xs, ys = [], []
        for record in records:
            if phase in record:
                for query in record[phase]["queries"]:
                    if query["matched_target_index"] is not None:
                        xs.append(record["step"])
                        ys.append(query["query_index"])
        axes[1].scatter(xs, ys, s=12, marker=marker, label=label)
    axes[1].set(ylabel="Matched query index")
    axes[1].legend()
    for layer in range(len(records[0]["eval"]["decoder_layers"])):
        distances = [
            max(
                1e-9,
                1 - record["eval"]["decoder_layers"][layer]["mean_pairwise_cosine"],
            )
            for record in records
        ]
        axes[2].plot(steps, distances, label=f"Decoder layer {layer + 1}")
    axes[2].set(
        yscale="log",
        ylabel="1 - mean feature cosine",
        xlabel="Additional optimizer steps",
    )
    axes[2].legend()
    for axis in axes:
        axis.grid(alpha=0.2)
    fig.savefig(output_file, dpi=160)
    plt.close(fig)
