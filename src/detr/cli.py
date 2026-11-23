"""学習・推論を実行するコマンドライン入口。"""

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.transforms import functional as TF

from detr.checkpoint import load_model, save_checkpoint
from detr.criterion import SetCriterion
from detr.dataset import CocoDetectionDataset
from detr.demo_data import make_demo_data
from detr.diagnostics import prediction_diagnostics, read_ground_truth
from detr.engine import collate_fn, train_one_epoch
from detr.evaluation import evaluate_model
from detr.model import DETR
from detr.plot_training import run_plot_training
from detr.postprocess import postprocess
from detr.preprocessing import prepare_image
from detr.query_trace import query_snapshot, run_trace
from detr.rendering import draw_predictions


def train(args: argparse.Namespace) -> None:
    if args.epochs <= 0 or args.batch_size <= 0:
        raise ValueError("epochs and batch_size must be positive")
    if args.save_every <= 0:
        raise ValueError("save-every must be positive")
    if args.eval_every <= 0:
        raise ValueError("eval-every must be positive")
    if bool(args.val_image_dir) != bool(args.val_annotations):
        raise ValueError("val-image-dir and val-annotations must be provided together")
    torch.manual_seed(args.seed)
    checkpoint = None
    start_epoch = 0
    if args.resume:
        model, checkpoint = load_model(args.resume, args.device)
        start_epoch = checkpoint["epoch"]
        if args.epochs <= start_epoch:
            raise ValueError(
                "epochs must exceed checkpoint epoch (epochs is the final epoch)"
            )
        args.short_side = checkpoint["preprocessing"]["short_side"]
        args.max_size = checkpoint["preprocessing"]["max_size"]
    dataset_options = {}
    if args.train_scales is not None:
        dataset_options["train_scales"] = args.train_scales
    dataset = CocoDetectionDataset(
        args.image_dir,
        args.annotations,
        training=not args.no_augmentation,
        short_side=args.short_side,
        max_size=args.max_size,
        **dataset_options,
    )
    if len(dataset) == 0:
        raise ValueError("training dataset must not be empty")
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=collate_fn,
    )
    config = {
        "num_classes": dataset.num_classes,
        "num_queries": args.num_queries,
        "d_model": args.d_model,
        "nhead": args.nhead,
        "dim_feedforward": args.dim_feedforward,
        "num_encoder_layers": args.encoder_layers,
        "num_decoder_layers": args.decoder_layers,
        "pretrained_backbone": args.pretrained_backbone,
        "aux_loss": True,
        "dropout": args.dropout,
    }
    if checkpoint is not None:
        config = checkpoint["model_config"]
        if (
            dataset.label_to_category_id != checkpoint["label_to_category_id"]
            or dataset.class_names != checkpoint["class_names"]
        ):
            raise ValueError("training categories must match checkpoint")
    else:
        model = DETR(**config).to(args.device)
    criterion = SetCriterion(dataset.num_classes)
    # Backboneは小さい学習率で更新する。
    optimizer = torch.optim.AdamW(
        [
            {
                "params": [
                    p
                    for name, p in model.named_parameters()
                    if not name.startswith("backbone.")
                ],
                "lr": args.lr,
            },
            {"params": model.backbone.parameters(), "lr": args.lr_backbone},
        ],
        weight_decay=1e-4,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if checkpoint is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
        if "rng_state" in checkpoint:
            torch.set_rng_state(checkpoint["rng_state"])
    val_loader = None
    if args.val_annotations:
        validation = CocoDetectionDataset(
            args.val_image_dir,
            args.val_annotations,
            short_side=args.short_side,
            max_size=args.max_size,
        )
        if (
            validation.label_to_category_id != dataset.label_to_category_id
            or validation.class_names != dataset.class_names
        ):
            raise ValueError("validation categories must match training categories")
        val_loader = DataLoader(
            validation,
            batch_size=args.batch_size,
            collate_fn=collate_fn,
            num_workers=args.workers,
        )
    best_ap = checkpoint.get("best_ap") if checkpoint is not None else None
    best_ap = -1.0 if best_ap is None else best_ap
    if not (output_dir / "best.pt").exists():
        best_ap = -1.0
    if not args.resume:
        for name in ("validation.jsonl", "query-probe.jsonl"):
            (output_dir / name).write_text("", encoding="utf-8")

    def save(path, epoch):
        save_checkpoint(
            path,
            model,
            optimizer,
            model_config=config,
            class_names=dataset.class_names,
            label_to_category_id=dataset.label_to_category_id,
            preprocessing={"short_side": args.short_side, "max_size": args.max_size},
            epoch=epoch,
            best_ap=best_ap,
        )

    def validate(epoch):
        nonlocal best_ap
        metrics = evaluate_model(
            model,
            criterion,
            val_loader,
            annotation_file=args.val_annotations,
            label_to_category_id=dataset.label_to_category_id,
        )
        record = json.dumps({"epoch": epoch, **metrics}, allow_nan=False)
        with (output_dir / "validation.jsonl").open("a", encoding="utf-8") as val_log:
            val_log.write(record + "\n")
        print("Validation: " + record, flush=True)
        probe_image, probe_target = validation[0]
        probe = query_snapshot(model, criterion, probe_image, probe_target)
        with (output_dir / "query-probe.jsonl").open(
            "a", encoding="utf-8"
        ) as probe_log:
            probe_log.write(
                json.dumps(
                    {
                        "epoch": epoch,
                        "image_id": probe_target["image_id"].item(),
                        **probe,
                    },
                    allow_nan=False,
                )
                + "\n"
            )
        if metrics["AP"] > best_ap or not (output_dir / "best.pt").exists():
            best_ap = max(best_ap, metrics["AP"])
            save(output_dir / "best.pt", epoch)

    log_path = output_dir / "metrics.jsonl"
    if args.resume and log_path.exists():
        lines = [line for line in log_path.read_text().splitlines() if line.strip()]
        if lines and json.loads(lines[-1])["epoch"] != start_epoch:
            raise ValueError(
                "output log epoch differs from checkpoint; use a new output directory"
            )
    if val_loader is not None and checkpoint is not None:
        validate(start_epoch)
    with log_path.open("a" if args.resume else "w", encoding="utf-8") as log:
        for epoch in range(start_epoch + 1, args.epochs + 1):
            metrics = train_one_epoch(model, criterion, loader, optimizer)
            record = json.dumps({"epoch": epoch, **metrics}, allow_nan=False)
            print(record, flush=True)
            log.write(record + "\n")
            log.flush()
            if val_loader is not None and (
                epoch % args.eval_every == 0 or epoch == args.epochs
            ):
                validate(epoch)
            if epoch % args.save_every != 0 and epoch != args.epochs:
                continue
            save(output_dir / "checkpoint.pt", epoch)


def predict(args: argparse.Namespace) -> None:
    if args.scale <= 0 or (args.top_k is not None and args.top_k <= 0):
        raise ValueError("scale and top-k must be positive")
    model, checkpoint = load_model(args.checkpoint, args.device)
    with Image.open(args.image) as source:
        image = source.convert("RGB")
    tensor = prepare_image(TF.pil_to_tensor(image), **checkpoint["preprocessing"])
    with torch.no_grad():
        outputs = model([tensor.to(args.device)])
    result = postprocess(
        outputs, torch.tensor([[image.height, image.width]]), threshold=args.threshold
    )[0]
    result = {key: value.cpu() for key, value in result.items()}
    ground_truth = None
    report = None
    if args.annotations:
        ground_truth = read_ground_truth(args.annotations, args.image, args.image_id)
        report = prediction_diagnostics(
            outputs,
            (image.height, image.width),
            ground_truth,
            checkpoint["class_names"],
            checkpoint["label_to_category_id"],
        )
    records = []
    for score, label, box in zip(
        result["scores"][: args.top_k],
        result["labels"][: args.top_k],
        result["boxes"][: args.top_k],
    ):
        label_id = label.item()
        name = checkpoint["class_names"][label_id]
        coordinates = box.tolist()
        records.append(
            {
                "label": label_id,
                "category_id": checkpoint["label_to_category_id"][label_id],
                "class_name": name,
                "score": score.item(),
                "box_xyxy": coordinates,
            }
        )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    image = draw_predictions(image, records, ground_truth, scale=args.scale)
    image.save(output)
    if report is not None:
        report_path = output.with_name(output.stem + ".diagnostics.json")
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(
            f"Saved ground truth (green), predictions (red), and diagnostics to {report_path}"
        )
    output.with_suffix(".json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Saved {len(records)} predictions to {output}")


def demo_data(args: argparse.Namespace) -> None:
    root = make_demo_data(
        args.output_dir,
        train_count=args.train_count,
        val_count=args.val_count,
        seed=args.seed,
    )
    print(f"Created demo data at {root}")


def evaluate(args: argparse.Namespace) -> None:
    model, checkpoint = load_model(args.checkpoint, args.device)
    dataset = CocoDetectionDataset(
        args.image_dir, args.annotations, training=False, **checkpoint["preprocessing"]
    )
    if (
        dataset.label_to_category_id != checkpoint["label_to_category_id"]
        or dataset.class_names != checkpoint["class_names"]
    ):
        raise ValueError("evaluation categories must match checkpoint categories")
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.workers,
    )
    metrics = evaluate_model(
        model,
        SetCriterion(dataset.num_classes),
        loader,
        annotation_file=args.annotations,
        label_to_category_id=dataset.label_to_category_id,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    record = json.dumps(metrics, indent=2, allow_nan=False)
    output.write_text(record, encoding="utf-8")
    print(record)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="DETRの学習と推論")
    commands = parser.add_subparsers(dest="command", required=True)
    plotting = commands.add_parser(
        "plot-training", help="保存済みログから4分割グラフを出力"
    )
    plotting.add_argument("--log-dirs", nargs="+", required=True)
    plotting.add_argument("--output", default="output/progress.png")
    plotting.set_defaults(run=run_plot_training)
    trace = commands.add_parser("trace-queries")
    trace.add_argument("--checkpoint", required=True)
    trace.add_argument("--image-dir", required=True)
    trace.add_argument("--annotations", required=True)
    trace.add_argument("--output-dir", default="output/query-trace")
    trace.add_argument("--steps", type=int, default=100)
    trace.add_argument("--seed", type=int, default=42)
    trace.add_argument("--device", default="cpu")
    trace.add_argument(
        "--dropout", type=float, help="重みはそのままでDropout率を変更して比較"
    )
    trace.set_defaults(run=run_trace)
    demo = commands.add_parser("make-demo-data")
    demo.add_argument("--output-dir", default="data/demo")
    demo.add_argument("--train-count", type=int, default=16)
    demo.add_argument("--val-count", type=int, default=4)
    demo.add_argument("--seed", type=int, default=42)
    demo.set_defaults(run=demo_data)
    training = commands.add_parser("train")
    training.add_argument("--image-dir", required=True)
    training.add_argument("--annotations", required=True)
    training.add_argument("--output-dir", default="output/train")
    training.add_argument("--epochs", type=int, default=1)
    training.add_argument(
        "--save-every",
        type=int,
        default=1,
        help="重みの保存間隔（エポック数）。最終エポックは必ず保存",
    )
    training.add_argument("--batch-size", type=int, default=2)
    training.add_argument("--workers", type=int, default=0)
    training.add_argument("--device", default="cpu")
    training.add_argument("--seed", type=int, default=42)
    training.add_argument("--lr", type=float, default=1e-4)
    training.add_argument("--lr-backbone", type=float, default=1e-5)
    training.add_argument("--short-side", type=int, default=800)
    training.add_argument("--max-size", type=int, default=1333)
    training.add_argument("--train-scales", type=int, nargs="+")
    training.add_argument("--no-augmentation", action="store_true")
    training.add_argument("--pretrained-backbone", action="store_true")
    training.add_argument("--num-queries", type=int, default=100)
    training.add_argument("--d-model", type=int, default=256)
    training.add_argument("--nhead", type=int, default=8)
    training.add_argument("--dim-feedforward", type=int, default=2048)
    training.add_argument("--encoder-layers", type=int, default=6)
    training.add_argument("--decoder-layers", type=int, default=6)
    training.add_argument("--dropout", type=float, default=0.1)
    training.add_argument(
        "--resume", help="モデル構成・Optimizer・エポックを復元（epochsは最終番号）"
    )
    training.add_argument("--val-image-dir")
    training.add_argument("--val-annotations")
    training.add_argument("--eval-every", type=int, default=5)
    training.set_defaults(run=train)
    prediction = commands.add_parser("predict")
    prediction.add_argument("--checkpoint", required=True)
    prediction.add_argument("--image", required=True)
    prediction.add_argument("--output", default="output/prediction.png")
    prediction.add_argument("--threshold", type=float, default=0.5)
    prediction.add_argument("--device", default="cpu")
    prediction.add_argument(
        "--annotations", help="正解を緑で描画し、IoU・全候補の確率を保存"
    )
    prediction.add_argument(
        "--image-id", type=int, help="注釈内でファイル名が重複する場合に指定"
    )
    prediction.add_argument("--scale", type=int, default=1, help="描画画像の拡大倍率")
    prediction.add_argument(
        "--top-k", type=int, help="描画・予測JSONを上位k候補に限定（診断は全候補）"
    )
    prediction.set_defaults(run=predict)
    evaluation = commands.add_parser("evaluate")
    evaluation.add_argument("--checkpoint", required=True)
    evaluation.add_argument("--image-dir", required=True)
    evaluation.add_argument("--annotations", required=True)
    evaluation.add_argument("--output", default="output/evaluation.json")
    evaluation.add_argument("--batch-size", type=int, default=2)
    evaluation.add_argument("--workers", type=int, default=0)
    evaluation.add_argument("--device", default="cpu")
    evaluation.set_defaults(run=evaluate)
    args = parser.parse_args(argv)
    args.run(args)


if __name__ == "__main__":
    main()
