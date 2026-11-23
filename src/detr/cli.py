"""学習・推論を実行するコマンドライン入口。"""

import argparse
import json
from pathlib import Path

import torch
from PIL import Image, ImageDraw
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
from detr.postprocess import postprocess
from detr.preprocessing import prepare_image


def train(args: argparse.Namespace) -> None:
    if args.epochs <= 0 or args.batch_size <= 0:
        raise ValueError("epochs and batch_size must be positive")
    torch.manual_seed(args.seed)
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
    }
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
    with (output_dir / "metrics.jsonl").open("w", encoding="utf-8") as log:
        for epoch in range(1, args.epochs + 1):
            metrics = train_one_epoch(model, criterion, loader, optimizer)
            record = json.dumps({"epoch": epoch, **metrics}, allow_nan=False)
            print(record, flush=True)
            log.write(record + "\n")
            log.flush()
            save_checkpoint(
                output_dir / "checkpoint.pt",
                model,
                optimizer,
                model_config=config,
                class_names=dataset.class_names,
                label_to_category_id=dataset.label_to_category_id,
                preprocessing={
                    "short_side": args.short_side,
                    "max_size": args.max_size,
                },
                epoch=epoch,
            )


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
    image = image.resize(
        (image.width * args.scale, image.height * args.scale), Image.Resampling.NEAREST
    )
    drawing = ImageDraw.Draw(image)
    if ground_truth is not None:
        for target in ground_truth:
            coordinates = [coordinate * args.scale for coordinate in target["box_xyxy"]]
            drawing.rectangle(coordinates, outline="lime", width=3)
    records = []
    for score, label, box in zip(
        result["scores"][: args.top_k],
        result["labels"][: args.top_k],
        result["boxes"][: args.top_k],
    ):
        label_id = label.item()
        name = checkpoint["class_names"][label_id]
        coordinates = box.tolist()
        # 描画だけ画像の境界に制限し、JSONには予測座標をそのまま残す。
        x1, y1, x2, y2 = [coordinate * args.scale for coordinate in coordinates]
        x1, x2 = [min(max(x, 0), image.width - 1) for x in (x1, x2)]
        y1, y2 = [min(max(y, 0), image.height - 1) for y in (y1, y2)]
        drawing.rectangle((x1, y1, x2, y2), outline="red", width=2)
        drawing.text((x1, y1), f"{name} {score.item():.2f}", fill="red")
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
