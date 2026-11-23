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
from detr.engine import collate_fn, train_one_epoch
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
    drawing = ImageDraw.Draw(image)
    records = []
    for score, label, box in zip(result["scores"], result["labels"], result["boxes"]):
        label_id = label.item()
        name = checkpoint["class_names"][label_id]
        coordinates = box.tolist()
        # 描画だけ画像の境界に制限し、JSONには予測座標をそのまま残す。
        x1, y1, x2, y2 = coordinates
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
    output.with_suffix(".json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Saved {len(records)} predictions to {output}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="DETRの学習と推論")
    commands = parser.add_subparsers(dest="command", required=True)
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
    prediction.set_defaults(run=predict)
    args = parser.parse_args(argv)
    args.run(args)


if __name__ == "__main__":
    main()
