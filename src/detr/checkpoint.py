"""モデル構成・クラス対応を含むチェックポイントの保存と読み込み。"""

from pathlib import Path

import torch
from torch.optim import Optimizer

from detr.model import DETR


def save_checkpoint(
    path: str | Path,
    model: DETR,
    optimizer: Optimizer,
    *,
    model_config: dict,
    class_names: list[str],
    label_to_category_id: dict[int, int],
    preprocessing: dict[str, int],
    epoch: int,
    best_ap: float | None = None,
) -> None:
    """学習再開に必要なOptimizer状態と乱数状態も保存する。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # 保存中断で既存のcheckpointを壊さないよう、保存後に置き換える。
    temporary = path.with_name(path.name + ".tmp")
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "model_config": model_config,
            "class_names": class_names,
            "label_to_category_id": label_to_category_id,
            "preprocessing": preprocessing,
            "epoch": epoch,
            "best_ap": best_ap,
            "rng_state": torch.get_rng_state(),
        },
        temporary,
    )
    temporary.replace(path)


def load_model(path: str | Path, device: str = "cpu") -> tuple[DETR, dict]:
    """保存した構成を復元し、重みの再ダウンロードなしで推論モデルを返す。"""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    config = {**checkpoint["model_config"], "pretrained_backbone": False}
    model = DETR(**config)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device).eval()
    return model, checkpoint
