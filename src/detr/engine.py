"""単一プロセス・float32でのDETR学習ループ。"""

import math
from collections.abc import Iterable, Mapping, Sequence

import torch
from torch import Tensor, nn
from torch.optim import Optimizer


def collate_fn(
    samples: Sequence[tuple[Tensor, Mapping[str, Tensor]]],
) -> tuple[list[Tensor], list[Mapping[str, Tensor]]]:
    """DataLoader用。画像サイズと物体数が違うのでstackせずリストで返す。

    各サンプルは(image, target)。画像のパディングはDETR.forwardが行う。
    targetはlabelsと、各画像サイズで正規化したcxcywhのboxesを含む。
    """
    images, targets = zip(*samples)
    return list(images), list(targets)


def train_one_epoch(
    model: nn.Module,
    criterion: nn.Module,
    data_loader: Iterable[tuple[Sequence[Tensor], Sequence[Mapping[str, Tensor]]]],
    optimizer: Optimizer,
    *,
    max_norm: float = 0.1,
) -> dict[str, float]:
    """1エポック学習し、損失ごとのバッチ平均（Python float）を返す。

    modelはoptimizer作成前に実行deviceへ移動しておく。画像・正解と
    criterionはそのdeviceへ移動する。各バッチでloss_totalだけを逆伝播。
    max_normは勾配の全体L2ノルム上限。0ならクリッピングを無効化する。
    optimizerや学習率スケジューラの作成、チェックポイント保存は呼び出し側。
    """
    if not math.isfinite(max_norm) or max_norm < 0:
        raise ValueError("max_norm must be finite and non-negative")
    device = next(model.parameters()).device
    model.train()
    criterion.to(device).train()
    totals: dict[str, float] = {}
    num_batches = 0
    for images, targets in data_loader:
        images = [image.to(device) for image in images]
        targets = [
            {key: value.to(device) for key, value in target.items()}
            for target in targets
        ]

        optimizer.zero_grad(set_to_none=True)
        outputs = model(images)
        losses = criterion(outputs, targets)
        loss = losses["loss_total"]
        if not torch.isfinite(loss).item():
            raise FloatingPointError("loss_total is not finite; optimizer step skipped")
        loss.backward()
        if max_norm > 0:
            nn.utils.clip_grad_norm_(
                model.parameters(), max_norm, error_if_nonfinite=True
            )
        optimizer.step()

        # 計算グラフを保持せず、表示用の数値だけを集計する。
        for name, value in losses.items():
            totals[name] = totals.get(name, 0.0) + value.detach().item()
        num_batches += 1

    if num_batches == 0:
        raise ValueError("data_loader must contain at least one batch")
    return {name: value / num_batches for name, value in totals.items()}
