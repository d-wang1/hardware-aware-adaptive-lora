from __future__ import annotations

import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


@torch.no_grad()
def evaluate(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device | str,
) -> dict[str, float]:
    was_training = model.training
    model.eval()

    loss_fn = nn.CrossEntropyLoss(reduction="sum")
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    try:
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch["labels"]
            inputs = {k: v for k, v in batch.items() if k != "labels"}
            out = model(**inputs)
            logits = out.logits
            total_loss += loss_fn(logits, labels).item()
            preds = logits.argmax(dim=-1)
            total_correct += (preds == labels).sum().item()
            total_examples += labels.size(0)
    finally:
        if was_training:
            model.train()

    if total_examples == 0:
        return {"val_loss": float("nan"), "val_accuracy": float("nan")}

    return {
        "val_loss": total_loss / total_examples,
        "val_accuracy": total_correct / total_examples,
    }


class TargetAccuracyTracker:
    """Locks step + wall-clock the first time val_accuracy crosses target."""

    def __init__(self, target: float, start_time: float | None = None) -> None:
        self.target = target
        self._start_time = (
            start_time if start_time is not None else time.perf_counter()
        )
        self.steps_to_target: int | None = None
        self.wall_clock_to_target: float | None = None

    def update(self, step: int, val_accuracy: float) -> bool:
        if self.steps_to_target is not None:
            return False
        if val_accuracy >= self.target:
            self.steps_to_target = step
            self.wall_clock_to_target = time.perf_counter() - self._start_time
            return True
        return False

    @property
    def reached(self) -> bool:
        return self.steps_to_target is not None
