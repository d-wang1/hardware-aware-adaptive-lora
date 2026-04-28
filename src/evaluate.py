"""Validation loop and target-accuracy tracker.

``evaluate()`` is shared by every training method: it produces ``val_loss`` and
``val_accuracy`` for the JSONL log. ``TargetAccuracyTracker`` records the first
step (and wall-clock seconds) at which validation accuracy crosses a configured
target — this is how the README's "time-to-quality" / "steps-to-target" metrics
become real numbers.
"""
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
    """Run one validation pass and return ``{"val_loss", "val_accuracy"}``.

    Loss is summed across examples and divided by the example count, which
    matches the standard CE-mean reported during training. Model train/eval
    state is restored on exit so callers can drop this anywhere in the loop.
    Empty loaders return ``nan`` for both metrics rather than dividing by zero.
    """
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
        # Only re-enter train mode if we were already there; don't surprise
        # callers who deliberately put the model in eval mode.
        if was_training:
            model.train()

    if total_examples == 0:
        return {"val_loss": float("nan"), "val_accuracy": float("nan")}

    return {
        "val_loss": total_loss / total_examples,
        "val_accuracy": total_correct / total_examples,
    }


class TargetAccuracyTracker:
    """Record the first step and wall-clock time that ``val_accuracy >= target``.

    Pair this with ``HardwareLogger``: pass ``logger``-relevant ``start_time``
    so that ``wall_clock_to_target`` is measured from the same anchor as
    everything else in the run. Once the threshold is crossed the recorded
    values are locked — later high-accuracy evaluations don't overwrite them,
    which matters for fair time-to-quality comparison across methods.
    """

    def __init__(self, target: float, start_time: float | None = None) -> None:
        self.target = target
        self._start_time = (
            start_time if start_time is not None else time.perf_counter()
        )
        self.steps_to_target: int | None = None
        self.wall_clock_to_target: float | None = None

    def update(self, step: int, val_accuracy: float) -> bool:
        """Record this evaluation. Returns ``True`` only on the *first* crossing."""
        if self.steps_to_target is not None:
            return False  # already locked in
        if val_accuracy >= self.target:
            self.steps_to_target = step
            self.wall_clock_to_target = time.perf_counter() - self._start_time
            return True
        return False

    @property
    def reached(self) -> bool:
        return self.steps_to_target is not None
