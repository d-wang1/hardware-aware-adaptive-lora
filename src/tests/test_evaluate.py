"""Tests for src.evaluate.

Uses tiny stub models so the test set runs in <1s without HF / network.
"""
from __future__ import annotations

import math
import time
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.evaluate import TargetAccuracyTracker, evaluate


# --- Stub models ----------------------------------------------------------


class _LabelEchoClassifier(nn.Module):
    """Returns logits whose argmax equals ``input_ids[:, 0]``.

    Pair this with batches where input_ids[:, 0] == labels and you get a
    perfect classifier — useful for asserting the eval loop reads logits
    correctly and that loss is small when predictions are confident.
    """

    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.num_classes = num_classes
        # at least one parameter so .train() / .eval() are observable on a real model
        self._dummy = nn.Parameter(torch.zeros(1))

    def forward(self, input_ids, **kwargs):
        labels = input_ids[:, 0]
        bsz = labels.size(0)
        logits = torch.full(
            (bsz, self.num_classes), -10.0, device=labels.device, dtype=torch.float32
        )
        logits.scatter_(1, labels.unsqueeze(1), 10.0)
        return SimpleNamespace(logits=logits)


class _AlwaysClassZero(nn.Module):
    """Uniform-zero logits => argmax == 0 for every example."""

    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.num_classes = num_classes
        self._dummy = nn.Parameter(torch.zeros(1))

    def forward(self, input_ids, **kwargs):
        bsz = input_ids.size(0)
        return SimpleNamespace(
            logits=torch.zeros(
                bsz, self.num_classes, device=input_ids.device, dtype=torch.float32
            )
        )


def _make_loader(input_ids: torch.Tensor, labels: torch.Tensor, batch_size: int = 4):
    """Build a DataLoader yielding {input_ids, labels} dicts (HF-collator-style)."""
    ds = TensorDataset(input_ids, labels)

    def collate(batch):
        ids = torch.stack([b[0] for b in batch])
        lbls = torch.stack([b[1] for b in batch])
        return {"input_ids": ids, "labels": lbls}

    return DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate)


# --- evaluate() -----------------------------------------------------------


def test_evaluate_perfect_predictions():
    model = _LabelEchoClassifier(num_classes=2)
    labels = torch.tensor([0, 1, 0, 1, 1, 0, 1, 0], dtype=torch.long)
    input_ids = torch.zeros(len(labels), 5, dtype=torch.long)
    input_ids[:, 0] = labels  # encode the label in the input
    loader = _make_loader(input_ids, labels, batch_size=3)

    metrics = evaluate(model, loader, device="cpu")
    assert metrics["val_accuracy"] == pytest.approx(1.0)
    # logits are 10 vs -10, very confident -> CE loss tiny
    assert metrics["val_loss"] < 1e-6


def test_evaluate_deterministic_50pct_accuracy():
    """Always-class-zero model on balanced labels gives exactly 50%."""
    model = _AlwaysClassZero(num_classes=2)
    labels = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1], dtype=torch.long)
    input_ids = torch.zeros(len(labels), 5, dtype=torch.long)
    loader = _make_loader(input_ids, labels, batch_size=4)

    metrics = evaluate(model, loader, device="cpu")
    assert metrics["val_accuracy"] == pytest.approx(0.5)
    # Uniform logits -> CE = log(num_classes) = log(2) ~= 0.693
    assert metrics["val_loss"] == pytest.approx(math.log(2), rel=1e-5)


def test_evaluate_restores_train_mode_when_started_in_train():
    model = _AlwaysClassZero()
    model.train()
    labels = torch.tensor([0, 1], dtype=torch.long)
    input_ids = torch.zeros(2, 5, dtype=torch.long)
    loader = _make_loader(input_ids, labels, batch_size=2)

    evaluate(model, loader, device="cpu")
    assert model.training is True


def test_evaluate_does_not_force_train_when_was_eval():
    model = _AlwaysClassZero()
    model.eval()
    labels = torch.tensor([0, 1], dtype=torch.long)
    input_ids = torch.zeros(2, 5, dtype=torch.long)
    loader = _make_loader(input_ids, labels, batch_size=2)

    evaluate(model, loader, device="cpu")
    assert model.training is False


def test_evaluate_empty_loader_returns_nan():
    model = _AlwaysClassZero()
    loader = _make_loader(
        torch.zeros(0, 5, dtype=torch.long),
        torch.zeros(0, dtype=torch.long),
    )
    metrics = evaluate(model, loader, device="cpu")
    assert math.isnan(metrics["val_loss"])
    assert math.isnan(metrics["val_accuracy"])


# --- TargetAccuracyTracker -----------------------------------------------


def test_target_accuracy_tracker_records_first_crossing():
    tracker = TargetAccuracyTracker(target=0.9)
    assert tracker.reached is False
    assert tracker.steps_to_target is None
    assert tracker.wall_clock_to_target is None

    assert tracker.update(step=100, val_accuracy=0.5) is False
    assert tracker.steps_to_target is None

    assert tracker.update(step=200, val_accuracy=0.85) is False
    assert tracker.steps_to_target is None

    assert tracker.update(step=300, val_accuracy=0.92) is True
    assert tracker.steps_to_target == 300
    assert tracker.wall_clock_to_target is not None
    assert tracker.wall_clock_to_target >= 0.0
    assert tracker.reached is True


def test_target_accuracy_tracker_locks_after_first_crossing():
    """A second crossing returns False and does not overwrite recorded values."""
    tracker = TargetAccuracyTracker(target=0.9)
    tracker.update(step=100, val_accuracy=0.95)
    first_step = tracker.steps_to_target
    first_time = tracker.wall_clock_to_target

    assert tracker.update(step=200, val_accuracy=0.99) is False
    assert tracker.steps_to_target == first_step
    assert tracker.wall_clock_to_target == first_time


def test_target_accuracy_tracker_never_reached():
    tracker = TargetAccuracyTracker(target=0.9)
    tracker.update(step=100, val_accuracy=0.5)
    tracker.update(step=200, val_accuracy=0.7)
    tracker.update(step=300, val_accuracy=0.85)
    assert tracker.steps_to_target is None
    assert tracker.wall_clock_to_target is None
    assert tracker.reached is False


def test_target_accuracy_tracker_exact_threshold_counts():
    """``>=`` semantics: hitting target exactly is a crossing."""
    tracker = TargetAccuracyTracker(target=0.9)
    assert tracker.update(step=100, val_accuracy=0.9) is True


def test_target_accuracy_tracker_uses_provided_start_time():
    """Caller-supplied ``start_time`` aligns wall-clock with HardwareLogger."""
    fake_start = time.perf_counter() - 10.0  # pretend training began 10s ago
    tracker = TargetAccuracyTracker(target=0.9, start_time=fake_start)
    tracker.update(step=50, val_accuracy=0.95)
    assert tracker.wall_clock_to_target is not None
    assert tracker.wall_clock_to_target >= 10.0
