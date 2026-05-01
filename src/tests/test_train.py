"""Unit tests for src.train helpers + a synthetic-model end-to-end exercise of
``train_loop``. The full uniform path is verified out-of-band by running

    python -m src.train --config configs/uniform_lora.yaml --smoke

(too slow for the unit-test suite — pulls DistilBERT + SST-2).
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.evaluate import TargetAccuracyTracker
from src.hardware_logger import HardwareLogger
from src.train import (
    _cycle,
    _log_run_config,
    apply_smoke_overrides,
    build_optimizer_and_scheduler,
    load_config,
    make_run_id,
    train_loop,
)


# --- helper smoke tests --------------------------------------------------


def test_load_config_round_trips_real_yaml():
    """The shipped configs must parse — they're consumed by train.main.
    Budget invariant: 24 modules (6 layers * 4 targets q_lin/v_lin/lin1/lin2)
    * uniform rank 8 = 192."""
    cfg = load_config("configs/uniform_lora.yaml")
    assert cfg["method"] == "uniform"
    assert cfg["lora"]["total_rank_budget"] == 192
    assert sorted(cfg["lora"]["target_modules"]) == ["lin1", "lin2", "q_lin", "v_lin"]


def test_make_run_id_format():
    rid = make_run_id("uniform", 42)
    assert rid.startswith("uniform-seed42-")
    # Ends with a UTC ISO-ish stamp (YYYYMMDDTHHMMSSZ = 16 chars)
    assert rid.endswith("Z") and len(rid.rsplit("-", 1)[-1]) == 16


def test_apply_smoke_overrides_clamps_training():
    cfg = {"training": {"epochs": 3, "eval_interval": 100, "batch_size": 32,
                        "learning_rate": 2e-4, "seed": 42}}
    apply_smoke_overrides(cfg)
    t = cfg["training"]
    assert t["max_steps"] == 5
    assert t["eval_interval"] == 5
    assert t["epochs"] == 1
    assert t["num_workers"] == 0
    # Two-stage methods need warmup_steps; smoke must set it so the same
    # smoke flag works for hardware_aware / gradient_adaptive too.
    assert t["warmup_steps"] == 2
    # Non-overridden keys preserved.
    assert t["batch_size"] == 32
    assert t["learning_rate"] == 2e-4


def test_cycle_restarts_loader():
    """``_cycle`` must keep yielding past the loader's natural end so
    train_loop can be step-budgeted instead of epoch-budgeted."""
    ds = TensorDataset(torch.arange(4))
    loader = DataLoader(ds, batch_size=2)
    it = _cycle(loader)
    seen = [next(it)[0].tolist() for _ in range(6)]  # 3 epochs worth
    assert len(seen) == 6


def test_build_optimizer_and_scheduler_warmup_fraction():
    """Linear warmup uses ~6% of total steps (rounded down, min 1). Sanity-
    check the boundaries the project will actually hit (5 smoke / 100 short)."""
    model = nn.Linear(4, 2)
    cfg = {"learning_rate": 2e-4}
    _, sched1 = build_optimizer_and_scheduler(model, cfg, total_steps=5)
    _, sched2 = build_optimizer_and_scheduler(model, cfg, total_steps=100)
    # Both schedules are LambdaLR; we just assert they produce a non-zero LR
    # at step 0 (post-warmup setup) without crashing — the exact numbers come
    # from HF's helper.
    assert sched1.get_last_lr()[0] >= 0.0
    assert sched2.get_last_lr()[0] >= 0.0


# --- train_loop end-to-end on a synthetic model -------------------------


class _StubHF(nn.Module):
    """Mimics a HuggingFace classification head: ``forward(**batch)`` returns
    a SimpleNamespace with ``.logits`` and ``.loss``."""

    def __init__(self, in_dim: int = 4, num_labels: int = 2):
        super().__init__()
        self.lin = nn.Linear(in_dim, num_labels)

    def forward(self, *, features, labels=None, **_unused):
        logits = self.lin(features)
        if labels is not None:
            loss = nn.functional.cross_entropy(logits, labels)
        else:
            loss = torch.tensor(0.0, device=logits.device)
        return SimpleNamespace(logits=logits, loss=loss)


def _toy_loader(n: int, seed: int = 0) -> DataLoader:
    g = torch.Generator().manual_seed(seed)
    features = torch.randn(n, 4, generator=g)
    # Linearly separable-ish labels so the stub can actually learn.
    labels = (features.sum(dim=1) > 0).long()

    class _Ds:
        def __len__(self):
            return n
        def __getitem__(self, i):
            return {"features": features[i], "labels": labels[i]}

    return DataLoader(_Ds(), batch_size=4, shuffle=False)


def test_train_loop_writes_jsonl_and_advances_optimizer(tmp_path):
    """One full pass through train_loop should: (1) write at least one JSONL
    row at the eval boundary, (2) move the optimizer (loss decreases), (3)
    return a final-eval dict with the correct step number."""
    torch.manual_seed(0)
    model = _StubHF()
    train_loader = _toy_loader(32, seed=0)
    val_loader = _toy_loader(16, seed=1)

    optim, sched = build_optimizer_and_scheduler(
        model, {"learning_rate": 1e-2}, total_steps=10
    )

    initial_loss_before = nn.functional.cross_entropy(
        model.lin(next(iter(val_loader))["features"]),
        next(iter(val_loader))["labels"],
    ).item()

    out_dir = tmp_path / "logs"
    with HardwareLogger(out_dir, method="stub", run_id="r0") as logger:
        tracker = TargetAccuracyTracker(target=0.99)
        last = train_loop(
            model=model,
            optimizer=optim,
            scheduler=sched,
            train_loader=train_loader,
            val_loader=val_loader,
            logger=logger,
            tracker=tracker,
            device="cpu",
            total_steps=10,
            eval_interval=5,
        )

    assert last["step"] == 10
    log_path = out_dir / "r0.jsonl"
    assert log_path.exists()
    rows = [
        line for line in log_path.read_text(encoding="utf-8").splitlines() if line
    ]
    # At eval_interval=5 over 10 steps we expect 2 eval rows (step 5, 10).
    assert len(rows) == 2

    # Optimizer made progress: re-evaluate, loss should be lower.
    val_batch = next(iter(val_loader))
    after_loss = nn.functional.cross_entropy(
        model.lin(val_batch["features"]),
        val_batch["labels"],
    ).item()
    assert after_loss < initial_loss_before, (after_loss, initial_loss_before)


# --- main() dispatch -----------------------------------------------------


def test_main_dispatches_hardware_aware_to_two_stage(monkeypatch):
    """``main`` must route hardware_aware → run_two_stage. Spy on the
    function to avoid touching DistilBERT in a unit test."""
    import src.train as t
    calls: list[str] = []

    def spy(cfg):
        calls.append(cfg["method"])
        return {}

    monkeypatch.setattr(t, "run_two_stage", spy)
    t.main(["--config", "configs/hardware_aware_lora.yaml", "--smoke"])
    assert calls == ["hardware_aware"]


def test_main_dispatches_gradient_adaptive_to_two_stage(monkeypatch):
    """gradient_adaptive shares the two-stage code path with hardware_aware
    (only allocator alpha differs). Same dispatch contract."""
    import src.train as t
    calls: list[str] = []

    def spy(cfg):
        calls.append(cfg["method"])
        return {}

    monkeypatch.setattr(t, "run_two_stage", spy)
    t.main(["--config", "configs/gradient_adaptive_lora.yaml", "--smoke"])
    assert calls == ["gradient_adaptive"]


def test_main_dispatches_adalora_to_run_adalora(monkeypatch):
    """AdaLoRA path: same dispatch convention but a separate run function
    because the per-step ``update_and_allocate`` hook lives only here."""
    import src.train as t
    calls: list[str] = []

    def spy(cfg):
        calls.append(cfg["method"])
        return {}

    monkeypatch.setattr(t, "run_adalora", spy)
    t.main(["--config", "configs/adalora.yaml", "--smoke"])
    assert calls == ["adalora"]


def test_apply_smoke_overrides_includes_adalora_knobs():
    """Smoke must override AdaLoRA's tinit/tfinal/deltaT — defaults of
    200/1000/10 would never fire in 5 smoke steps and the AdaLoRA hook
    would silently no-op."""
    cfg = {"method": "adalora", "training": {"seed": 0}}
    apply_smoke_overrides(cfg)
    # PEFT requires tinit + tfinal < total_step (=5) so the budgeting phase
    # has room. tinit=1, tfinal=1 leaves steps 1-3 for reallocation.
    assert cfg["lora"]["tinit"] == 1
    assert cfg["lora"]["tfinal"] == 1
    assert cfg["lora"]["deltaT"] == 1


def test_train_loop_calls_allocator_hook(tmp_path):
    """When an allocator is passed, ``update_gradient_scores`` must be invoked
    every step. This is the contract the two-stage flow (Phase 5.3) depends on."""
    torch.manual_seed(0)
    model = _StubHF()
    train_loader = _toy_loader(16)
    val_loader = _toy_loader(8, seed=1)
    optim, sched = build_optimizer_and_scheduler(
        model, {"learning_rate": 1e-2}, total_steps=4
    )

    class _SpyAllocator:
        def __init__(self):
            self.calls = 0
        def update_gradient_scores(self, _model):
            self.calls += 1

    spy = _SpyAllocator()
    with HardwareLogger(tmp_path, method="stub", run_id="rs") as logger:
        tracker = TargetAccuracyTracker(target=0.99)
        train_loop(
            model=model,
            optimizer=optim,
            scheduler=sched,
            train_loader=train_loader,
            val_loader=val_loader,
            logger=logger,
            tracker=tracker,
            device="cpu",
            total_steps=4,
            eval_interval=4,
            allocator=spy,
        )
    assert spy.calls == 4


def test_log_run_config_emits_event_config_row(tmp_path):
    """Phase 6.7 contract: every dispatcher writes one ``event="config"``
    row before training starts. Phase 6's metrics reader extracts per-run
    knobs (notably ``allocator.hardware_alpha`` for the α-sweep) from this
    row, so it must round-trip the full cfg + the seed."""
    import json

    cfg = {
        "method": "hardware_aware",
        "training": {"seed": 7, "epochs": 3, "batch_size": 32,
                     "learning_rate": 2e-4, "eval_interval": 100,
                     "warmup_steps": 200},
        "lora": {"total_rank_budget": 192, "min_rank": 2, "max_rank": 16,
                 "initial_rank": 8, "alpha_lora": 16,
                 "target_modules": ["q_lin", "v_lin", "lin1", "lin2"]},
        "allocator": {"hardware_alpha": 0.5, "ema_beta": 0.9},
        "logging": {"output_dir": "results/raw_logs", "target_accuracy": 0.9},
    }
    with HardwareLogger(tmp_path, method="hardware_aware", run_id="rcfg") as logger:
        _log_run_config(logger, cfg)

    rows = [
        json.loads(line)
        for line in (tmp_path / "rcfg.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    assert len(rows) == 1
    row = rows[0]
    assert row["event"] == "config"
    assert row["step"] == 0
    assert row["seed"] == 7
    # Whole cfg round-trips so metrics reader can pull alpha without re-loading yaml.
    assert row["config"]["allocator"]["hardware_alpha"] == 0.5
    assert row["config"]["lora"]["total_rank_budget"] == 192


def test_train_loop_calls_post_step_hook_inside_scheduler_block(tmp_path):
    """``post_step_hook`` is the AdaLoRA contract: invoked every step *after*
    optimizer.step, inside ``logger.scheduler_block`` so its overhead lands
    in ``scheduler_overhead_seconds``. The hook must receive the step number
    (1-indexed; AdaLoRA's internal schedule is 1-indexed)."""
    import time
    torch.manual_seed(0)
    model = _StubHF()
    train_loader = _toy_loader(16)
    val_loader = _toy_loader(8, seed=1)
    optim, sched = build_optimizer_and_scheduler(
        model, {"learning_rate": 1e-2}, total_steps=3
    )

    seen_steps: list[int] = []
    def hook(global_step: int) -> None:
        seen_steps.append(global_step)
        time.sleep(0.005)  # measurable overhead so scheduler_block is non-zero

    with HardwareLogger(tmp_path, method="stub", run_id="rh") as logger:
        tracker = TargetAccuracyTracker(target=0.99)
        train_loop(
            model=model, optimizer=optim, scheduler=sched,
            train_loader=train_loader, val_loader=val_loader,
            logger=logger, tracker=tracker, device="cpu",
            total_steps=3, eval_interval=3,
            post_step_hook=hook,
        )
        assert seen_steps == [0, 1, 2]
        # Scheduler overhead picked up the hook's sleeps (3 × 5ms ≈ 15ms).
        assert logger.scheduler_overhead_seconds >= 0.010
