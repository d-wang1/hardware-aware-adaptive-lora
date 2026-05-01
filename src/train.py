"""Unified training entry point. Dispatches by config ``method`` field:
``{uniform, adalora, gradient_adaptive, hardware_aware}``.

CLAUDE.md "Things to not do" requires that the four methods share optimizer,
LR, batch size, and max-step settings — those live once in
``build_optimizer_and_scheduler`` + ``train_loop`` and are wired identically
for every method. Method-specific code only differs in *what model is built*
(and, for the adaptive variants, the warmup→stage-2 hand-off).

Usage::

    python -m src.train --config configs/uniform_lora.yaml --seed 42
    python -m src.train --config configs/uniform_lora.yaml --seed 42 --smoke

``--smoke`` clamps the data + steps for a fast CPU sanity run; real
experiments require CUDA (see CLAUDE.md "Hardware target").
"""
from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path
from typing import Any, Iterator

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from src.data import load_sst2, make_dataloaders, set_seed
from src.evaluate import TargetAccuracyTracker, evaluate
from src.hardware_logger import HardwareLogger
from peft import AdaLoraConfig, TaskType, get_peft_model

from src.lora_utils import build_non_uniform_lora_model, build_uniform_lora_model
from src.models import (
    count_parameters,
    find_lora_target_module_names,
    load_model_and_tokenizer,
)
from src.rank_allocator import HardwareAwareRankAllocator


# --- config / id --------------------------------------------------------


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def make_run_id(method: str, seed: int) -> str:
    """Sortable timestamped id used for the JSONL filename. UTC so runs
    from different machines line up in lexicographic order."""
    stamp = dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    return f"{method}-seed{seed}-{stamp}"


def apply_smoke_overrides(cfg: dict[str, Any]) -> None:
    """Mutate ``cfg`` in place to a fast CPU-tolerant configuration.

    Forces ``num_workers=0`` (CLAUDE.md: Windows DataLoader reentry constraint)
    and clamps train/val/step counts so the entire run finishes in seconds.
    """
    training = cfg.setdefault("training", {})
    training["max_train_samples"] = 64
    training["max_val_samples"] = 32
    training["num_workers"] = 0
    training["max_steps"] = 5
    training["eval_interval"] = 5
    training["epochs"] = 1
    # Two-stage methods read warmup_steps; override here so the smoke flag is
    # method-agnostic. With max_steps=5 → 2 warmup + 3 stage-2.
    training["warmup_steps"] = 2
    # AdaLoRA-specific knobs (default tinit=200/tfinal=1000/deltaT=10 would
    # never fire in a 5-step smoke). PEFT requires tinit + tfinal < total_step
    # so the budgeting phase has room (here: steps 1-3 reallocate, step 4 is
    # the final-phase tail). Other methods ignore these keys.
    lora = cfg.setdefault("lora", {})
    lora["tinit"] = 1
    lora["tfinal"] = 1
    lora["deltaT"] = 1


# --- run-config provenance ---------------------------------------------


def _log_run_config(logger: HardwareLogger, cfg: dict[str, Any]) -> None:
    """Write a one-time ``event="config"`` row carrying the resolved ``cfg``.

    Phase 6's metrics reader uses this to recover per-run knobs (notably
    ``cfg["allocator"]["hardware_alpha"]`` for the α-sweep ablation) without
    having to guess the source yaml from the run_id. Step is 0: this row
    fires before any training step. ``logger.log`` auto-fills schema columns
    with ``None`` so train_loss/val_loss/val_accuracy do not need to be
    passed here.
    """
    logger.log(
        0,
        event="config",
        config=cfg,
        seed=cfg["training"]["seed"],
    )


# --- optimizer / scheduler ---------------------------------------------


def build_optimizer_and_scheduler(
    model: nn.Module,
    training_cfg: dict[str, Any],
    total_steps: int,
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]:
    """AdamW + linear warmup over 6% of total steps, then linear decay.

    Single source of truth for the optimizer hyperparameters across all four
    methods — only `learning_rate` is read from the config so divergence
    between methods is impossible by construction.
    """
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(training_cfg["learning_rate"]),
    )
    warmup_steps = max(1, int(0.06 * total_steps))
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    return optimizer, scheduler


# --- shared inner loop -------------------------------------------------


def _cycle(loader: DataLoader) -> Iterator[dict[str, torch.Tensor]]:
    """Yield batches forever; restart the loader on epoch boundaries.

    train_loop is step-budgeted, not epoch-budgeted, so we don't need to
    surface epoch transitions to the caller.
    """
    while True:
        for batch in loader:
            yield batch


def train_loop(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    train_loader: DataLoader,
    val_loader: DataLoader,
    logger: HardwareLogger,
    tracker: TargetAccuracyTracker,
    device: torch.device | str,
    total_steps: int,
    eval_interval: int,
    allocator: Any | None = None,
    post_step_hook: Any | None = None,
    start_step: int = 0,
) -> dict[str, float | int]:
    """Run ``total_steps`` optimizer steps, evaluating every ``eval_interval``.

    Step 0 is *not* evaluated — the optimizer hasn't moved yet so the
    pretrained baseline isn't informative for time-to-target. The final
    step is always evaluated regardless of interval alignment so the run
    has a guaranteed last-row ``val_accuracy`` for Phase 6 aggregation.

    Two scheduling hooks, both charged to ``scheduler_overhead_seconds`` for
    fair cross-method comparison:

    * ``allocator.update_gradient_scores(model)`` — called between
      ``loss.backward()`` and ``optimizer.step()`` (the only window where
      LoRA gradients exist). Used by the two-stage methods.
    * ``post_step_hook(global_step)`` — called after ``optimizer.step()`` /
      ``scheduler.step()`` / ``zero_grad``. Used by AdaLoRA's
      ``update_and_allocate``, which needs to inspect the *post-update*
      weights to recompute importance.
    """
    model.train()
    step = start_step
    end_step = start_step + total_steps
    train_iter = _cycle(train_loader)
    last_eval: dict[str, float | int] | None = None

    while step < end_step:
        batch = next(train_iter)

        logger.step_start()
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(**batch)
        loss = out.loss
        loss.backward()

        if allocator is not None:
            with logger.scheduler_block():
                allocator.update_gradient_scores(model)

        optimizer.step()
        scheduler.step()

        # AdaLoRA's update_and_allocate computes ``p * p.grad`` for importance
        # scoring, so the hook must fire BEFORE zero_grad while gradients are
        # still attached. zero_grad goes last; the next step's loss.backward()
        # is the only consumer and it doesn't care which side of the loop the
        # zero happened on.
        if post_step_hook is not None:
            with logger.scheduler_block():
                post_step_hook(step)

        optimizer.zero_grad()
        logger.step_end(num_examples=batch["labels"].size(0))
        step += 1

        is_eval_boundary = (step % eval_interval == 0) or (step == end_step)
        if is_eval_boundary:
            metrics = evaluate(model, val_loader, device)
            tracker.update(step, metrics["val_accuracy"])
            logger.log(
                step,
                train_loss=float(loss.item()),
                val_loss=metrics["val_loss"],
                val_accuracy=metrics["val_accuracy"],
                trainable_parameters=count_parameters(model, trainable_only=True),
            )
            last_eval = {"step": step, **metrics}

    if last_eval is None:
        # total_steps == 0 (shouldn't happen with a real config, but keep the
        # contract honest for the unit tests).
        last_eval = {"step": start_step, "val_loss": float("nan"),
                     "val_accuracy": float("nan")}
    return last_eval


# --- method dispatchers ------------------------------------------------


def _resolve_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _build_loaders(
    cfg: dict[str, Any], tokenizer
) -> tuple[DataLoader, DataLoader]:
    training = cfg["training"]
    splits = load_sst2(
        tokenizer,
        max_train_samples=training.get("max_train_samples"),
        max_val_samples=training.get("max_val_samples"),
    )
    return make_dataloaders(
        splits,
        tokenizer,
        batch_size=training["batch_size"],
        num_workers=training.get("num_workers", 2),
    )


def run_uniform(cfg: dict[str, Any]) -> dict[str, Any]:
    """Train uniform LoRA. The simplest method — no allocator, no two-stage.

    Returns the last evaluation row + the run_id so callers (tests, sweeps)
    can pick the JSONL file back up.
    """
    set_seed(cfg["training"]["seed"])
    device = _resolve_device()

    base_model, tokenizer = load_model_and_tokenizer(
        cfg["model"]["name"], num_labels=2
    )
    train_loader, val_loader = _build_loaders(cfg, tokenizer)

    lora_cfg = cfg["lora"]
    targets = list(lora_cfg["target_modules"])
    n_modules = len(
        find_lora_target_module_names(base_model, target_keywords=targets)
    )
    expected_budget = int(lora_cfg["total_rank_budget"])
    actual_budget = n_modules * int(lora_cfg["rank"])
    if actual_budget != expected_budget:
        raise ValueError(
            f"uniform rank * n_modules ({n_modules} * {lora_cfg['rank']} = "
            f"{actual_budget}) does not match total_rank_budget "
            f"({expected_budget}). Adjust config or model targets."
        )

    peft_model = build_uniform_lora_model(
        base_model,
        target_modules=targets,
        rank=int(lora_cfg["rank"]),
        alpha=int(lora_cfg["alpha_lora"]),
        dropout=float(lora_cfg.get("dropout", 0.0)),
    )
    peft_model.to(device)

    training = cfg["training"]
    max_steps = training.get("max_steps")
    if max_steps is not None:
        total_steps = int(max_steps)
    else:
        total_steps = int(training["epochs"]) * len(train_loader)

    optimizer, scheduler = build_optimizer_and_scheduler(
        peft_model, training, total_steps
    )

    run_id = make_run_id("uniform", training["seed"])
    out_dir = cfg["logging"]["output_dir"]

    with HardwareLogger(out_dir, method="uniform", run_id=run_id) as logger:
        _log_run_config(logger, cfg)
        tracker = TargetAccuracyTracker(
            target=float(cfg["logging"]["target_accuracy"])
        )
        last_eval = train_loop(
            model=peft_model,
            optimizer=optimizer,
            scheduler=scheduler,
            train_loader=train_loader,
            val_loader=val_loader,
            logger=logger,
            tracker=tracker,
            device=device,
            total_steps=total_steps,
            eval_interval=int(training["eval_interval"]),
        )
        # Sentinel "final" row carries the time-to-target summary so Phase 6
        # can pull it out without scanning every per-step row.
        logger.log(
            total_steps,
            event="final",
            train_loss=None,
            val_loss=last_eval["val_loss"],
            val_accuracy=last_eval["val_accuracy"],
            trainable_parameters=count_parameters(peft_model, trainable_only=True),
            steps_to_target_accuracy=tracker.steps_to_target,
            wall_clock_to_target=tracker.wall_clock_to_target,
        )

    return {"run_id": run_id, "log_path": str(logger.path), **last_eval}


def run_two_stage(cfg: dict[str, Any]) -> dict[str, Any]:
    """Two-stage warmup→reallocate→fine-tune for ``hardware_aware`` and
    ``gradient_adaptive``. Both methods share this code path; the only
    difference is ``allocator.alpha`` (read from ``allocator.hardware_alpha``).

    Stage 1: build uniform LoRA at ``initial_rank``, run ``warmup_steps`` of
    ``train_loop`` with the allocator hook engaged so EMA gradient scores
    accumulate while the optimizer is also moving (warmup gradients reflect
    *active* training dynamics, not just the gradient at init).

    Reallocation: compute ``rank_dict`` inside ``logger.scheduler_block()`` so
    the allocator's cost is honestly attributed to ``scheduler_overhead_seconds``,
    matching how AdaLoRA's allocate-step will be wrapped in Phase 5.4. A
    JSONL row with ``event="reallocation"`` carries the chosen ranks.

    Stage 2: rebuild the base model from scratch (warmup LoRA weights are
    discarded — README §"Two-Stage Version"), wrap with
    ``build_non_uniform_lora_model(rank_dict)``, and re-create the optimizer
    + scheduler around the new (different-shape) parameters. Run
    ``train_loop`` for the remaining ``stage2_steps`` with ``allocator=None``.
    """
    method = cfg["method"]  # "hardware_aware" or "gradient_adaptive"
    set_seed(cfg["training"]["seed"])
    device = _resolve_device()

    base_model_a, tokenizer = load_model_and_tokenizer(
        cfg["model"]["name"], num_labels=2
    )
    train_loader, val_loader = _build_loaders(cfg, tokenizer)

    lora_cfg = cfg["lora"]
    targets = list(lora_cfg["target_modules"])
    initial_rank = int(lora_cfg["initial_rank"])
    alpha_lora = int(lora_cfg["alpha_lora"])
    dropout = float(lora_cfg.get("dropout", 0.0))

    # --- step accounting ----------------------------------------------------
    training = cfg["training"]
    max_steps = training.get("max_steps")
    if max_steps is not None:
        total_steps = int(max_steps)
        # Smoke override or explicit max: warmup is half rounded down (≥1).
        warmup_steps = int(training.get("warmup_steps", max(1, total_steps // 2)))
    else:
        total_steps = int(training["epochs"]) * len(train_loader)
        warmup_steps = int(training["warmup_steps"])
    stage2_steps = total_steps - warmup_steps
    if warmup_steps < 1 or stage2_steps < 1:
        raise ValueError(
            f"need ≥1 step each in warmup ({warmup_steps}) and stage 2 "
            f"({stage2_steps}); total={total_steps}"
        )

    # --- Stage 1: uniform LoRA + allocator EMA hook -------------------------
    warmup_model = build_uniform_lora_model(
        base_model_a, target_modules=targets,
        rank=initial_rank, alpha=alpha_lora, dropout=dropout,
    ).to(device)

    allocator_cfg = cfg["allocator"]
    allocator = HardwareAwareRankAllocator(
        total_budget=int(lora_cfg["total_rank_budget"]),
        min_rank=int(lora_cfg["min_rank"]),
        max_rank=int(lora_cfg["max_rank"]),
        alpha=float(allocator_cfg["hardware_alpha"]),
        ema_beta=float(allocator_cfg["ema_beta"]),
    )

    stage1_optim, stage1_sched = build_optimizer_and_scheduler(
        warmup_model, training, total_steps=warmup_steps
    )

    run_id = make_run_id(method, training["seed"])
    out_dir = cfg["logging"]["output_dir"]

    with HardwareLogger(out_dir, method=method, run_id=run_id) as logger:
        _log_run_config(logger, cfg)
        tracker = TargetAccuracyTracker(
            target=float(cfg["logging"]["target_accuracy"])
        )
        train_loop(
            model=warmup_model,
            optimizer=stage1_optim, scheduler=stage1_sched,
            train_loader=train_loader, val_loader=val_loader,
            logger=logger, tracker=tracker, device=device,
            total_steps=warmup_steps,
            eval_interval=int(training["eval_interval"]),
            allocator=allocator,
        )

        # --- Reallocation (charge to scheduler_overhead) --------------------
        with logger.scheduler_block():
            rank_dict = allocator.allocate(warmup_model)
        logger.log(
            warmup_steps,
            event="reallocation",
            rank_dict=rank_dict,
            gradient_scores=allocator.gradient_scores,
            train_loss=None, val_loss=None, val_accuracy=None,
        )

        # Free Stage 1 state before allocating Stage 2 (helps on tight VRAM).
        del warmup_model, stage1_optim, stage1_sched
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # --- Stage 2: non-uniform LoRA, fresh optimizer ---------------------
        base_model_b, _ = load_model_and_tokenizer(
            cfg["model"]["name"], num_labels=2
        )
        stage2_model = build_non_uniform_lora_model(
            base_model_b, target_modules=targets,
            rank_dict=rank_dict, alpha=alpha_lora, dropout=dropout,
        ).to(device)
        stage2_optim, stage2_sched = build_optimizer_and_scheduler(
            stage2_model, training, total_steps=stage2_steps
        )
        last_eval = train_loop(
            model=stage2_model,
            optimizer=stage2_optim, scheduler=stage2_sched,
            train_loader=train_loader, val_loader=val_loader,
            logger=logger, tracker=tracker, device=device,
            total_steps=stage2_steps,
            eval_interval=int(training["eval_interval"]),
            allocator=None,
            start_step=warmup_steps,
        )

        logger.log(
            total_steps,
            event="final",
            train_loss=None,
            val_loss=last_eval["val_loss"],
            val_accuracy=last_eval["val_accuracy"],
            trainable_parameters=count_parameters(stage2_model, trainable_only=True),
            steps_to_target_accuracy=tracker.steps_to_target,
            wall_clock_to_target=tracker.wall_clock_to_target,
            rank_dict=rank_dict,
        )

    return {
        "run_id": run_id,
        "log_path": str(logger.path),
        "rank_dict": rank_dict,
        **last_eval,
    }


def run_adalora(cfg: dict[str, Any]) -> dict[str, Any]:
    """Train with PEFT's AdaLoRA — the strong adaptive baseline.

    PEFT runs its own importance-based reallocation inside
    ``update_and_allocate(global_step)``, called every step. We invoke that
    via ``train_loop``'s ``post_step_hook`` so the call lands inside
    ``logger.scheduler_block()`` — AdaLoRA's per-step overhead is then
    attributed to ``scheduler_overhead_seconds`` the same way the two-stage
    methods' allocator work is, making the cross-method comparison fair.

    Budget is matched at the *target_r * n_modules* level rather than
    init_r — AdaLoRA starts at higher init_r and prunes down. With 24
    modules and target_r=8, the post-pruning budget matches uniform's 192.
    """
    set_seed(cfg["training"]["seed"])
    device = _resolve_device()

    base_model, tokenizer = load_model_and_tokenizer(
        cfg["model"]["name"], num_labels=2
    )
    train_loader, val_loader = _build_loaders(cfg, tokenizer)

    lora_cfg = cfg["lora"]
    targets = list(lora_cfg["target_modules"])

    training = cfg["training"]
    max_steps = training.get("max_steps")
    if max_steps is not None:
        total_steps = int(max_steps)
    else:
        total_steps = int(training["epochs"]) * len(train_loader)

    adalora_config = AdaLoraConfig(
        init_r=int(lora_cfg["init_r"]),
        target_r=int(lora_cfg["target_r"]),
        tinit=int(lora_cfg["tinit"]),
        tfinal=int(lora_cfg["tfinal"]),
        deltaT=int(lora_cfg["deltaT"]),
        beta1=float(lora_cfg["beta1"]),
        beta2=float(lora_cfg["beta2"]),
        lora_alpha=int(lora_cfg["alpha_lora"]),
        lora_dropout=float(lora_cfg.get("dropout", 0.0)),
        target_modules=targets,
        total_step=total_steps,  # PEFT 0.19 uses this internally for the schedule
        bias="none",
        task_type=TaskType.SEQ_CLS,
    )
    peft_model = get_peft_model(base_model, adalora_config).to(device)

    optimizer, scheduler = build_optimizer_and_scheduler(
        peft_model, training, total_steps
    )

    run_id = make_run_id("adalora", training["seed"])
    out_dir = cfg["logging"]["output_dir"]

    def adalora_step_hook(global_step: int) -> None:
        # PEFT's AdaLoRA uses 1-indexed steps internally; train_loop counts
        # the just-completed step as ``step`` (post-increment), so passing
        # ``global_step`` directly matches AdaLoRA's expectation.
        peft_model.base_model.update_and_allocate(global_step)

    with HardwareLogger(out_dir, method="adalora", run_id=run_id) as logger:
        _log_run_config(logger, cfg)
        tracker = TargetAccuracyTracker(
            target=float(cfg["logging"]["target_accuracy"])
        )
        last_eval = train_loop(
            model=peft_model,
            optimizer=optimizer, scheduler=scheduler,
            train_loader=train_loader, val_loader=val_loader,
            logger=logger, tracker=tracker, device=device,
            total_steps=total_steps,
            eval_interval=int(training["eval_interval"]),
            post_step_hook=adalora_step_hook,
        )
        logger.log(
            total_steps,
            event="final",
            train_loss=None,
            val_loss=last_eval["val_loss"],
            val_accuracy=last_eval["val_accuracy"],
            trainable_parameters=count_parameters(peft_model, trainable_only=True),
            steps_to_target_accuracy=tracker.steps_to_target,
            wall_clock_to_target=tracker.wall_clock_to_target,
        )

    return {"run_id": run_id, "log_path": str(logger.path), **last_eval}


# --- CLI ---------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Unified training entry point for the four LoRA methods.",
    )
    parser.add_argument("--config", required=True, help="path to method yaml")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="override config seed (useful for multi-seed sweeps)",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="tiny subset / few steps / num_workers=0 for fast CPU sanity",
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    if args.seed is not None:
        cfg.setdefault("training", {})["seed"] = args.seed
    if args.smoke:
        apply_smoke_overrides(cfg)

    method = cfg.get("method")
    if method == "uniform":
        run_uniform(cfg)
    elif method in {"hardware_aware", "gradient_adaptive"}:
        run_two_stage(cfg)
    elif method == "adalora":
        run_adalora(cfg)
    else:
        raise ValueError(f"unknown method: {method!r}")
    return 0


if __name__ == "__main__":  # CLAUDE.md: required on Windows for DataLoader workers
    raise SystemExit(main())
