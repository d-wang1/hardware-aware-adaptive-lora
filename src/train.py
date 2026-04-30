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
from src.lora_utils import build_uniform_lora_model
from src.models import (
    count_parameters,
    find_lora_target_module_names,
    load_model_and_tokenizer,
)


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
    start_step: int = 0,
) -> dict[str, float | int]:
    """Run ``total_steps`` optimizer steps, evaluating every ``eval_interval``.

    Step 0 is *not* evaluated — the optimizer hasn't moved yet so the
    pretrained baseline isn't informative for time-to-target. The final
    step is always evaluated regardless of interval alignment so the run
    has a guaranteed last-row ``val_accuracy`` for Phase 6 aggregation.

    ``allocator``: if provided, ``update_gradient_scores(model)`` is called
    after ``loss.backward()`` and before ``optimizer.step()`` — that's the
    only window where gradients exist on the LoRA parameters.
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
            allocator.update_gradient_scores(model)

        optimizer.step()
        scheduler.step()
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
        raise NotImplementedError(
            f"method '{method}' lands in Phase 5.3 (two-stage adaptive)"
        )
    elif method == "adalora":
        raise NotImplementedError("method 'adalora' lands in Phase 5.4")
    else:
        raise ValueError(f"unknown method: {method!r}")
    return 0


if __name__ == "__main__":  # CLAUDE.md: required on Windows for DataLoader workers
    raise SystemExit(main())
