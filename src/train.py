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
    find_lora_targets,
    load_model_and_tokenizer,
)
from src.rank_allocator import HardwareAwareRankAllocator


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def make_run_id(method: str, seed: int) -> str:
    # UTC so logs from different machines sort consistently
    stamp = dt.datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    return f"{method}-seed{seed}-{stamp}"


def apply_smoke_overrides(cfg: dict[str, Any]) -> None:
    training = cfg.setdefault("training", {})
    training["max_train_samples"] = 64
    training["max_val_samples"] = 32
    training["num_workers"] = 0
    training["max_steps"] = 5
    training["eval_interval"] = 5
    training["epochs"] = 1
    # two-stage methods consume warmup_steps; set here so smoke is method-agnostic
    training["warmup_steps"] = 2
    # AdaLoRA needs tinit + tfinal < total_step to have a budgeting phase
    lora = cfg.setdefault("lora", {})
    lora["tinit"] = 1
    lora["tfinal"] = 1
    lora["deltaT"] = 1


def _log_config(logger: HardwareLogger, cfg: dict[str, Any]) -> None:
    logger.log(
        0,
        event="config",
        config=cfg,
        seed=cfg["training"]["seed"],
    )


def build_optimizer_and_scheduler(
    model: nn.Module,
    training_cfg: dict[str, Any],
    total_steps: int,
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]:
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(training_cfg["learning_rate"]),
    )
    warmup_steps = max(1, int(0.06 * total_steps))
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    return optimizer, scheduler


def _cycle(loader: DataLoader) -> Iterator[dict[str, torch.Tensor]]:
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
            # only window where LoRA grads exist (between backward and zero_grad)
            with logger.scheduler_block():
                allocator.update_gradient_scores(model)

        optimizer.step()
        scheduler.step()

        # AdaLoRA's update_and_allocate reads p * p.grad, so fire before zero_grad
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
        last_eval = {"step": start_step, "val_loss": float("nan"),
                     "val_accuracy": float("nan")}
    return last_eval


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
    set_seed(cfg["training"]["seed"])
    device = _resolve_device()

    base_model, tokenizer = load_model_and_tokenizer(
        cfg["model"]["name"], num_labels=2
    )
    train_loader, val_loader = _build_loaders(cfg, tokenizer)

    lora_cfg = cfg["lora"]
    targets = list(lora_cfg["target_modules"])
    n_modules = len(
        find_lora_targets(base_model, target_keywords=targets)
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
        _log_config(logger, cfg)
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
    method = cfg["method"]
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

    training = cfg["training"]
    max_steps = training.get("max_steps")
    if max_steps is not None:
        total_steps = int(max_steps)
        warmup_steps = int(training.get("warmup_steps", max(1, total_steps // 2)))
    else:
        total_steps = int(training["epochs"]) * len(train_loader)
        warmup_steps = int(training["warmup_steps"])
    stage2_steps = total_steps - warmup_steps
    if warmup_steps < 1 or stage2_steps < 1:
        raise ValueError(
            f"need >=1 step each in warmup ({warmup_steps}) and stage 2 "
            f"({stage2_steps}); total={total_steps}"
        )

    # stage 1: uniform LoRA, allocator EMA hook engaged
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
        _log_config(logger, cfg)
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

        # reallocation; charge to scheduler_overhead for fair comparison
        with logger.scheduler_block():
            rank_dict = allocator.allocate(warmup_model)
        logger.log(
            warmup_steps,
            event="reallocation",
            rank_dict=rank_dict,
            gradient_scores=allocator.gradient_scores,
            train_loss=None, val_loss=None, val_accuracy=None,
        )

        # free stage 1 state before allocating stage 2
        del warmup_model, stage1_optim, stage1_sched
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # stage 2: rebuild from fresh base; warmup LoRA weights are discarded
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
        total_step=total_steps,
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
        peft_model.base_model.update_and_allocate(global_step)

    with HardwareLogger(out_dir, method="adalora", run_id=run_id) as logger:
        _log_config(logger, cfg)
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Train one LoRA method on SST-2.",
    )
    parser.add_argument("--config", required=True, help="path to method yaml")
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="override config seed",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="tiny subset / few steps / num_workers=0 for a fast CPU sanity",
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


if __name__ == "__main__":  # required on Windows for DataLoader workers
    raise SystemExit(main())
