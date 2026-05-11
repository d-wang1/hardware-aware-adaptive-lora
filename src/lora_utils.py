from __future__ import annotations

from typing import Iterable, Mapping

import torch.nn as nn
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from peft.tuners.lora import Linear as LoraLinear

_PEFT_PREFIX = "base_model.model."


def parameter_cost(in_dim: int, out_dim: int) -> int:
    # each extra rank adds one A row (in_dim) + one B column (out_dim)
    return in_dim + out_dim


def enumerate_lora_modules(peft_model: PeftModel) -> dict[str, dict]:
    """Walk a PEFT model and return per-target {A, B, in_dim, out_dim, rank}."""
    out: dict[str, dict] = {}
    for fqname, module in peft_model.named_modules():
        if not isinstance(module, LoraLinear):
            continue
        a_weight = module.lora_A["default"].weight  # (rank, in_features)
        b_weight = module.lora_B["default"].weight  # (out_features, rank)
        rank, in_dim = a_weight.shape
        out_dim, rank_b = b_weight.shape
        if rank != rank_b:
            raise RuntimeError(
                f"LoRA A/B rank mismatch at {fqname}: A has rank {rank}, "
                f"B has rank {rank_b}"
            )
        out[fqname] = {
            "A": a_weight,
            "B": b_weight,
            "in_dim": in_dim,
            "out_dim": out_dim,
            "rank": rank,
        }
    return out


def build_uniform_lora_model(
    base_model: nn.Module,
    target_modules: Iterable[str],
    rank: int,
    alpha: int,
    dropout: float = 0.0,
    task_type: TaskType | str | None = TaskType.SEQ_CLS,
) -> PeftModel:
    config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=list(target_modules),
        lora_dropout=dropout,
        bias="none",
        task_type=task_type,
    )
    return get_peft_model(base_model, config)


def lora_grad_norms(peft_model: PeftModel) -> dict[str, float]:
    """{fqname: ||grad(A)||_F + ||grad(B)||_F}; modules with no grad yet return 0."""
    # sum A and B norms because either alone undercounts how much the bottleneck moves
    out: dict[str, float] = {}
    for fqname, info in enumerate_lora_modules(peft_model).items():
        a, b = info["A"], info["B"]
        a_norm = float(a.grad.norm()) if a.grad is not None else 0.0
        b_norm = float(b.grad.norm()) if b.grad is not None else 0.0
        out[fqname] = a_norm + b_norm
    return out


def build_non_uniform_lora_model(
    base_model: nn.Module,
    target_modules: Iterable[str],
    rank_dict: Mapping[str, int],
    alpha: int,
    dropout: float = 0.0,
    task_type: TaskType | str | None = TaskType.SEQ_CLS,
) -> PeftModel:
    """Wrap base_model with per-module ranks via LoraConfig.rank_pattern."""
    if not rank_dict:
        raise ValueError(
            "rank_dict is empty; pass at least one module->rank entry"
        )
    if any(r < 1 for r in rank_dict.values()):
        raise ValueError(f"all ranks must be >= 1; got {dict(rank_dict)}")
    default_r = max(rank_dict.values())
    # rank_pattern keys match pre-wrap paths, so strip PEFT's wrapper prefix
    peft_rank_pattern = {
        k.removeprefix(_PEFT_PREFIX): v for k, v in rank_dict.items()
    }
    config = LoraConfig(
        r=default_r,
        lora_alpha=alpha,
        target_modules=list(target_modules),
        lora_dropout=dropout,
        bias="none",
        task_type=task_type,
        rank_pattern=peft_rank_pattern,
    )
    return get_peft_model(base_model, config)
