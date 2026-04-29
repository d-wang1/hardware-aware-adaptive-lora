"""LoRA module enumeration, gradient access, and (non-)uniform attachment.

Sits between PEFT and the rank allocator. The allocator (Phase 4b) needs:

- A way to walk the model and read per-module gradients (``lora_grad_norms``).
- A way to map "module name" -> "per-rank parameter cost" (``parameter_cost``
  applied to each module's in/out dims from ``enumerate_lora_modules``).
- A way to materialize a chosen rank assignment back into a fresh PEFT model
  (``build_non_uniform_lora_model``).
"""
from __future__ import annotations

from typing import Iterable, Mapping

import torch.nn as nn
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from peft.tuners.lora import Linear as LoraLinear


def parameter_cost(in_dim: int, out_dim: int) -> int:
    """Per-rank trainable parameter cost: every additional rank adds one row
    of A (length ``in_dim``) and one column of B (length ``out_dim``), so the
    cost is ``in_dim + out_dim``. The Phase 4b allocator uses this as the
    denominator in ``s_i = g_i / c_i^α``.
    """
    return in_dim + out_dim


def enumerate_lora_modules(peft_model: PeftModel) -> dict[str, dict]:
    """Return ``{fqname: {"A", "B", "in_dim", "out_dim", "rank"}}`` for every
    LoRA-injected target in ``peft_model``.

    ``in_dim`` / ``out_dim`` are the wrapped ``nn.Linear``'s features (used for
    the cost proxy), not the bottleneck rank — those stay separate as ``rank``.
    Looks up the ``"default"`` adapter (PEFT's standard single-adapter slot);
    if the project ever uses named adapters this needs revisiting.
    """
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
    """Wrap ``base_model`` in PEFT with the same rank on every target.

    ``target_modules`` accepts either suffixes (``["q_lin", "v_lin"]``) or
    fully-qualified names — PEFT's matcher handles both. Used directly for
    the uniform baseline and for Stage 1 (warmup) of the adaptive methods.
    """
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
    """Return ``{fqname: ||grad(A)||_F + ||grad(B)||_F}`` over LoRA modules.

    This is the per-module gradient signal ``g_i`` that the Phase 4b allocator
    consumes (combined into ``s_i = g_i / c_i^α``). The Frobenius norms of A
    and B are summed because both halves of the low-rank product carry signal
    about how much the bottleneck is being used; either alone undercounts.

    Any tensor whose ``.grad`` is ``None`` (e.g. before the first backward
    pass, or when frozen) contributes ``0.0`` rather than raising — the
    allocator should be safe to call at any point in training, including
    warmup step 0 when the EMA is being seeded.
    """
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
    """Wrap ``base_model`` in PEFT with per-module ranks taken from ``rank_dict``.

    Used for Stage 2 of the two-stage adaptive flow: after the allocator
    decides each module's rank, we throw away the warmup PEFT wrapper and
    rebuild a fresh one with ``LoraConfig.rank_pattern``. Keys in
    ``rank_dict`` are the same fqnames produced by ``enumerate_lora_modules``;
    PEFT regex-matches them against module names after wrapping.

    The default ``r`` is set to ``max(rank_dict.values())`` so any unmatched
    module — there shouldn't be any if the dict came straight from
    ``enumerate_lora_modules`` — still gets a sensible (and never zero) rank
    rather than crashing or silently dropping. ``target_modules`` still has
    to be passed explicitly so PEFT knows which leaves to wrap; the
    ``rank_pattern`` only chooses *what rank* to use among already-targeted
    modules, not *whether* to target them.
    """
    if not rank_dict:
        raise ValueError(
            "rank_dict is empty; pass at least one module->rank entry"
        )
    if any(r < 1 for r in rank_dict.values()):
        raise ValueError(f"all ranks must be >= 1; got {dict(rank_dict)}")
    default_r = max(rank_dict.values())
    config = LoraConfig(
        r=default_r,
        lora_alpha=alpha,
        target_modules=list(target_modules),
        lora_dropout=dropout,
        bias="none",
        task_type=task_type,
        rank_pattern=dict(rank_dict),
    )
    return get_peft_model(base_model, config)
