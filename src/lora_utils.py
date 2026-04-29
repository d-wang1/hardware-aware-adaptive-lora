"""LoRA module enumeration, gradient access, and (non-)uniform attachment.

Sits between PEFT and the rank allocator. The allocator (Phase 4b) needs:

- A way to walk the model and read per-module gradients (``lora_grad_norms``).
- A way to map "module name" -> "per-rank parameter cost" (``parameter_cost``
  applied to each module's in/out dims from ``enumerate_lora_modules``).
- A way to materialize a chosen rank assignment back into a fresh PEFT model
  (``build_non_uniform_lora_model``).

Sub-unit 4a.1.A (this file as of writing) implements the static side:
``parameter_cost``, ``enumerate_lora_modules``, and the uniform-rank builder
used in Stage 1 of the two-stage adaptive flow. Sub-unit 4a.1.B will add
``lora_grad_norms`` and ``build_non_uniform_lora_model``.
"""
from __future__ import annotations

from typing import Iterable

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
