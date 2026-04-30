"""Pretrained model loading and LoRA target-module identification.

DistilBERT's attention exposes ``q_lin`` and ``v_lin`` (768→768) and its
FFN exposes ``lin1`` (768→3072) and ``lin2`` (3072→768). The production
config targets all four across 6 transformer layers = 24 LoRA modules,
which sets the rank budget invariant: ``total_rank_budget = 192``
distributes to a uniform rank of 8 per module. The mixed-cost target
list (1536 for attention vs 3840 for FFN) is what makes the
hardware-aware allocator's ``s_i = g_i / c_i^α`` non-trivial — see
CLAUDE.md "Repo conventions".

``find_lora_target_module_names`` defaults to ``("q_lin", "v_lin")``
(historical; preserved for backward compatibility with attention-only
test fixtures); production callers pass all four keywords explicitly
through the YAML configs.
"""
from __future__ import annotations

from typing import Iterable

import torch.nn as nn
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)


def load_model_and_tokenizer(
    model_name: str,
    num_labels: int,
) -> tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """Load a HuggingFace sequence-classification model and its tokenizer."""
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer


def find_lora_target_module_names(
    model: nn.Module,
    target_keywords: Iterable[str] = ("q_lin", "v_lin"),
) -> list[str]:
    """Return fully-qualified names of nn.Linear modules whose final segment
    matches one of ``target_keywords``.

    Final-segment matching (rather than substring) avoids accidentally picking
    up unrelated modules whose path happens to contain "q_lin"/"v_lin".
    """
    keywords = set(target_keywords)
    names: list[str] = []
    for fqname, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        last = fqname.rsplit(".", 1)[-1]
        if last in keywords:
            names.append(fqname)
    return names


def module_dims(model: nn.Module, fqname: str) -> tuple[int, int]:
    """Return (in_features, out_features) of the nn.Linear at ``fqname``.

    The hardware-aware allocator's cost proxy is c_i = in_features + out_features.
    """
    module = model.get_submodule(fqname)
    if not isinstance(module, nn.Linear):
        raise TypeError(
            f"{fqname} is not nn.Linear (got {type(module).__name__})"
        )
    return module.in_features, module.out_features


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """Count parameters in ``model``; defaults to trainable-only (the
    LoRA-relevant count after PEFT freezes the backbone)."""
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())
