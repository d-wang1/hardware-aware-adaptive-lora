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
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=num_labels
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer


def find_lora_targets(
    model: nn.Module,
    target_keywords: Iterable[str] = ("q_lin", "v_lin"),
) -> list[str]:
    # match on the final path segment, not substring, to avoid spurious hits
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
    module = model.get_submodule(fqname)
    if not isinstance(module, nn.Linear):
        raise TypeError(
            f"{fqname} is not nn.Linear (got {type(module).__name__})"
        )
    return module.in_features, module.out_features


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())
