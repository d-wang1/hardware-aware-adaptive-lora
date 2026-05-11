"""Quick CPU walkthrough of LoRA enumeration + gradient + non-uniform attach.

Targets attention-only (q_lin, v_lin) so the demo arithmetic stays simple:
12 modules * rank 8 = 96. Production uses q/v/lin1/lin2 = 24 modules, budget 192.

Run from repo root: python demo_lora_grads.py
"""
from __future__ import annotations

import torch
import torch.nn as nn

from src.lora_utils import (
    build_non_uniform_lora_model,
    build_uniform_lora_model,
    enumerate_lora_modules,
    lora_grad_norms,
    parameter_cost,
)
from src.models import count_parameters, load_model_and_tokenizer


def main() -> None:
    print("=== 1. Load DistilBERT + build uniform LoRA (rank=8) ===")
    model, _ = load_model_and_tokenizer("distilbert-base-uncased", num_labels=2)
    peft_model = build_uniform_lora_model(
        model,
        target_modules=["q_lin", "v_lin"],
        rank=8,
        alpha=16,
    )
    enumerated = enumerate_lora_modules(peft_model)
    print(f"LoRA modules: {len(enumerated)}")
    print(f"Trainable params (uniform r=8): {count_parameters(peft_model):,}")

    print("\n=== 2. Grad norms BEFORE any backward (should all be 0.0) ===")
    norms_before = lora_grad_norms(peft_model)
    nonzero_before = sum(1 for v in norms_before.values() if v != 0.0)
    print(f"Modules with nonzero grad: {nonzero_before} / {len(norms_before)}")
    for name, v in list(norms_before.items())[:3]:
        print(f"  {name[-50:]:>50}  {v:.6f}")

    print("\n=== 3. One forward+backward, then re-check ===")
    peft_model.train()
    # synthetic batch: 4 sequences, length 16, labels 0/1
    input_ids = torch.randint(0, 30000, (4, 16))
    attention_mask = torch.ones_like(input_ids)
    labels = torch.tensor([0, 1, 0, 1])
    out = peft_model(
        input_ids=input_ids, attention_mask=attention_mask, labels=labels
    )
    out.loss.backward()
    norms_after = lora_grad_norms(peft_model)
    positive = sum(1 for v in norms_after.values() if v > 0.0)
    print(f"Modules with positive grad: {positive} / {len(norms_after)}")
    print(f"Sample (first 3, last 3):")
    items = list(norms_after.items())
    for name, v in items[:3] + items[-3:]:
        print(f"  {name[-50:]:>50}  {v:.6f}")
    g_min = min(norms_after.values())
    g_max = max(norms_after.values())
    print(f"min grad: {g_min:.6f}   max grad: {g_max:.6f}")

    print("\n=== 4. Build NON-uniform model from a synthetic rank_dict ===")
    # pretend the allocator decided some modules need more rank than others
    fqnames = sorted(enumerated.keys())
    # first 6 modules rank=4 and last 6 rank=12 -> sum = 24 + 72 = 96
    rank_dict = {n: 4 for n in fqnames[:6]} | {n: 12 for n in fqnames[6:]}
    print(f"Rank dict (showing 4 entries):")
    for name, r in list(rank_dict.items())[:4]:
        print(f"  {name[-50:]:>50}  rank={r}")
    print(f"sum(rank_dict.values()) = {sum(rank_dict.values())} (demo budget: 96)")

    # need a fresh base model; get_peft_model mutates in place
    base_again, _ = load_model_and_tokenizer(
        "distilbert-base-uncased", num_labels=2
    )
    nonuniform = build_non_uniform_lora_model(
        base_again,
        target_modules=["q_lin", "v_lin"],
        rank_dict=rank_dict,
        alpha=16,
    )
    actual_ranks = sorted(
        info["rank"] for info in enumerate_lora_modules(nonuniform).values()
    )
    expected_ranks = sorted(rank_dict.values())
    print(f"actual ranks  : {actual_ranks}")
    print(f"expected ranks: {expected_ranks}")
    assert actual_ranks == expected_ranks, "rank_pattern did not take effect!"

    # LoRA-only param count: sum_i r_i * (in+out) = 96 * 1536 = 147,456
    cost = parameter_cost(768, 768)
    expected_lora_params = sum(rank_dict.values()) * cost
    print(f"Predicted LoRA params: {expected_lora_params:,}")
    print(f"Trainable in non-uniform model (LoRA + classifier head): "
          f"{count_parameters(nonuniform):,}")

    print("\nDone.")


if __name__ == "__main__":
    main()
