"""Tests for src.lora_utils.

Uses real DistilBERT (cached after the first test_models run) for the
end-to-end count check, and a tiny stub model for fast structural assertions
that don't need a full HF model.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from src.lora_utils import (
    build_non_uniform_lora_model,
    build_uniform_lora_model,
    enumerate_lora_modules,
    lora_grad_norms,
    parameter_cost,
)


# --- Stub model (fast, deterministic) ------------------------------------


class _MiniBackbone(nn.Module):
    """Two attention-style projections named like DistilBERT targets."""

    def __init__(self, in_dim: int = 64, out_dim: int = 64):
        super().__init__()
        self.q_lin = nn.Linear(in_dim, out_dim)
        self.v_lin = nn.Linear(in_dim, out_dim)
        self.classifier = nn.Linear(out_dim, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.q_lin(x) + self.v_lin(x))


# --- parameter_cost ------------------------------------------------------


def test_parameter_cost_basic():
    assert parameter_cost(768, 768) == 1536
    assert parameter_cost(64, 256) == 320
    assert parameter_cost(1, 1) == 2


# --- build_uniform_lora_model + enumerate_lora_modules -------------------


def test_uniform_lora_attaches_to_q_and_v_on_stub():
    model = _MiniBackbone(in_dim=64, out_dim=64)
    peft_model = build_uniform_lora_model(
        model,
        target_modules=["q_lin", "v_lin"],
        rank=4,
        alpha=8,
        task_type=None,  # plain nn.Module, no HF task head
    )
    enumerated = enumerate_lora_modules(peft_model)
    assert len(enumerated) == 2, list(enumerated)
    for fqname, info in enumerated.items():
        assert fqname.endswith(("q_lin", "v_lin")), fqname
        assert info["rank"] == 4
        assert info["in_dim"] == 64
        assert info["out_dim"] == 64


def test_uniform_lora_a_and_b_shapes():
    """A is (rank, in); B is (out, rank). Matters because allocator code
    will index into .shape[0] / .shape[1] when computing grad norms.
    """
    model = _MiniBackbone(in_dim=64, out_dim=128)
    peft_model = build_uniform_lora_model(
        model,
        target_modules=["q_lin", "v_lin"],
        rank=8,
        alpha=16,
        task_type=None,
    )
    for info in enumerate_lora_modules(peft_model).values():
        assert info["A"].shape == (8, 64), info["A"].shape
        assert info["B"].shape == (128, 8), info["B"].shape


def test_uniform_lora_classifier_not_targeted():
    """Only q_lin / v_lin are touched; the classifier head is left alone."""
    model = _MiniBackbone()
    peft_model = build_uniform_lora_model(
        model,
        target_modules=["q_lin", "v_lin"],
        rank=4,
        alpha=8,
        task_type=None,
    )
    fqnames = list(enumerate_lora_modules(peft_model).keys())
    assert all(not n.endswith("classifier") for n in fqnames), fqnames


def test_trainable_param_count_matches_lora_geometry():
    """After PEFT freezes the backbone, trainable params = rank * (in + out)
    per target module + the LoRA dropout/identity bookkeeping (zero floats)."""
    in_dim, out_dim, rank = 64, 64, 4
    model = _MiniBackbone(in_dim=in_dim, out_dim=out_dim)
    peft_model = build_uniform_lora_model(
        model,
        target_modules=["q_lin", "v_lin"],
        rank=rank,
        alpha=8,
        task_type=None,
    )
    trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    # 2 targets * rank * (in_dim + out_dim) — one A row + one B col per rank
    expected_lora = 2 * rank * (in_dim + out_dim)
    assert trainable == expected_lora, (trainable, expected_lora)


# --- Real DistilBERT integration (slow first run; cached after) ----------


@pytest.fixture(scope="module")
def distilbert_uniform():
    """Real DistilBERT + uniform LoRA. Cached for tests in this module."""
    from src.models import load_model_and_tokenizer

    model, _ = load_model_and_tokenizer("distilbert-base-uncased", num_labels=2)
    peft_model = build_uniform_lora_model(
        model,
        target_modules=["q_lin", "v_lin"],
        rank=8,
        alpha=16,
    )
    return peft_model


def test_distilbert_uniform_produces_12_lora_modules(distilbert_uniform):
    enumerated = enumerate_lora_modules(distilbert_uniform)
    assert len(enumerated) == 12, list(enumerated)


def test_distilbert_uniform_module_dims_are_768(distilbert_uniform):
    for info in enumerate_lora_modules(distilbert_uniform).values():
        assert info["in_dim"] == 768
        assert info["out_dim"] == 768
        assert info["rank"] == 8


# --- lora_grad_norms -----------------------------------------------------


def test_grad_norms_zero_before_backward():
    """Before any backward pass, every param's .grad is None and we return 0.0
    (rather than raising) so the allocator is safe to call during warmup."""
    model = _MiniBackbone(in_dim=64, out_dim=64)
    peft_model = build_uniform_lora_model(
        model,
        target_modules=["q_lin", "v_lin"],
        rank=4,
        alpha=8,
        task_type=None,
    )
    norms = lora_grad_norms(peft_model)
    assert len(norms) == 2, list(norms)
    assert all(v == 0.0 for v in norms.values()), norms


def test_grad_norms_positive_after_backward():
    """A real forward+backward should make every targeted module's grad norm
    strictly positive — both q_lin and v_lin are on the gradient path of the
    classifier loss, so neither A nor B can stay at zero grad."""
    torch.manual_seed(0)
    model = _MiniBackbone(in_dim=64, out_dim=64)
    peft_model = build_uniform_lora_model(
        model,
        target_modules=["q_lin", "v_lin"],
        rank=4,
        alpha=8,
        task_type=None,
    )
    x = torch.randn(8, 64)
    target = torch.zeros(8, dtype=torch.long)
    logits = peft_model(x)
    loss = nn.functional.cross_entropy(logits, target)
    loss.backward()

    norms = lora_grad_norms(peft_model)
    assert len(norms) == 2, list(norms)
    for fqname, value in norms.items():
        assert value > 0.0, (fqname, value)


# --- build_non_uniform_lora_model ----------------------------------------


def test_non_uniform_builder_rejects_empty_dict():
    model = _MiniBackbone()
    with pytest.raises(ValueError, match="empty"):
        build_non_uniform_lora_model(
            model,
            target_modules=["q_lin", "v_lin"],
            rank_dict={},
            alpha=8,
            task_type=None,
        )


def test_non_uniform_builder_rejects_zero_rank():
    model = _MiniBackbone()
    with pytest.raises(ValueError, match=">= 1"):
        build_non_uniform_lora_model(
            model,
            target_modules=["q_lin", "v_lin"],
            rank_dict={"q_lin": 0, "v_lin": 4},
            alpha=8,
            task_type=None,
        )


def test_non_uniform_builder_assigns_per_module_ranks():
    """Probe the post-wrap fqnames first, then build with two distinct ranks
    and verify each module ends up at the rank the dict asked for. This is
    what Stage 2 of the adaptive flow relies on."""
    in_dim, out_dim = 64, 64
    probe_model = _MiniBackbone(in_dim=in_dim, out_dim=out_dim)
    probe = build_uniform_lora_model(
        probe_model,
        target_modules=["q_lin", "v_lin"],
        rank=4,
        alpha=8,
        task_type=None,
    )
    fqnames = sorted(enumerate_lora_modules(probe).keys())
    assert len(fqnames) == 2, fqnames
    rank_dict = {fqnames[0]: 2, fqnames[1]: 6}

    model = _MiniBackbone(in_dim=in_dim, out_dim=out_dim)
    peft_model = build_non_uniform_lora_model(
        model,
        target_modules=["q_lin", "v_lin"],
        rank_dict=rank_dict,
        alpha=8,
        task_type=None,
    )
    enumerated = enumerate_lora_modules(peft_model)
    actual = {n: info["rank"] for n, info in enumerated.items()}
    assert sorted(actual.values()) == sorted(rank_dict.values()), (
        actual,
        rank_dict,
    )


def test_non_uniform_builder_param_count_matches_rank_dict():
    """Trainable LoRA params == sum_i r_i * (in_dim + out_dim).
    This is the budget-invariant the allocator depends on: total trainable
    LoRA params is fully determined by sum(rank_dict.values()) once the
    geometry is fixed."""
    in_dim, out_dim = 64, 64
    probe = build_uniform_lora_model(
        _MiniBackbone(in_dim=in_dim, out_dim=out_dim),
        target_modules=["q_lin", "v_lin"],
        rank=4,
        alpha=8,
        task_type=None,
    )
    fqnames = sorted(enumerate_lora_modules(probe).keys())
    rank_dict = {fqnames[0]: 2, fqnames[1]: 6}

    model = _MiniBackbone(in_dim=in_dim, out_dim=out_dim)
    peft_model = build_non_uniform_lora_model(
        model,
        target_modules=["q_lin", "v_lin"],
        rank_dict=rank_dict,
        alpha=8,
        task_type=None,
    )
    trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    expected = sum(r * (in_dim + out_dim) for r in rank_dict.values())
    assert trainable == expected, (trainable, expected, rank_dict)


def test_distilbert_uniform_trainable_params_match_budget(distilbert_uniform):
    """12 targets * rank=8 * (768 + 768) = 147,456 LoRA params. The classifier
    head is also trainable on a SEQ_CLS PEFT model, which adds a known amount;
    we just check the LoRA contribution is at least the geometric minimum.
    """
    trainable = sum(
        p.numel() for p in distilbert_uniform.parameters() if p.requires_grad
    )
    expected_lora_min = 12 * 8 * (768 + 768)  # 147,456
    assert trainable >= expected_lora_min
