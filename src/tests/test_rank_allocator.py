"""Tests for src.rank_allocator.

Most tests are pure-Python on synthetic scores; the integration tests wire
the allocator against a tiny stub model + DistilBERT to confirm the fqname
keys round-trip cleanly into PEFT's ``rank_pattern`` via
``build_non_uniform_lora_model``.
"""
from __future__ import annotations

import random

import pytest
import torch
import torch.nn as nn

import src.rank_allocator as ra
from src.lora_utils import (
    build_non_uniform_lora_model,
    build_uniform_lora_model,
    enumerate_lora_modules,
)
from src.rank_allocator import HardwareAwareRankAllocator


class _MiniBackbone(nn.Module):
    """Same shape as the lora_utils stub (q_lin / v_lin / classifier)."""

    def __init__(self, in_dim: int = 64, out_dim: int = 64):
        super().__init__()
        self.q_lin = nn.Linear(in_dim, out_dim)
        self.v_lin = nn.Linear(in_dim, out_dim)
        self.classifier = nn.Linear(out_dim, 2)

    def forward(self, x):
        return self.classifier(self.q_lin(x) + self.v_lin(x))


# --- __init__ guards -----------------------------------------------------


def test_init_rejects_min_rank_below_one():
    with pytest.raises(ValueError, match="min_rank"):
        HardwareAwareRankAllocator(96, 0, 16, 1.0, 0.9)


def test_init_rejects_max_below_min():
    with pytest.raises(ValueError, match="max_rank"):
        HardwareAwareRankAllocator(96, 4, 2, 1.0, 0.9)


def test_init_rejects_ema_beta_out_of_range():
    with pytest.raises(ValueError, match="ema_beta"):
        HardwareAwareRankAllocator(96, 2, 16, 1.0, 1.0)


# --- update_gradient_scores ---------------------------------------------


def test_update_initializes_at_first_observation():
    """First update should seed _g_ema at the observed value rather than
    blending with an implicit zero prior — mirrors the throughput-EMA
    seeding in HardwareLogger."""
    torch.manual_seed(0)
    model = _MiniBackbone()
    peft = build_uniform_lora_model(
        model, target_modules=["q_lin", "v_lin"], rank=4, alpha=8, task_type=None
    )
    x = torch.randn(8, 64)
    target = torch.zeros(8, dtype=torch.long)
    nn.functional.cross_entropy(peft(x), target).backward()

    alloc = HardwareAwareRankAllocator(96, 2, 16, alpha=1.0, ema_beta=0.9)
    alloc.update_gradient_scores(peft)
    snap = alloc.gradient_scores
    assert len(snap) == 2
    assert all(v > 0.0 for v in snap.values()), snap


def test_update_blends_with_beta(monkeypatch):
    """Second update should produce ``beta * prior + (1 - beta) * new``.
    Monkeypatch ``lora_grad_norms`` so the math is exact rather than
    backward-pass dependent."""
    fake_norms = {"a": 3.0, "b": 5.0}
    monkeypatch.setattr(ra, "lora_grad_norms", lambda _peft: fake_norms)

    alloc = HardwareAwareRankAllocator(96, 2, 16, alpha=1.0, ema_beta=0.5)
    alloc._g_ema = {"a": 1.0, "b": 1.0}
    alloc.update_gradient_scores(peft_model=None)  # arg unused via monkeypatch

    assert alloc.gradient_scores == pytest.approx({"a": 2.0, "b": 3.0})


# --- compute_costs / compute_scores -------------------------------------


def test_compute_costs_matches_in_plus_out():
    model = _MiniBackbone(in_dim=64, out_dim=128)
    peft = build_uniform_lora_model(
        model, target_modules=["q_lin", "v_lin"], rank=4, alpha=8, task_type=None
    )
    alloc = HardwareAwareRankAllocator(96, 2, 16, alpha=1.0, ema_beta=0.9)
    costs = alloc.compute_costs(peft)
    assert len(costs) == 2
    assert all(c == 192 for c in costs.values()), costs  # 64 + 128


def test_compute_scores_alpha_zero_equals_gradient():
    """With alpha=0, c**alpha == 1, so scores collapse to the EMA values
    regardless of cost. This is the 'gradient-only' ablation."""
    alloc = HardwareAwareRankAllocator(96, 2, 16, alpha=0.0, ema_beta=0.9)
    alloc._g_ema = {"a": 5.0, "b": 7.0}
    scores = alloc.compute_scores({"a": 100, "b": 1})
    assert scores == pytest.approx({"a": 5.0, "b": 7.0})


def test_compute_scores_alpha_one_divides_by_cost():
    alloc = HardwareAwareRankAllocator(96, 2, 16, alpha=1.0, ema_beta=0.9)
    alloc._g_ema = {"a": 4.0, "b": 4.0}
    scores = alloc.compute_scores({"a": 1, "b": 4})
    assert scores == pytest.approx({"a": 4.0, "b": 1.0})


def test_compute_scores_unobserved_module_is_zero():
    alloc = HardwareAwareRankAllocator(96, 2, 16, alpha=1.0, ema_beta=0.9)
    alloc._g_ema = {"a": 5.0}
    scores = alloc.compute_scores({"a": 1, "b": 2})
    assert scores["b"] == 0.0


# --- allocate_ranks ------------------------------------------------------


def test_allocate_sum_invariant_random_scores():
    """For a range of random-but-bounded score vectors, the allocation must
    always sum to total_budget exactly. This is the project-wide invariant
    CLAUDE.md flags as 'must not drift'."""
    rng = random.Random(0)
    alloc = HardwareAwareRankAllocator(96, 2, 16, alpha=1.0, ema_beta=0.9)
    keys = [f"m{i}" for i in range(12)]
    for _ in range(50):
        scores = {k: rng.random() for k in keys}
        ranks = alloc.allocate_ranks(scores)
        assert sum(ranks.values()) == 96, ranks


def test_allocate_respects_bounds():
    alloc = HardwareAwareRankAllocator(96, 2, 16, alpha=1.0, ema_beta=0.9)
    # Heavily skewed: one module dominates. Without the cap it would absorb
    # most of the budget.
    scores = {f"m{i}": 0.001 for i in range(12)}
    scores["m0"] = 1000.0
    ranks = alloc.allocate_ranks(scores)
    for k, r in ranks.items():
        assert 2 <= r <= 16, (k, r)


def test_allocate_alpha_monotonicity():
    """Equal gradients, unequal costs: alpha=0 gives equal-ish ranks (the
    gradient-only ablation), alpha=1 strictly prefers the cheaper module."""
    costs = {"cheap": 10, "expensive": 100}
    g_ema = {"cheap": 1.0, "expensive": 1.0}

    alloc0 = HardwareAwareRankAllocator(20, 2, 16, alpha=0.0, ema_beta=0.9)
    alloc0._g_ema = dict(g_ema)
    ranks0 = alloc0.allocate_ranks(alloc0.compute_scores(costs))
    assert ranks0["cheap"] == ranks0["expensive"], ranks0

    alloc1 = HardwareAwareRankAllocator(20, 2, 16, alpha=1.0, ema_beta=0.9)
    alloc1._g_ema = dict(g_ema)
    ranks1 = alloc1.allocate_ranks(alloc1.compute_scores(costs))
    assert ranks1["cheap"] > ranks1["expensive"], ranks1
    assert sum(ranks1.values()) == 20


def test_allocate_zero_total_score_spreads_evenly():
    """All-zero scores (e.g. allocator queried before any backward pass) must
    still produce a valid allocation that sums to budget."""
    alloc = HardwareAwareRankAllocator(96, 2, 16, alpha=1.0, ema_beta=0.9)
    scores = {f"m{i}": 0.0 for i in range(12)}
    ranks = alloc.allocate_ranks(scores)
    assert sum(ranks.values()) == 96
    # 96 / 12 = 8 per module exactly
    assert all(r == 8 for r in ranks.values()), ranks


def test_allocate_nan_score_treated_as_zero():
    """NaN coerces to 0 so all weight goes to ``b``. ``b`` saturates at
    max_rank=16 and the leftover budget falls back to ``a`` via the
    rebalance step (a is the only under-cap candidate left)."""
    alloc = HardwareAwareRankAllocator(20, 2, 16, alpha=1.0, ema_beta=0.9)
    scores = {"a": float("nan"), "b": 1.0}
    ranks = alloc.allocate_ranks(scores)
    assert ranks["b"] == 16, ranks
    assert ranks["a"] == 4, ranks
    assert sum(ranks.values()) == 20


def test_allocate_infeasible_min_too_high():
    alloc = HardwareAwareRankAllocator(20, 8, 16, alpha=1.0, ema_beta=0.9)
    scores = {f"m{i}": 1.0 for i in range(4)}  # 4 * 8 = 32 > 20
    with pytest.raises(ValueError, match="min_rank"):
        alloc.allocate_ranks(scores)


def test_allocate_infeasible_max_too_low():
    alloc = HardwareAwareRankAllocator(96, 2, 4, alpha=1.0, ema_beta=0.9)
    scores = {f"m{i}": 1.0 for i in range(12)}  # 12 * 4 = 48 < 96
    with pytest.raises(ValueError, match="max_rank"):
        alloc.allocate_ranks(scores)


def test_allocate_empty_scores_raises():
    alloc = HardwareAwareRankAllocator(96, 2, 16, alpha=1.0, ema_beta=0.9)
    with pytest.raises(ValueError, match="empty"):
        alloc.allocate_ranks({})


# --- end-to-end: stub + non_uniform builder ------------------------------


def test_allocate_then_build_non_uniform_round_trips_on_stub():
    """The keys produced by the allocator (via enumerate_lora_modules on the
    warmup PEFT model) must feed straight into build_non_uniform_lora_model
    on a freshly loaded base model. This is the contract Stage 2 relies on."""
    torch.manual_seed(0)
    in_dim, out_dim = 64, 64

    warmup_model = _MiniBackbone(in_dim=in_dim, out_dim=out_dim)
    warmup_peft = build_uniform_lora_model(
        warmup_model,
        target_modules=["q_lin", "v_lin"],
        rank=4,
        alpha=8,
        task_type=None,
    )
    x = torch.randn(8, in_dim)
    target = torch.zeros(8, dtype=torch.long)
    nn.functional.cross_entropy(warmup_peft(x), target).backward()

    alloc = HardwareAwareRankAllocator(
        total_budget=12, min_rank=2, max_rank=10, alpha=1.0, ema_beta=0.9
    )
    alloc.update_gradient_scores(warmup_peft)
    rank_dict = alloc.allocate(warmup_peft)
    assert sum(rank_dict.values()) == 12
    assert set(rank_dict.keys()) == set(enumerate_lora_modules(warmup_peft).keys())

    fresh_model = _MiniBackbone(in_dim=in_dim, out_dim=out_dim)
    stage2 = build_non_uniform_lora_model(
        fresh_model,
        target_modules=["q_lin", "v_lin"],
        rank_dict=rank_dict,
        alpha=8,
        task_type=None,
    )
    actual = {n: info["rank"] for n, info in enumerate_lora_modules(stage2).items()}
    assert sorted(actual.values()) == sorted(rank_dict.values()), (actual, rank_dict)


# --- end-to-end: real DistilBERT (slow first run; cached after) ---------


@pytest.fixture(scope="module")
def distilbert_warmup_peft():
    """Warmed-up DistilBERT + uniform LoRA after one backward pass on a tiny
    synthetic batch. Reused across the integration tests in this module."""
    from src.models import load_model_and_tokenizer

    torch.manual_seed(0)
    model, tokenizer = load_model_and_tokenizer(
        "distilbert-base-uncased", num_labels=2
    )
    peft = build_uniform_lora_model(
        model, target_modules=["q_lin", "v_lin"], rank=8, alpha=16
    )
    enc = tokenizer(
        ["a positive sentence", "a negative sentence"],
        padding=True,
        return_tensors="pt",
    )
    out = peft(**enc, labels=torch.tensor([1, 0]))
    out.loss.backward()
    return peft


def test_distilbert_allocate_sums_to_budget(distilbert_warmup_peft):
    alloc = HardwareAwareRankAllocator(96, 2, 16, alpha=1.0, ema_beta=0.9)
    alloc.update_gradient_scores(distilbert_warmup_peft)
    rank_dict = alloc.allocate(distilbert_warmup_peft)
    assert len(rank_dict) == 12, rank_dict
    assert sum(rank_dict.values()) == 96
    assert all(2 <= r <= 16 for r in rank_dict.values()), rank_dict


def test_distilbert_allocate_then_build_stage2(distilbert_warmup_peft):
    """Stage-1 → allocator → Stage-2 round trip on the real model. Confirms
    the PEFT prefix stripping in build_non_uniform_lora_model still works
    when the rank dict comes from the allocator (not hand-written)."""
    from src.models import load_model_and_tokenizer

    alloc = HardwareAwareRankAllocator(96, 2, 16, alpha=1.0, ema_beta=0.9)
    alloc.update_gradient_scores(distilbert_warmup_peft)
    rank_dict = alloc.allocate(distilbert_warmup_peft)

    fresh_model, _ = load_model_and_tokenizer(
        "distilbert-base-uncased", num_labels=2
    )
    stage2 = build_non_uniform_lora_model(
        fresh_model,
        target_modules=["q_lin", "v_lin"],
        rank_dict=rank_dict,
        alpha=16,
    )
    stage2_ranks = {
        n: info["rank"] for n, info in enumerate_lora_modules(stage2).items()
    }
    assert len(stage2_ranks) == 12
    assert sum(stage2_ranks.values()) == 96
