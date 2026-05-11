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


class _MiniBackbone(nn.Module):

    def __init__(self, in_dim: int = 64, out_dim: int = 64):
        super().__init__()
        self.q_lin = nn.Linear(in_dim, out_dim)
        self.v_lin = nn.Linear(in_dim, out_dim)
        self.classifier = nn.Linear(out_dim, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.q_lin(x) + self.v_lin(x))


def test_parameter_cost_basic():
    assert parameter_cost(768, 768) == 1536
    assert parameter_cost(64, 256) == 320
    assert parameter_cost(1, 1) == 2


def test_uniform_lora_attaches_to_q_and_v_on_stub():
    model = _MiniBackbone(in_dim=64, out_dim=64)
    peft_model = build_uniform_lora_model(
        model,
        target_modules=["q_lin", "v_lin"],
        rank=4,
        alpha=8,
        task_type=None,
    )
    enumerated = enumerate_lora_modules(peft_model)
    assert len(enumerated) == 2, list(enumerated)
    for fqname, info in enumerated.items():
        assert fqname.endswith(("q_lin", "v_lin")), fqname
        assert info["rank"] == 4
        assert info["in_dim"] == 64
        assert info["out_dim"] == 64


def test_uniform_lora_a_and_b_shapes():
    # A is (rank, in); B is (out, rank)
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
    expected_lora = 2 * rank * (in_dim + out_dim)
    assert trainable == expected_lora, (trainable, expected_lora)


@pytest.fixture(scope="module")
def distilbert_uniform():
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


def test_grad_norms_zero_before_backward():
    # before any backward, .grad is None on every param -> 0.0 instead of raising
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
    # 12 targets * rank=8 * (768 + 768) = 147,456 LoRA params
    trainable = sum(
        p.numel() for p in distilbert_uniform.parameters() if p.requires_grad
    )
    expected_lora_min = 12 * 8 * (768 + 768)
    assert trainable >= expected_lora_min
