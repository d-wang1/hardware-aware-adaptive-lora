"""Smoke tests for src.models.

Loads DistilBERT once per test module — the first run downloads ~250 MB
to the HuggingFace cache; subsequent runs reuse the cache.
"""
from __future__ import annotations

import pytest

from src.models import (
    count_parameters,
    find_lora_target_module_names,
    load_model_and_tokenizer,
    module_dims,
)

MODEL_NAME = "distilbert-base-uncased"


@pytest.fixture(scope="module")
def distilbert():
    return load_model_and_tokenizer(MODEL_NAME, num_labels=2)


def test_distilbert_has_12_lora_targets(distilbert):
    model, _ = distilbert
    names = find_lora_target_module_names(model)
    assert len(names) == 12, names
    for n in names:
        assert n.endswith(("q_lin", "v_lin"))


def test_module_dims_q_lin(distilbert):
    model, _ = distilbert
    names = find_lora_target_module_names(model)
    for n in names:
        in_dim, out_dim = module_dims(model, n)
        assert in_dim == 768
        assert out_dim == 768


def test_count_parameters_full_model_has_distilbert_size(distilbert):
    model, _ = distilbert
    total = count_parameters(model, trainable_only=False)
    # DistilBERT base ~67M params; classifier head adds a small amount.
    assert 60_000_000 < total < 80_000_000


def test_count_parameters_trainable_equals_total_on_fresh_model(distilbert):
    model, _ = distilbert
    # Before PEFT freezes the backbone, every parameter is trainable.
    assert count_parameters(model, trainable_only=True) == count_parameters(
        model, trainable_only=False
    )


def test_module_dims_rejects_non_linear(distilbert):
    model, _ = distilbert
    # The top-level module is nn.Module, not nn.Linear.
    with pytest.raises(TypeError):
        module_dims(model, "distilbert.embeddings")
