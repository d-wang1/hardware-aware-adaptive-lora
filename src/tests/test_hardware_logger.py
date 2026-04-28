"""Tests for src.hardware_logger.HardwareLogger.

Covers schema invariants, JSONL round-trip, throughput EMA, scheduler-overhead
accumulation, and CPU vs CUDA peak-memory behavior.
"""
from __future__ import annotations

import json
import time
import warnings as warnings_mod
from pathlib import Path

import pytest
import torch

from src.hardware_logger import SCHEMA_FIELDS, HardwareLogger


def _read_rows(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def test_jsonl_roundtrip(tmp_path: Path):
    logger = HardwareLogger(tmp_path, method="uniform", run_id="r1")
    logger.log(0, train_loss=0.71)
    logger.log(50, train_loss=0.42, val_loss=0.40, val_accuracy=0.81)
    logger.log(100, train_loss=0.30, val_loss=0.32, val_accuracy=0.88)
    logger.close()

    rows = _read_rows(tmp_path / "r1.jsonl")
    assert len(rows) == 3
    for row in rows:
        for field in SCHEMA_FIELDS:
            assert field in row, f"missing schema field: {field}"
    assert rows[0]["train_loss"] == pytest.approx(0.71)
    assert rows[2]["val_accuracy"] == pytest.approx(0.88)
    assert rows[2]["method"] == "uniform"
    assert rows[2]["step"] == 100


def test_missing_fields_serialize_as_null(tmp_path: Path):
    logger = HardwareLogger(tmp_path, method="adalora", run_id="r2")
    logger.log(0, train_loss=0.5)
    logger.close()

    [row] = _read_rows(tmp_path / "r2.jsonl")
    # Caller didn't supply val metrics or trainable_parameters
    assert row["val_loss"] is None
    assert row["val_accuracy"] is None
    assert row["trainable_parameters"] is None


def test_extra_fields_accepted(tmp_path: Path):
    """Two-stage methods log a marker row with ``event`` and ``rank_dict``."""
    logger = HardwareLogger(tmp_path, method="hardware_aware", run_id="r_extra")
    logger.log(
        200,
        event="reallocation",
        rank_dict={"layer.0.q_lin": 8, "layer.0.v_lin": 8},
    )
    logger.close()

    [row] = _read_rows(tmp_path / "r_extra.jsonl")
    assert row["event"] == "reallocation"
    assert row["rank_dict"] == {"layer.0.q_lin": 8, "layer.0.v_lin": 8}


def test_scheduler_overhead_accumulates(tmp_path: Path):
    logger = HardwareLogger(tmp_path, method="hardware_aware", run_id="r3")
    assert logger.scheduler_overhead_seconds == 0.0

    with logger.scheduler_block():
        time.sleep(0.05)
    first = logger.scheduler_overhead_seconds
    assert first >= 0.04  # generous slack for scheduler jitter

    with logger.scheduler_block():
        time.sleep(0.05)
    second = logger.scheduler_overhead_seconds
    assert second >= first + 0.04

    logger.close()


def test_examples_per_second_positive_after_step(tmp_path: Path):
    logger = HardwareLogger(tmp_path, method="uniform", run_id="r4")
    assert logger.examples_per_second is None

    logger.step_start()
    time.sleep(0.01)
    logger.step_end(num_examples=32)

    eps = logger.examples_per_second
    assert eps is not None and eps > 0.0
    # 32 examples in ~10ms is ~3200 ex/sec; allow a wide envelope
    assert eps < 1e6

    logger.close()


def test_examples_per_second_ema_smooths(tmp_path: Path):
    """EMA should not collapse to the latest sample (i.e. β actually applies)."""
    logger = HardwareLogger(tmp_path, method="uniform", run_id="r4b", ema_beta=0.9)

    logger.step_start()
    time.sleep(0.01)
    logger.step_end(num_examples=32)
    fast = logger.examples_per_second

    # Now do a much slower "step" — the EMA should move down but not to the
    # new instantaneous rate immediately because β=0.9.
    logger.step_start()
    time.sleep(0.05)
    logger.step_end(num_examples=32)
    smoothed = logger.examples_per_second

    assert smoothed < fast  # EMA pulled down by the slower step
    # The latest *instantaneous* rate is ~640; EMA should still be much higher.
    assert smoothed > 1000

    logger.close()


def test_step_end_without_start_raises(tmp_path: Path):
    logger = HardwareLogger(tmp_path, method="uniform", run_id="r5")
    with pytest.raises(RuntimeError):
        logger.step_end(num_examples=1)
    logger.close()


def test_log_returns_row(tmp_path: Path):
    logger = HardwareLogger(tmp_path, method="uniform", run_id="r6")
    row = logger.log(7, train_loss=0.123)
    assert row["step"] == 7
    assert row["train_loss"] == pytest.approx(0.123)
    assert row["method"] == "uniform"
    logger.close()


def test_context_manager_closes_file(tmp_path: Path):
    with HardwareLogger(tmp_path, method="uniform", run_id="r7") as logger:
        logger.log(0, train_loss=0.5)
    assert logger._fh.closed
    rows = _read_rows(tmp_path / "r7.jsonl")
    assert len(rows) == 1


@pytest.mark.skipif(
    torch.cuda.is_available(),
    reason="CPU-only check; on CUDA peak memory is real and may be non-zero",
)
def test_peak_memory_zero_on_cpu_with_one_warning(tmp_path: Path):
    logger = HardwareLogger(tmp_path, method="uniform", run_id="r8")
    with pytest.warns(RuntimeWarning, match="without CUDA"):
        assert logger.peak_memory_mb() == 0.0

    # The second call must NOT emit the warning again.
    with warnings_mod.catch_warnings():
        warnings_mod.simplefilter("error")  # any warning -> exception
        assert logger.peak_memory_mb() == 0.0

    logger.close()


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="Requires CUDA",
)
def test_peak_memory_nonneg_on_cuda(tmp_path: Path):
    logger = HardwareLogger(tmp_path, method="uniform", run_id="r9")
    # Allocate something on the device so peak memory is unambiguously > 0.
    t = torch.zeros(1024, 1024, device="cuda")
    del t
    assert logger.peak_memory_mb() > 0.0
    logger.close()
