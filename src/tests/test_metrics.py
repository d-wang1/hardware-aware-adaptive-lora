"""Phase 6 — unit tests for src.metrics.

Synthetic JSONL fixtures only; no real DistilBERT / training (consistent
with test_train.py + test_lora_utils.py: the convention is fast unit tests
+ out-of-band ``--smoke`` verification for the real pipeline).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from src.metrics import (
    ALPHA_SWEEP_VARIANTS,
    PRIMARY_VARIANTS,
    _attention_rank_share,
    _short_module_name,
    alpha_sweep_table,
    hardware_table,
    load_run_records,
    parse_run_jsonl,
    statistical_table,
    summarize,
    systems_tradeoff_table,
    variant_key,
    write_all_figures,
    write_all_tables,
    write_table,
)


# --- fixture builders ---------------------------------------------------


def _module_names() -> list[str]:
    """24 PEFT-style FQ module names matching the Phase 5.6 production
    layout: 6 layers × {q_lin, v_lin, lin1, lin2}."""
    out: list[str] = []
    for layer in range(6):
        for leaf in ("q_lin", "v_lin", "lin1", "lin2"):
            out.append(
                f"base_model.model.distilbert.transformer.layer.{layer}"
                f".{('attention' if leaf in ('q_lin', 'v_lin') else 'ffn')}"
                f".{leaf}"
            )
    return out


def _write_run(
    path: Path,
    method: str,
    seed: int,
    *,
    alpha: float | None = None,
    final_val_accuracy: float = 0.92,
    final_val_loss: float = 0.35,
    peak_memory_mb: float = 2048.0,
    examples_per_second: float = 50.0,
    trainable_parameters: int = 700_000,
    scheduler_overhead_seconds: float = 0.5,
    steps_to_target: int | None = 200,
    wall_clock_to_target: float | None = 30.0,
    rank_dict: dict[str, int] | None = None,
    eval_curve: list[tuple[int, float, float, float]] | None = None,
) -> None:
    """Write a JSONL log file matching the schema produced by HardwareLogger
    + src.train. ``eval_curve`` is a list of (step, dt_from_t0, val_loss,
    val_accuracy) for the per-eval-interval rows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    cfg: dict = {
        "method": method,
        "lora": {"total_rank_budget": 192, "rank": 8},
        "training": {"seed": seed, "batch_size": 32,
                     "learning_rate": 2e-4, "epochs": 3},
        "logging": {"output_dir": str(path.parent),
                    "target_accuracy": 0.90},
    }
    if method in {"hardware_aware", "gradient_adaptive"} or alpha is not None:
        cfg["allocator"] = {"hardware_alpha": alpha if alpha is not None
                            else (0.0 if method == "gradient_adaptive" else 1.0),
                            "ema_beta": 0.9}

    rows: list[dict] = []
    rows.append({
        "method": method, "step": 0, "timestamp": t0,
        "train_loss": None, "val_loss": None, "val_accuracy": None,
        "examples_per_second": None, "peak_memory_mb": 0.0,
        "trainable_parameters": None, "scheduler_overhead_seconds": 0.0,
        "event": "config", "config": cfg, "seed": seed,
    })

    for step, dt, vl, va in (eval_curve or [(100, 10.0, 0.5, 0.85),
                                            (200, 20.0, 0.4, 0.91)]):
        rows.append({
            "method": method, "step": step, "timestamp": t0 + dt,
            "train_loss": 0.6, "val_loss": vl, "val_accuracy": va,
            "examples_per_second": examples_per_second,
            "peak_memory_mb": peak_memory_mb,
            "trainable_parameters": trainable_parameters,
            "scheduler_overhead_seconds": scheduler_overhead_seconds,
        })

    if method in {"hardware_aware", "gradient_adaptive"} and rank_dict:
        rows.append({
            "method": method, "step": 100, "timestamp": t0 + 9.5,
            "train_loss": None, "val_loss": None, "val_accuracy": None,
            "examples_per_second": examples_per_second,
            "peak_memory_mb": peak_memory_mb,
            "trainable_parameters": trainable_parameters,
            "scheduler_overhead_seconds": scheduler_overhead_seconds,
            "event": "reallocation", "rank_dict": rank_dict,
            "gradient_scores": {k: 0.1 for k in rank_dict},
        })

    final_total_dt = (eval_curve[-1][1] if eval_curve else 20.0) + 1.0
    rows.append({
        "method": method, "step": 300, "timestamp": t0 + final_total_dt,
        "train_loss": None,
        "val_loss": final_val_loss, "val_accuracy": final_val_accuracy,
        "examples_per_second": examples_per_second,
        "peak_memory_mb": peak_memory_mb,
        "trainable_parameters": trainable_parameters,
        "scheduler_overhead_seconds": scheduler_overhead_seconds,
        "event": "final",
        "steps_to_target_accuracy": steps_to_target,
        "wall_clock_to_target": wall_clock_to_target,
        **({"rank_dict": rank_dict} if rank_dict else {}),
    })

    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _attn_skewed_rank_dict(total: int = 192) -> dict[str, int]:
    """Rank dict that puts most rank on attention modules (the post-Phase-5.6
    expected pattern for hardware-aware: cheap attention modules absorb
    rank because c_i is smaller). Sum equals ``total``."""
    mods = _module_names()
    attn_modules = [m for m in mods if "q_lin" in m or "v_lin" in m]
    ffn_modules = [m for m in mods if "lin1" in m or "lin2" in m]
    # Give attention rank 12 each (12*12 = 144) and FFN rank 4 each (12*4 = 48).
    rd = {m: 12 for m in attn_modules}
    rd.update({m: 4 for m in ffn_modules})
    assert sum(rd.values()) == total
    return rd


def _ffn_skewed_rank_dict(total: int = 192) -> dict[str, int]:
    """Inverse of the above — gradient-only's expected pattern (FFN gradients
    are larger, so without the cost penalty rank flows there)."""
    mods = _module_names()
    attn_modules = [m for m in mods if "q_lin" in m or "v_lin" in m]
    ffn_modules = [m for m in mods if "lin1" in m or "lin2" in m]
    rd = {m: 4 for m in attn_modules}
    rd.update({m: 12 for m in ffn_modules})
    assert sum(rd.values()) == total
    return rd


def _build_full_sweep(tmp_path: Path) -> Path:
    """3 seeds × {uniform, adalora, gradient_adaptive, hardware_aware α=1.0,
    hardware_aware α=0.5}. 15 runs total; mirrors the planned Phase 6.8 sweep."""
    base = tmp_path / "raw_logs"
    for seed in (42, 43, 44):
        _write_run(base / "uniform" / f"uniform-seed{seed}.jsonl",
                   "uniform", seed,
                   final_val_accuracy=0.91 + 0.005 * (seed - 42),
                   peak_memory_mb=2000.0,
                   examples_per_second=55.0,
                   scheduler_overhead_seconds=0.0)
        _write_run(base / "adalora" / f"adalora-seed{seed}.jsonl",
                   "adalora", seed,
                   final_val_accuracy=0.93 + 0.003 * (seed - 42),
                   peak_memory_mb=2200.0,
                   examples_per_second=42.0,
                   scheduler_overhead_seconds=12.0)
        _write_run(base / "gradient_adaptive"
                        / f"gradient_adaptive-seed{seed}.jsonl",
                   "gradient_adaptive", seed,
                   final_val_accuracy=0.91,
                   peak_memory_mb=2050.0,
                   examples_per_second=53.0,
                   scheduler_overhead_seconds=0.6,
                   rank_dict=_ffn_skewed_rank_dict())
        _write_run(base / "hardware_aware"
                        / f"hardware_aware-seed{seed}-alpha1.jsonl",
                   "hardware_aware", seed, alpha=1.0,
                   final_val_accuracy=0.92,
                   peak_memory_mb=2100.0,
                   examples_per_second=52.0,
                   scheduler_overhead_seconds=0.5,
                   rank_dict=_attn_skewed_rank_dict())
        _write_run(base / "hardware_aware"
                        / f"hardware_aware-seed{seed}-alpha0_5.jsonl",
                   "hardware_aware", seed, alpha=0.5,
                   final_val_accuracy=0.915,
                   peak_memory_mb=2080.0,
                   examples_per_second=52.5,
                   scheduler_overhead_seconds=0.5,
                   rank_dict=_attn_skewed_rank_dict())
    return base


# --- reader / parser ----------------------------------------------------


def test_parse_run_jsonl_extracts_config_alpha_and_final(tmp_path):
    p = tmp_path / "hw.jsonl"
    _write_run(
        p, "hardware_aware", seed=42, alpha=0.5,
        final_val_accuracy=0.93, final_val_loss=0.31,
        peak_memory_mb=2150.0, examples_per_second=51.0,
        scheduler_overhead_seconds=0.6, steps_to_target=180,
        wall_clock_to_target=27.5, rank_dict=_attn_skewed_rank_dict(),
    )
    rec = parse_run_jsonl(p)
    assert rec.method == "hardware_aware"
    assert rec.seed == 42
    assert rec.alpha == 0.5
    assert rec.final_val_accuracy == 0.93
    assert rec.steps_to_target == 180
    assert rec.wall_clock_to_target == 27.5
    assert rec.rank_dict is not None
    assert sum(rec.rank_dict.values()) == 192
    assert rec.eval_curve, "eval rows must be populated"
    # eval_curve excludes config / reallocation / final rows.
    assert all(
        not isinstance(p.val_accuracy, type(None)) for p in rec.eval_curve
    )


def test_parse_run_jsonl_rejects_missing_config_row(tmp_path):
    p = tmp_path / "broken.jsonl"
    with p.open("w", encoding="utf-8") as f:
        # final row only — no event=config
        f.write(json.dumps({
            "method": "uniform", "step": 5, "timestamp": time.time(),
            "event": "final", "val_loss": 0.4, "val_accuracy": 0.9,
            "peak_memory_mb": 100.0, "examples_per_second": 1.0,
            "trainable_parameters": 1, "scheduler_overhead_seconds": 0.0,
            "steps_to_target_accuracy": None, "wall_clock_to_target": None,
        }) + "\n")
    with pytest.raises(ValueError, match="event=\"config\""):
        parse_run_jsonl(p)


def test_parse_run_jsonl_rejects_missing_final_row(tmp_path):
    """Run that crashed mid-training (no ``event="final"`` row) should be
    skipped with a clear error rather than silently aggregated."""
    p = tmp_path / "crashed.jsonl"
    t0 = time.time()
    with p.open("w", encoding="utf-8") as f:
        f.write(json.dumps({
            "method": "uniform", "step": 0, "timestamp": t0,
            "event": "config", "seed": 0, "config": {"method": "uniform"},
        }) + "\n")
        # one eval row, then nothing
        f.write(json.dumps({
            "method": "uniform", "step": 100, "timestamp": t0 + 5.0,
            "val_loss": 0.5, "val_accuracy": 0.8,
            "peak_memory_mb": 100.0, "examples_per_second": 1.0,
            "trainable_parameters": 1, "scheduler_overhead_seconds": 0.0,
        }) + "\n")
    with pytest.raises(ValueError, match="event=\"final\""):
        parse_run_jsonl(p)


def test_load_run_records_skips_broken_files_with_warning(tmp_path):
    base = tmp_path / "raw_logs"
    _write_run(base / "uniform" / "good.jsonl", "uniform", 42)
    bad = base / "uniform" / "bad.jsonl"
    bad.parent.mkdir(parents=True, exist_ok=True)
    bad.write_text("not json\n", encoding="utf-8")
    with pytest.warns():
        records = load_run_records(base)
    assert len(records) == 1
    assert records[0].method == "uniform"


# --- variant grouping ---------------------------------------------------


def test_variant_key_splits_hardware_aware_by_alpha(tmp_path):
    p1 = tmp_path / "hw1.jsonl"
    p05 = tmp_path / "hw05.jsonl"
    _write_run(p1, "hardware_aware", seed=0, alpha=1.0)
    _write_run(p05, "hardware_aware", seed=0, alpha=0.5)
    assert variant_key(parse_run_jsonl(p1)) == "hardware_aware_alpha1"
    assert variant_key(parse_run_jsonl(p05)) == "hardware_aware_alpha0.5"


def test_variant_key_uses_method_for_non_hardware_aware(tmp_path):
    p = tmp_path / "u.jsonl"
    _write_run(p, "uniform", seed=0)
    assert variant_key(parse_run_jsonl(p)) == "uniform"


# --- aggregation + tables ----------------------------------------------


def test_summarize_groups_runs_by_variant(tmp_path):
    base = _build_full_sweep(tmp_path)
    records = load_run_records(base)
    primary = summarize(records, PRIMARY_VARIANTS)
    # 4 primary variants × 3 seeds each.
    assert set(primary.keys()) == set(PRIMARY_VARIANTS.keys())
    for vk, s in primary.items():
        assert s.seeds == 3, vk


def test_statistical_table_has_one_row_per_primary_method(tmp_path):
    base = _build_full_sweep(tmp_path)
    records = load_run_records(base)
    summaries = summarize(records, PRIMARY_VARIANTS)
    rows = statistical_table(summaries)
    assert rows[0] == ["Method", "Final Val Loss", "Final Val Accuracy",
                       "Steps to Target"]
    assert len(rows) == 1 + 4  # header + 4 methods
    # Steps-to-target column should be the (n/n) format
    last_col = [r[-1] for r in rows[1:]]
    assert all("(3/3)" in c for c in last_col)


def test_hardware_table_includes_wall_clock(tmp_path):
    base = _build_full_sweep(tmp_path)
    records = load_run_records(base)
    summaries = summarize(records, PRIMARY_VARIANTS)
    rows = hardware_table(summaries)
    assert rows[0] == ["Method", "Peak Memory MB", "Examples/sec",
                       "Wall-Clock (s)", "Trainable Params"]
    assert len(rows) == 5


def test_systems_tradeoff_table_excludes_gradient_adaptive(tmp_path):
    """README §1082-1088: only Uniform / AdaLoRA / Hardware-Aware in this
    table. Gradient-adaptive must NOT appear — it's an ablation."""
    base = _build_full_sweep(tmp_path)
    records = load_run_records(base)
    summaries = summarize(records, PRIMARY_VARIANTS)
    rows = systems_tradeoff_table(summaries)
    assert len(rows) == 1 + 3  # header + 3 methods
    method_col = [r[0] for r in rows[1:]]
    assert "Gradient" not in " ".join(method_col)


def test_alpha_sweep_table_has_three_alpha_rows(tmp_path):
    base = _build_full_sweep(tmp_path)
    records = load_run_records(base)
    rows = alpha_sweep_table(records)
    assert len(rows) == 1 + 3
    # Attn rank share column — α=1.0 (attn-skewed) > α=0.0 (ffn-skewed)
    grad_share = float(rows[1][3].split(" ")[0])  # α=0.0 row
    hw1_share = float(rows[3][3].split(" ")[0])   # α=1.0 row
    assert hw1_share > grad_share, (
        "hardware-aware α=1.0 must skew toward attention more than "
        "gradient-only — see CLAUDE.md post-Phase-5.6 expected pattern"
    )


def test_attention_rank_share_matches_expected_pattern(tmp_path):
    """Attn-skewed rank_dict → share > 0.5; FFN-skewed → share < 0.5."""
    p_attn = tmp_path / "hw.jsonl"
    p_ffn = tmp_path / "grad.jsonl"
    _write_run(p_attn, "hardware_aware", seed=0, alpha=1.0,
               rank_dict=_attn_skewed_rank_dict())
    _write_run(p_ffn, "gradient_adaptive", seed=0,
               rank_dict=_ffn_skewed_rank_dict())
    rec_attn = parse_run_jsonl(p_attn)
    rec_ffn = parse_run_jsonl(p_ffn)
    assert _attention_rank_share(rec_attn) > 0.5
    assert _attention_rank_share(rec_ffn) < 0.5


def test_short_module_name_strips_prefix():
    fq = "base_model.model.distilbert.transformer.layer.3.attention.q_lin"
    assert _short_module_name(fq) == "L3.attn.q_lin"
    fq2 = "base_model.model.distilbert.transformer.layer.5.ffn.lin2"
    assert _short_module_name(fq2) == "L5.ffn.lin2"


# --- end-to-end driver --------------------------------------------------


def test_write_all_tables_produces_csv_and_md(tmp_path):
    base = _build_full_sweep(tmp_path)
    records = load_run_records(base)
    out = tmp_path / "summaries"
    write_all_tables(records, out)
    for name in ("statistical", "hardware", "systems_tradeoff", "alpha_sweep"):
        assert (out / f"{name}.csv").exists(), name
        assert (out / f"{name}.md").exists(), name


def test_write_all_figures_produces_png(tmp_path):
    base = _build_full_sweep(tmp_path)
    records = load_run_records(base)
    out = tmp_path / "figures"
    write_all_figures(records, out)
    expected = {
        "val_accuracy_vs_walltime.png",
        "rank_allocation_heatmap.png",
        "peak_memory_bars.png",
        "examples_per_second_bars.png",
        "scheduler_overhead_bars.png",
    }
    actual = {p.name for p in out.glob("*.png")}
    assert expected.issubset(actual), expected - actual


def test_write_table_round_trips_csv(tmp_path):
    rows = [["a", "b"], ["1,2", "x"], ["plain", "y"]]
    csv_p = tmp_path / "t.csv"
    md_p = tmp_path / "t.md"
    write_table(rows, csv_p, md_p)
    text = csv_p.read_text(encoding="utf-8")
    # The "1,2" value must be quoted to survive CSV parsing.
    assert '"1,2"' in text
    md_text = md_p.read_text(encoding="utf-8")
    assert md_text.startswith("| a | b |")
