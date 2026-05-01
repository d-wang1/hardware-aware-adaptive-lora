"""Phase 6 — aggregate JSONL training logs into the README's three result
tables (Statistical / Hardware / Systems Tradeoff), the α-sweep ablation
table, and the project's headline figures (val-accuracy-vs-walltime, rank
heatmap, peak-memory / throughput / scheduler-overhead bars).

CLI::

    python -m src.metrics --logs-dir results/raw_logs \\
        --summaries-dir results/summaries --figures-dir results/figures

Schema contract — every row is the JSONL produced by ``HardwareLogger`` plus
the markers written by ``src.train``:

* one ``event="config"`` row at step 0 carrying ``config`` (the resolved cfg
  dict) and ``seed`` — Phase 6.7 added this; older logs without it are
  skipped with a warning.
* per-eval rows (no ``event`` key) with ``train_loss``, ``val_loss``,
  ``val_accuracy``, ``trainable_parameters``.
* optional ``event="reallocation"`` row (two-stage methods only) carrying
  ``rank_dict``.
* one terminal ``event="final"`` row with ``val_loss``, ``val_accuracy``,
  ``trainable_parameters``, ``steps_to_target_accuracy``,
  ``wall_clock_to_target``, and (for two-stage) ``rank_dict``.

α-sweep: hardware_aware runs are split into variants by
``cfg["allocator"]["hardware_alpha"]``. The primary tables include only
α=1.0 (the canonical hardware-aware setting); the α-sweep table covers
α ∈ {0.0 (= gradient_adaptive), 0.5, 1.0}.
"""
from __future__ import annotations

import argparse
import json
import math
import statistics
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any


# --- record + reader ----------------------------------------------------


@dataclass
class EvalPoint:
    """One eval-interval row, used by the val-accuracy-vs-walltime curves."""
    step: int
    wall_clock_s: float
    val_loss: float
    val_accuracy: float
    examples_per_second: float | None
    peak_memory_mb: float


@dataclass
class RunRecord:
    run_id: str
    log_path: Path
    method: str
    seed: int
    alpha: float | None
    final_val_loss: float
    final_val_accuracy: float
    steps_to_target: int | None
    wall_clock_to_target: float | None
    peak_memory_mb: float
    examples_per_second: float
    trainable_parameters: int
    scheduler_overhead_seconds: float
    rank_dict: dict[str, int] | None
    total_wall_clock_s: float
    eval_curve: list[EvalPoint]
    config: dict[str, Any]


def parse_run_jsonl(path: Path) -> RunRecord:
    """Parse one run's JSONL file into a ``RunRecord``.

    Raises ``ValueError`` if the file is empty, missing the config row
    (older logs / pre-Phase-6.7), or missing the final row (run crashed).
    """
    with path.open(encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    if not rows:
        raise ValueError(f"empty log: {path}")

    config_row = next((r for r in rows if r.get("event") == "config"), None)
    if config_row is None:
        raise ValueError(
            f"no event=\"config\" row in {path.name}; older log format. "
            f"Re-run with current src/train.py."
        )

    final_rows = [r for r in rows if r.get("event") == "final"]
    if not final_rows:
        raise ValueError(
            f"no event=\"final\" row in {path.name}; run may have crashed."
        )
    if len(final_rows) > 1:
        warnings.warn(
            f"{path.name} has {len(final_rows)} \"final\" rows; using the "
            f"last one. May indicate the same run_id was logged twice "
            f"(append-mode). See CLAUDE.md JSONL note.",
            RuntimeWarning,
            stacklevel=2,
        )
    final_row = final_rows[-1]

    realloc_row = next(
        (r for r in rows if r.get("event") == "reallocation"), None
    )

    cfg = config_row["config"]
    t0 = config_row["timestamp"]

    eval_curve: list[EvalPoint] = []
    for r in rows:
        if r.get("event") in {"config", "reallocation", "final"}:
            continue
        if r.get("val_accuracy") is None:
            continue
        eval_curve.append(EvalPoint(
            step=int(r["step"]),
            wall_clock_s=float(r["timestamp"] - t0),
            val_loss=float(r["val_loss"]),
            val_accuracy=float(r["val_accuracy"]),
            examples_per_second=r.get("examples_per_second"),
            peak_memory_mb=float(r.get("peak_memory_mb") or 0.0),
        ))

    rank_dict = final_row.get("rank_dict")
    if rank_dict is None and realloc_row is not None:
        rank_dict = realloc_row.get("rank_dict")

    return RunRecord(
        run_id=path.stem,
        log_path=path,
        method=cfg["method"],
        seed=int(config_row["seed"]),
        alpha=(cfg.get("allocator") or {}).get("hardware_alpha"),
        final_val_loss=float(final_row["val_loss"]),
        final_val_accuracy=float(final_row["val_accuracy"]),
        steps_to_target=final_row.get("steps_to_target_accuracy"),
        wall_clock_to_target=final_row.get("wall_clock_to_target"),
        peak_memory_mb=float(final_row.get("peak_memory_mb") or 0.0),
        examples_per_second=float(final_row.get("examples_per_second") or 0.0),
        trainable_parameters=int(final_row.get("trainable_parameters") or 0),
        scheduler_overhead_seconds=float(
            final_row.get("scheduler_overhead_seconds") or 0.0
        ),
        rank_dict=rank_dict,
        total_wall_clock_s=float(rows[-1]["timestamp"] - t0),
        eval_curve=eval_curve,
        config=cfg,
    )


def _dedupe_runs(records: list[RunRecord]) -> list[RunRecord]:
    """For runs sharing ``(method, seed, alpha)``, keep only the
    lexicographically latest ``run_id`` and warn about the rest.

    Duplicates usually mean the same configuration was re-run (e.g. a
    pre-sweep verification + the sweep itself, both at seed=42). Including
    both biases the mean/std for that variant. ``run_id`` ends with a UTC
    timestamp so lexicographic sort = chronological sort.
    """
    by_key: dict[tuple[str, int, float | None], list[RunRecord]] = {}
    for rec in records:
        key = (rec.method, rec.seed, rec.alpha)
        by_key.setdefault(key, []).append(rec)

    out: list[RunRecord] = []
    for key, group in by_key.items():
        if len(group) == 1:
            out.append(group[0])
            continue
        group.sort(key=lambda r: r.run_id)
        kept = group[-1]
        out.append(kept)
        for stale in group[:-1]:
            warnings.warn(
                f"de-duplicating {stale.run_id}: keeping {kept.run_id} "
                f"(both have method={key[0]}, seed={key[1]}, alpha={key[2]})",
                RuntimeWarning, stacklevel=2,
            )
    return out


def load_run_records(logs_dir: Path) -> list[RunRecord]:
    """Walk ``logs_dir`` recursively, returning one record per readable run.

    Files without an ``event="config"`` row are skipped with a warning.
    Runs sharing ``(method, seed, alpha)`` are de-duped (latest kept) so
    pre-sweep verification runs don't poison the aggregation.
    """
    records: list[RunRecord] = []
    for path in sorted(Path(logs_dir).rglob("*.jsonl")):
        try:
            records.append(parse_run_jsonl(path))
        except ValueError as exc:
            warnings.warn(
                f"skipping {path}: {exc}", RuntimeWarning, stacklevel=2
            )
    return _dedupe_runs(records)


# --- variant grouping ---------------------------------------------------


def variant_key(rec: RunRecord) -> str:
    """Stable id for the (method, α) combo. Hardware-aware is split by α
    so the α-sweep ablation can isolate the cost-penalty axis without
    polluting the four-method primary tables."""
    if rec.method == "hardware_aware":
        # ``f"{1.0:g}"`` → "1"; ``f"{0.5:g}"`` → "0.5".
        return f"hardware_aware_alpha{rec.alpha:g}"
    return rec.method


# Order is the row order in the primary tables.
PRIMARY_VARIANTS: dict[str, str] = {
    "uniform": "Uniform LoRA",
    "adalora": "AdaLoRA",
    "gradient_adaptive": "Gradient-Adaptive LoRA (α=0.0)",
    "hardware_aware_alpha1": "Hardware-Aware LoRA (α=1.0)",
}

ALPHA_SWEEP_VARIANTS: dict[str, str] = {
    "gradient_adaptive": "α=0.0 (gradient-only)",
    "hardware_aware_alpha0.5": "α=0.5",
    "hardware_aware_alpha1": "α=1.0 (full hardware penalty)",
}


# --- aggregation --------------------------------------------------------


@dataclass
class VariantSummary:
    label: str
    seeds: int
    final_val_loss: list[float]
    final_val_accuracy: list[float]
    steps_to_target: list[int | None]
    wall_clock_to_target: list[float | None]
    peak_memory_mb: list[float]
    examples_per_second: list[float]
    trainable_parameters: list[int]
    scheduler_overhead_seconds: list[float]
    total_wall_clock_s: list[float]


def summarize(
    records: list[RunRecord], variants: dict[str, str]
) -> dict[str, VariantSummary]:
    out: dict[str, VariantSummary] = {}
    for vk, label in variants.items():
        runs = [r for r in records if variant_key(r) == vk]
        if not runs:
            continue
        out[vk] = VariantSummary(
            label=label,
            seeds=len(runs),
            final_val_loss=[r.final_val_loss for r in runs],
            final_val_accuracy=[r.final_val_accuracy for r in runs],
            steps_to_target=[r.steps_to_target for r in runs],
            wall_clock_to_target=[r.wall_clock_to_target for r in runs],
            peak_memory_mb=[r.peak_memory_mb for r in runs],
            examples_per_second=[r.examples_per_second for r in runs],
            trainable_parameters=[r.trainable_parameters for r in runs],
            scheduler_overhead_seconds=[
                r.scheduler_overhead_seconds for r in runs
            ],
            total_wall_clock_s=[r.total_wall_clock_s for r in runs],
        )
    return out


def _mean_std(values: list[float]) -> tuple[float, float]:
    vs = [
        v for v in values
        if v is not None and not (isinstance(v, float) and math.isnan(v))
    ]
    if not vs:
        return float("nan"), float("nan")
    if len(vs) == 1:
        return float(vs[0]), 0.0
    return statistics.fmean(vs), statistics.stdev(vs)


def _fmt_mean_std(values: list[float], digits: int = 3) -> str:
    m, s = _mean_std(values)
    if math.isnan(m):
        return "—"
    return f"{m:.{digits}f} ± {s:.{digits}f}"


def _fmt_partial(values: list[Any], digits: int = 1) -> str:
    """Format for fields where some seeds may be ``None`` (target never crossed).

    ``"1850 ± 120 (3/3)"`` if all crossed, ``"never (0/3)"`` if none did.
    """
    n = len(values)
    crossed = [float(v) for v in values if v is not None]
    if not crossed:
        return f"never (0/{n})"
    m, s = _mean_std(crossed)
    return f"{m:.{digits}f} ± {s:.{digits}f} ({len(crossed)}/{n})"


# --- table writers ------------------------------------------------------


def _csv_escape(s: str) -> str:
    if "," in s or '"' in s or "\n" in s:
        return '"' + s.replace('"', '""') + '"'
    return s


def write_table(rows: list[list[str]], out_csv: Path, out_md: Path) -> None:
    """Write the same table to ``out_csv`` and ``out_md``. ``rows[0]`` is the header."""
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(",".join(_csv_escape(c) for c in row) + "\n")
    with out_md.open("w", encoding="utf-8") as f:
        if not rows:
            return
        f.write("| " + " | ".join(rows[0]) + " |\n")
        f.write("|" + "|".join("---" for _ in rows[0]) + "|\n")
        for row in rows[1:]:
            f.write("| " + " | ".join(row) + " |\n")


def statistical_table(
    summaries: dict[str, VariantSummary]
) -> list[list[str]]:
    rows = [["Method", "Final Val Loss", "Final Val Accuracy",
             "Steps to Target"]]
    for vk in PRIMARY_VARIANTS:
        if vk not in summaries:
            continue
        s = summaries[vk]
        rows.append([
            s.label,
            _fmt_mean_std(s.final_val_loss),
            _fmt_mean_std(s.final_val_accuracy),
            _fmt_partial(s.steps_to_target, digits=0),
        ])
    return rows


def hardware_table(
    summaries: dict[str, VariantSummary]
) -> list[list[str]]:
    rows = [["Method", "Peak Memory MB", "Examples/sec",
             "Wall-Clock (s)", "Trainable Params"]]
    for vk in PRIMARY_VARIANTS:
        if vk not in summaries:
            continue
        s = summaries[vk]
        rows.append([
            s.label,
            _fmt_mean_std(s.peak_memory_mb, digits=1),
            _fmt_mean_std(s.examples_per_second, digits=1),
            _fmt_mean_std(s.total_wall_clock_s, digits=1),
            _fmt_mean_std([float(x) for x in s.trainable_parameters], digits=0),
        ])
    return rows


def systems_tradeoff_table(
    summaries: dict[str, VariantSummary]
) -> list[list[str]]:
    """Per README §1082-1088 this table is Uniform / AdaLoRA / Hardware-Aware
    only — gradient_adaptive is an ablation, not a method-vs-method baseline."""
    rows = [["Method", "Accuracy per MB", "Time to Target (s)",
             "Scheduler Overhead (s)"]]
    for vk in ("uniform", "adalora", "hardware_aware_alpha1"):
        if vk not in summaries:
            continue
        s = summaries[vk]
        # Per-seed ratio first, then mean ± std (avoids dividing means).
        acc_per_mb = [
            (a / m) if m and m > 0 else float("nan")
            for a, m in zip(s.final_val_accuracy, s.peak_memory_mb)
        ]
        rows.append([
            s.label,
            _fmt_mean_std(acc_per_mb, digits=5),
            _fmt_partial(s.wall_clock_to_target, digits=1),
            _fmt_mean_std(s.scheduler_overhead_seconds, digits=2),
        ])
    return rows


def _attention_rank_share(rec: RunRecord) -> float | None:
    """Fraction of total rank assigned to attention modules (q_lin / v_lin).

    Matches CLAUDE.md's post-Phase-5.6 expected pattern: hardware-aware
    skews toward attention (cheaper per parameter) → high share; gradient-
    only skews toward FFN → low share.
    """
    if rec.rank_dict is None:
        return None
    attn = sum(
        rk for k, rk in rec.rank_dict.items()
        if "q_lin" in k or "v_lin" in k
    )
    total = sum(rec.rank_dict.values())
    if total == 0:
        return None
    return attn / total


def alpha_sweep_table(records: list[RunRecord]) -> list[list[str]]:
    """``α | Final Val Accuracy | Time to Target | Attn Rank Share``."""
    summaries = summarize(records, ALPHA_SWEEP_VARIANTS)
    rows = [["α / Variant", "Final Val Accuracy", "Time to Target (s)",
             "Attn Rank Share"]]
    for vk, label in ALPHA_SWEEP_VARIANTS.items():
        if vk not in summaries:
            continue
        s = summaries[vk]
        runs = [r for r in records if variant_key(r) == vk]
        attn_shares = [
            _attention_rank_share(r) for r in runs
            if _attention_rank_share(r) is not None
        ]
        rows.append([
            label,
            _fmt_mean_std(s.final_val_accuracy),
            _fmt_partial(s.wall_clock_to_target, digits=1),
            _fmt_mean_std([a for a in attn_shares if a is not None], digits=3)
            if attn_shares else "—",
        ])
    return rows


# --- figures -----------------------------------------------------------


_PRIMARY_COLORS = {
    "uniform": "tab:blue",
    "adalora": "tab:orange",
    "gradient_adaptive": "tab:green",
    "hardware_aware_alpha1": "tab:red",
}


def figure_val_accuracy_vs_walltime(
    records: list[RunRecord], out_path: Path
) -> None:
    """README §1099 primary figure. One faded line per seed, one solid mean
    line per method. X-axis is wall-clock to make the systems claim visible."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    plotted_any = False
    for vk, label in PRIMARY_VARIANTS.items():
        runs = [r for r in records if variant_key(r) == vk]
        if not runs:
            continue
        color = _PRIMARY_COLORS.get(vk, "gray")
        for r in runs:
            xs = [p.wall_clock_s for p in r.eval_curve]
            ys = [p.val_accuracy for p in r.eval_curve]
            if xs:
                ax.plot(xs, ys, color=color, alpha=0.25, linewidth=0.8)
        # Mean curve aligned by step (eval_interval is identical across seeds).
        steps_set = sorted({p.step for r in runs for p in r.eval_curve})
        xs_mean: list[float] = []
        ys_mean: list[float] = []
        for st in steps_set:
            wallclocks: list[float] = []
            accs: list[float] = []
            for r in runs:
                for p in r.eval_curve:
                    if p.step == st:
                        wallclocks.append(p.wall_clock_s)
                        accs.append(p.val_accuracy)
            if wallclocks:
                xs_mean.append(statistics.fmean(wallclocks))
                ys_mean.append(statistics.fmean(accs))
        if xs_mean:
            ax.plot(xs_mean, ys_mean, color=color, linewidth=2.0, label=label)
            plotted_any = True

    ax.set_xlabel("Wall-clock (s)")
    ax.set_ylabel("Validation accuracy")
    ax.set_title("Validation accuracy vs wall-clock")
    if plotted_any:
        ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _short_module_name(fqname: str) -> str:
    """Strip the long PEFT prefix so heatmap x-tick labels stay readable.

    ``base_model.model.distilbert.transformer.layer.3.attention.q_lin``
    → ``L3.attn.q_lin``; ``...layer.5.ffn.lin2`` → ``L5.ffn.lin2``.
    """
    parts = fqname.split(".")
    layer = next(
        (parts[i + 1] for i in range(len(parts) - 1) if parts[i] == "layer"),
        "?",
    )
    leaf = parts[-1]
    section = "attn" if leaf in {"q_lin", "v_lin", "k_lin", "out_lin"} else "ffn"
    return f"L{layer}.{section}.{leaf}"


def figure_rank_allocation_heatmap(
    records: list[RunRecord], out_path: Path
) -> None:
    """Module × variant matrix of mean rank. Uniform's row is filled from
    cfg["lora"]["rank"] since uniform doesn't log a rank_dict; AdaLoRA's row
    is left blank — its 'effective rank' depends on PEFT's internal pruning
    schedule and is non-trivial to extract post-hoc (Phase 6 deferral)."""
    import matplotlib.pyplot as plt
    import numpy as np

    variants_to_plot = [
        "uniform",
        "gradient_adaptive",
        "hardware_aware_alpha0.5",
        "hardware_aware_alpha1",
    ]
    pretty = {
        "uniform": "Uniform",
        "gradient_adaptive": "Grad (α=0.0)",
        "hardware_aware_alpha0.5": "HW-Aware (α=0.5)",
        "hardware_aware_alpha1": "HW-Aware (α=1.0)",
    }

    all_modules: set[str] = set()
    for r in records:
        if r.rank_dict:
            all_modules.update(r.rank_dict.keys())
    if not all_modules:
        return  # nothing to plot
    modules = sorted(all_modules)
    matrix = np.full((len(variants_to_plot), len(modules)), np.nan)

    for i, vk in enumerate(variants_to_plot):
        runs = [r for r in records if variant_key(r) == vk]
        if not runs:
            continue
        if vk == "uniform":
            # No rank_dict logged; fill with the constant rank from cfg.
            uniform_rank = int(runs[0].config["lora"]["rank"])
            for j, _ in enumerate(modules):
                matrix[i, j] = float(uniform_rank)
            continue
        for j, mod in enumerate(modules):
            vals = [
                r.rank_dict[mod] for r in runs
                if r.rank_dict and mod in r.rank_dict
            ]
            if vals:
                matrix[i, j] = statistics.fmean(vals)

    fig, ax = plt.subplots(figsize=(max(8.0, 0.4 * len(modules)), 4.0))
    im = ax.imshow(matrix, aspect="auto", cmap="viridis")
    ax.set_yticks(range(len(variants_to_plot)))
    ax.set_yticklabels([pretty[v] for v in variants_to_plot])
    ax.set_xticks(range(len(modules)))
    ax.set_xticklabels(
        [_short_module_name(m) for m in modules], rotation=90, fontsize=7
    )
    ax.set_title("Rank allocation by module (mean over seeds)")
    fig.colorbar(im, ax=ax, label="rank")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def figure_metric_bars(
    records: list[RunRecord],
    metric_name: str,
    ylabel: str,
    out_path: Path,
) -> None:
    """Bar chart of one per-run scalar metric across the four primary
    methods, with per-seed scatter overlay. ``metric_name`` is a field on
    ``VariantSummary`` (e.g. ``"peak_memory_mb"``)."""
    import matplotlib.pyplot as plt
    import numpy as np

    summaries = summarize(records, PRIMARY_VARIANTS)
    if not summaries:
        return

    labels: list[str] = []
    means: list[float] = []
    stds: list[float] = []
    points: list[list[float]] = []
    colors: list[str] = []
    for vk in PRIMARY_VARIANTS:
        if vk not in summaries:
            continue
        s = summaries[vk]
        vs = getattr(s, metric_name)
        valid = [
            float(v) for v in vs
            if v is not None
            and not (isinstance(v, float) and math.isnan(v))
        ]
        m, sd = _mean_std(valid)
        labels.append(s.label)
        means.append(0.0 if math.isnan(m) else m)
        stds.append(0.0 if math.isnan(sd) else sd)
        points.append(valid)
        colors.append(_PRIMARY_COLORS.get(vk, "gray"))

    fig, ax = plt.subplots(figsize=(7, 5))
    xs = np.arange(len(labels))
    ax.bar(xs, means, yerr=stds, alpha=0.7, capsize=5, color=colors)
    for i, pts in enumerate(points):
        if pts:
            ax.scatter([i] * len(pts), pts,
                       color="black", alpha=0.6, s=20, zorder=3)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{ylabel} by method")
    ax.grid(alpha=0.3, axis="y")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# --- top-level driver ---------------------------------------------------


def write_all_tables(
    records: list[RunRecord], summaries_dir: Path
) -> None:
    primary = summarize(records, PRIMARY_VARIANTS)
    write_table(
        statistical_table(primary),
        summaries_dir / "statistical.csv",
        summaries_dir / "statistical.md",
    )
    write_table(
        hardware_table(primary),
        summaries_dir / "hardware.csv",
        summaries_dir / "hardware.md",
    )
    write_table(
        systems_tradeoff_table(primary),
        summaries_dir / "systems_tradeoff.csv",
        summaries_dir / "systems_tradeoff.md",
    )
    write_table(
        alpha_sweep_table(records),
        summaries_dir / "alpha_sweep.csv",
        summaries_dir / "alpha_sweep.md",
    )


def write_all_figures(
    records: list[RunRecord], figures_dir: Path
) -> None:
    figure_val_accuracy_vs_walltime(
        records, figures_dir / "val_accuracy_vs_walltime.png"
    )
    figure_rank_allocation_heatmap(
        records, figures_dir / "rank_allocation_heatmap.png"
    )
    figure_metric_bars(
        records, "peak_memory_mb", "Peak GPU memory (MB)",
        figures_dir / "peak_memory_bars.png",
    )
    figure_metric_bars(
        records, "examples_per_second", "Throughput (examples/sec)",
        figures_dir / "examples_per_second_bars.png",
    )
    figure_metric_bars(
        records, "scheduler_overhead_seconds", "Scheduler overhead (s)",
        figures_dir / "scheduler_overhead_bars.png",
    )


# --- CLI ---------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate JSONL training logs into the README's three result "
            "tables, the α-sweep table, and the project's figures."
        ),
    )
    parser.add_argument("--logs-dir", required=True, type=Path)
    parser.add_argument("--summaries-dir", required=True, type=Path)
    parser.add_argument("--figures-dir", required=True, type=Path)
    args = parser.parse_args(argv)

    records = load_run_records(args.logs_dir)
    if not records:
        print(f"no JSONL logs found under {args.logs_dir}")
        return 1
    print(f"loaded {len(records)} runs from {args.logs_dir}")
    counts: dict[str, int] = {}
    for r in records:
        counts[variant_key(r)] = counts.get(variant_key(r), 0) + 1
    for vk, n in sorted(counts.items()):
        print(f"  {vk}: {n} runs")

    write_all_tables(records, args.summaries_dir)
    write_all_figures(records, args.figures_dir)
    print(f"wrote summaries to {args.summaries_dir}")
    print(f"wrote figures to {args.figures_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
