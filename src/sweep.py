"""Phase 6.5 — cross-platform multi-seed × multi-method sweep driver.

Python equivalent of ``experiments/run_sweep.sh`` for hosts without bash
(Windows cmd / PowerShell). Same behavior: shells out to
``python -m src.train`` once per (method, seed), then auto-aggregates
via ``python -m src.metrics``.

CLI::

    python -m src.sweep                                  # 5 methods × 3 seeds (default)
    python -m src.sweep --methods uniform adalora        # subset of methods
    python -m src.sweep --seeds 42                       # single seed
    python -m src.sweep --skip-aggregate                 # don't auto-call src.metrics

Returns non-zero exit code if any individual run failed; the sweep does
not abort early so a single transient failure doesn't waste the rest of
the queue. Failed (method, seed) pairs are listed at the end.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent

# Method label → config yaml path. Keep in sync with experiments/*.sh.
# These five are the Phase 6.8 production sweep set; the α=0.5 entry is the
# only one without a corresponding experiments/run_<method>.sh because the
# method label itself is non-canonical (it's still "hardware_aware" in the
# yaml — just with a different alpha).
METHOD_CONFIGS: dict[str, str] = {
    "uniform":                 "configs/uniform_lora.yaml",
    "adalora":                 "configs/adalora.yaml",
    "gradient_adaptive":       "configs/gradient_adaptive_lora.yaml",
    "hardware_aware":          "configs/hardware_aware_lora.yaml",
    "hardware_aware_alpha0_5": "configs/hardware_aware_lora_alpha0_5.yaml",
}

DEFAULT_METHODS = list(METHOD_CONFIGS.keys())
DEFAULT_SEEDS = [42, 43, 44]


def run_one(method: str, seed: int) -> int:
    cfg = METHOD_CONFIGS[method]
    print(f"\n===== {method} seed={seed} =====", flush=True)
    return subprocess.call(
        [sys.executable, "-m", "src.train",
         "--config", cfg, "--seed", str(seed)],
        cwd=REPO_ROOT,
    )


def aggregate() -> int:
    print("\n===== aggregating =====", flush=True)
    return subprocess.call(
        [sys.executable, "-m", "src.metrics",
         "--logs-dir", "results/raw_logs",
         "--summaries-dir", "results/summaries",
         "--figures-dir", "results/figures"],
        cwd=REPO_ROOT,
    )


def _validate_configs(methods: list[str]) -> list[str]:
    """Fail fast on a typo / missing config rather than 14 successful runs
    followed by one mystery failure at the end."""
    missing: list[str] = []
    for m in methods:
        cfg_path = REPO_ROOT / METHOD_CONFIGS[m]
        if not cfg_path.is_file():
            missing.append(f"{m}: {cfg_path}")
    return missing


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run a multi-seed × multi-method LoRA training sweep, then "
            "aggregate the resulting JSONL logs into the README's tables "
            "and figures. Cross-platform alternative to "
            "experiments/run_sweep.sh."
        ),
    )
    parser.add_argument(
        "--methods", nargs="+", default=DEFAULT_METHODS,
        choices=list(METHOD_CONFIGS.keys()),
        help="subset of methods to run (default: all 5)",
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=DEFAULT_SEEDS,
        help="seeds to run for each method (default: 42 43 44)",
    )
    parser.add_argument(
        "--skip-aggregate", action="store_true",
        help="don't auto-call src.metrics after the sweep finishes",
    )
    args = parser.parse_args(argv)

    missing = _validate_configs(args.methods)
    if missing:
        for line in missing:
            print(f"missing config: {line}", file=sys.stderr)
        return 2

    total = len(args.methods) * len(args.seeds)
    print(f"sweep: {len(args.methods)} methods × {len(args.seeds)} seeds "
          f"= {total} runs")

    start = time.time()
    failed: list[tuple[str, int, int]] = []
    for m in args.methods:
        for s in args.seeds:
            rc = run_one(m, s)
            if rc != 0:
                print(f"[!] {m} seed={s} exited {rc}", file=sys.stderr)
                failed.append((m, s, rc))
    elapsed = int(time.time() - start)
    ok = total - len(failed)
    print(f"\n===== sweep complete in {elapsed}s; {ok}/{total} runs OK =====")
    if failed:
        for m, s, rc in failed:
            print(f"  failed: {m} seed={s} (exit {rc})", file=sys.stderr)
        return 1

    if not args.skip_aggregate:
        return aggregate()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
