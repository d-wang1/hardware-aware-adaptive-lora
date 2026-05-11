"""Multi-seed x multi-method sweep driver.

Shells out to `python -m src.train` once per (method, seed), then runs
`python -m src.metrics` to populate results/summaries and results/figures.
Cross-platform; on Windows use this instead of experiments/run_sweep.sh.

Examples:
    python -m src.sweep
    python -m src.sweep --methods uniform adalora
    python -m src.sweep --seeds 42 --skip-aggregate
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent

# method label -> config yaml. Names aren't uniform so map explicitly.
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
    # fail fast on a typo rather than 14 good runs followed by a mystery failure
    missing: list[str] = []
    for m in methods:
        cfg_path = REPO_ROOT / METHOD_CONFIGS[m]
        if not cfg_path.is_file():
            missing.append(f"{m}: {cfg_path}")
    return missing


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run a multi-seed x multi-method LoRA training sweep, then "
            "aggregate the JSONL logs into tables and figures."
        ),
    )
    parser.add_argument(
        "--methods", nargs="+", default=DEFAULT_METHODS,
        choices=list(METHOD_CONFIGS.keys()),
        help="subset of methods (default: all 5)",
    )
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=DEFAULT_SEEDS,
        help="seeds (default: 42 43 44)",
    )
    parser.add_argument(
        "--skip-aggregate", action="store_true",
        help="don't run src.metrics after the sweep",
    )
    args = parser.parse_args(argv)

    missing = _validate_configs(args.methods)
    if missing:
        for line in missing:
            print(f"missing config: {line}", file=sys.stderr)
        return 2

    total = len(args.methods) * len(args.seeds)
    print(f"sweep: {len(args.methods)} methods x {len(args.seeds)} seeds "
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
