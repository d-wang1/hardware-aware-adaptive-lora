#!/usr/bin/env bash
# Phase 6.5 — multi-seed × multi-method sweep driver.
#
# Default sweep: 5 method-configs × 3 seeds = 15 runs.
#   primary methods: uniform, adalora, gradient_adaptive (α=0.0),
#                    hardware_aware (α=1.0)
#   ablation:        hardware_aware_alpha0_5 (α=0.5)
#
# After all runs land, automatically calls ``python -m src.metrics`` to
# populate results/summaries and results/figures.
#
# Override via env vars (e.g. CI smoke or alpha-only):
#   SEEDS="42" METHODS="uniform" bash experiments/run_sweep.sh
#   METHODS="hardware_aware hardware_aware_alpha0_5" bash experiments/run_sweep.sh
set -euo pipefail
cd "$(dirname "$0")/.."

SEEDS="${SEEDS:-42 43 44}"
METHODS="${METHODS:-uniform adalora gradient_adaptive hardware_aware hardware_aware_alpha0_5}"

# Map method labels → existing per-config shell scripts. Naming is not
# uniform (run_adalora.sh has no _lora suffix; the α=0.5 ablation has its
# own script) so we map explicitly rather than string-substituting.
script_for_method() {
    case "$1" in
        uniform)                  echo "experiments/run_uniform_lora.sh" ;;
        adalora)                  echo "experiments/run_adalora.sh" ;;
        gradient_adaptive)        echo "experiments/run_gradient_adaptive_lora.sh" ;;
        hardware_aware)           echo "experiments/run_hardware_aware_lora.sh" ;;
        hardware_aware_alpha0_5)  echo "experiments/run_hardware_aware_lora_alpha0_5.sh" ;;
        *) echo "unknown method: $1" >&2; return 1 ;;
    esac
}

start=$(date +%s)
for m in $METHODS; do
    script=$(script_for_method "$m")
    for s in $SEEDS; do
        echo
        echo "===== $m seed=$s ====="
        bash "$script" "$s"
    done
done
end=$(date +%s)

echo
echo "===== sweep complete in $((end - start))s; aggregating ====="
python -m src.metrics \
    --logs-dir results/raw_logs \
    --summaries-dir results/summaries \
    --figures-dir results/figures
