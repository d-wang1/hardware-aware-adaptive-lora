#!/usr/bin/env bash
# Multi-seed x multi-method sweep driver.
# Default: 5 method-configs x 3 seeds = 15 runs, then aggregate.
#
# Override via env vars:
#   SEEDS="42" METHODS="uniform" bash experiments/run_sweep.sh
#   METHODS="hardware_aware hardware_aware_alpha0_5" bash experiments/run_sweep.sh
set -euo pipefail
cd "$(dirname "$0")/.."

SEEDS="${SEEDS:-42 43 44}"
METHODS="${METHODS:-uniform adalora gradient_adaptive hardware_aware hardware_aware_alpha0_5}"

# script names aren't uniform, so map explicitly
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
