#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
SEED="${1:-42}"
python -m src.train --config configs/uniform_lora.yaml --seed "$SEED"
