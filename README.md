# Hardware-Aware Adaptive LoRA Rank Allocation

A lightweight rank allocator for LoRA fine-tuning that picks per-module
ranks from the score

    s_i = g_i / c_i ** alpha

where `g_i` is an EMA of LoRA gradient Frobenius norms and `c_i` is the
per-rank parameter cost (`in_dim + out_dim`). `alpha = 0` collapses to
gradient-only allocation; `alpha = 1` is the full hardware-aware
variant. Compared against uniform LoRA and AdaLoRA on SST-2 with
DistilBERT under a fixed rank budget.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: source .venv/Scripts/activate
pip install -r requirements.txt
```

On Windows with NVIDIA, the default torch wheel from PyPI is CPU-only;
swap it after the install:

```bash
pip uninstall -y torch
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

## Running

A fast CPU smoke test (clamps data + steps to a few seconds):

```bash
python -m src.train --config configs/uniform_lora.yaml --seed 42 --smoke
```

One method at full settings:

```bash
python -m src.train --config configs/hardware_aware_lora.yaml --seed 42
```

The full sweep (5 method-configs x 3 seeds, then aggregate):

```bash
python -m src.sweep                  # cross-platform
bash experiments/run_sweep.sh        # Mac / Linux / Git Bash
make sweep                           # same thing
```

Re-aggregate existing JSONL logs without retraining:

```bash
python -m src.metrics --logs-dir results/raw_logs \
    --summaries-dir results/summaries --figures-dir results/figures
```

Tests:

```bash
python -m pytest src/tests -q
```

## Layout

```
configs/         per-method yamls (5)
src/             data, models, LoRA helpers, allocator, training, metrics
src/tests/       unit + integration tests
experiments/     per-method shell wrappers + sweep driver
notebooks/       Phase 1 demo notebook
demo_lora_grads.py   CPU-only LoRA enumeration + non-uniform attach demo
results/         raw_logs (per-run JSONL), summaries (tables), figures (PNGs)
```

Each training run writes one JSONL under `results/raw_logs/<method>/`
with an `event="config"` row at step 0, eval rows at the eval interval,
an `event="reallocation"` row for two-stage methods, and a terminal
`event="final"` row.
