# CLAUDE.md

Context for Claude Code sessions working in this repo. Loaded automatically — keep it tight, current, and pruned.

## Project

Hardware-Aware Adaptive LoRA Rank Allocation. A fixed total rank budget is distributed across LoRA modules using `s_i = g_i / c_i^α` (gradient-norm EMA over per-rank parameter cost). Compared against uniform LoRA, AdaLoRA, and a gradient-only ablation on SST-2 / DistilBERT. The contribution is the allocator + a systems-level evaluation (wall-clock, peak memory, throughput, scheduler overhead), **not** a claim of beating AdaLoRA on accuracy. Full spec in [README.md](README.md). Each Claude session approves its own implementation plan under `~/.claude/plans/<slug>.md` (per-machine; not in repo). Progress is recorded in the commit log — `git log --grep='^Phase'` is the source of truth.

## Current state

- **Phase 0** (bootstrap) ✅ — directory tree, configs for all four methods, run scripts, requirements, Makefile.
- **Phase 1.1** (`src/data.py`) ✅ — SST-2 loader, tokenizer, `set_seed`, dataloaders.
- **Phase 1.2** (`src/models.py`) ✅ — DistilBERT loader, target-module enumeration (12 `q_lin`/`v_lin` modules at 768×768), `module_dims` for the Phase 4b cost proxy, `count_parameters`.
- **Phase 2** (`src/hardware_logger.py`) ✅ — JSONL writer matching the README schema, throughput EMA (β=0.9, initialized at first instantaneous rate), peak memory via `torch.cuda.max_memory_allocated`, `scheduler_block()` context manager that accumulates `scheduler_overhead_seconds`. Extra fields like `event="reallocation"` and `rank_dict={...}` round-trip — Phase 5.3 relies on this.
- **Phase 3** (`src/evaluate.py`) ✅ — `evaluate()` returning `{val_loss, val_accuracy}`, plus `TargetAccuracyTracker` recording the first step + wall-clock time at which `val_accuracy >= target` (locks on first crossing for fair time-to-quality across methods).
- **Tests** under [src/tests/](src/tests/): `test_models.py`, `test_hardware_logger.py`, `test_evaluate.py` (~31 tests; all passing on the CUDA box, with one CPU-only peak-memory test correctly skipped).
- **Demo notebook**: [notebooks/demo_phase_1.ipynb](notebooks/demo_phase_1.ipynb) — exercises Phase 1.1 + 1.2 end-to-end with a real DistilBERT forward pass on an SST-2 batch.
- **Next: Phase 4a** — `src/lora_utils.py`: enumerate LoRA modules under PEFT (`peft.tuners.lora.Linear` walker), `lora_grad_norms` (Frobenius norm of A + B grads), `parameter_cost`, and `build_uniform_lora_model` / `build_non_uniform_lora_model` (the latter via `LoraConfig.rank_pattern`). Prerequisite for Phase 4b's `HardwareAwareRankAllocator`.

When updating this section: replace, don't append. The reader needs to know what's true *now*.

## How to work on this project

### Cadence
Break each phase into the smallest sensible work units (one file, one config, one function group). After each unit:
1. Stop and explain *exactly* what was implemented — file by file, function by function, with any non-obvious choice called out.
2. Suggest a concrete way to inspect or exercise the new code right now (`python -c` import, `pytest -k …`, file:line pointer, smoke run).
3. Wait for explicit "continue" / "next" before moving on.

This applies *within* phases as well as between them.

### Workflow
- **Solo project.** Commit and push directly to `main`. No PRs by default.
- **Phase milestones**: tag with `git tag phase-N-complete && git push --tags`.
- **Pushing is still a separate step** that needs the user's say-so — committing locally does not imply pushing.
- **Don't auto-stage untracked files.** `git add <specific-file>`, never `git add .`.

### Environments
One repo, multiple venvs (one per machine; `requirements.txt` stays cross-platform with no CUDA pins):

- **Mac dev venv** — `python3 -m venv .venv && pip install -r requirements.txt`. CPU-only Torch from PyPI. For editing, syntax checks, `pytest`, smoke runs on tiny subsets. Activate with `source .venv/bin/activate`.
- **Linux CUDA box** — same setup, CUDA Torch resolved automatically by pip on Linux. Activate with `source .venv/bin/activate`.
- **Windows CUDA box** (RTX 4070 Laptop, driver 581.x, CUDA 13 capable) — `python -m venv .venv` then activate via `source .venv/Scripts/activate` (Git Bash) or `.\.venv\Scripts\Activate.ps1` (PowerShell). The default PyPI torch wheel is **CPU-only on Windows**, so install the CUDA wheel explicitly after `pip install -r requirements.txt`:
  ```
  pip uninstall -y torch
  pip install torch --index-url https://download.pytorch.org/whl/cu124
  ```
  Verified working: `torch 2.6.0+cu124`, `torch.cuda.is_available() == True`. **Jupyter gotcha**: the kernel can segfault on `import torch` in a notebook (plain `python -c` works fine) due to duplicate OpenMP DLLs (`libiomp5md.dll`) when numpy / scipy were installed before the torch wheel was swapped. Fix is `pip install --force-reinstall --no-deps numpy scipy`. Quick diagnostic bypass (don't ship): `KMP_DUPLICATE_LIB_OK=TRUE` before `jupyter lab`.

### Hardware target
CUDA. Use `torch.cuda.max_memory_allocated()` as the peak-memory metric in `src/hardware_logger.py`. Don't write MPS/CPU fallback as the primary path. Smoke runs may execute on CPU; real experiments require CUDA.

## Repo conventions

- Configs are YAML in `configs/<method>_lora.yaml`. All four configs share `total_rank_budget: 96` — this is the **rank-budget invariant** for fair comparison and must not drift.
- Run scripts in `experiments/` are thin wrappers: `python -m src.train --config configs/X.yaml --seed $1`.
- Logs go to `results/raw_logs/<method>/<run_id>.jsonl` (append-only JSONL, schema in README §"hardware_logger.py").
- Commit messages use a descriptive title and a body explaining *why*. Phase work is prefixed `Phase N.M:`.
- LoRA target modules for DistilBERT: `q_lin`, `v_lin`. Document deviations.

## Things to not do

- Do not propose dynamic in-training rank reallocation before Phase 4's two-stage version is working — it's a Phase 7 stretch goal (README Risk 2).
- Do not reimplement LoRA / AdaLoRA. Use PEFT's `LoraConfig`, `AdaLoraConfig`, `get_peft_model`.
- Do not let the four methods diverge on optimizer, LR schedule, batch size, or max steps — fairness in Phase 5 depends on these being identical.
- Do not claim "our method beats AdaLoRA." Frame as "explores whether hardware-aware allocation improves practical efficiency under constrained budgets" (README §"Final Notes").

## Pointers

- Project spec: [README.md](README.md)
- Method configs: [configs/](configs/)
- Run scripts: [experiments/](experiments/)
- Source: [src/](src/) — `data.py` `models.py` `evaluate.py` `hardware_logger.py` (done); `lora_utils.py` `rank_allocator.py` `train.py` `metrics.py` (stubs).
- Tests: [src/tests/](src/tests/) — one file per implemented module. Run with `python -m pytest src/tests -q` from the repo root.
- Demo notebooks: [notebooks/](notebooks/) — `demo_phase_1.ipynb` shows the model loader and dataloader end-to-end. Re-run after each phase to watch progress (e.g. trainable param count drops sharply once Unit 4a.1 wraps the model with PEFT).
- Results land in: [results/raw_logs/](results/raw_logs/), [results/figures/](results/figures/), [results/summaries/](results/summaries/)
