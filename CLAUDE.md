# CLAUDE.md

Context for Claude Code sessions in this repo. Loaded automatically — keep tight, current, pruned.

## Project

Hardware-Aware Adaptive LoRA Rank Allocation. Distributes a fixed rank budget across LoRA modules using `s_i = g_i / c_i^α` (gradient-norm EMA over per-rank parameter cost). Compared against uniform LoRA, AdaLoRA, and a gradient-only ablation on SST-2 / DistilBERT. Contribution = allocator + systems-level evaluation (wall-clock, peak memory, throughput, scheduler overhead), **not** beating AdaLoRA on accuracy. Full spec in [README.md](README.md). Per-session plan at `~/.claude/plans/<slug>.md` (per-machine, not in repo). Progress in commit log: `git log --grep='^Phase'`.

## Current state

- Phase 0 ✅ — scaffolding.
- Phase 1.1 ✅ — `src/data.py`: SST-2 loader, tokenizer, `set_seed`, dataloaders.
- Phase 1.2 ✅ — `src/models.py`: DistilBERT loader, `find_lora_target_module_names` (12 q/v_lin @ 768×768), `module_dims` (Phase 4b cost proxy), `count_parameters`.
- Phase 2 ✅ — `src/hardware_logger.py`: JSONL writer, throughput EMA, peak memory via `torch.cuda.max_memory_allocated`, `scheduler_block()` ctx manager. Round-trips arbitrary fields (Phase 5.3 logs `event="reallocation"` + `rank_dict`).
- Phase 3 ✅ — `src/evaluate.py`: `evaluate()` → `{val_loss, val_accuracy}`; `TargetAccuracyTracker` locks first crossing of `val_accuracy >= target`.
- Phase 4a ✅ — `src/lora_utils.py`: `parameter_cost`, `enumerate_lora_modules`, `build_uniform_lora_model` (4a.1.A); `lora_grad_norms` (Frobenius A+B grads, 0.0 for None grads), `build_non_uniform_lora_model` (`LoraConfig.rank_pattern`) (4a.1.B).
- Phase 4b ✅ — `src/rank_allocator.py`: `HardwareAwareRankAllocator` with EMA gradient scores (seeded at first observation), `parameter_cost`-based costs, `s_i = g_i / c_i^α`, and budget-preserving allocation (floor at min_rank → proportional split → cap at max_rank → deterministic 1-rank rebalance to exact budget). One-shot `allocate(peft_model)` for the warmup→stage-2 hand-off.
- Tests in [src/tests/](src/tests/) — all passing on CUDA box (one CPU-only test correctly skipped).
- Demos: [notebooks/demo_phase_1.ipynb](notebooks/demo_phase_1.ipynb), `demo.py`, `demo_lora.py`, `demo_lora_grads.py` at repo root.
- Phase 5 ✅ — `src/train.py`: shared harness (`load_config`, `make_run_id`, `apply_smoke_overrides`, `build_optimizer_and_scheduler`, `train_loop`); `run_uniform`; `run_two_stage` for `hardware_aware` + `gradient_adaptive` (warmup with allocator hook → `scheduler_block`-charged reallocation → fresh-base-model stage 2 with `build_non_uniform_lora_model`); `run_adalora` via `peft.AdaLoraConfig` with `update_and_allocate` plumbed through `train_loop`'s `post_step_hook` so its per-step cost is charged to `scheduler_overhead_seconds` (AdaLoRA's overhead is ~360× the two-stage methods' on the smoke baseline — the project's headline systems-level finding); `--smoke` flag works for all four methods.
- **Next: Phase 6** — `src/metrics.py`: aggregate `results/raw_logs/**/*.jsonl` into the README's three tables (Statistical / Hardware / Systems Tradeoff) and figures (`val_accuracy_vs_walltime.png`, rank allocation heatmap, peak-memory bars).

When updating: replace, don't append.

## How to work on this project

### Cadence
Smallest sensible work units (one file / function group). After each: stop, explain file-by-file with non-obvious choices called out, suggest a verify command, wait for explicit "continue". User prefers paste-ready `demo_X.py` files at repo root over multi-line `python -c "..."` blocks (shell escaping breaks on copy-paste).

### Workflow
Solo project: commit + push directly to `main`, no PRs. Tag phase milestones (`git tag phase-N-complete && git push --tags`). Pushing requires user say-so. Use `git add <file>`, never `git add .`.

### Environments
One repo, multiple venvs. `requirements.txt` is cross-platform; upper bounds pin **numpy<2 + pyarrow<15 + datasets<3** to avoid Windows DLL traps (see below).

- **Mac dev venv** — `python3 -m venv .venv && pip install -r requirements.txt`. CPU-only torch from PyPI.
- **Linux CUDA box** — same; CUDA torch resolves automatically.
- **Windows CUDA box** (RTX 4070 Laptop, driver 581.x) — `python -m venv .venv` then `source .venv/Scripts/activate` (Git Bash). Default PyPI torch is CPU-only on Windows; swap after install:
  ```
  pip uninstall -y torch
  pip install torch --index-url https://download.pytorch.org/whl/cu124
  ```
  Verified: `torch 2.6.0+cu124`.

#### Windows gotchas
- **The pyarrow triangle**: pyarrow≥15 silently segfaults (exit 139) when torch is imported first (DLL conflict; `KMP_DUPLICATE_LIB_OK` does *not* help). pyarrow<15 breaks numpy 2.x ABI (`_ARRAY_API not found`). datasets≥3 calls `pa.json_()` missing in pyarrow<15. Hence the triple pin (`numpy<2 + pyarrow<15 + datasets<3`). Diagnose with `pip show numpy pyarrow datasets`.
  - **Recovery — always use `--no-deps`** when reinstalling any one of the three; `pip install 'pyarrow<15' --force-reinstall` *without* `--no-deps` will pull in numpy 2.x as a build dep and break the other side. Working sequence: `pip install 'pyarrow<15' --force-reinstall --no-deps && pip install 'numpy<2' --force-reinstall --no-deps`.
- **Jupyter `import torch` segfault** (duplicate OpenMP DLLs): `pip install --force-reinstall --no-deps numpy scipy`. Bypass-only: `KMP_DUPLICATE_LIB_OK=TRUE`.
- **Standalone scripts** with DataLoader workers must wrap entry in `if __name__ == "__main__":` — Windows re-imports per worker, otherwise the script silently hangs/exits. `--smoke` in `train.py` will use `num_workers=0`.

### Hardware target
CUDA. Peak-memory metric is `torch.cuda.max_memory_allocated()`. No MPS/CPU fallback as primary path. Smoke runs may run CPU; real experiments require CUDA.

### What `--smoke` is for
Plumbing-only sanity: 5 total steps, 64/32 train/val samples, `num_workers=0`, `eval_interval=5`. Per-method overrides: two-stage methods get `warmup_steps=2`; AdaLoRA gets `tinit=1, tfinal=1, deltaT=1` (PEFT requires `tinit + tfinal < total_step` for a non-empty budgeting phase). Verifies the run produces a JSONL with the right schema. **Not** data-meaningful — gradient EMAs from 2 warmup steps are noise, val_accuracy hovers around chance, time-to-target is always `null`. Real experiments need the full configured `warmup_steps: 200` + `epochs: 3`.

## Repo conventions
- Configs in `configs/<method>_lora.yaml`. All share `total_rank_budget: 96` — **invariant**, must not drift.
- Run scripts: `python -m src.train --config configs/X.yaml --seed $1`.
- Logs: `results/raw_logs/<method>/<run_id>.jsonl` (schema in README §"hardware_logger.py"). `run_id = <method>-seed<seed>-<utc-stamp>`; `HardwareLogger` opens append-mode, so multiple invocations with the same id concatenate (don't `cat *.jsonl` and treat as one run).
- **Two-stage JSONL row order** (`hardware_aware`, `gradient_adaptive`): (1) eval at warmup endpoint, (2) `event="reallocation"` carrying `rank_dict` + `gradient_scores`, (3) per-eval-interval rows in stage 2, (4) `event="final"` summary (also re-carries `rank_dict` for convenience). Phase 6 aggregation can read `rank_dict` from either the reallocation row or the final row.
- Commits: descriptive title + *why* in body. Phase work prefixed `Phase N.M:`.
- LoRA targets for DistilBERT: `q_lin`, `v_lin`.

## Things to not do
- No dynamic in-training rank reallocation before Phase 4's two-stage version works (Phase 7 stretch; README Risk 2).
- Don't reimplement LoRA/AdaLoRA. Use PEFT (`LoraConfig`, `AdaLoraConfig`, `get_peft_model`).
- Don't let the four methods diverge on optimizer / LR / batch size / max steps — `build_optimizer_and_scheduler` + `train_loop` in `src/train.py` are single-source by design; Phase 5 fairness depends on it.
- Don't "optimize" the stage-2 base-model reload in `run_two_stage` — it's intentional per README §"Two-Stage Version" (warmup LoRA weights are *discarded*; stage 2 starts from fresh base + non-uniform LoRA).
- Don't add real-DistilBERT integration tests to the unit suite — the convention (set in `test_lora_utils` + `test_train`) is fast unit tests + out-of-band verification via `python -m src.train ... --smoke`.
- Don't claim beating AdaLoRA. Frame as "explores whether hardware-aware allocation improves practical efficiency under constrained budgets" (README §"Final Notes").

## Pointers
- Spec: [README.md](README.md). Plan: `~/.claude/plans/<slug>.md`.
- Code: [src/](src/), [configs/](configs/), [experiments/](experiments/).
- Tests: [src/tests/](src/tests/) — `python -m pytest src/tests -q` from repo root.
- Demos: [notebooks/](notebooks/), `demo.py`, `demo_lora.py` (paste-ready files preferred).
- Results: [results/raw_logs/](results/raw_logs/), [results/figures/](results/figures/), [results/summaries/](results/summaries/).
