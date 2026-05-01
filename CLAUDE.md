# CLAUDE.md

Context for Claude Code sessions in this repo. Loaded automatically — keep tight, current, pruned.

## Project

Hardware-Aware Adaptive LoRA Rank Allocation. Distributes a fixed rank budget across LoRA modules using `s_i = g_i / c_i^α` (gradient-norm EMA over per-rank parameter cost). Compared against uniform LoRA, AdaLoRA, and a gradient-only ablation on SST-2 / DistilBERT. Contribution = allocator + systems-level evaluation (wall-clock, peak memory, throughput, scheduler overhead), **not** beating AdaLoRA on accuracy. Full spec in [README.md](README.md). Per-session plan at `~/.claude/plans/<slug>.md` (per-machine, not in repo). Progress in commit log: `git log --grep='^Phase'`.

## Current state

- Phase 0 ✅ — scaffolding.
- Phase 1.1 ✅ — `src/data.py`: SST-2 loader, tokenizer, `set_seed`, dataloaders.
- Phase 1.2 ✅ — `src/models.py`: DistilBERT loader, `find_lora_target_module_names` (default keywords are q_lin/v_lin only — production targets are `q_lin, v_lin, lin1, lin2` for 24 mixed-cost modules), `module_dims` (Phase 4b cost proxy), `count_parameters`.
- Phase 2 ✅ — `src/hardware_logger.py`: JSONL writer, throughput EMA, peak memory via `torch.cuda.max_memory_allocated`, `scheduler_block()` ctx manager. Round-trips arbitrary fields (Phase 5.3 logs `event="reallocation"` + `rank_dict`).
- Phase 3 ✅ — `src/evaluate.py`: `evaluate()` → `{val_loss, val_accuracy}`; `TargetAccuracyTracker` locks first crossing of `val_accuracy >= target`.
- Phase 4a ✅ — `src/lora_utils.py`: `parameter_cost`, `enumerate_lora_modules`, `build_uniform_lora_model` (4a.1.A); `lora_grad_norms` (Frobenius A+B grads, 0.0 for None grads), `build_non_uniform_lora_model` (`LoraConfig.rank_pattern`) (4a.1.B).
- Phase 4b ✅ — `src/rank_allocator.py`: `HardwareAwareRankAllocator` with EMA gradient scores (seeded at first observation), `parameter_cost`-based costs, `s_i = g_i / c_i^α`, and budget-preserving allocation (floor at min_rank → proportional split → cap at max_rank → deterministic 1-rank rebalance to exact budget). One-shot `allocate(peft_model)` for the warmup→stage-2 hand-off.
- Tests in [src/tests/](src/tests/) — all passing on CUDA box (one CPU-only test correctly skipped).
- Demos: [notebooks/demo_phase_1.ipynb](notebooks/demo_phase_1.ipynb), `demo.py`, `demo_lora.py`, `demo_lora_grads.py` at repo root.
- Phase 5 ✅ — `src/train.py`: shared harness (`load_config`, `make_run_id`, `apply_smoke_overrides`, `build_optimizer_and_scheduler`, `train_loop`); `run_uniform`; `run_two_stage` for `hardware_aware` + `gradient_adaptive` (warmup with allocator hook → `scheduler_block`-charged reallocation → fresh-base-model stage 2 with `build_non_uniform_lora_model`); `run_adalora` via `peft.AdaLoraConfig` with `update_and_allocate` plumbed through `train_loop`'s `post_step_hook` so its per-step cost is charged to `scheduler_overhead_seconds` (AdaLoRA's per-step overhead is several× the two-stage one-shot cost on smoke; the gap will widen on real runs since AdaLoRA's hook fires every step and two-stage's only once at the warmup→stage-2 transition); `--smoke` flag works for all four methods.
- Phase 5.6 ✅ — extended LoRA targets to FFN (`q_lin`, `v_lin`, `lin1`, `lin2`); 24 modules at mixed cost (1536 for attention vs 3840 for FFN, 2.5× heterogeneity ratio) so the gradient-only ablation is no longer degenerate vs hardware-aware. Smoke confirms the two methods now produce strikingly different rank dicts: hardware-aware concentrates rank in attention (cheaper per parameter), gradient-only pulls toward FFN (larger gradient magnitudes). Same 192-rank budget produces different *parameter* budgets per method (a new memory-systems finding).
- Phase 6.6 ✅ — `configs/hardware_aware_lora_alpha0_5.yaml` (α=0.5 mid-point for the README §272-275 α-sweep). Same 24 targets, same 192 budget, only `allocator.hardware_alpha: 0.5` differs from `hardware_aware_lora.yaml`. Logs co-mingle with α=1.0 in `results/raw_logs/hardware_aware/`; the metrics reader disambiguates via the `event="config"` row's `cfg["allocator"]["hardware_alpha"]`.
- Phase 6.7 ✅ — `src/train.py` now writes a single `event="config"` row at step 0 in every run via `_log_run_config(logger, cfg)` (called inside the `with HardwareLogger(...)` block in all three dispatchers). The row carries the resolved cfg dict + seed so JSONL files are self-describing — Phase 6's aggregator no longer has to guess the source yaml from `run_id`. **Older logs without the config row are skipped by the reader with a warning** (`parse_run_jsonl` raises `ValueError`); re-run with current `src/train.py` if you have pre-Phase-6.7 logs.
- Phase 6 ✅ — `src/metrics.py`: full aggregator. `parse_run_jsonl` → `RunRecord` (one per run), `summarize` → `VariantSummary` (per `(method, α)` variant). Writes `results/summaries/{statistical,hardware,systems_tradeoff,alpha_sweep}.{csv,md}` and `results/figures/{val_accuracy_vs_walltime,rank_allocation_heatmap,peak_memory_bars,examples_per_second_bars,scheduler_overhead_bars}.png`. matplotlib is lazy-imported inside figure functions so the test suite is fast. Variant key convention: `hardware_aware` is split as `hardware_aware_alpha0.5` / `hardware_aware_alpha1` to keep the α-sweep ablation out of the four-method primary tables; the canonical hardware-aware row uses α=1.0. AdaLoRA has no `rank_dict` in its log so the rank heatmap leaves its row blank (post-pruning effective rank is non-trivial to extract from PEFT post-hoc; deferred). CLI: `python -m src.metrics --logs-dir results/raw_logs --summaries-dir results/summaries --figures-dir results/figures` or `make metrics`.
- Phase 6.5 ✅ — `experiments/run_sweep.sh` + `make sweep`: 5 method-configs × 3 seeds (default) = 15 runs in series, then auto-aggregate. Override via env: `SEEDS="42 43" METHODS="uniform adalora" make sweep`. Method→script mapping is explicit (case statement) because the per-config shell scripts have inconsistent naming (`run_adalora.sh` has no `_lora` suffix; the α=0.5 ablation has its own `run_hardware_aware_lora_alpha0_5.sh`). New `make hwaware05` target also added for single α=0.5 runs.
- **Next: Phase 6.8** — execute the real sweep (`make clean && make sweep` on the CUDA box; ~3-5 hours wall-clock). Acceptance: 15 JSONL files, populated `results/summaries/*.csv` and `results/figures/*.png`, hardware-aware row of the rank heatmap shows attention skew vs gradient-adaptive's FFN skew.
- **Then: Phase 6.9** — `REPORT.md` writeup; structure in plan file. Phase 7 (dynamic in-training reallocation, README Risk 2) is **out of scope** by user choice.

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
- Configs in `configs/<method>_lora.yaml`. All share `total_rank_budget: 192` — **invariant**, must not drift. (24 modules × uniform rank 8: 12 attention `q_lin`/`v_lin` at cost 1536 + 12 FFN `lin1`/`lin2` at cost 3840. The mixed-cost target list exists so the hardware-aware allocator's `s_i = g_i / c_i^α` is non-trivial; on attention-only targets every cost would be identical and α=0 vs α=1 would produce bit-identical allocations, making the gradient-only ablation degenerate.)
- Run scripts: `python -m src.train --config configs/X.yaml --seed $1`.
- Logs: `results/raw_logs/<method>/<run_id>.jsonl` (schema in README §"hardware_logger.py"). `run_id = <method>-seed<seed>-<utc-stamp>`; `HardwareLogger` opens append-mode, so multiple invocations with the same id concatenate (don't `cat *.jsonl` and treat as one run; `parse_run_jsonl` warns and uses the *last* `event="final"` row when this happens).
- **JSONL row order** (post-Phase-6.7, every run): (1) `event="config"` at step 0 carrying `config` (resolved cfg dict) + `seed`, (2) per-eval-interval rows, (3) [two-stage only] `event="reallocation"` carrying `rank_dict` + `gradient_scores`, (4) [two-stage only] stage-2 per-eval-interval rows, (5) `event="final"` summary (re-carries `rank_dict` for two-stage). The aggregator reads `rank_dict` from either the reallocation or final row.
- **α-sweep variant disambiguation**: `hardware_aware` runs at α=0.5 and α=1.0 share `results/raw_logs/hardware_aware/`. The metrics reader splits them via `cfg["allocator"]["hardware_alpha"]` from the config row → variant keys `hardware_aware_alpha0.5` and `hardware_aware_alpha1`. Don't try to disambiguate via filename — `run_id` only encodes method + seed + stamp.
- Commits: descriptive title + *why* in body. Phase work prefixed `Phase N.M:`.
- LoRA targets for DistilBERT: `q_lin`, `v_lin`, `lin1`, `lin2` — **all four** in production. The historic q_lin/v_lin-only setup survives in test fixtures (`distilbert_uniform`, `distilbert_warmup_peft`) which test the builders abstractly.

## Things to not do
- No dynamic in-training rank reallocation. Phase 7 (README Risk 2) is **explicitly out of scope** for this project — user-confirmed in the Phase 6 plan. Don't start.
- Don't reimplement LoRA/AdaLoRA. Use PEFT (`LoraConfig`, `AdaLoraConfig`, `get_peft_model`).
- Don't let the four methods diverge on optimizer / LR / batch size / max steps — `build_optimizer_and_scheduler` + `train_loop` in `src/train.py` are single-source by design; Phase 5 fairness depends on it.
- Don't "optimize" the stage-2 base-model reload in `run_two_stage` — it's intentional per README §"Two-Stage Version" (warmup LoRA weights are *discarded*; stage 2 starts from fresh base + non-uniform LoRA).
- Don't add real-DistilBERT integration tests to the unit suite — the convention (set in `test_lora_utils` + `test_train` + `test_metrics`) is fast unit tests + out-of-band verification via `python -m src.train ... --smoke`.
- Don't claim beating AdaLoRA. Frame as "explores whether hardware-aware allocation improves practical efficiency under constrained budgets" (README §"Final Notes").
- Don't remove `_log_run_config` or stop emitting `event="config"`. The metrics reader treats it as required and skips runs without it. If you need to migrate the schema, also bump the parser to accept old logs explicitly.
- Don't add `gradient_adaptive` to the Systems Tradeoff table — README §1082-1088 limits that table to Uniform / AdaLoRA / Hardware-Aware (3 rows). It's an ablation, not a method-vs-method baseline.

## Pointers
- Spec: [README.md](README.md). Plan: `~/.claude/plans/<slug>.md`.
- Code: [src/](src/), [configs/](configs/), [experiments/](experiments/).
- Tests: [src/tests/](src/tests/) — `python -m pytest src/tests -q` from repo root.
- Demos: [notebooks/](notebooks/), `demo.py`, `demo_lora.py` (paste-ready files preferred).
- Results: [results/raw_logs/](results/raw_logs/), [results/figures/](results/figures/), [results/summaries/](results/summaries/).
