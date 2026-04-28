# CLAUDE.md

Context for Claude Code sessions working in this repo. Loaded automatically — keep it tight, current, and pruned.

## Project

Hardware-Aware Adaptive LoRA Rank Allocation. A fixed total rank budget is distributed across LoRA modules using `s_i = g_i / c_i^α` (gradient-norm EMA over per-rank parameter cost). Compared against uniform LoRA, AdaLoRA, and a gradient-only ablation on SST-2 / DistilBERT. The contribution is the allocator + a systems-level evaluation (wall-clock, peak memory, throughput, scheduler overhead), **not** a claim of beating AdaLoRA on accuracy. Full spec in [README.md](README.md). Approved plan at `~/.claude/plans/read-the-readme-md-file-expressive-sparrow.md` (per-machine; not in repo).

## Current state

- **Phase 0** (bootstrap) ✅ — directory tree, configs for all four methods, run scripts, requirements, Makefile.
- **Phase 1.1** (`src/data.py`) ✅ — SST-2 loader, tokenizer, `set_seed`, dataloaders.
- **Next: Phase 1.2** — `src/models.py`: DistilBERT loader + LoRA target-module identification.

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
Two venvs, one repo:
- **Mac dev venv** (this laptop) — `python3 -m venv .venv && pip install -r requirements.txt`. CPU-only Torch from PyPI. For editing, syntax checks, `pytest`, smoke runs on tiny subsets.
- **CUDA box** (the Nvidia laptop) — same setup, Linux+CUDA Torch resolved automatically by pip. For Phase-5 experiment matrix and any time peak-memory / wall-clock numbers matter.

Both activate with `source .venv/bin/activate`. `requirements.txt` is cross-platform (no CUDA pins).

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
- Source: [src/](src/) — `data.py` (done), `models.py` `train.py` `evaluate.py` `lora_utils.py` `rank_allocator.py` `hardware_logger.py` `metrics.py` (stubs)
- Results land in: [results/raw_logs/](results/raw_logs/), [results/figures/](results/figures/), [results/summaries/](results/summaries/)
