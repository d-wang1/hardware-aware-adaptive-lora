# REPORT.md — outline

Working outline for the Phase 6.9 writeup. Resolved before drafting prose so
the structure, narrative beats, and numerical anchors are agreed first.

Target length: ≤1500 words of prose; tables and figures carry the load.
Framing rules: README §1267-1293 (no "beats AdaLoRA" anywhere). Numbers
must come verbatim from `results/summaries/*.md` — no rounding drift.

---

## 1. Problem & motivation (~150 words)

**Purpose:** set up the rank-budget framing in one paragraph and end on the
exact README L1280 sentence.

**Beats:**
- LoRA reduces trainable params, but a fixed rank budget is itself a resource
  that has to be allocated across modules.
- Two existing answers: Uniform (split evenly) and AdaLoRA (reallocate by
  importance via SVD pruning).
- Gap: neither accounts for the *hardware cost per rank* — different modules
  produce different numbers of trainable parameters (and activation
  footprint) per unit of rank.
- This work asks whether mixing gradient signal with hardware cost in the
  allocator changes the practical-efficiency picture.

**Closes with verbatim README L1280:** "Our method explores whether
hardware-aware rank allocation can improve practical efficiency compared
with AdaLoRA under constrained training budgets."

---

## 2. Method (~250 words)

### 2.1 The allocator rule
- `s_i = g_i / c_i^α` per LoRA-target module `i`.
- `g_i`: EMA of Frobenius gradient norms over the `(A_i, B_i)` LoRA pair
  during a warmup phase.
- `c_i`: `parameter_cost` — number of trainable parameters added per unit
  of rank at module `i` (= `in_features + out_features`).
- α ∈ {0.0, 0.5, 1.0}: cost penalty exponent. α=0 reduces to gradient-only
  (the ablation method, "Gradient-Adaptive"); α=1 is the headline
  Hardware-Aware variant.
- Budget-preserving allocation: floor at `min_rank` → proportional split
  → cap at `max_rank` → deterministic 1-rank rebalance to exact budget.

### 2.2 Two-stage training
- Stage 1: warmup with uniform-rank LoRA for 200 steps, accumulating
  gradient EMAs.
- Reallocation: one-shot, charged to `scheduler_overhead_seconds` via the
  logger's `scheduler_block` context manager.
- Stage 2: discard warmup LoRA weights, rebuild from a fresh base model
  with `rank_pattern` set from the allocator output, train to completion.
- Why discard stage-1 weights: README §"Two-Stage Version"; warmup is for
  *signal*, not parameters.

### 2.3 Out of scope
- Dynamic in-training reallocation (Phase 7) is explicitly deferred. One
  sentence, no apologetics.

---

## 3. Experimental setup (~200 words)

**Beats:**
- Task: SST-2 binary sentiment, 67k train / 872 val.
- Model: DistilBERT-base-uncased, 6 transformer layers.
- LoRA targets: `q_lin`, `v_lin`, `lin1`, `lin2` — 24 modules total
  (12 attention at cost 1536, 12 FFN at cost 3840). The 2.5× cost
  heterogeneity is what makes the gradient-only ablation non-degenerate.
- Rank budget: **192 — invariant across all four methods**. (24 modules ×
  uniform rank 8.)
- Optimizer / LR / batch size / max steps: identical across methods,
  enforced by the shared `train_loop` in [src/train.py](src/train.py).
- Methods: Uniform LoRA, AdaLoRA, Gradient-Adaptive (α=0.0),
  Hardware-Aware (α=1.0). Plus α=0.5 for the ablation table.
- Seeds: 42, 43, 44.
- Hardware: single RTX 4070 Laptop, CUDA 12.4, torch 2.6.0+cu124.
- Reproducibility: `python -m src.sweep` runs the full 15-run sweep
  (~4.4h) and re-aggregates. Footnote: `_dedupe_runs` keeps the latest
  `run_id` per `(method, seed, α)`; 4 stale pre-Phase-6.7 logs in
  `results/raw_logs/` are skipped (they don't affect any reported number).

---

## 4. Results (~600 words — the meat)

**Structure:** four subsections, each leading with a table or figure then
walking through what's interesting in 2-3 sentences. Honest mixed-result
framing throughout.

### 4.1 Statistical performance
**Embed table:** `results/summaries/statistical.md` verbatim.

**Embed figure:** `results/figures/val_accuracy_vs_walltime.png`.

**Beats to make:**
- All three rank-budget-respecting methods land in a tight 0.902–0.906
  accuracy band; the four methods are essentially tied on accuracy.
- Hardware-Aware has the lowest seed-to-seed variance (±0.001 vs ±0.005
  for Gradient-Adaptive and ±0.006 for Uniform) — worth one sentence.
- AdaLoRA gets the lowest **loss** (0.268) but never crosses the 0.90
  accuracy target in 3/3 seeds, while the other three methods cross it
  every time. State this as observed; don't editorialize.

### 4.2 Hardware performance
**Embed table:** `results/summaries/hardware.md` verbatim.

**Embed figures:** `results/figures/peak_memory_bars.png`,
`results/figures/examples_per_second_bars.png`.

**Beats:**
- *Trainable parameters*: Hardware-Aware uses **1,081,346** — fewest of all
  four, ~2.5% under Uniform (1,108,226), ~21% under AdaLoRA (1,366,562),
  at the same 192-rank budget. Direct evidence the cost term is doing what
  it's designed to do (concentrate rank on cheaper modules).
- *Throughput*: Uniform / Gradient-Adaptive / Hardware-Aware all ~920 ex/s;
  AdaLoRA at 602 ex/s, ~35% slower. The gap reflects AdaLoRA's per-step
  `update_and_allocate` work.
- *Peak memory* — be honest: Uniform 949 MB, AdaLoRA 964 MB, but
  Hardware-Aware 1208 MB and Gradient-Adaptive 1211 MB. Both two-stage
  methods carry a ~250 MB peak-memory penalty despite using fewer
  parameters than AdaLoRA. Plausible cause: rank concentrating on
  attention grows those A/B matrices and `max_memory_allocated` reflects
  activation+optimizer-state peaks during training, not parameter count.
  One sentence — surface it, don't over-explain.

### 4.3 Systems tradeoff
**Embed table:** `results/summaries/systems_tradeoff.md` verbatim
(3 rows: Uniform / AdaLoRA / Hardware-Aware, per README §1082-1088).

**Embed figure:** `results/figures/scheduler_overhead_bars.png`.

**Beats — the systems-flavor finding the project is built on:**
- Scheduler overhead: Uniform 0.0s, Hardware-Aware 4.0s, AdaLoRA 76.7s.
  The two-stage one-shot reallocation is ~19× cheaper than AdaLoRA's
  per-step approach because it fires once at the warmup→stage-2 transition,
  not every training step.
- Time-to-target: Uniform 225s (3/3), Hardware-Aware 294s (3/3), AdaLoRA
  never (0/3). Mixed result vs Uniform — Uniform is genuinely faster
  here.
- Accuracy-per-MB: Uniform 0.00095 wins, driven entirely by its lower peak
  memory. State as data.

### 4.4 α-sweep ablation
**Embed table:** `results/summaries/alpha_sweep.md` verbatim.

**Embed figure:** `results/figures/rank_allocation_heatmap.png`.

**Beats — the cleanest single-method evidence:**
- Attention rank share moves monotonically with α: **39.1% (α=0) → 47.6%
  (α=0.5) → 56.1% (α=1)**. The cost term in `s_i = g_i / c_i^α` is
  isolated from gradient signal here — α is the only thing that changes —
  so this is direct evidence the denominator is doing real work.
- Final accuracy stays flat (0.904 / 0.906 / 0.906) across α. The systems
  reshaping is essentially "free" on accuracy for this task.
- Heatmap visually confirms attention skew (cross-check spot-check from
  the plan).

---

## 5. Limitations (~150 words, bulleted)

- **Single task, single model**: SST-2 / DistilBERT only. Generalization
  to GLUE-large or to RoBERTa/Llama-class models is not validated.
- **Two-stage one-shot reallocation only**: the dynamic in-training
  variant (Phase 7) is deferred; we don't know whether continuous
  reallocation would close the time-to-target gap with Uniform.
- **3 seeds**: too few for paired significance testing. Mean ± std is
  reported and the reader is left to judge; no t-tests.
- **Mixed practical-efficiency picture vs Uniform**: Hardware-Aware uses
  fewer parameters and has lower seed variance, but Uniform is faster to
  target and lower peak memory. The headline systems wins are vs AdaLoRA,
  not vs Uniform.
- **Peak memory not modeled in the cost function**: `c_i =
  in_features + out_features` captures parameter cost per rank, not
  activation memory — likely why Hardware-Aware's peak memory exceeds
  Uniform's despite fewer trainable params.

---

## 6. Closing framing (~100 words)

**Beats:**
- One sentence summary: the cost term reshapes allocations meaningfully
  (clean α-sweep monotonicity, fewest trainable params), produces
  competitive accuracy, and substantially reduces scheduler overhead vs
  AdaLoRA, but does not deliver an across-the-board practical-efficiency
  win over Uniform LoRA on this task.
- Verbatim README L1280: "Our method explores whether hardware-aware rank
  allocation can improve practical efficiency compared with AdaLoRA under
  constrained training budgets."
- The course-fit framing list verbatim from README L1287-1292: parameter-
  efficient fine-tuning, adaptive optimization, memory-aware training,
  hardware-performance measurement, empirical systems evaluation.

---

## Word-budget rollup

| Section | Target words |
|---|---|
| 1. Problem & motivation | ~150 |
| 2. Method | ~250 |
| 3. Experimental setup | ~200 |
| 4. Results (4 subsections) | ~600 |
| 5. Limitations | ~150 |
| 6. Closing framing | ~100 |
| **Total prose** | **~1450** |

Tables (4) and figures (5) on top of this. All prose word counts are soft
caps — better to be tighter than longer.

---

## Open questions before drafting

1. Embed figures inline at section breaks, or collect them into a
   "Figures" appendix at the end? Default plan: inline at section breaks.
2. Cite specific code paths (e.g. `[src/rank_allocator.py:42](...)`) in
   the Method section, or keep prose clean and link only the file? Default
   plan: link the file, no line numbers (the code is the contract).
3. AdaLoRA's never-crosses-0.90 result — is this a genuine "AdaLoRA is
   tuned for loss, not accuracy" finding, or a quirk of the 3-seed
   sample? We have no data to disambiguate; default plan is to report it
   factually with no causal claim.
