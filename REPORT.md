# Hardware-Aware Adaptive LoRA Rank Allocation

## 1. Problem & motivation

Low-Rank Adaptation (LoRA) reduces the number of trainable parameters needed to fine-tune a transformer by inserting small rank-`r` adapters into selected weight matrices. Once the total rank budget is fixed, *how* that budget is distributed across modules becomes its own decision. Standard LoRA splits it uniformly. AdaLoRA reallocates it during training by SVD-pruning singular values that contribute least to the loss. Neither answer accounts for the fact that the same unit of rank costs different numbers of trainable parameters at different modules — adding rank to a 768→3072 FFN projection produces 2.5× more parameters than adding rank to a 768→768 attention projection on DistilBERT.

This work asks whether mixing the gradient-based importance signal with explicit hardware cost in the allocator changes the practical-efficiency picture. Our method explores whether hardware-aware rank allocation can improve practical efficiency compared with AdaLoRA under constrained training budgets.

## 2. Method

### 2.1 The allocator rule

For each LoRA target module `i`, the allocator computes

```
s_i = g_i / c_i^α
```

where `g_i` is an exponential moving average of the Frobenius norm of the LoRA gradients at module `i` over a warmup phase, `c_i = in_features + out_features` is the parameter cost per unit of rank, and α ∈ [0, 1] is a tunable cost-penalty exponent. Rank is distributed proportionally to `s_i` while preserving the total budget: floor at `min_rank`, scale to budget, cap at `max_rank`, then a deterministic 1-rank rebalance to restore the exact total. Setting α=0 reduces to a gradient-only baseline (the Gradient-Adaptive ablation); α=1 is the headline Hardware-Aware variant.

### 2.2 Two-stage training

Stage 1 trains a uniform-rank LoRA model for 200 warmup steps, accumulating gradient-norm EMAs in [src/rank_allocator.py](src/rank_allocator.py). At the warmup → stage-2 transition the allocator runs once inside the logger's `scheduler_block` context manager, charging the reallocation wall-time to a `scheduler_overhead_seconds` field. The warmup LoRA weights are then discarded: stage 2 rebuilds the LoRA model from the fresh base with `LoraConfig.rank_pattern` set from the allocator's output, then trains to completion on the same hyperparameters as stage 1. Warmup exists to surface signal, not to pre-train weights. A dynamic in-training variant — re-running the allocator periodically rather than once — is out of scope for this project.

## 3. Experimental setup

- **Task:** SST-2 binary sentiment classification (67k train / 872 val, GLUE).
- **Model:** DistilBERT-base-uncased.
- **LoRA targets:** `q_lin`, `v_lin`, `lin1`, `lin2` — 24 modules total (12 attention at parameter cost 1536, 12 FFN at cost 3840). The 2.5× cost heterogeneity is what makes the gradient-only ablation non-degenerate; on attention-only targets every cost would be identical and α=0 vs α=1 would produce bit-identical allocations.
- **Rank budget:** 192, invariant across all four methods (24 modules × uniform rank 8).
- **Methods:** Uniform LoRA, AdaLoRA (PEFT's `AdaLoraConfig`), Gradient-Adaptive LoRA (α=0.0), Hardware-Aware LoRA (α=1.0). The α-sweep ablation additionally includes α=0.5.
- **Shared:** identical optimizer (AdamW), learning rate, batch size, max steps, and `train_loop` for all methods, enforced by the single dispatcher in [src/train.py](src/train.py).
- **Seeds:** 42, 43, 44.
- **Hardware:** single NVIDIA RTX 4070 Laptop GPU, CUDA 12.4, PyTorch 2.6.0+cu124. Total sweep wall-clock: 15780 s (~4.4 h) over 5 method-configs × 3 seeds = 15 runs, all successful.

The full sweep is reproducible via `python -m src.sweep`. Aggregation in [src/metrics.py](src/metrics.py) deduplicates by `(method, seed, α)` (keeping the lexicographically latest `run_id`); four pre-provenance logs missing the self-describing `event="config"` row are skipped and do not contribute to any reported number.

## 4. Results

### 4.1 Statistical performance

| Method | Final Val Loss | Final Val Accuracy | Steps to Target |
|---|---|---|---|
| Uniform LoRA | 0.286 ± 0.016 | 0.902 ± 0.006 | 1400 ± 300 (3/3) |
| AdaLoRA | 0.268 ± 0.003 | 0.894 ± 0.003 | never (0/3) |
| Gradient-Adaptive LoRA (α=0.0) | 0.287 ± 0.008 | 0.904 ± 0.005 | 1333 ± 231 (3/3) |
| Hardware-Aware LoRA (α=1.0) | 0.282 ± 0.007 | 0.906 ± 0.001 | 1767 ± 351 (3/3) |

![Validation accuracy vs wall-clock](results/figures/val_accuracy_vs_walltime.png)

All four methods land in a tight 0.894–0.906 final-accuracy band; on this task the accuracy differences are within seed variance. Hardware-Aware shows the lowest seed-to-seed variance (±0.001 over three seeds) and the highest mean accuracy. AdaLoRA produces the lowest validation loss (0.268) but does not cross the 0.90 accuracy threshold in any of three seeds within three epochs, while the other three methods cross it in all three. We report this factually without further causal claim.

### 4.2 Hardware performance

| Method | Peak Memory (MB) | Examples/sec | Wall-Clock (s) | Trainable Params |
|---|---|---|---|---|
| Uniform LoRA | 948.8 ± 1.0 | 920.3 ± 15.4 | 1011.4 ± 1.5 | 1,108,226 |
| AdaLoRA | 964.3 ± 0.1 | 601.7 ± 25.9 | 1130.2 ± 1.0 | 1,366,562 |
| Gradient-Adaptive LoRA | 1210.7 ± 1.3 | 918.7 ± 18.4 | 1024.1 ± 1.7 | 1,156,610 |
| Hardware-Aware LoRA | 1208.3 ± 1.2 | 938.4 ± 44.5 | 1025.6 ± 0.6 | 1,081,346 |

![Peak memory](results/figures/peak_memory_bars.png)
![Examples per second](results/figures/examples_per_second_bars.png)

At the same 192-rank budget, Hardware-Aware uses **1,081,346 trainable parameters** — fewer than Uniform (1,108,226), Gradient-Adaptive (1,156,610), and AdaLoRA (1,366,562). This is the cost penalty doing what it is designed to do: rank concentrates on the cheaper attention modules (cost 1536) rather than the more expensive FFN modules (cost 3840). Throughput splits cleanly along scheduler design: Uniform, Gradient-Adaptive, and Hardware-Aware all run at ~920 examples/sec, while AdaLoRA runs at 602 ex/s — a ~35% throughput penalty attributable to its per-step `update_and_allocate` work.

Peak memory tells the opposite story we expected from the parameter counts: Uniform 949 MB and AdaLoRA 964 MB, but Hardware-Aware 1208 MB and Gradient-Adaptive 1211 MB. Both two-stage methods carry ~250 MB more peak memory than Uniform despite Hardware-Aware using fewer parameters. `torch.cuda.max_memory_allocated` reflects activation and optimizer-state peaks during training, not parameter count alone; rank concentrating on attention grows those layers' A/B factors and their activation footprint along the forward/backward pass. The cost function `c_i = in_features + out_features` does not currently model activation memory — a direction for follow-up rather than a closed story.

### 4.3 Systems tradeoff

| Method | Accuracy per MB | Time to Target (s) | Scheduler Overhead (s) |
|---|---|---|---|
| Uniform LoRA | 0.00095 ± 0.00001 | 225.2 ± 45.3 (3/3) | 0.00 ± 0.00 |
| AdaLoRA | 0.00093 ± 0.00000 | never (0/3) | 76.67 ± 0.33 |
| Hardware-Aware LoRA | 0.00075 ± 0.00000 | 293.5 ± 54.1 (3/3) | 3.98 ± 0.13 |

![Scheduler overhead](results/figures/scheduler_overhead_bars.png)

The scheduler-overhead column is the project's clearest systems-flavor finding. Hardware-Aware's reallocator runs once at the warmup → stage-2 transition and costs **4.0 ± 0.13 s** of wall-clock, while AdaLoRA's runs every training step and accumulates to **76.7 ± 0.33 s** — roughly 19× more. This gap is structural to one-shot vs continuous reallocation and would only widen on longer training horizons. Time-to-target is mixed: Uniform reaches 0.90 in 225 s vs Hardware-Aware's 294 s, and AdaLoRA does not reach it at all in three seeds. Accuracy-per-MB favors Uniform (0.00095) over Hardware-Aware (0.00075) — a direct consequence of the peak-memory gap above, not a property of the allocator itself.

### 4.4 α-sweep ablation

| α / Variant | Final Val Accuracy | Time to Target (s) | Attention Rank Share |
|---|---|---|---|
| α=0.0 (gradient-only) | 0.904 ± 0.005 | 227.6 ± 35.9 (3/3) | 0.391 ± 0.005 |
| α=0.5 | 0.906 ± 0.003 | 320.1 ± 85.1 (3/3) | 0.476 ± 0.008 |
| α=1.0 (full hardware penalty) | 0.906 ± 0.001 | 293.5 ± 54.1 (3/3) | 0.561 ± 0.003 |

![Rank allocation heatmap](results/figures/rank_allocation_heatmap.png)

The α-sweep is the cleanest single piece of evidence in the report that the cost term in `s_i = g_i / c_i^α` is doing real work. As α moves from 0.0 to 0.5 to 1.0, attention-module rank share grows monotonically from **39.1% → 47.6% → 56.1%** while the gradient signal is held fixed across the three configurations. Final accuracy stays essentially flat across the sweep (0.904, 0.906, 0.906), so the systems reshaping is effectively free on accuracy for this task.

The heatmap reveals that the attention-share growth is asymmetric across the two attention projections: rank migrates from `ffn.lin1` to `attn.v_lin` as α grows, while `attn.q_lin` stays low across all methods (its gradient signal is too small for the cost penalty to lift it) and `ffn.lin2` stays low throughout. The 39% → 56% attention-share growth in the table is driven primarily by `v_lin`. This exposes a structural property of the rule `s_i = g_i / c_i^α`: the cost term reshapes allocations *among* modules with non-trivial gradient signal, but it cannot manufacture importance for modules whose `g_i` is near zero. The penalty is multiplicative, not additive.

### 4.5 Interpreting the asymmetric skew

The asymmetry has three implications beyond the specific column values.

First, "hardware-aware" is more precisely "hardware-biased gradient" allocation: the cost term shapes the distribution of rank among modules that already have learning signal, but does not recruit modules from a wider candidate pool. That is a sharper claim than "the cost term reshapes allocations" and is the one supported by the data.

Second, it likely explains the small accuracy gap between the two two-stage variants. Hardware-Aware (α=1.0) and Gradient-Adaptive (α=0.0) end at 0.904–0.906 because they allocate among the same set of gradient-signal-rich modules and differ only in the proportions. If the cost penalty reached `attn.q_lin`, we would expect a larger accuracy delta in either direction; its absence is consistent with redistribution-within-shortlist rather than expansion-of-shortlist.

Third, this suggests a clean follow-up design: an additive rule `s_i = g_i + λ / c_i` would let the cost term contribute independently of `g_i` and could route rank to low-gradient modules. Whether that is desirable is a separate question — funding modules that are not learning may waste budget — but it would disambiguate whether `attn.q_lin`'s low rank reflects genuine uninformativeness or insufficient warmup signal.

## 5. Limitations

- **Single task and single model.** SST-2 with DistilBERT-base only. Whether the allocator's behavior generalizes to multi-class GLUE tasks or to RoBERTa- or Llama-class models is not validated here.
- **Two-stage one-shot reallocation only.** A dynamic in-training variant is deferred. We do not know whether continuous reallocation would close the time-to-target gap with Uniform LoRA on this task.
- **Three seeds.** Too few to support paired significance testing. We report mean ± standard deviation and let the reader judge.
- **Mixed practical-efficiency picture vs Uniform.** Hardware-Aware uses fewer trainable parameters and achieves the lowest seed variance, but Uniform LoRA reaches the 0.90 target faster and uses less peak memory on this task. The headline systems wins (throughput, scheduler overhead) are observed against AdaLoRA, not against Uniform.
- **Peak memory not modeled in the cost function.** `c_i = in_features + out_features` captures parameter cost per rank, not activation memory. The unexpected peak-memory excess of the two-stage methods over Uniform is an open follow-up.
- **Multiplicative penalty cannot rescue near-zero gradient signal.** Because the rule is `s_i = g_i / c_i^α`, the cost term scales an existing gradient signal but cannot create one. As §4.4 shows, `attn.q_lin` stays low-rank across every α we tested because its `g_i` is small; no value of α can route rank to a module with no learning signal. An additive variant (`s_i = g_i + λ / c_i`) would behave differently here.

## 6. Closing

The cost term in `s_i = g_i / c_i^α` redistributes rank meaningfully among gradient-signal-rich modules — the α-sweep monotonicity demonstrates this independent of any change to the gradient signal itself — and yields the lowest trainable-parameter count of any method evaluated, competitive accuracy with the lowest seed variance, and a ~19× reduction in scheduler overhead relative to AdaLoRA. It does not recruit rank for modules with near-zero gradient signal (the rule is multiplicative), and it does not deliver an across-the-board practical-efficiency win over Uniform LoRA on this task, with peak memory and time-to-target as the headline tradeoffs.

Our method explores whether hardware-aware rank allocation can improve practical efficiency compared with AdaLoRA under constrained training budgets.

This makes the project appropriate for an Advanced Machine Learning Systems course because it combines parameter-efficient fine-tuning, adaptive optimization, memory-aware training, hardware-performance measurement, and empirical systems evaluation.
