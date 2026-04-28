# Hardware-Aware Adaptive LoRA Rank Allocation

## Overview

This project explores a hardware-aware alternative to adaptive LoRA rank allocation for efficient fine-tuning of large neural networks.

Low-Rank Adaptation (LoRA) is a parameter-efficient fine-tuning method that freezes a pretrained model and inserts small trainable low-rank matrices into selected layers. Instead of updating the full model weight matrix \(W\), LoRA learns a low-rank update:

\[
W' = W + \Delta W
\]

where

\[
\Delta W = BA
\]

with \(A \in \mathbb{R}^{r \times d}\), \(B \in \mathbb{R}^{k \times r}\), and \(r\) is the LoRA rank.

A common LoRA setup assigns the same rank \(r\) to every adapted layer. However, this uniform allocation may be inefficient because not all layers contribute equally to downstream task performance. AdaLoRA improves on this idea by adaptively assigning rank budget according to parameter importance.

This project investigates a related but distinct question:

> Can a lightweight hardware-aware LoRA rank allocation rule achieve better practical efficiency than uniform LoRA and potentially AdaLoRA under constrained training budgets?

Instead of asking only which layers are most important for model quality, this project asks:

> Which layers provide the best improvement per unit of hardware cost?

The goal is not to prove that this method universally outperforms AdaLoRA. Rather, the project studies whether a simpler, hardware-aware rank scheduler can outperform AdaLoRA in certain practical metrics such as wall-clock time to target validation loss, memory-normalized accuracy, throughput, or scheduler overhead.

---

## Motivation

Modern fine-tuning methods are often evaluated primarily on final task accuracy or validation loss. However, in real ML systems, hardware constraints often matter just as much as model quality.

For example, a method may have slightly better final accuracy but require:

- more GPU memory,
- lower training throughput,
- higher scheduling overhead,
- longer wall-clock time,
- or more complicated implementation.

In many practical settings, especially small-budget fine-tuning, student projects, edge devices, and resource-constrained research environments, the most useful method may not be the method with the best final accuracy. Instead, it may be the method that reaches acceptable quality fastest or with the least memory.

AdaLoRA adaptively allocates rank according to importance scores. This is powerful, but it is not explicitly optimized for measured hardware cost. Our project explores whether incorporating a hardware-cost proxy into the allocation rule can improve practical efficiency.

---

## Research Question

The main research question is:

> Under a fixed LoRA rank budget, can a lightweight hardware-aware rank allocation method achieve better time-to-quality or memory-normalized performance than uniform LoRA and AdaLoRA?

A secondary research question is:

> Does incorporating hardware cost into the rank allocation score improve training efficiency compared with rank allocation based only on gradient or importance signals?

---

## Hypothesis

We hypothesize that a hardware-aware LoRA rank allocator can outperform uniform LoRA and may outperform AdaLoRA under certain constrained training regimes.

Specifically, we expect the proposed method to perform well when evaluated by:

- validation performance per trainable parameter,
- validation performance per MB of peak memory,
- wall-clock time to reach a target validation loss,
- examples/sec or tokens/sec,
- and scheduler overhead.

We do not necessarily expect the method to beat AdaLoRA in final validation accuracy after long training. Instead, the core hypothesis is that a simpler hardware-aware method may provide a better systems-level tradeoff.

---

## Background

### LoRA

LoRA fine-tunes a pretrained model by freezing the original weights and adding low-rank trainable updates to selected linear layers.

For a linear layer with weight matrix:

\[
W \in \mathbb{R}^{k \times d}
\]

LoRA represents the update as:

\[
\Delta W = BA
\]

where:

\[
A \in \mathbb{R}^{r \times d}
\]

and

\[
B \in \mathbb{R}^{k \times r}
\]

The rank \(r\) controls the capacity and cost of the adaptation. Larger rank usually gives the model more expressive power but increases trainable parameters, memory usage, and compute.

### Uniform Rank Allocation

Standard LoRA often uses the same rank for every adapted layer, such as:

```text
q_proj: r = 8
v_proj: r = 8
k_proj: r = 8
o_proj: r = 8
mlp_up: r = 8
mlp_down: r = 8
````

This is simple but may waste rank budget on layers that do not need much adaptation.

### AdaLoRA

AdaLoRA improves on uniform LoRA by dynamically allocating rank budget across layers. It estimates the importance of different low-rank components and prunes less important components over time.

The key distinction is that AdaLoRA primarily focuses on importance-aware allocation.

Our project instead studies hardware-aware allocation:

```text
importance-aware allocation:
    allocate rank to layers that seem most useful for task performance

hardware-aware allocation:
    allocate rank to layers that seem most useful per unit of hardware cost
```

---

## Proposed Method

The proposed method is called:

> Hardware-Aware Gradient LoRA Allocation

The method assigns LoRA rank to each adapted module using a score that combines a task-importance proxy and a hardware-cost proxy.

At a high level:

```text
score(layer) = task_importance(layer) / hardware_cost(layer)
```

Layers with high score receive more rank. Layers with low score receive less rank.

---

## Importance Proxy

The first version of the project will use the LoRA gradient norm as the task-importance proxy.

For each LoRA module (i), we compute:

[
g_i = \text{EMA}(|\nabla_{\theta_i} L|)
]

where:

* (\theta_i) are the trainable LoRA parameters for module (i),
* (L) is the training loss,
* and EMA is an exponential moving average over training steps.

The intuition is that layers with larger LoRA gradient norms may be receiving stronger learning signals and may benefit from additional rank capacity.

Possible gradient signals include:

```text
||grad(A_i)||_F
||grad(B_i)||_F
||grad(A_i)||_F + ||grad(B_i)||_F
||grad(Delta W_i)||_F
```

The simplest implementation will use:

```text
g_i = ||grad(A_i)||_F + ||grad(B_i)||_F
```

with exponential smoothing.

---

## Hardware-Cost Proxy

The hardware-cost proxy estimates how expensive it is to allocate rank to a given layer.

Possible cost proxies include:

### 1. Parameter Cost

For a LoRA module applied to a linear layer (W_i \in \mathbb{R}^{k_i \times d_i}), each additional rank adds:

[
d_i + k_i
]

trainable parameters.

So the parameter cost per rank is:

[
c_i = d_i + k_i
]

This is simple, deterministic, and easy to implement.

### 2. Activation-Aware Cost

A more hardware-aware cost proxy can include the size of the input/output activations.

For a batch with sequence length (s), batch size (b), input dimension (d_i), and output dimension (k_i), the cost may roughly scale with:

[
b \cdot s \cdot (d_i + k_i)
]

This better reflects actual compute and memory pressure during training.

### 3. Measured Runtime Cost

The most systems-oriented version measures wall-clock time contribution directly. For example, the training script can profile each adapted module or benchmark short training windows with different rank allocations.

This is more complex and may be added as an extension.

### Initial Choice

The first implementation will use parameter cost:

[
c_i = d_i + k_i
]

This is easy to compute for every LoRA module and gives a clear hardware-related cost estimate.

---

## Allocation Score

The proposed rank allocation score is:

[
s_i = \frac{g_i}{c_i^\alpha}
]

where:

* (g_i) is the gradient-based importance score,
* (c_i) is the hardware-cost proxy,
* and (\alpha) controls how strongly hardware cost is penalized.

Possible values:

```text
alpha = 0.0   # importance-only allocation
alpha = 0.5   # mild hardware penalty
alpha = 1.0   # full importance-per-cost allocation
```

This gives a direct experiment:

```text
Does hardware-aware scoring improve the quality/hardware tradeoff compared with importance-only scoring?
```

---

## Rank Budget

The method uses a fixed total rank budget.

Suppose there are (n) LoRA modules and a total rank budget (B). The allocation must satisfy:

[
\sum_{i=1}^{n} r_i = B
]

with constraints:

[
r_{\min} \leq r_i \leq r_{\max}
]

For example:

```text
number of LoRA modules: 24
uniform baseline rank: 8
total rank budget: 24 * 8 = 192
minimum rank: 2
maximum rank: 16
```

Then the adaptive method must allocate ranks across modules while keeping the same total rank budget as uniform LoRA.

This ensures that any performance difference is not simply caused by using more trainable parameters.

---

## Algorithm

### High-Level Algorithm

```text
Inputs:
    pretrained model
    downstream dataset
    LoRA target modules
    total rank budget B
    minimum rank r_min
    maximum rank r_max
    allocation update interval K
    hardware penalty alpha

Initialize:
    assign initial rank r_i to each LoRA module
    usually r_i = B / number_of_modules
    initialize EMA gradient score g_i = 0 for each module

Training:
    for each training step t:
        run forward pass
        compute loss
        run backward pass

        for each LoRA module i:
            compute current gradient norm
            update EMA gradient score g_i

        if t % K == 0 and t is after warmup:
            compute hardware cost c_i for each module
            compute score s_i = g_i / c_i^alpha
            reallocate rank budget B according to scores
            resize LoRA modules if necessary

        optimizer step
        log statistical and hardware metrics
```

---

## Practical Simplification

Dynamically resizing LoRA matrices during training can be annoying because changing rank changes parameter shapes and optimizer states.

Therefore, the first version may use a simpler two-stage allocation procedure.

### Two-Stage Version

```text
Stage 1: Warmup
    Train uniform LoRA for W steps.
    Track gradient norms for each LoRA module.

Stage 2: Reallocation
    Compute hardware-aware scores.
    Assign nonuniform ranks under the same total rank budget.
    Reinitialize LoRA modules with the selected ranks.
    Fine-tune using the fixed nonuniform allocation.

Evaluation:
    Compare the final nonuniform allocation against uniform LoRA and AdaLoRA.
```

This version is much easier to implement and still captures the core idea.

The project can start with the two-stage version and optionally add dynamic reallocation later.

---

## Alternative Variants

Several variants can be tested.

### Variant A: Uniform LoRA

Every adapted module receives the same rank.

```text
r_i = r_uniform
```

This is the main baseline.

### Variant B: Gradient-Only Adaptive LoRA

Rank is allocated according to gradient norm only.

```text
score_i = gradient_norm_i
```

This tests whether adaptation alone helps.

### Variant C: Hardware-Aware Gradient LoRA

Rank is allocated according to gradient norm divided by hardware cost.

```text
score_i = gradient_norm_i / cost_i
```

This is the proposed method.

### Variant D: AdaLoRA

Use AdaLoRA as a stronger adaptive baseline.

This comparison is important because the instructor specifically asked whether the method has the potential to outperform AdaLoRA.

---

## Why This Method Could Outperform AdaLoRA

The project should not claim that the proposed method is generally superior to AdaLoRA.

A better claim is:

> The proposed method may outperform AdaLoRA under hardware-constrained or short-budget fine-tuning regimes.

Possible advantages include:

### 1. Lower Allocation Overhead

AdaLoRA uses a more complex adaptive mechanism involving importance estimation and rank pruning. A simple gradient-per-cost allocator may have less scheduling overhead.

### 2. Hardware-Aware Objective

AdaLoRA focuses primarily on parameter importance. The proposed method explicitly considers hardware cost.

This may improve:

```text
accuracy per MB of memory
validation loss per second
time to target validation loss
throughput at comparable validation loss
```

### 3. Better Short-Horizon Behavior

In short fine-tuning runs, complex adaptive methods may not have enough time to estimate stable importance scores. A warmup-based gradient allocator may be easier to use in small-budget settings.

### 4. Simpler Implementation

The proposed method may be easier to implement, inspect, and modify than AdaLoRA. This matters for practical ML systems.

---

## Experimental Setup

### Models

Possible model choices:

```text
distilbert-base-uncased
bert-base-uncased
roberta-base
gpt2
t5-small
TinyLlama or another small causal LM, if compute allows
```

For feasibility, the first implementation should use a small or medium model.

Recommended starting point:

```text
distilbert-base-uncased
```

or

```text
bert-base-uncased
```

These are easier to fine-tune and evaluate quickly than larger language models.

### Tasks

Possible tasks:

```text
SST-2 sentiment classification
AG News classification
MRPC paraphrase classification
QNLI natural language inference
small instruction-following dataset, if using a causal LM
```

Recommended starting point:

```text
SST-2
```

because it is small, fast, and easy to evaluate.

A stronger final report could include two tasks:

```text
SST-2 for classification
AG News for multiclass classification
```

---

## Baselines

The project should compare at least three methods:

### Baseline 1: Uniform LoRA

Standard LoRA with the same rank for all adapted modules.

Example:

```text
rank = 8 for every target module
```

### Baseline 2: AdaLoRA

Adaptive LoRA using an existing implementation, likely through PEFT.

This is the strong adaptive baseline.

### Method 3: Hardware-Aware Adaptive LoRA

The proposed method.

Optional additional baseline:

### Baseline 4: Gradient-Only Adaptive LoRA

This helps isolate whether the hardware-cost term actually matters.

---

## Evaluation Metrics

The course requires evaluation in two ways:

1. statistical performance,
2. hardware performance.

This project will evaluate both.

---

## Statistical Performance Metrics

Depending on the task, possible statistical metrics include:

```text
validation accuracy
validation loss
F1 score
steps to reach target validation accuracy
steps to reach target validation loss
```

For SST-2:

```text
primary metric: validation accuracy
secondary metric: validation loss
```

For MRPC:

```text
primary metric: F1
secondary metric: accuracy
```

For language modeling:

```text
primary metric: validation perplexity
secondary metric: validation loss
```

Recommended initial metric:

```text
validation accuracy on SST-2
```

---

## Hardware Performance Metrics

The project should record:

```text
peak GPU memory or device memory
training wall-clock time
average time per training step
examples/sec
tokens/sec, if using language modeling
trainable parameter count
scheduler overhead
time to target validation score
```

The most important hardware-aware metrics are:

```text
wall-clock time to target validation accuracy
peak memory
examples/sec
```

If GPU memory measurement is difficult on Apple Silicon or CPU, the project can still measure:

```text
wall-clock time
examples/sec
trainable parameter count
estimated LoRA parameter memory
```

---

## Time-to-Quality Metric

A central metric should be:

```text
time to reach target validation accuracy
```

For example:

```text
time to reach 90% validation accuracy on SST-2
```

or:

```text
time to reach validation loss below 0.35
```

This metric is useful because the proposed method may not have the best final accuracy but may reach a useful quality threshold faster.

---

## Memory-Normalized Metric

Another useful metric is:

```text
validation accuracy / peak memory
```

or:

```text
validation loss improvement per MB
```

For example:

[
\text{memory-normalized score} = \frac{\text{validation accuracy}}{\text{peak memory in MB}}
]

A more interpretable metric may be:

```text
best validation accuracy under a fixed memory budget
```

---

## Expected Results

We expect:

### Uniform LoRA

Uniform LoRA should be simple and stable but may waste rank on less useful layers.

Expected behavior:

```text
good baseline quality
low scheduler overhead
possibly worse quality per trainable parameter
```

### AdaLoRA

AdaLoRA should be strong in validation performance because it adaptively reallocates rank based on importance.

Expected behavior:

```text
strong final validation accuracy
possibly higher adaptive scheduling complexity
possibly more overhead than uniform LoRA
```

### Hardware-Aware Adaptive LoRA

The proposed method may not always beat AdaLoRA in final accuracy, but may perform better in practical systems metrics.

Expected behavior:

```text
competitive validation accuracy
better time-to-quality than AdaLoRA in some settings
better memory-normalized performance
lower scheduling overhead
simpler implementation
```

---

## Main Experiment Plan

### Hypothesis

A hardware-aware LoRA rank allocation rule can achieve better wall-clock efficiency or memory-normalized validation performance than uniform LoRA and may outperform AdaLoRA under short training budgets.

### Proxy Statement

The main statistical proxy will be validation accuracy and validation loss.

The main hardware proxies will be:

```text
training wall-clock time
examples/sec
peak memory
trainable parameter count
time to target validation accuracy
```

### Protocol

1. Choose a pretrained transformer model.
2. Fine-tune it on SST-2 or another small NLP benchmark.
3. Compare:

   * uniform LoRA,
   * AdaLoRA,
   * gradient-only adaptive LoRA,
   * hardware-aware adaptive LoRA.
4. Use the same total rank budget across methods.
5. Train each method for the same maximum number of steps.
6. Log validation metrics and hardware metrics.
7. Compare final quality and time-to-quality.

### Expected Result

The proposed hardware-aware method is expected to achieve similar validation performance to AdaLoRA while improving one or more practical hardware metrics, especially wall-clock time to target validation score or validation accuracy per trainable parameter.

---

## Implementation Details

### Recommended Libraries

```text
Python
PyTorch
Hugging Face Transformers
Hugging Face Datasets
PEFT
NumPy
Pandas
Matplotlib
```

Optional:

```text
Weights & Biases
TensorBoard
PyTorch Profiler
```

---

## Repository Structure

Suggested repository structure:

```text
hardware-aware-lora/
│
├── README.md
├── requirements.txt
├── configs/
│   ├── uniform_lora.yaml
│   ├── adalora.yaml
│   ├── gradient_adaptive_lora.yaml
│   └── hardware_aware_lora.yaml
│
├── src/
│   ├── train.py
│   ├── evaluate.py
│   ├── data.py
│   ├── models.py
│   ├── lora_utils.py
│   ├── rank_allocator.py
│   ├── hardware_logger.py
│   └── metrics.py
│
├── experiments/
│   ├── run_uniform_lora.sh
│   ├── run_adalora.sh
│   ├── run_gradient_adaptive_lora.sh
│   └── run_hardware_aware_lora.sh
│
├── results/
│   ├── raw_logs/
│   ├── summaries/
│   └── figures/
│
└── paper/
    ├── main.tex
    └── references.bib
```

---

## Important Components

### `rank_allocator.py`

This file should contain the core contribution.

Responsibilities:

```text
track gradient norms
compute hardware cost proxies
compute allocation scores
assign rank budget
save final rank allocation
```

Example interface:

```python
class HardwareAwareRankAllocator:
    def __init__(
        self,
        total_budget,
        min_rank,
        max_rank,
        alpha,
        ema_beta,
    ):
        ...

    def update_gradient_scores(self, model):
        ...

    def compute_costs(self, model):
        ...

    def allocate_ranks(self):
        ...

    def apply_allocation(self, model):
        ...
```

---

### `hardware_logger.py`

This file should record hardware metrics.

Responsibilities:

```text
measure wall-clock time
measure examples/sec
measure peak memory if available
count trainable parameters
estimate LoRA parameter memory
measure scheduler overhead
```

Example metrics to log per run:

```json
{
  "method": "hardware_aware_lora",
  "step": 500,
  "train_loss": 0.42,
  "val_loss": 0.36,
  "val_accuracy": 0.89,
  "examples_per_second": 130.5,
  "peak_memory_mb": 3120,
  "trainable_parameters": 884736,
  "scheduler_overhead_seconds": 2.3
}
```

---

## Pseudocode

### Two-Stage Hardware-Aware Allocation

```python
def train_hardware_aware_lora(model, train_loader, val_loader, config):
    # Stage 1: warmup with uniform LoRA
    model = add_uniform_lora(
        model,
        rank=config.initial_rank,
        target_modules=config.target_modules,
    )

    allocator = HardwareAwareRankAllocator(
        total_budget=config.total_rank_budget,
        min_rank=config.min_rank,
        max_rank=config.max_rank,
        alpha=config.hardware_alpha,
        ema_beta=config.ema_beta,
    )

    for step in range(config.warmup_steps):
        loss = training_step(model, train_loader)
        loss.backward()

        allocator.update_gradient_scores(model)

        optimizer.step()
        optimizer.zero_grad()

    # Compute final nonuniform rank allocation
    costs = allocator.compute_costs(model)
    scores = allocator.compute_scores(costs)
    rank_allocation = allocator.allocate_ranks(scores)

    # Stage 2: train with nonuniform LoRA allocation
    model = reload_base_model()
    model = add_nonuniform_lora(
        model,
        rank_allocation=rank_allocation,
        target_modules=config.target_modules,
    )

    for step in range(config.finetune_steps):
        loss = training_step(model, train_loader)
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        if step % config.eval_interval == 0:
            evaluate_and_log(model, val_loader)
```

---

## Rank Allocation Procedure

One simple allocation rule:

```python
def allocate_ranks(scores, total_budget, min_rank, max_rank):
    n = len(scores)

    # Start with minimum rank for every module
    ranks = {module: min_rank for module in scores}

    remaining_budget = total_budget - n * min_rank

    # Normalize scores into probabilities
    total_score = sum(scores.values())
    probabilities = {
        module: score / total_score
        for module, score in scores.items()
    }

    # Allocate remaining rank budget proportionally
    for module, p in probabilities.items():
        extra_rank = round(p * remaining_budget)
        ranks[module] += extra_rank

    # Enforce max rank constraint
    for module in ranks:
        ranks[module] = min(ranks[module], max_rank)

    # Fix rounding errors until sum(ranks) == total_budget
    ranks = adjust_to_exact_budget(ranks, scores, total_budget, min_rank, max_rank)

    return ranks
```

---

## Example Configuration

```yaml
model:
  name: distilbert-base-uncased
  task: sst2

lora:
  target_modules:
    - q_lin
    - v_lin
  initial_rank: 8
  min_rank: 2
  max_rank: 16
  total_rank_budget: 96
  alpha: 1.0

training:
  batch_size: 32
  learning_rate: 2e-4
  epochs: 3
  warmup_steps: 200
  eval_interval: 100
  seed: 42

hardware:
  log_memory: true
  log_wall_clock: true
  log_examples_per_second: true
```

---

## Reporting Results

The final report should include tables like the following.

### Statistical Performance Table

| Method                 | Final Val Loss | Final Val Accuracy | Steps to Target Accuracy |
| ---------------------- | -------------: | -----------------: | -----------------------: |
| Uniform LoRA           |            TBD |                TBD |                      TBD |
| AdaLoRA                |            TBD |                TBD |                      TBD |
| Gradient-Adaptive LoRA |            TBD |                TBD |                      TBD |
| Hardware-Aware LoRA    |            TBD |                TBD |                      TBD |

### Hardware Performance Table

| Method                 | Peak Memory MB | Examples/sec | Wall-Clock Time | Trainable Params |
| ---------------------- | -------------: | -----------: | --------------: | ---------------: |
| Uniform LoRA           |            TBD |          TBD |             TBD |              TBD |
| AdaLoRA                |            TBD |          TBD |             TBD |              TBD |
| Gradient-Adaptive LoRA |            TBD |          TBD |             TBD |              TBD |
| Hardware-Aware LoRA    |            TBD |          TBD |             TBD |              TBD |

### Systems Tradeoff Table

| Method              | Accuracy per MB | Time to Target Accuracy | Scheduler Overhead |
| ------------------- | --------------: | ----------------------: | -----------------: |
| Uniform LoRA        |             TBD |                     TBD |                TBD |
| AdaLoRA             |             TBD |                     TBD |                TBD |
| Hardware-Aware LoRA |             TBD |                     TBD |                TBD |

---

## Figures

Useful figures for the final paper:

```text
validation accuracy vs training step
validation loss vs training step
validation accuracy vs wall-clock time
peak memory by method
examples/sec by method
rank allocation heatmap by layer
scheduler overhead by method
```

The most important figure is probably:

```text
validation accuracy vs wall-clock time
```

because it directly shows whether the proposed method improves practical training efficiency.

Another useful figure:

```text
rank allocation by layer
```

This helps explain what the adaptive method actually did.

---

## Preliminary Work

The proposal should include a small amount of preliminary work. This can be minimal.

Possible preliminary work to report:

```text
I verified that LoRA fine-tuning can be run on a small transformer model using Hugging Face Transformers and PEFT.
I inspected the adapted linear modules and confirmed that different layers have different input/output dimensions, which motivates a hardware-cost-aware allocation rule.
I drafted the rank allocation algorithm and identified gradient norm and parameter count as feasible signals to collect during training.
```

If you run even a tiny experiment, you can say:

```text
As preliminary work, I ran a small LoRA fine-tuning script on a subset of SST-2 using DistilBERT. The run confirmed that LoRA gradients can be collected per adapted module and that trainable parameter counts can be computed for each LoRA layer. This supports the feasibility of implementing a rank allocator based on gradient norm divided by hardware cost.
```

---

## Risks and Mitigations

### Risk 1: The proposed method does not beat AdaLoRA in final accuracy.

This is acceptable. The project is about systems tradeoffs, not only final accuracy.

Mitigation:

```text
Evaluate wall-clock time, throughput, memory, and scheduler overhead.
Frame success as better practical efficiency rather than universal accuracy improvement.
```

### Risk 2: Dynamic rank resizing is difficult.

Mitigation:

```text
Use a two-stage allocation method:
1. warm up with uniform LoRA,
2. compute ranks,
3. restart fine-tuning with fixed nonuniform ranks.
```

### Risk 3: Hardware memory measurements are unavailable on some devices.

Mitigation:

```text
Record wall-clock time, examples/sec, trainable parameter count, and estimated LoRA memory.
If CUDA is available, also record torch.cuda.max_memory_allocated().
```

### Risk 4: Gradient norms are noisy.

Mitigation:

```text
Use exponential moving average smoothing.
Use a warmup period before allocation.
Run multiple random seeds if time allows.
```

---

## Success Criteria

The project is successful if it demonstrates a clear empirical comparison between:

```text
uniform LoRA
AdaLoRA
hardware-aware adaptive LoRA
```

and reports both:

```text
statistical performance
hardware performance
```

The method does not need to beat AdaLoRA on every metric. A meaningful result could be:

```text
AdaLoRA achieves the best final accuracy, but hardware-aware LoRA reaches 95% of AdaLoRA's accuracy with lower wall-clock time and simpler scheduling.
```

or:

```text
Hardware-aware LoRA achieves similar validation accuracy to AdaLoRA but with lower scheduler overhead and better accuracy per trainable parameter.
```

or:

```text
The hardware-aware cost term improves time-to-quality compared with gradient-only adaptive allocation.
```

---

## Suggested Proposal Framing

A concise project description could be:

> This project implements a hardware-aware adaptive LoRA rank allocation system for efficient transformer fine-tuning. Standard LoRA often assigns the same rank to every adapted layer, while AdaLoRA adaptively reallocates rank according to parameter importance. This project explores a different allocation objective: validation improvement per unit hardware cost. The proposed method tracks gradient norms of LoRA modules during a warmup period and allocates a fixed total rank budget according to a score that divides gradient-based importance by a hardware-cost proxy such as per-rank parameter count. The method will be compared against uniform LoRA and AdaLoRA on a small NLP benchmark. Statistical performance will be measured using validation loss and accuracy, while hardware performance will be measured using wall-clock training time, examples/sec, peak memory when available, trainable parameter count, and time to target validation accuracy.

---

## Relevant Papers

The final paper should cite at least:

```text
LoRA:
Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models."

AdaLoRA:
Zhang et al., "Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning."

QLoRA:
Dettmers et al., "QLoRA: Efficient Finetuning of Quantized LLMs."

LoRA+:
Hayou et al., "LoRA+: Efficient Low Rank Adaptation of Large Models."

GaLore:
Zhao et al., "GaLore: Memory-Efficient LLM Training by Gradient Low-Rank Projection."
```

Potentially useful related concepts:

```text
gradient-based importance scoring
parameter-efficient fine-tuning
low-rank optimization
memory-efficient training
hardware-aware neural network design
```

---

## Final Notes

The project should be presented carefully.

Do not claim:

```text
Our method is better than AdaLoRA.
```

Instead, claim:

```text
Our method explores whether hardware-aware rank allocation can improve practical efficiency compared with AdaLoRA under constrained training budgets.
```

The strongest argument is that this is not just a LoRA-vs-QLoRA comparison. The core implementation is a custom rank allocation system that changes how trainable capacity is distributed across the model based on both learning signal and hardware cost.

This makes the project appropriate for an Advanced Machine Learning Systems course because it combines:

```text
parameter-efficient fine-tuning
adaptive optimization
memory-aware training
hardware-performance measurement
empirical systems evaluation
```

```