"""HardwareAwareRankAllocator — gradient EMA, cost proxy, score, allocation.

Phase 4b core contribution. Pure-Python; no training-loop dependencies. The
allocator is fed LoRA gradient norms once per step during a warmup stage,
then asked to materialize a per-module rank dict that respects the
project-wide ``total_rank_budget`` invariant (192 in production: 24
DistilBERT LoRA modules × uniform rank 8). The allocator itself is
budget-agnostic — pass any feasible (budget, min_rank, max_rank) triple.

The allocation score is::

    s_i = g_i / c_i ** alpha

where ``g_i`` is the EMA of LoRA Frobenius gradient norms (updated by
``update_gradient_scores``) and ``c_i = in_dim + out_dim`` is the per-rank
parameter cost (from ``enumerate_lora_modules`` + ``parameter_cost``).
``alpha = 0.0`` collapses to a gradient-only allocator (the
``gradient_adaptive`` ablation); ``alpha = 1.0`` is the proposed
hardware-aware variant.
"""
from __future__ import annotations

import math
from typing import Mapping

from peft import PeftModel

from src.lora_utils import (
    enumerate_lora_modules,
    lora_grad_norms,
    parameter_cost,
)


class HardwareAwareRankAllocator:
    """Two-stage rank allocator for the hardware-aware LoRA experiments.

    Usage during the warmup stage::

        allocator = HardwareAwareRankAllocator(96, 2, 16, alpha=1.0, ema_beta=0.9)
        for step in range(warmup_steps):
            loss.backward()
            allocator.update_gradient_scores(peft_model)
            optimizer.step()
        rank_dict = allocator.allocate(peft_model)

    The returned ``rank_dict`` has the same fqname keys as
    ``enumerate_lora_modules`` and feeds directly into
    ``build_non_uniform_lora_model`` for stage 2.
    """

    def __init__(
        self,
        total_budget: int,
        min_rank: int,
        max_rank: int,
        alpha: float,
        ema_beta: float,
    ) -> None:
        if min_rank < 1:
            raise ValueError(f"min_rank must be >= 1; got {min_rank}")
        if max_rank < min_rank:
            raise ValueError(
                f"max_rank ({max_rank}) must be >= min_rank ({min_rank})"
            )
        if not 0.0 <= ema_beta < 1.0:
            raise ValueError(
                f"ema_beta must be in [0.0, 1.0); got {ema_beta}"
            )
        self.total_budget = int(total_budget)
        self.min_rank = int(min_rank)
        self.max_rank = int(max_rank)
        self.alpha = float(alpha)
        self.ema_beta = float(ema_beta)
        self._g_ema: dict[str, float] = {}

    # --- gradient EMA --------------------------------------------------------

    @property
    def gradient_scores(self) -> dict[str, float]:
        """Read-only snapshot of the current EMA per module."""
        return dict(self._g_ema)

    def update_gradient_scores(self, peft_model: PeftModel) -> None:
        """Fold one step of LoRA gradient norms into the EMA.

        Seeds at the observed value on a module's first appearance (mirrors
        the throughput-EMA seeding in [hardware_logger.py:111](src/hardware_logger.py:111))
        so warmup step 0 isn't biased toward zero by an implicit zero prior.
        """
        norms = lora_grad_norms(peft_model)
        for fqname, value in norms.items():
            if fqname in self._g_ema:
                self._g_ema[fqname] = (
                    self.ema_beta * self._g_ema[fqname]
                    + (1.0 - self.ema_beta) * value
                )
            else:
                self._g_ema[fqname] = value

    # --- costs and scores ----------------------------------------------------

    def compute_costs(self, peft_model: PeftModel) -> dict[str, int]:
        """Per-rank parameter cost ``c_i = in_dim + out_dim`` for each module."""
        return {
            fqname: parameter_cost(info["in_dim"], info["out_dim"])
            for fqname, info in enumerate_lora_modules(peft_model).items()
        }

    def compute_scores(self, costs: Mapping[str, int]) -> dict[str, float]:
        """``s_i = g_ema_i / c_i ** alpha``.

        Modules with no EMA entry yet score 0.0. ``alpha = 0.0`` makes
        ``c_i ** alpha == 1`` so this reduces to gradient-only scoring.
        """
        return {
            fqname: self._g_ema.get(fqname, 0.0) / (cost ** self.alpha)
            for fqname, cost in costs.items()
        }

    # --- allocation ----------------------------------------------------------

    def allocate_ranks(self, scores: Mapping[str, float]) -> dict[str, int]:
        """Turn allocation scores into a per-module rank dict.

        Implements README §"Rank Allocation Procedure" (lines 988-1017): floor
        every module at ``min_rank``, distribute the remaining budget
        proportionally to scores, clip to ``max_rank``, then nudge one rank at
        a time (helper below) until the sum is exactly ``total_budget``.

        NaN / negative scores are coerced to 0 so an un-observed module
        doesn't poison the proportional split.
        """
        n = len(scores)
        if n == 0:
            raise ValueError("scores is empty")
        if self.min_rank * n > self.total_budget:
            raise ValueError(
                f"infeasible: min_rank*n ({self.min_rank}*{n}) exceeds "
                f"total_budget ({self.total_budget})"
            )
        if self.max_rank * n < self.total_budget:
            raise ValueError(
                f"infeasible: max_rank*n ({self.max_rank}*{n}) below "
                f"total_budget ({self.total_budget})"
            )

        safe = {
            k: 0.0 if (math.isnan(float(v)) or float(v) < 0.0) else float(v)
            for k, v in scores.items()
        }
        ranks = {k: self.min_rank for k in scores}
        remaining = self.total_budget - n * self.min_rank
        total = sum(safe.values())

        if total <= 0.0:
            # No information yet — spread the leftover budget as evenly as
            # possible so we don't dump it all on one arbitrary module.
            base, leftover = divmod(remaining, n)
            for i, k in enumerate(scores):
                extra = base + (1 if i < leftover else 0)
                ranks[k] = min(self.min_rank + extra, self.max_rank)
        else:
            for k in scores:
                extra = round(safe[k] / total * remaining)
                ranks[k] = min(self.min_rank + extra, self.max_rank)

        self._rebalance_to_exact_budget(ranks, safe)
        return ranks

    def _rebalance_to_exact_budget(
        self,
        ranks: dict[str, int],
        scores: Mapping[str, float],
    ) -> None:
        """Add (or remove) one rank at a time from the most (or least)
        deserving module until ``sum(ranks) == total_budget``. Ties are broken
        on name so the loop is deterministic across Python invocations.
        """
        while True:
            diff = self.total_budget - sum(ranks.values())
            if diff == 0:
                return
            if diff > 0:
                candidates = [k for k, r in ranks.items() if r < self.max_rank]
                if not candidates:
                    raise ValueError(
                        f"cannot reach budget {self.total_budget}: all modules "
                        f"already at max_rank {self.max_rank}"
                    )
                target = max(candidates, key=lambda k: (scores.get(k, 0.0), k))
                ranks[target] += 1
            else:
                candidates = [k for k, r in ranks.items() if r > self.min_rank]
                if not candidates:
                    raise ValueError(
                        f"cannot reach budget {self.total_budget}: all modules "
                        f"already at min_rank {self.min_rank}"
                    )
                target = min(candidates, key=lambda k: (scores.get(k, 0.0), k))
                ranks[target] -= 1

    def allocate(self, peft_model: PeftModel) -> dict[str, int]:
        """One-shot warmup→stage-2 helper: read costs from ``peft_model``,
        score the current EMA, return the rank dict ready for
        ``build_non_uniform_lora_model``."""
        costs = self.compute_costs(peft_model)
        scores = self.compute_scores(costs)
        return self.allocate_ranks(scores)
