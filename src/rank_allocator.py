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
    """Per-module rank allocator: s_i = g_i / c_i ** alpha.

    g_i is an EMA over LoRA Frobenius grad norms (update_gradient_scores);
    c_i is per-rank parameter cost (in_dim + out_dim). alpha=0 -> gradient-only.
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

    @property
    def gradient_scores(self) -> dict[str, float]:
        return dict(self._g_ema)

    def update_gradient_scores(self, peft_model: PeftModel) -> None:
        # seed at the observed value on first appearance, otherwise blend
        norms = lora_grad_norms(peft_model)
        for fqname, value in norms.items():
            if fqname in self._g_ema:
                self._g_ema[fqname] = (
                    self.ema_beta * self._g_ema[fqname]
                    + (1.0 - self.ema_beta) * value
                )
            else:
                self._g_ema[fqname] = value

    def compute_costs(self, peft_model: PeftModel) -> dict[str, int]:
        return {
            fqname: parameter_cost(info["in_dim"], info["out_dim"])
            for fqname, info in enumerate_lora_modules(peft_model).items()
        }

    def compute_scores(self, costs: Mapping[str, int]) -> dict[str, float]:
        return {
            fqname: self._g_ema.get(fqname, 0.0) / (cost ** self.alpha)
            for fqname, cost in costs.items()
        }

    def allocate_ranks(self, scores: Mapping[str, float]) -> dict[str, int]:
        """Floor at min_rank, distribute leftover proportionally, clip, rebalance."""
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

        # coerce nan/negative to 0 so an unobserved module doesn't poison the split
        safe = {
            k: 0.0 if (math.isnan(float(v)) or float(v) < 0.0) else float(v)
            for k, v in scores.items()
        }
        ranks = {k: self.min_rank for k in scores}
        remaining = self.total_budget - n * self.min_rank
        total = sum(safe.values())

        if total <= 0.0:
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
        # break ties on name so the loop is deterministic
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
        costs = self.compute_costs(peft_model)
        scores = self.compute_scores(costs)
        return self.allocate_ranks(scores)
