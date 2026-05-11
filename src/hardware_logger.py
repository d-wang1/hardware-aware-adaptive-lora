from __future__ import annotations

import json
import time
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import torch


# every row gets these fields; missing ones serialize as null
SCHEMA_FIELDS: tuple[str, ...] = (
    "method",
    "step",
    "train_loss",
    "val_loss",
    "val_accuracy",
    "examples_per_second",
    "peak_memory_mb",
    "trainable_parameters",
    "scheduler_overhead_seconds",
)


class HardwareLogger:

    def __init__(
        self,
        output_dir: str | Path,
        method: str,
        run_id: str,
        ema_beta: float = 0.9,
    ) -> None:
        self.method = method
        self.run_id = run_id
        self.ema_beta = ema_beta

        self.path = Path(output_dir) / f"{run_id}.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("a", encoding="utf-8")

        self._step_started_at: float | None = None
        self._examples_per_second: float | None = None
        self._scheduler_overhead_s: float = 0.0
        self._cpu_memory_warned: bool = False

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    def step_start(self) -> None:
        self._step_started_at = time.perf_counter()

    def step_end(self, num_examples: int) -> None:
        if self._step_started_at is None:
            raise RuntimeError("step_end() called without a matching step_start()")
        elapsed = max(time.perf_counter() - self._step_started_at, 1e-9)
        instant = num_examples / elapsed
        if self._examples_per_second is None:
            # seed at the instantaneous rate so the first row isn't biased toward 0
            self._examples_per_second = instant
        else:
            self._examples_per_second = (
                self.ema_beta * self._examples_per_second
                + (1.0 - self.ema_beta) * instant
            )
        self._step_started_at = None

    @contextmanager
    def scheduler_block(self) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            self._scheduler_overhead_s += time.perf_counter() - start

    def peak_memory_mb(self) -> float:
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / (1024**2)
        if not self._cpu_memory_warned:
            warnings.warn(
                "HardwareLogger.peak_memory_mb() called without CUDA; "
                "returning 0.0.",
                RuntimeWarning,
                stacklevel=2,
            )
            self._cpu_memory_warned = True
        return 0.0

    @property
    def examples_per_second(self) -> float | None:
        return self._examples_per_second

    @property
    def scheduler_overhead_seconds(self) -> float:
        return self._scheduler_overhead_s

    def log(self, step: int, **fields: Any) -> dict[str, Any]:
        row: dict[str, Any] = {field: None for field in SCHEMA_FIELDS}
        row["method"] = self.method
        row["step"] = step
        row["timestamp"] = time.time()
        row["examples_per_second"] = self._examples_per_second
        row["peak_memory_mb"] = self.peak_memory_mb()
        row["scheduler_overhead_seconds"] = self._scheduler_overhead_s
        row.update(fields)
        self._fh.write(json.dumps(row) + "\n")
        # flush per row so a kill -9 mid-run doesn't lose anything
        self._fh.flush()
        return row

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.flush()
            self._fh.close()

    def __enter__(self) -> "HardwareLogger":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
