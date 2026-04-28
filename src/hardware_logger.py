"""JSONL training instrumentation: wall-clock, memory, throughput, scheduler overhead.

Used by every training method (uniform / AdaLoRA / gradient_adaptive /
hardware_aware) so the Phase 6 aggregator only has to read one schema.
Schema (per the README):

    method, step, train_loss, val_loss, val_accuracy,
    examples_per_second, peak_memory_mb, trainable_parameters,
    scheduler_overhead_seconds

Missing fields are written as ``null``. Extra caller-supplied fields (e.g.
``event="reallocation"``, ``rank_dict={...}``) round-trip through pandas in
Phase 6 and are how the two-stage adaptive methods record their reallocation
events.

Typical use inside a training loop::

    with HardwareLogger(out_dir, method="uniform", run_id=run_id) as logger:
        for step, batch in enumerate(loader):
            logger.step_start()
            train_loss = train_one_step(batch)
            logger.step_end(num_examples=batch["input_ids"].size(0))

            if step % eval_interval == 0:
                with logger.scheduler_block():
                    # any allocator / reallocation work goes here so the
                    # cost is attributed honestly in scheduler_overhead_seconds
                    ...
                metrics = evaluate(model, val_loader, device)
                logger.log(step,
                           train_loss=train_loss,
                           val_loss=metrics["val_loss"],
                           val_accuracy=metrics["val_accuracy"],
                           trainable_parameters=trainable_count)
"""
from __future__ import annotations

import json
import time
import warnings
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import torch


# Schema columns the README guarantees. ``log()`` emits these for every row,
# filling absent fields with ``None`` so Phase 6 can rely on their presence.
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
    """Append-only JSONL writer with throughput / peak-memory / overhead probes.

    The logger is responsible for *measuring* throughput and scheduler overhead
    (the caller wraps the relevant code in ``step_start/step_end`` and
    ``scheduler_block``); the caller is responsible for supplying losses,
    val accuracy, and the trainable parameter count via ``log(...)``.
    """

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

        # internal probe state
        self._step_started_at: float | None = None
        self._examples_per_second: float | None = None  # EMA, None until first step
        self._scheduler_overhead_s: float = 0.0
        self._cpu_memory_warned: bool = False

        # peak-memory baseline; subsequent peak_memory_mb() calls are relative to here
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    # --- step timing ---------------------------------------------------------

    def step_start(self) -> None:
        """Mark the start of one optimizer step (call right before forward)."""
        self._step_started_at = time.perf_counter()

    def step_end(self, num_examples: int) -> None:
        """Finalize the step started by ``step_start()`` and update the
        throughput EMA. ``num_examples`` is typically the per-step batch size.
        """
        if self._step_started_at is None:
            raise RuntimeError("step_end() called without a matching step_start()")
        elapsed = max(time.perf_counter() - self._step_started_at, 1e-9)
        instant = num_examples / elapsed
        if self._examples_per_second is None:
            # First step: initialize the EMA at the instantaneous rate so we
            # don't bias early rows toward zero.
            self._examples_per_second = instant
        else:
            self._examples_per_second = (
                self.ema_beta * self._examples_per_second
                + (1.0 - self.ema_beta) * instant
            )
        self._step_started_at = None

    # --- scheduler overhead --------------------------------------------------

    @contextmanager
    def scheduler_block(self) -> Iterator[None]:
        """Context manager that adds elapsed seconds to the running scheduler
        overhead total. Wrap any allocator / reallocation work inside this.
        """
        start = time.perf_counter()
        try:
            yield
        finally:
            self._scheduler_overhead_s += time.perf_counter() - start

    # --- probes --------------------------------------------------------------

    def peak_memory_mb(self) -> float:
        """Peak GPU memory (MB) since logger construction; 0.0 on CPU.

        Emits one ``RuntimeWarning`` per logger instance when called without
        CUDA, so it's clear that any per-step ``peak_memory_mb`` field in the
        log is meaningless from a Mac/CPU smoke run.
        """
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / (1024**2)
        if not self._cpu_memory_warned:
            warnings.warn(
                "HardwareLogger.peak_memory_mb() called without CUDA; "
                "returning 0.0. Real peak-memory numbers require the CUDA box.",
                RuntimeWarning,
                stacklevel=2,
            )
            self._cpu_memory_warned = True
        return 0.0

    @property
    def examples_per_second(self) -> float | None:
        """Current EMA throughput; ``None`` until the first ``step_end``."""
        return self._examples_per_second

    @property
    def scheduler_overhead_seconds(self) -> float:
        """Cumulative seconds spent inside ``scheduler_block`` contexts."""
        return self._scheduler_overhead_s

    # --- writer --------------------------------------------------------------

    def log(self, step: int, **fields: Any) -> dict[str, Any]:
        """Write one JSONL row and return it.

        Auto-fills ``method``, ``step``, ``timestamp``, ``examples_per_second``,
        ``peak_memory_mb`` and ``scheduler_overhead_seconds`` from the logger's
        internal state. Caller-supplied ``fields`` override these defaults and
        may add arbitrary extras (e.g. ``event="reallocation"``).
        """
        row: dict[str, Any] = {field: None for field in SCHEMA_FIELDS}
        row["method"] = self.method
        row["step"] = step
        row["timestamp"] = time.time()
        row["examples_per_second"] = self._examples_per_second
        row["peak_memory_mb"] = self.peak_memory_mb()
        row["scheduler_overhead_seconds"] = self._scheduler_overhead_s
        row.update(fields)
        self._fh.write(json.dumps(row) + "\n")
        # Flush every row so logs survive a kill -9 / OOM mid-run.
        self._fh.flush()
        return row

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.flush()
            self._fh.close()

    # --- context manager interface ------------------------------------------

    def __enter__(self) -> "HardwareLogger":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
