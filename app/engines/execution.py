"""CPU execution budget shared by engines and pipeline schedulers."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ExecutionBudget:
    """Prevent nested engine and pipeline parallelism from oversubscribing CPU."""

    worker_count: int
    kernel_parallel: bool

    @classmethod
    def for_pipeline(cls, worker_count: int) -> "ExecutionBudget":
        workers = max(1, int(worker_count))
        # Parallel kernels are reserved for the final single-worker reduction.
        # Frame/leaf parallelism otherwise owns the available CPU cores.
        return cls(worker_count=workers, kernel_parallel=workers == 1)
