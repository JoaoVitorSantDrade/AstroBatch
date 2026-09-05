"""CPU execution budget shared by engines and pipeline schedulers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math


def science_frame_bytes(path: Path) -> int:
    """Float32 science-plane size, inspecting headers without decoding pixels."""
    from astropy.io import fits
    with fits.open(path, memmap=False, lazy_load_hdus=True) as hdul:
        for hdu in hdul:
            if hdu.is_image and hdu.shape and hdu.name not in {"VALID_MASK", "SAT_MASK", "DISAGREE"}:
                return math.prod(hdu.shape) * 4
    raise ValueError(f"No science image: {path}")


@dataclass(frozen=True, slots=True)
class ExecutionBudget:
    """Prevent nested engine and pipeline parallelism from oversubscribing CPU."""

    worker_count: int
    kernel_parallel: bool
    in_flight_limit: int | None = None

    @classmethod
    def for_pipeline(cls, worker_count: int) -> "ExecutionBudget":
        workers = max(1, int(worker_count))
        # Parallel kernels are reserved for the final single-worker reduction.
        # Frame/leaf parallelism otherwise owns the available CPU cores.
        return cls(worker_count=workers, kernel_parallel=workers == 1)

    @classmethod
    def for_frame_pipeline(cls, requested_workers: int, memory_budget_mb: int,
                           frame_bytes: int, reserved_frames: int = 2) -> "ExecutionBudget":
        """Return a conservative CPU/in-flight budget for frame pipelines."""
        workers = max(1, int(requested_workers))
        budget = max(64, int(memory_budget_mb)) * 1024 * 1024
        slots = int(budget // max(int(frame_bytes), 1))
        available = slots - max(int(reserved_frames), 0)
        if available < 1:
            required = math.ceil((max(int(reserved_frames), 0)+1)*int(frame_bytes)/1024**2)
            raise ValueError(f"Frame working set requires approximately {required} MiB; increase memory_budget_mb (currently {memory_budget_mb}).")
        # max_in_flight is two worker results, so account for both halves.
        workers = min(workers, max(1, available // 2))
        return cls(worker_count=max(1, workers), kernel_parallel=workers == 1,
                   in_flight_limit=min(available, max(1, workers)*2))

    @property
    def max_in_flight(self) -> int:
        return self.in_flight_limit if self.in_flight_limit is not None else self.worker_count * 2
