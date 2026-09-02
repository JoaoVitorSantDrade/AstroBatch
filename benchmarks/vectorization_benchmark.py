"""Deterministic CPU benchmark for the vectorized processing paths.

Run directly with ``python benchmarks/vectorization_benchmark.py``.  Timings
are informational rather than test assertions, so CI is not made flaky by
machine-specific performance.
"""

from __future__ import annotations

import time
import sys
from pathlib import Path

import numpy as np
import psutil

# Make the repository root importable when this file is executed directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import astroflow_logic as flow
import stacking_logic as stacking
from cpu_kernels import warm_cpu_kernels

try:
    import astroalign_logic as align
except ModuleNotFoundError:
    align = None


def _best_seconds(callback, repeats: int = 5) -> float:
    callback()
    timings = []
    for _ in range(repeats):
        started = time.perf_counter()
        callback()
        timings.append(time.perf_counter() - started)
    return min(timings)


def _peak_rss() -> int:
    info = psutil.Process().memory_info()
    return int(getattr(info, "peak_wset", info.rss))


def main() -> None:
    started = time.perf_counter()
    warm_cpu_kernels()
    print(f"Numba warm cache: {time.perf_counter() - started:.6f}s")
    rng = np.random.default_rng(20260902)
    stars = rng.uniform(0, 2048, size=(180, 2)).astype(np.float32)
    values = rng.normal(1000, 20, size=(15, 256, 256)).astype(np.float32)
    masks = rng.random(values.shape) > 0.08
    print(
        "stack band allocation: "
        f"values={values.nbytes / 1024 / 1024:.2f} MiB, "
        f"masks={masks.nbytes / 1024 / 1024:.2f} MiB"
    )
    workloads = {
        "flow incremental matching": lambda: flow._match_incremental_stars(
            stars, stars + (4.25, -2.5), (-4.25, 2.5), 2.0
        ),
        "flow asterism hashing": lambda: flow._extract_asterism_database(stars, 60),
        "stack masked mean": lambda: stacking.reject_and_combine_block(
            values, masks, "Mean", "None", 3.0, 3.0
        ),
        "stack sigma rejection": lambda: stacking._reject_cpu(
            values, masks, "SigmaClip", 3.0, 3.0
        ),
    }
    if align is not None:
        image = rng.uniform(0, 65535, size=(512, 512, 3)).astype(np.float32)
        matrix = np.asarray([[1, 0, 2.25], [0, 1, -1.5]], dtype=np.float32)
        workloads["align RGB warp"] = lambda: align.warp_frame(
            image, matrix, "bilinear", False
        )

    for name, callback in workloads.items():
        before = _peak_rss()
        elapsed = _best_seconds(callback)
        after = _peak_rss()
        print(f"{name}: {elapsed:.6f}s; peak RSS delta={(after - before) / 1024 / 1024:.2f} MiB")


if __name__ == "__main__":
    main()
