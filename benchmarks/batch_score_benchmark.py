"""Repeatable benchmark for the batch comparison scorer.

Run directly with ``.venv\\Scripts\\python.exe benchmarks/batch_score_benchmark.py``.
The embedded reference is the pre-optimization implementation.  Timings are
informational and use the median after warmup so machine-specific noise does
not turn into a test failure.
"""

from __future__ import annotations

import math
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import batch_logic


def _reference_score(current: np.ndarray, previous: np.ndarray) -> float:
    diff = cv2.subtract(current, previous)
    if np.isnan(diff).any() or np.isinf(diff).any():
        valid = np.isfinite(diff)
        if not valid.any():
            return math.nan
        diff = diff[valid]
    diff -= np.mean(diff)
    return float(np.linalg.norm(diff) / math.sqrt(diff.size))


def _median_ms(callback, repeats: int = 9) -> float:
    callback()
    samples = []
    for _ in range(repeats):
        started = time.perf_counter()
        callback()
        samples.append((time.perf_counter() - started) * 1000)
    return statistics.median(samples)


def _make_case(size: int, mixed_invalid: bool) -> tuple[np.ndarray, np.ndarray]:
    current = np.random.default_rng(20260904 + size).normal(
        1000, 20, size=(size, size)
    ).astype(np.float32)
    previous = np.random.default_rng(20261001 + size).normal(
        1000, 20, size=(size, size)
    ).astype(np.float32)
    if mixed_invalid:
        current[::37, ::43] = np.nan
        previous[::53, ::47] = np.inf
    return current, previous


def main() -> None:
    print("case,reference_ms,optimized_ms,speedup,reference_score,optimized_score")
    for size in (512, 2048):
        for mixed_invalid in (False, True):
            current, previous = _make_case(size, mixed_invalid)
            reference = _reference_score(current, previous)
            optimized = batch_logic.comparison_score(current, previous)
            reference_ms = _median_ms(lambda: _reference_score(current, previous))
            optimized_ms = _median_ms(
                lambda: batch_logic.comparison_score(current, previous)
            )
            label = f"{size}x{size}-{'mixed' if mixed_invalid else 'finite'}"
            print(
                f"{label},{reference_ms:.3f},{optimized_ms:.3f},"
                f"{reference_ms / optimized_ms:.2f}x,{reference:.8f},{optimized:.8f}"
            )


if __name__ == "__main__":
    main()
