"""Repeatable CPU-profile benchmark for V2 engines.

Run in the project environment after installing requirements:
``python benchmarks/engine_benchmark.py``.
The script reports median elapsed seconds, never declares a Fast engine
automatic merely because it exists.
"""

from __future__ import annotations

import json
import sys
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def median_seconds(action: Callable[[], object], repeats: int = 5) -> float:
    samples: list[float] = []
    action()  # warm imported/JIT paths before timing
    for _ in range(repeats):
        started = time.perf_counter()
        action()
        samples.append(time.perf_counter() - started)
    return float(np.median(samples))


def run() -> dict[str, float]:
    from astroalign_logic import warp_frame
    from astroflow_logic import detect_stars
    from stacking_logic import reject_and_combine_block

    rng = np.random.default_rng(42)
    image = rng.normal(1000, 20, (1024, 1024)).astype(np.float32)
    for x, y in rng.integers(20, 1004, size=(180, 2)):
        image[y - 1 : y + 2, x - 1 : x + 2] += 500
    values = rng.normal(1000, 10, (16, 256, 256)).astype(np.float32)
    masks = np.ones(values.shape, dtype=bool)
    matrix = np.array([[1.0, 0.001, 2.0], [-0.001, 1.0, -3.0], [0.0, 0.0, 1.0]])

    return {
        "flow_stable_seconds": median_seconds(lambda: detect_stars(image, 4.0, 4.0, 250, "dao", "Stable")),
        "flow_fast_seconds": median_seconds(lambda: detect_stars(image, 4.0, 4.0, 250, "opencv-components", "Fast")),
        "align_stable_seconds": median_seconds(lambda: warp_frame(image, matrix, "lanczos", engine_profile="Stable")),
        "align_fast_seconds": median_seconds(lambda: warp_frame(image, matrix, "lanczos", engine_profile="Fast")),
        "stack_stable_seconds": median_seconds(lambda: reject_and_combine_block(values, masks, "Mean", "None", 3.0, 3.0, engine_profile="Stable")),
        "stack_fast_seconds": median_seconds(lambda: reject_and_combine_block(values, masks, "Mean", "None", 3.0, 3.0, engine_profile="Fast", kernel_parallel=True)),
    }


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, sort_keys=True))
