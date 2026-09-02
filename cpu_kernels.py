"""CPU kernels with deterministic, cacheable Numba compilation.

Kernels deliberately avoid ``fastmath`` and parallel reductions: numerical
compatibility takes priority over reassociation-based SIMD speedups.
"""

from __future__ import annotations

import numpy as np
from numba import njit


def as_c_float32(values: np.ndarray) -> np.ndarray:
    """Return a C-contiguous float32 array, copying only when required."""
    return np.ascontiguousarray(values, dtype=np.float32)


def as_c_uint8(values: np.ndarray) -> np.ndarray:
    """Return a compact C-contiguous validity array."""
    return np.ascontiguousarray(values, dtype=np.uint8)


@njit(cache=True)
def _calibrate_inplace(
    data: np.ndarray,
    dark: np.ndarray,
    flat: np.ndarray,
    use_dark: bool,
    use_flat: bool,
) -> None:
    height, width = data.shape
    for y in range(height):
        for x in range(width):
            value = data[y, x]
            if use_dark:
                value = value - dark[y, x]
            if use_flat:
                flat_value = flat[y, x]
                if np.isfinite(flat_value) and flat_value > np.float32(0.01):
                    value = value / flat_value
            data[y, x] = value


def calibrate_inplace(
    data: np.ndarray,
    master_dark: np.ndarray | None,
    master_flat: np.ndarray | None,
) -> np.ndarray:
    """Apply calibration in place using the established elementwise order."""
    if data.dtype != np.float32 or not data.flags.c_contiguous:
        raise ValueError("calibration data must be a C-contiguous float32 array")
    dark = (
        as_c_float32(master_dark)
        if master_dark is not None
        else np.empty((0, 0), dtype=np.float32)
    )
    flat = (
        as_c_float32(master_flat)
        if master_flat is not None
        else np.empty((0, 0), dtype=np.float32)
    )
    _calibrate_inplace(data, dark, flat, master_dark is not None, master_flat is not None)
    return data


@njit(cache=True)
def _masked_extrema(
    values: np.ndarray, masks: np.ndarray, maximum: bool
) -> np.ndarray:
    frames, height, width = values.shape
    result = np.empty((height, width), dtype=np.float32)
    for y in range(height):
        for x in range(width):
            found = False
            candidate = np.float32(0.0)
            for frame in range(frames):
                value = values[frame, y, x]
                if masks[frame, y, x] != 0 and not np.isnan(value):
                    if not found or (value > candidate if maximum else value < candidate):
                        candidate = value
                        found = True
            result[y, x] = candidate if found else np.nan
    return result


def masked_extrema(values: np.ndarray, masks: np.ndarray, maximum: bool) -> np.ndarray:
    """Bitwise-safe masked min/max: selections, not reordered reductions."""
    return _masked_extrema(as_c_float32(values), as_c_uint8(masks), maximum)


def warm_cpu_kernels() -> None:
    """Compile representative signatures in a background startup worker."""
    sample = np.ones((4, 4), dtype=np.float32)
    calibrate_inplace(sample.copy(), sample, sample)
    values = np.ones((2, 4, 4), dtype=np.float32)
    masks = np.ones(values.shape, dtype=np.uint8)
    masked_extrema(values, masks, True)
