"""CPU kernels with deterministic, cacheable Numba compilation.

Kernels deliberately avoid ``fastmath`` and parallel reductions: numerical
compatibility takes priority over reassociation-based SIMD speedups.
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange


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


@njit(cache=True)
def _masked_sum_count(values: np.ndarray, masks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    frames, height, width = values.shape
    totals = np.zeros((height, width), dtype=np.float32)
    counts = np.zeros((height, width), dtype=np.uint32)
    for frame in range(frames):
        for y in range(height):
            for x in range(width):
                value = values[frame, y, x]
                if masks[frame, y, x] != 0 and not np.isnan(value):
                    totals[y, x] += value
                    counts[y, x] += 1
    return totals, counts


@njit(cache=True, parallel=True)
def _masked_sum_count_parallel(
    values: np.ndarray, masks: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    frames, height, width = values.shape
    totals = np.zeros((height, width), dtype=np.float32)
    counts = np.zeros((height, width), dtype=np.uint32)
    for y in prange(height):
        for x in range(width):
            total = np.float32(0.0)
            count = np.uint32(0)
            for frame in range(frames):
                value = values[frame, y, x]
                if masks[frame, y, x] != 0 and not np.isnan(value):
                    total += value
                    count += 1
            totals[y, x] = total
            counts[y, x] = count
    return totals, counts


def masked_sum_count(
    values: np.ndarray, masks: np.ndarray, parallel: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """One-pass masked reduction without allocating a temporary ``where`` array."""
    prepared_values = as_c_float32(values)
    prepared_masks = as_c_uint8(masks)
    if parallel:
        return _masked_sum_count_parallel(prepared_values, prepared_masks)
    return _masked_sum_count(prepared_values, prepared_masks)


@njit(cache=True)
def _apply_scale_and_mask(data: np.ndarray, mask: np.ndarray, factor: np.float32) -> None:
    height, width = data.shape
    for y in range(height):
        for x in range(width):
            if mask[y, x] != 0:
                data[y, x] *= factor
            else:
                data[y, x] = np.float32(0.0)


def apply_scale_and_mask_inplace(
    data: np.ndarray, mask: np.ndarray, factor: float
) -> np.ndarray:
    """Normalize and zero invalid pixels in one cacheable CPU kernel."""
    if data.dtype != np.float32 or not data.flags.c_contiguous:
        raise ValueError("stack data must be C-contiguous float32")
    _apply_scale_and_mask(data, as_c_uint8(mask), np.float32(factor))
    return data


@njit(cache=True, parallel=True)
def _weighted_merge_mono(values: np.ndarray, counts: np.ndarray) -> np.ndarray:
    leaves, height, width = values.shape
    result = np.zeros((height, width), dtype=np.float32)
    for y in prange(height):
        for x in range(width):
            numerator = np.float32(0.0)
            total = np.uint32(0)
            for leaf in range(leaves):
                count = counts[leaf, y, x]
                numerator += values[leaf, y, x] * count
                total += count
            if total > 0:
                result[y, x] = numerator / total
    return result


@njit(cache=True, parallel=True)
def _weighted_merge_rgb(values: np.ndarray, counts: np.ndarray) -> np.ndarray:
    leaves, channels, height, width = values.shape
    result = np.zeros((channels, height, width), dtype=np.float32)
    for y in prange(height):
        for x in range(width):
            total = np.uint32(0)
            for leaf in range(leaves):
                total += counts[leaf, y, x]
            if total > 0:
                for channel in range(channels):
                    numerator = np.float32(0.0)
                    for leaf in range(leaves):
                        numerator += values[leaf, channel, y, x] * counts[leaf, y, x]
                    result[channel, y, x] = numerator / total
    return result


def weighted_merge(values: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """Merge substack means by their valid-pixel counts."""
    values = as_c_float32(values)
    counts = np.ascontiguousarray(counts, dtype=np.uint32)
    if values.ndim == 3:
        return _weighted_merge_mono(values, counts)
    if values.ndim == 4:
        return _weighted_merge_rgb(values, counts)
    raise ValueError("substack values must be mono or RGB")


def warm_cpu_kernels() -> None:
    """Compile representative signatures in a background startup worker."""
    sample = np.ones((4, 4), dtype=np.float32)
    calibrate_inplace(sample.copy(), sample, sample)
    values = np.ones((2, 4, 4), dtype=np.float32)
    masks = np.ones(values.shape, dtype=np.uint8)
    masked_extrema(values, masks, True)
    masked_sum_count(values, masks)
    apply_scale_and_mask_inplace(sample.copy(), np.ones(sample.shape, dtype=np.uint8), 1.0)
    weighted_merge(values, np.ones((2, 4, 4), dtype=np.uint32))
