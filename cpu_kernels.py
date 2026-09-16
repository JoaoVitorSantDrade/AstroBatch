"""CPU kernels with deterministic, cacheable Numba compilation.

Kernels deliberately avoid ``fastmath`` and keep every per-pixel frame/leaf
traversal in its established order. Independent output pixels may be split
with ``prange`` once the working set is large enough to amortize scheduling.
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange

from cpu_runtime import configure_numba_threads, configure_opencv_threads


# Parallel scheduling is worthwhile only once a leaf contains several million
# pixel comparisons; below that point the launch overhead is slower than the
# serial, cache-friendly traversal.
_PARALLEL_MIN_ELEMENTS = 4 * 1024 * 1024


def as_c_float32(values: np.ndarray) -> np.ndarray:
    """Return a C-contiguous float32 array, copying only when required."""
    return np.ascontiguousarray(values, dtype=np.float32)


def as_c_uint8(values: np.ndarray) -> np.ndarray:
    """Return a compact C-contiguous validity array."""
    return np.ascontiguousarray(values, dtype=np.uint8)


@njit(cache=True, nogil=True)
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


@njit(cache=True, nogil=True)
def _calibrate_inplace_rgb_hwc(
    data: np.ndarray,
    dark: np.ndarray,
    flat: np.ndarray,
    use_dark: bool,
    use_flat: bool,
) -> None:
    height, width, channels = data.shape
    for y in range(height):
        for x in range(width):
            for channel in range(channels):
                value = data[y, x, channel]
                if use_dark:
                    value = value - dark[y, x, channel]
                if use_flat:
                    flat_value = flat[y, x, channel]
                    if np.isfinite(flat_value) and flat_value > np.float32(0.01):
                        value = value / flat_value
                data[y, x, channel] = value


@njit(cache=True, nogil=True)
def _calibrate_inplace_rgb_chw(
    data: np.ndarray,
    dark: np.ndarray,
    flat: np.ndarray,
    use_dark: bool,
    use_flat: bool,
) -> None:
    channels, height, width = data.shape
    for channel in range(channels):
        for y in range(height):
            for x in range(width):
                value = data[channel, y, x]
                if use_dark:
                    value = value - dark[channel, y, x]
                if use_flat:
                    flat_value = flat[channel, y, x]
                    if np.isfinite(flat_value) and flat_value > np.float32(0.01):
                        value = value / flat_value
                data[channel, y, x] = value


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
        else np.empty((0,) * data.ndim, dtype=np.float32)
    )
    flat = (
        as_c_float32(master_flat)
        if master_flat is not None
        else np.empty((0,) * data.ndim, dtype=np.float32)
    )
    use_dark = master_dark is not None
    use_flat = master_flat is not None
    if data.ndim == 2:
        _calibrate_inplace(data, dark, flat, use_dark, use_flat)
    elif data.ndim == 3:
        channels_first = data.shape[0] in (3, 4) and data.shape[-1] not in (3, 4)
        if channels_first:
            _calibrate_inplace_rgb_chw(data, dark, flat, use_dark, use_flat)
        else:
            _calibrate_inplace_rgb_hwc(data, dark, flat, use_dark, use_flat)
    else:
        raise ValueError("calibration data must be mono 2D or RGB 3D")
    return data


@njit(cache=True, nogil=True)
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


@njit(cache=True, parallel=True, nogil=True)
def _masked_extrema_parallel(
    values: np.ndarray, masks: np.ndarray, maximum: bool
) -> np.ndarray:
    """Independent-pixel extrema with stable frame traversal.

    ``prange`` distributes output rows, while the frame loop remains in the
    same order as ``_masked_extrema``.  This preserves bitwise selection
    semantics and lets Numba/LLVM use the host SIMD path for large planes.
    """

    frames, height, width = values.shape
    result = np.empty((height, width), dtype=np.float32)
    for y in prange(height):
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


def masked_extrema(
    values: np.ndarray,
    masks: np.ndarray,
    maximum: bool,
    parallel: bool = False,
) -> np.ndarray:
    """Bitwise-safe masked min/max: selections, not reordered reductions."""
    prepared_values = as_c_float32(values)
    prepared_masks = as_c_uint8(masks)
    if parallel and prepared_values.size >= _PARALLEL_MIN_ELEMENTS:
        return _masked_extrema_parallel(prepared_values, prepared_masks, maximum)
    return _masked_extrema(prepared_values, prepared_masks, maximum)


@njit(cache=True, nogil=True)
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


@njit(cache=True, parallel=True, nogil=True)
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
    if parallel and prepared_values.size >= _PARALLEL_MIN_ELEMENTS:
        return _masked_sum_count_parallel(prepared_values, prepared_masks)
    return _masked_sum_count(prepared_values, prepared_masks)


@njit(cache=True, nogil=True)
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


@njit(cache=True, parallel=True, nogil=True)
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


@njit(cache=True, parallel=True, nogil=True)
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


@njit(cache=True, nogil=True)
def _weighted_merge_mono_serial(values: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """Serial counterpart with the same leaf traversal as the parallel path."""

    leaves, height, width = values.shape
    result = np.zeros((height, width), dtype=np.float32)
    for y in range(height):
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


@njit(cache=True, nogil=True)
def _weighted_merge_rgb_serial(values: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """Serial RGB merge used below the parallel launch threshold."""

    leaves, channels, height, width = values.shape
    result = np.zeros((channels, height, width), dtype=np.float32)
    for y in range(height):
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
        if values.size >= _PARALLEL_MIN_ELEMENTS:
            return _weighted_merge_mono(values, counts)
        return _weighted_merge_mono_serial(values, counts)
    if values.ndim == 4:
        if values.size >= _PARALLEL_MIN_ELEMENTS:
            return _weighted_merge_rgb(values, counts)
        return _weighted_merge_rgb_serial(values, counts)
    raise ValueError("substack values must be mono or RGB")


def warm_cpu_kernels() -> None:
    """Compile representative signatures in a background startup worker."""
    configure_opencv_threads(1)
    configure_numba_threads()
    sample = np.ones((4, 4), dtype=np.float32)
    calibrate_inplace(sample.copy(), sample, sample)
    values = np.ones((2, 4, 4), dtype=np.float32)
    masks = np.ones(values.shape, dtype=np.uint8)
    masked_extrema(values, masks, True)
    # Cross the production threshold so the cached parallel dispatchers are
    # compiled during startup rather than on the first user stack.
    parallel_values = np.ones((16, 512, 512), dtype=np.float32)
    parallel_masks = np.ones(parallel_values.shape, dtype=np.uint8)
    _masked_extrema_parallel(parallel_values, parallel_masks, True)
    masked_sum_count(values, masks)
    masked_sum_count(parallel_values, parallel_masks, parallel=True)
    apply_scale_and_mask_inplace(sample.copy(), np.ones(sample.shape, dtype=np.uint8), 1.0)
    weighted_merge(values, np.ones((2, 4, 4), dtype=np.uint32))
