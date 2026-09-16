"""Production bounded-RAM reducer for the Intelligent Stack profile.

The Stable path keeps the disk-backed four-leaf topology for bit-for-bit
compatibility, even when Intelligent features are enabled.  Fast may use
contiguous bounded leaves, while both paths merge deterministically with a
binary-carry tree.  FITS files are opened read-only and only the final uint16
product plus its sidecars are written.
"""

from __future__ import annotations

from dataclasses import dataclass
import gc
import json
import math
import os
from pathlib import Path
import shutil
import time
from typing import Any, Callable
import uuid

import numpy as np
from astropy.io import fits

import stacking_logic as stack
from app.engines import EngineProfile, ExecutionBudget
from cpu_kernels import apply_scale_and_mask_inplace
from cpu_runtime import configure_numba_threads, configure_opencv_threads


MIB = 1024**2
GIB = 1024**3
MIN_SYSTEM_HEADROOM = 4 * GIB
MIN_LEAF_FRAMES = 3
RAM_LEAF_FRAMES = 16


@dataclass(slots=True)
class RamNode:
    data: np.ndarray
    valid_mask: np.ndarray
    coverage: np.ndarray
    frame_count: int
    weight_sums: np.ndarray | None = None
    order: int = 0


@dataclass(frozen=True, slots=True)
class RamTelemetry:
    requested_mb: int
    effective_mb: int
    initial_available_mb: int | None
    total_memory_mb: int | None
    minimum_available_mb: int | None
    peak_rss_mb: int | None
    estimated_peak_working_mb: int
    leaf_sizes: tuple[int, ...]

    def as_dict(self) -> dict[str, Any]:
        return {
            "memory_budget_requested_mb": self.requested_mb,
            "memory_budget_effective_mb": self.effective_mb,
            "initial_available_mb": self.initial_available_mb,
            "total_memory_mb": self.total_memory_mb,
            "minimum_available_mb": self.minimum_available_mb,
            "peak_process_rss_mb": self.peak_rss_mb,
            "estimated_peak_working_mb": self.estimated_peak_working_mb,
            "leaf_sizes": list(self.leaf_sizes),
            "leaf_count": len(self.leaf_sizes),
        }


def _memory_state() -> tuple[int, int, int] | None:
    try:
        import psutil

        vm = psutil.virtual_memory()
        rss = psutil.Process(os.getpid()).memory_info().rss
        return int(vm.available), int(vm.total), int(rss)
    except Exception:
        return None


def _effective_budget(requested_mb: int) -> tuple[int, int | None, int | None]:
    # The command layer accepts 64 MiB as the smallest explicit budget.  Do
    # not silently round a user's lower limit up to 1 GiB: honoring it lets
    # the preflight guards return a clean ``memory_insufficient`` result
    # instead of allocating beyond the requested ceiling.
    requested = max(64, int(requested_mb)) * MIB
    state = _memory_state()
    if state is None:
        return requested, None, None
    available, total, _rss = state
    reserve = max(MIN_SYSTEM_HEADROOM, int(total * 0.20))
    safe_available = max(0, available - reserve)
    return min(requested, int(safe_available * 0.75)), available, total


def _source_snapshot(paths: list[Path], flow_root: Path | None = None) -> dict[str, tuple[int, int]]:
    snapshot: dict[str, tuple[int, int]] = {}
    all_paths = list(paths)
    if flow_root is not None and flow_root.is_dir():
        all_paths.extend(flow_root.glob("batch_*/flow_local.json"))
        global_flow = flow_root / "global_flow.json"
        if global_flow.exists():
            all_paths.append(global_flow)
    for path in all_paths:
        try:
            stat = path.stat()
        except OSError:
            continue
        snapshot[str(path)] = (int(stat.st_size), int(stat.st_mtime_ns))
    return snapshot


def _bytes_per_frame(pixels: int, channels: int) -> int:
    # Input, corrected band, masks, rejection temporaries and allocator slack.
    return int(pixels * max(1, channels) * 20)


def _node_bytes(shape: tuple[int, ...]) -> int:
    pixels = int(np.prod(shape, dtype=np.int64))
    return pixels * 4 + (pixels // max(1, shape[0]) if len(shape) == 3 else pixels) * 1 + pixels * 4


def _merge_working_bytes(shape: tuple[int, ...], config: stack.StackingConfig) -> int:
    """Conservative transient estimate for one binary node merge.

    ``_combine_nodes`` keeps both child nodes alive while materialising the
    stacked values/masks and the output.  Rejection methods can also allocate
    a masked copy plus centre/dispersion arrays.  The estimate intentionally
    over-approximates that peak so a small user budget fails before NumPy can
    ask the OS for an unbounded allocation.
    """

    pixels = int(np.prod(shape, dtype=np.int64))
    channels = int(shape[0]) if len(shape) == 3 else 1
    plane_pixels = pixels if len(shape) == 2 else pixels // max(1, channels)
    # Two child data planes, stacked values, rejection copy, output and (for
    # QWM) weighted numerator/denominator.  Masks and coverage add roughly
    # one byte and four bytes per spatial pixel, respectively.
    data_factor = 6 if config.method == "QualityWeightedMean" else 5
    return int(
        plane_pixels * channels * 4 * data_factor
        + plane_pixels * 2 * 4  # stacked masks/coverage and output coverage
        + plane_pixels * 2  # valid-mask byte arrays
    )


def _band_rows(config: stack.StackingConfig, group_size: int, width: int, channels: int, height: int, budget: int) -> int:
    bytes_per_row = max(1, group_size * width * max(1, channels) * stack.STACK_BAND_BYTES_PER_PIXEL)
    workers = max(1, min(config.worker_count, 2))
    usable = max(8 * MIB, int(budget * 0.45))
    rows = int(usable / max(1, workers * bytes_per_row))
    return max(16, min(height, rows))


def _guard(required: int, budget: int) -> tuple[int | None, int | None]:
    state = _memory_state()
    if state is None:
        if required > int(budget * 0.90):
            raise MemoryError("estimativa da folha excede o orçamento RAM")
        return None, None
    available, total, rss = state
    reserve = max(MIN_SYSTEM_HEADROOM, int(total * 0.20))
    if available - reserve < int(required * 1.10):
        raise MemoryError(
            f"RAM segura insuficiente: requer {required / GIB:.2f} GiB, "
            f"disponível {(available - reserve) / GIB:.2f} GiB"
        )
    if required > int(budget * 0.90):
        raise MemoryError("estimativa da folha excede 90% do orçamento RAM")
    return int(available), int(rss)


def _read_band(
    frame: stack.FrameInfo,
    hdul: fits.HDUList,
    geometry: stack.FrameGeometry,
    y1: int,
    y2: int,
    channel: int,
    factor: float,
    shift: tuple[float, float] | None,
    engine_profile: str = "Stable",
) -> stack.BlockRead:
    width = geometry.width
    if shift is not None:
        return stack.read_frame_block(frame, geometry, y1, y2, 0, width, channel, factor, shift)
    hdu = hdul[geometry.hdu_index]
    if geometry.image_kind == "Mono":
        raw = stack._read_hdu_section(hdu, (slice(y1, y2), slice(0, width)))
    elif frame.path.suffix.casefold() not in stack.TIFF_SUFFIXES:
        raw = stack._read_hdu_section(hdu, (slice(channel, channel + 1), slice(y1, y2), slice(0, width)))[0]
    else:
        raw = stack._read_hdu_section(hdu, (slice(y1, y2), slice(0, width), slice(channel, channel + 1)))[:, :, 0]
    raw = np.ascontiguousarray(
        stack._restore_streaming_physical_values(raw, geometry),
        dtype=np.float32,
    )
    mask = np.isfinite(raw)
    if frame.valid_mask is not None:
        mask &= np.asarray(frame.valid_mask[y1:y2, :], dtype=bool)
    if EngineProfile.coerce(engine_profile) is EngineProfile.FAST:
        apply_scale_and_mask_inplace(raw, mask, factor)
    else:
        if factor != 1.0:
            raw *= np.float32(factor)
        raw[~mask] = np.float32(0.0)
    return stack.BlockRead(np.asarray(raw, dtype=np.float32), np.asarray(mask, dtype=bool))


def _read_mask_band(frame: stack.FrameInfo, hdul: fits.HDUList, geometry: stack.FrameGeometry, y1: int, y2: int, width: int) -> np.ndarray | None:
    if frame.valid_mask is not None:
        return np.asarray(frame.valid_mask[y1:y2, :], dtype=bool)
    if frame.path.suffix.casefold() in stack.TIFF_SUFFIXES:
        from image_io import read_tiff_sidecar_arrays
        arrays = read_tiff_sidecar_arrays(frame.path)
        valid = arrays.get("valid_mask")
        sat = arrays.get("sat_mask")
        result = None
        if valid is not None:
            result = np.asarray(valid[y1:y2, :], dtype=bool)
        if sat is not None:
            sat_values = ~np.asarray(sat[y1:y2, :], dtype=bool)
            result = sat_values if result is None else result & sat_values
        return result
    if geometry.mask_hdu_index is None:
        return None
    return stack._read_mask_section(hdul[geometry.mask_hdu_index], y1, y2, 0, width)


def _combine_leaf(
    values: np.ndarray,
    masks: np.ndarray,
    config: stack.StackingConfig,
    frame_weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray | None]:
    if config.method == "QualityWeightedMean":
        return stack._quality_weighted_reduce(
            values,
            masks,
            config.rejection_method,
            config.rejection_low,
            config.rejection_high,
            frame_weights,
        )
    return (
        stack.reject_and_combine_block(
            values,
            masks,
            config.method,
            config.rejection_method,
            config.rejection_low,
            config.rejection_high,
            engine_profile=config.engine_profile,
            reducer_engine=config.reducer_engine,
            kernel_parallel=False,
        ),
        None,
    )


def _build_leaf(
    frames: list[stack.FrameInfo],
    geometries: dict[Path, stack.FrameGeometry],
    factors: list[float],
    shifts: list[tuple[float, float] | None],
    frame_weights: list[float],
    config: stack.StackingConfig,
    height: int,
    width: int,
    channels: int,
    budget: int,
    leaf_index: int,
    progress: Callable[[str], None] | None = None,
) -> tuple[RamNode, dict[str, int]]:
    group_size = len(frames)
    node_shape = (height, width) if channels == 1 else (channels, height, width)
    band_rows = _band_rows(config, group_size, width, channels, height, budget)
    # The reducer streams row bands; sizing the guard from the whole frame
    # would unnecessarily split the Legacy four-leaf topology on large
    # sensors. Account for the actual in-flight band plus two output buffers.
    estimate = _bytes_per_frame(band_rows * width, channels) * group_size
    available, rss = _guard(estimate + _node_bytes(node_shape) * 2, budget)
    result = np.zeros(node_shape, dtype=np.float32)
    counts = np.zeros((height, width), dtype=np.uint32)
    weight_sums = np.zeros(node_shape, dtype=np.float32) if config.method == "QualityWeightedMean" else None
    handles: list[fits.HDUList] = []
    try:
        for frame in frames:
            handles.append(stack._open_streaming_fits(frame.path, geometries[frame.path]))
        for band_index, y1 in enumerate(range(0, height, band_rows), start=1):
            if progress is not None:
                progress(f"RAM folha {leaf_index + 1}: banda {band_index}")
            y2 = min(height, y1 + band_rows)
            band_masks = [
                _read_mask_band(frame, hdul, geometries[frame.path], y1, y2, width)
                for frame, hdul in zip(frames, handles, strict=True)
            ]
            values = np.empty((group_size, channels, y2 - y1, width), dtype=np.float32)
            masks = np.empty(values.shape, dtype=bool)
            for index, (frame, hdul) in enumerate(zip(frames, handles, strict=True)):
                for channel in range(channels):
                    block = _read_band(
                        frame,
                        hdul,
                        geometries[frame.path],
                        y1,
                        y2,
                        channel,
                        factors[index],
                        shifts[index],
                        config.engine_profile,
                    )
                    if band_masks[index] is not None:
                        block.mask &= band_masks[index]
                        block.data[~block.mask] = np.float32(0.0)
                    values[index, channel] = block.data
                    masks[index, channel] = block.mask
            for channel in range(channels):
                combined, effective = _combine_leaf(
                    values[:, channel],
                    masks[:, channel],
                    config,
                    np.asarray(frame_weights, dtype=np.float32),
                )
                if channels == 1:
                    result[y1:y2] = combined
                    if effective is not None:
                        weight_sums[y1:y2] = effective
                else:
                    result[channel, y1:y2] = combined
                    if effective is not None:
                        weight_sums[channel, y1:y2] = effective
            counts[y1:y2] = masks[:, 0].sum(axis=0, dtype=np.uint32)
            del values, masks, band_masks
    finally:
        for handle in handles:
            handle.close()
    coverage = counts
    return (
        RamNode(
            data=result,
            valid_mask=coverage > 0,
            coverage=coverage,
            frame_count=group_size,
            weight_sums=weight_sums,
            order=leaf_index,
        ),
        {
            "available": int(available or -1),
            "rss": int(rss or -1),
            "estimate": int(estimate),
        },
    )


def _combine_nodes(left: RamNode, right: RamNode, config: stack.StackingConfig) -> RamNode:
    values = np.stack((left.data, right.data), axis=0)
    masks = np.stack((left.valid_mask, right.valid_mask), axis=0)
    total = left.frame_count + right.frame_count
    # The legacy Stable reducer combines an un-rejected mean by the number of
    # valid source frames represented by each child, rather than by the number
    # of child nodes.  Reuse that exact arithmetic in the RAM tree; a plain
    # two-node mean would otherwise give RGB/partial-mask stacks a different
    # float32 rounding path even when the logical tree is identical.
    if config.method == "Mean" and config.rejection_method == "None":
        child_counts = np.stack((left.coverage, right.coverage), axis=0)
        if values.ndim == 4:
            weights = child_counts[:, None, :, :].astype(np.float32)
            denominator = np.sum(weights, axis=0, dtype=np.float32)
        else:
            weights = child_counts.astype(np.float32)
            denominator = np.sum(weights, axis=0, dtype=np.float32)
        numerator = np.sum(values * weights, axis=0, dtype=np.float32)
        result = np.divide(
            numerator,
            np.maximum(denominator, np.float32(1.0)),
            out=np.zeros_like(numerator, dtype=np.float32),
        )
        result_weights = None
    elif config.method == "QualityWeightedMean":
        if left.weight_sums is None or right.weight_sums is None:
            raise ValueError("nó sem weight_sums na média ponderada")
        weights = np.stack((left.weight_sums, right.weight_sums), axis=0)
        if values.ndim == 4:
            result = np.empty(values.shape[1:], dtype=np.float32)
            result_weights = np.empty(values.shape[1:], dtype=np.float32)
            valid = np.broadcast_to(masks[:, None, :, :], values.shape)
            for channel in range(values.shape[1]):
                result[channel], result_weights[channel] = stack._quality_weighted_merge(
                    values[:, channel],
                    valid[:, channel],
                    weights[:, channel],
                    config.rejection_method,
                    config.rejection_low,
                    config.rejection_high,
                )
        else:
            result, result_weights = stack._quality_weighted_merge(
                values,
                masks,
                weights,
                config.rejection_method,
                config.rejection_low,
                config.rejection_high,
            )
    else:
        if values.ndim == 4:
            result = np.empty(values.shape[1:], dtype=np.float32)
            valid = np.broadcast_to(masks[:, None, :, :], values.shape)
            for channel in range(values.shape[1]):
                result[channel] = stack.reject_and_combine_block(
                    values[:, channel],
                    valid[:, channel],
                    config.method,
                    config.rejection_method,
                    config.rejection_low,
                    config.rejection_high,
                    engine_profile=config.engine_profile,
                    reducer_engine=config.reducer_engine,
                    kernel_parallel=False,
                )
        else:
            result = stack.reject_and_combine_block(
                values,
                masks,
                config.method,
                config.rejection_method,
                config.rejection_low,
                config.rejection_high,
                engine_profile=config.engine_profile,
                reducer_engine=config.reducer_engine,
                kernel_parallel=False,
            )
        result_weights = None
    coverage = left.coverage + right.coverage
    valid_mask = coverage >= max(1, math.ceil(total * 0.70))
    del values, masks
    if config.method == "QualityWeightedMean":
        del weights
    return RamNode(
        data=np.asarray(result, dtype=np.float32),
        valid_mask=np.asarray(valid_mask, dtype=bool),
        coverage=np.asarray(coverage, dtype=np.uint32),
        frame_count=total,
        weight_sums=result_weights,
        order=min(left.order, right.order),
    )


def _load_node_reference(reference: RamNode | Path) -> RamNode:
    if isinstance(reference, RamNode):
        return reference
    with np.load(reference, allow_pickle=False) as payload:
        weights = payload["weight_sums"] if "weight_sums" in payload.files else None
        node = RamNode(
            data=np.asarray(payload["data"], dtype=np.float32).copy(),
            valid_mask=np.asarray(payload["valid_mask"], dtype=bool).copy(),
            coverage=np.asarray(payload["coverage"], dtype=np.uint32).copy(),
            frame_count=int(payload["frame_count"]),
            weight_sums=np.asarray(weights, dtype=np.float32).copy() if weights is not None else None,
            order=int(payload["order"]),
        )
    try:
        reference.unlink(missing_ok=True)
    except OSError:
        pass
    return node


def _write_spill_node(node: RamNode, directory: Path, index: int, limit_bytes: int) -> tuple[Path, int]:
    estimated = int(node.data.nbytes + node.valid_mask.nbytes + node.coverage.nbytes)
    if node.weight_sums is not None:
        estimated += int(node.weight_sums.nbytes)
    if estimated > limit_bytes:
        raise MemoryError("limite de spill insuficiente para uma folha")
    path = directory / f"node_{index:06d}.npz"
    temporary = path.with_suffix(path.suffix + ".tmp")
    kwargs: dict[str, Any] = {
        "data": np.asarray(node.data, dtype=np.float32),
        "valid_mask": np.asarray(node.valid_mask, dtype=np.uint8),
        "coverage": np.asarray(node.coverage, dtype=np.uint32),
        "frame_count": np.asarray(node.frame_count, dtype=np.int64),
        "order": np.asarray(node.order, dtype=np.int64),
    }
    if node.weight_sums is not None:
        kwargs["weight_sums"] = np.asarray(node.weight_sums, dtype=np.float32)
    np.savez_compressed(temporary, **kwargs)
    # numpy appends .npz when the suffix is not already .npz; our temporary
    # name ends in .tmp, so resolve the actual file before replacing.
    generated = temporary if temporary.exists() else Path(str(temporary) + ".npz")
    actual_size = generated.stat().st_size
    if actual_size > limit_bytes:
        generated.unlink(missing_ok=True)
        raise MemoryError("limite de spill excedido")
    os.replace(generated, path)
    return path, int(actual_size)


def _reduce_nodes(
    nodes: list[RamNode | Path],
    config: stack.StackingConfig,
    budget: int | None = None,
) -> RamNode:
    """Binary-carry reduction; logical order is independent of completion order."""

    levels: list[RamNode | None] = []
    for reference in nodes:
        carry = _load_node_reference(reference)
        level = 0
        while True:
            if level == len(levels):
                levels.append(carry)
                break
            previous = levels[level]
            if previous is None:
                levels[level] = carry
                break
            levels[level] = None
            if budget is not None:
                _guard(_merge_working_bytes(previous.data.shape, config), budget)
            carry = _combine_nodes(previous, carry, config)
            del previous
            gc.collect()
            level += 1
    root: RamNode | None = None
    for node in reversed(levels):
        if node is None:
            continue
        if root is None:
            root = node
        else:
            if budget is not None:
                _guard(_merge_working_bytes(node.data.shape, config), budget)
            # Binary-carry leaves are stored with the lowest-order (rightmost)
            # remainder at the lower levels.  The high-level node therefore
            # precedes ``node`` in logical frame order; keep it on the left so
            # non-associative rejection methods see the same sequence as the
            # disk reducer.
            root = _combine_nodes(root, node, config)
        gc.collect()
    if root is None:
        raise ValueError("nenhum nó RAM produzido")
    return root


def _push_ram_node(
    levels: list[RamNode | None],
    carry: RamNode,
    config: stack.StackingConfig,
    budget: int | None = None,
) -> None:
    """Insert one in-memory leaf while retaining one node per tree level.

    Keeping the binary-carry invariant during leaf construction prevents a
    RAM-only run from retaining every full-resolution leaf simultaneously.
    The insertion order is still the logical frame order, so the resulting
    tree is identical to :func:`_reduce_nodes` and remains bitwise Stable.
    """

    def check_resident(extra: RamNode) -> None:
        if budget is None:
            return
        resident = sum(
            _node_bytes(node.data.shape)
            for node in levels
            if node is not None
        )
        _guard(resident + _node_bytes(extra.data.shape), budget)

    level = 0
    while True:
        if level == len(levels):
            check_resident(carry)
            levels.append(carry)
            return
        previous = levels[level]
        if previous is None:
            check_resident(carry)
            levels[level] = carry
            return
        levels[level] = None
        if budget is not None:
            resident = sum(
                _node_bytes(node.data.shape)
                for node in levels
                if node is not None
            )
            required = resident + _node_bytes(carry.data.shape) + _merge_working_bytes(
                carry.data.shape, config
            )
            _guard(required, budget)
        carry = _combine_nodes(previous, carry, config)
        del previous
        gc.collect()
        level += 1


def _finish_ram_levels(
    levels: list[RamNode | None],
    config: stack.StackingConfig,
    budget: int | None = None,
) -> RamNode:
    """Finish an online binary-carry tree in deterministic left-to-right order."""

    root: RamNode | None = None
    # Clear each consumed slot before checking the resident set.  Keeping the
    # pointer in ``levels`` while it is also held by ``root`` would count the
    # same allocation twice and could reject a run that actually fits the
    # configured budget.
    for index in range(len(levels) - 1, -1, -1):
        node = levels[index]
        levels[index] = None
        if node is None:
            continue
        if root is None:
            root = node
            continue
        if budget is not None:
            resident = sum(
                _node_bytes(candidate.data.shape)
                for candidate in levels
                if candidate is not None
            )
            required = resident + _node_bytes(root.data.shape) + _merge_working_bytes(
                node.data.shape, config
            )
            _guard(required, budget)
        # ``root`` covers the earlier frames and ``node`` is the later
        # remainder from a lower tree level.  Reversing these operands would
        # preserve a commutative mean but change Stable Median/SigmaClip
        # boundaries for non-power-of-two leaf counts.
        root = _combine_nodes(root, node, config)
        gc.collect()
    if root is None:
        raise ValueError("nenhum nó RAM produzido")
    return root


def run_ram_stack(
    *,
    selected_frames: list[stack.FrameInfo],
    all_frames: list[stack.FrameInfo],
    geometries: dict[Path, stack.FrameGeometry],
    normalization_factors: list[float],
    dither_shifts: list[tuple[float, float] | None],
    frame_weights: list[float],
    config: stack.StackingConfig,
    report_path: Path,
    total_frames: int,
    n_batches: int,
    avg_quality: float,
    avg_star_count: float,
    avg_fwhm: float,
    progress_callback: Callable[[int, int, str], None] | None = None,
    status_callback: Callable[[str], None] | None = None,
    cancel_event: Any = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    source_paths = [frame.path for frame in all_frames]
    source_before = _source_snapshot(source_paths, config.base_dir)
    effective, initial_available, total_memory = _effective_budget(config.memory_budget_mb)
    if effective <= 0:
        return {"status": "error", "reason": "memory_insufficient", "selection_report": str(report_path)}
    configure_opencv_threads(1)
    configure_numba_threads(1)
    reference = selected_frames[0]
    geometry = geometries[reference.path]
    height, width, channels = geometry.height, geometry.width, geometry.channels
    shape = (height, width) if channels == 1 else (channels, height, width)
    minimum_band = _band_rows(config, MIN_LEAF_FRAMES, width, channels, height, effective)
    minimum = _bytes_per_frame(minimum_band * width, channels) * MIN_LEAF_FRAMES + _node_bytes(shape) * 2
    if minimum > int(effective * 0.90):
        return {
            "status": "error",
            "reason": "memory_insufficient",
            "required_memory_mb": int(math.ceil(minimum / MIB)),
            "selection_report": str(report_path),
        }
    merge_minimum = _merge_working_bytes(shape, config)
    if merge_minimum > int(effective * 0.90):
        return {
            "status": "error",
            "reason": "memory_insufficient_for_merge",
            "required_memory_mb": int(math.ceil(merge_minimum / MIB)),
            "selection_report": str(report_path),
        }
    if config.reduction_storage == "ram_spill" and config.spill_directory is None:
        return {"status": "error", "reason": "spill_directory_required", "selection_report": str(report_path)}
    if config.reduction_storage == "ram_spill":
        spill = Path(config.spill_directory).resolve()
        if spill == config.input_dir.resolve() or config.input_dir.resolve() in spill.parents:
            return {"status": "error", "reason": "spill_inside_input", "selection_report": str(report_path)}
        spill_session = spill / f".astrostack-spill-{os.getpid()}-{uuid.uuid4().hex}"
        spill_session.mkdir(parents=True, exist_ok=False)
        spill_limit_bytes = int(config.spill_limit_mb) * MIB
    else:
        spill_session = None
        spill_limit_bytes = 0
    stable_engine = EngineProfile.coerce(config.engine_profile) is EngineProfile.STABLE
    if config.feature_profile == "Legacy" or stable_engine:
        # Match the historical _create_substacks topology exactly: four
        # interleaved leaves followed by the same left-to-right tree. This is
        # required for Stable bit-for-bit compatibility even with the new
        # Intelligent feature policy; changing only reduction storage must not
        # change the rejection boundaries or float32 accumulation order.
        leaf_count = min(stack.DETERMINISTIC_LEAF_COUNT, len(selected_frames))
        groups = [
            (
                selected_frames[index::leaf_count],
                normalization_factors[index::leaf_count],
                dither_shifts[index::leaf_count],
                frame_weights[index::leaf_count],
            )
            for index in range(leaf_count)
        ]
    else:
        groups = [
            (
                selected_frames[index : index + RAM_LEAF_FRAMES],
                normalization_factors[index : index + RAM_LEAF_FRAMES],
                dither_shifts[index : index + RAM_LEAF_FRAMES],
                frame_weights[index : index + RAM_LEAF_FRAMES],
            )
            for index in range(0, len(selected_frames), RAM_LEAF_FRAMES)
        ]
    pending_groups = list(groups)
    initial_leaf_count = len(pending_groups)
    nodes: list[RamNode | Path] = []
    ram_levels: list[RamNode | None] = []
    spill_bytes = 0
    spill_index = 0
    leaf_sizes: list[int] = []
    leaf_estimates: list[int] = []
    minimum_available = initial_available
    peak_rss = 0
    processed_leaves = 0
    while pending_groups:
        group, factors, shifts, weights = pending_groups.pop(0)
        if cancel_event is not None and cancel_event.is_set():
            if spill_session is not None:
                shutil.rmtree(spill_session, ignore_errors=True)
            return {"status": "cancelled", "selection_report": str(report_path)}
        while True:
            try:
                node, stats = _build_leaf(
                    group,
                    geometries,
                    factors,
                    shifts,
                    weights,
                    config,
                    height,
                    width,
                    channels,
                    effective,
                    processed_leaves,
                    lambda message: status_callback(message + "\n") if status_callback else None,
                )
                break
            except MemoryError as exc:
                if len(group) <= MIN_LEAF_FRAMES:
                    if config.reduction_storage == "ram_spill":
                        if spill_session is not None:
                            shutil.rmtree(spill_session, ignore_errors=True)
                        return {
                            "status": "error",
                            "reason": "memory_insufficient_even_with_spill",
                            "error": str(exc),
                            "selection_report": str(report_path),
                        }
                    return {
                        "status": "error",
                        "reason": "memory_insufficient",
                        "error": str(exc),
                        "selection_report": str(report_path),
                    }
                new_size = max(MIN_LEAF_FRAMES, len(group) // 2)
                if new_size >= len(group):
                    return {
                        "status": "error",
                        "reason": "memory_insufficient",
                        "error": str(exc),
                        "selection_report": str(report_path),
                    }
                # Split the work instead of truncating the frame list.  Both
                # halves remain in logical order and are merged by the same
                # deterministic binary-carry tree.
                pending_groups.insert(
                    0,
                    (
                        group[new_size:],
                        factors[new_size:],
                        shifts[new_size:],
                        weights[new_size:],
                    ),
                )
                pending_groups.insert(0, (group[:new_size], factors[:new_size], shifts[:new_size], weights[:new_size]))
                group = []
                break
            except Exception:
                # FITS/read/decompression failures must not leave a partially
                # populated spill session behind.  The source tree remains
                # untouched; only this run's explicitly selected spill
                # directory is removed.
                if spill_session is not None:
                    shutil.rmtree(spill_session, ignore_errors=True)
                raise
        if not group:
            continue
        processed_leaves += 1
        leaf_frame_count = node.frame_count
        if spill_session is not None:
            try:
                spill_path, written = _write_spill_node(
                    node,
                    spill_session,
                    spill_index,
                    max(1, spill_limit_bytes - spill_bytes),
                )
            except (MemoryError, OSError) as exc:
                shutil.rmtree(spill_session, ignore_errors=True)
                return {
                    "status": "error",
                    "reason": "spill_limit_exceeded",
                    "error": str(exc),
                    "selection_report": str(report_path),
                }
            nodes.append(spill_path)
            spill_bytes += written
            spill_index += 1
            del node
            gc.collect()
        else:
            # RAM-only reductions consume each leaf immediately.  This keeps
            # at most one resident node per binary-tree level instead of
            # retaining all full-resolution leaves until the final merge.
            try:
                _push_ram_node(ram_levels, node, config, effective)
            except MemoryError as exc:
                del node
                return {
                    "status": "error",
                    "reason": "memory_insufficient",
                    "error": str(exc),
                    "selection_report": str(report_path),
                }
            del node
        leaf_sizes.append(leaf_frame_count)
        leaf_estimates.append(int(stats.get("estimate", 0)))
        if stats["available"] >= 0:
            minimum_available = stats["available"] if minimum_available is None else min(minimum_available, stats["available"])
        peak_rss = max(peak_rss, stats["rss"])
        if progress_callback:
            progress_callback(
                processed_leaves,
                max(initial_leaf_count, processed_leaves + len(pending_groups)),
                f"RAM: folha {processed_leaves}",
            )
    if sum(leaf_sizes) != len(selected_frames):
        if spill_session is not None:
            shutil.rmtree(spill_session, ignore_errors=True)
        return {
            "status": "error",
            "reason": "frame_partition_mismatch",
            "selection_report": str(report_path),
        }
    try:
        if spill_session is None:
            root = _finish_ram_levels(ram_levels, config, effective)
        else:
            root = _reduce_nodes(nodes, config, effective)
    except MemoryError as exc:
        if spill_session is not None:
            shutil.rmtree(spill_session, ignore_errors=True)
        return {
            "status": "error",
            "reason": "memory_insufficient",
            "error": str(exc),
            "selection_report": str(report_path),
        }
    except Exception:
        if spill_session is not None:
            shutil.rmtree(spill_session, ignore_errors=True)
        raise
    if spill_session is not None:
        # References are consumed and unlinked while loading; this removes
        # the now-empty session directory defensively.
        shutil.rmtree(spill_session, ignore_errors=True)
    root.valid_mask = root.coverage >= max(1, math.ceil(len(selected_frames) * 0.70))
    if root.data.ndim == 3:
        root.data[:, ~root.valid_mask] = 0.0
    else:
        root.data[~root.valid_mask] = 0.0
    root.data = np.nan_to_num(root.data, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
    if config.remove_background:
        # Keep the same scientific implementation as the legacy path, but
        # run it while the root is already resident so no intermediate FITS
        # plane is needed.
        root.data = stack.flatten_background(root.data, root.valid_mask)
        if root.data.ndim == 3:
            root.data[:, ~root.valid_mask] = 0.0
        else:
            root.data[~root.valid_mask] = 0.0
    header = stack.prepare_output_header(
        stack.load_source_header(reference.path),
        config,
        len(selected_frames),
        len(all_frames),
        n_batches,
        avg_quality,
        avg_star_count,
        avg_fwhm,
    )
    header["STACK_RAM"] = True
    header["STACK_STOR"] = config.reduction_storage
    header["STACK_WGTP"] = config.selection_profile
    header["STACK_WMIN"] = float(min(frame_weights) if frame_weights else 1.0)
    header["STACK_WMED"] = float(np.median(frame_weights) if frame_weights else 1.0)
    header["STACK_WMAX"] = float(max(frame_weights) if frame_weights else 1.0)
    output_path = config.output_dir / config.output_name
    output_data = stack.convert_output_dtype(root.data, config)
    stack.write_stack_output(output_path, output_data, root.valid_mask, header, config)
    final_state = _memory_state()
    if final_state is not None:
        available, _total, rss = final_state
        minimum_available = available if minimum_available is None else min(minimum_available, available)
        peak_rss = max(peak_rss, rss)
    source_after = _source_snapshot(source_paths, config.base_dir)
    telemetry = RamTelemetry(
        requested_mb=int(config.memory_budget_mb),
        effective_mb=int(effective / MIB),
        initial_available_mb=int(initial_available / MIB) if initial_available is not None else None,
        total_memory_mb=int(total_memory / MIB) if total_memory is not None else None,
        minimum_available_mb=int(minimum_available / MIB) if minimum_available is not None else None,
        peak_rss_mb=int(peak_rss / MIB) if peak_rss else None,
        estimated_peak_working_mb=int(math.ceil(max(leaf_estimates, default=0) / MIB)),
        leaf_sizes=tuple(leaf_sizes),
    )
    report = {
        "schema_version": 1,
        "storage": config.reduction_storage,
        "ram_only": config.reduction_storage == "ram",
        "files_unchanged": source_before == source_after,
        "output_path": str(output_path),
        "selection_report": str(report_path),
        "selected_frames": len(selected_frames),
        "input_frames": len(all_frames),
        "feature_profile": config.feature_profile,
        "method": config.method,
        "selection_profile": config.selection_profile,
        "spill_bytes_written": spill_bytes,
        "elapsed_seconds": time.perf_counter() - started,
        **telemetry.as_dict(),
    }
    report_file = config.output_dir / "stack_ram_report.json"
    report_file.parent.mkdir(parents=True, exist_ok=True)
    temporary = report_file.with_name(f".{report_file.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    os.replace(temporary, report_file)
    return {
        "status": "success" if report["files_unchanged"] else "error",
        "reason": None if report["files_unchanged"] else "source_changed",
        "output_path": str(output_path),
        "selection_report": str(report_path),
        "stack_ram_report": str(report_file),
        "n_frames": len(selected_frames),
        "n_frames_total": len(all_frames),
        "dtype": str(output_data.dtype),
        "bit_depth": config.output_bit_depth,
        "feature_profile": config.feature_profile,
        "reduction_storage": config.reduction_storage,
        "spill_bytes_written": spill_bytes,
        "quality_weighted": config.method == "QualityWeightedMean",
        "streaming": True,
        "hierarchical": True,
        **telemetry.as_dict(),
    }


__all__ = ["RamNode", "RamTelemetry", "run_ram_stack"]
