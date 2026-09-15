"""Bounded-RAM Lagoon SigmaClip diagnostic.

The aligned capture and its Flow metadata are read only.  Unlike
``lagoon_similarity_sigma_stack.py`` this reducer never materializes corrected
frames as FITS files: a small leaf is built in memory, reduced, and released
before the next leaf is read.  Only the final FITS and a JSON report are
written under the explicitly supplied output directory.

This is a diagnostic reduction.  The production Stack remains unchanged and
continues to own the Stable scientific path.  The in-memory tree keeps the
same Stable reducer and binary child-combine semantics, but the leaf count is
chosen from the RAM budget rather than the production four-leaf topology.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import time
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_DIR = PROJECT_ROOT / "benchmarks"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DIR))

import stacking_logic as stacking
from astroalign_logic import warp_frame
from cpu_runtime import configure_numba_threads, configure_opencv_threads
from lagoon_similarity_stack import (
    _apply_rgb_validity,
    _atomic_json,
    _read_flow,
    _read_rgb_and_mask,
    _source_snapshot,
)


GIB = 1024**3
MIB = 1024**2
MIN_SYSTEM_HEADROOM = 4 * GIB
MAX_LEAF_FRAMES = 64
MIN_LEAF_FRAMES = 3


@dataclass(slots=True)
class RamNode:
    data: np.ndarray
    valid_mask: np.ndarray
    coverage: np.ndarray
    frame_count: int
    leaf: bool = False


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True, help="aligned FITS tree (read-only)")
    parser.add_argument("--flow-root", type=Path, required=True, help="Flow tree (read-only)")
    parser.add_argument("--output-dir", type=Path, required=True, help="output directory outside both sources")
    parser.add_argument("--roundness", type=float, default=0.5953124412305257)
    parser.add_argument("--max-frames", type=int, default=0, help="smoke-test limit; zero means all selected")
    parser.add_argument(
        "--memory-budget-mb",
        type=int,
        default=8192,
        help="hard working-set budget; the reducer applies an additional safety fraction",
    )
    parser.add_argument(
        "--chunk-frames",
        type=int,
        default=0,
        help="maximum frames per RAM leaf; zero selects a safe size automatically",
    )
    parser.add_argument(
        "--rgb-mode",
        choices=("translation", "similarity", "hybrid"),
        default="similarity",
        help="RGB correction: hybrid uses translation for R and similarity for B",
    )
    return parser.parse_args()


def _outside(path: Path, roots: tuple[Path, ...]) -> None:
    for root in roots:
        try:
            path.relative_to(root)
        except ValueError:
            continue
        raise ValueError(f"output-dir must be outside read-only source: {path}")


def _records(source: Path, flow_root: Path, threshold: float) -> tuple[list[Path], int, dict[tuple[str, str], dict]]:
    flow = _read_flow(flow_root)
    paths = sorted(source.rglob("*.fits"), key=lambda p: (str(p.parent).casefold(), p.name.casefold()))
    selected: list[Path] = []
    for path in paths:
        frame = flow.get((path.parent.name, path.name), {})
        status = str(frame.get("status", "unknown"))
        try:
            roundness = float(frame.get("roundness"))
        except (TypeError, ValueError):
            roundness = float("nan")
        if status not in {"accepted", "master", "unknown"}:
            continue
        if math.isfinite(roundness) and roundness < threshold:
            continue
        selected.append(path)
    return selected, len(paths), flow


def _memory_state() -> tuple[int, int, int] | None:
    """Return (available, total, process_rss), or None without psutil."""

    try:
        import psutil

        vm = psutil.virtual_memory()
        rss = psutil.Process(os.getpid()).memory_info().rss
        return int(vm.available), int(vm.total), int(rss)
    except Exception:
        return None


def _effective_budget(requested_mb: int) -> tuple[int, int | None, int | None]:
    requested = int(requested_mb) * MIB
    state = _memory_state()
    if state is None:
        return requested, None, None
    available, total, _rss = state
    # Never reserve the entire visible free RAM.  Keep both a fixed floor and
    # a percentage of physical RAM for Windows, FITS decompression and other
    # applications.  The configured budget remains the upper bound.
    headroom = max(MIN_SYSTEM_HEADROOM, int(total * 0.20))
    safe_available = max(0, available - headroom)
    effective = min(requested, int(safe_available * 0.75))
    return max(0, effective), available, total


def _bytes_per_frame(pixels: int) -> int:
    # Conservative estimate: source + corrected image, channels-first leaf,
    # masks, and several Stable SigmaClip temporaries.  It intentionally
    # overestimates the resident set so a leaf is reduced well before RAM is
    # pressured by the OS.
    return int(pixels * 64)


def _choose_leaf_size(
    remaining: int,
    pixels: int,
    budget_bytes: int,
    requested: int,
) -> int:
    per_frame = _bytes_per_frame(pixels)
    allocation_budget = int(budget_bytes * 0.60)
    automatic = allocation_budget // max(1, per_frame)
    if requested > 0:
        size = min(requested, automatic if automatic else requested)
    else:
        size = automatic
    size = max(MIN_LEAF_FRAMES, min(MAX_LEAF_FRAMES, int(size)))
    return min(remaining, size)


def _next_leaf_size(remaining: int, maximum: int) -> int:
    """Choose a bounded leaf without leaving a one/two-frame tail."""

    maximum = max(MIN_LEAF_FRAMES, int(maximum))
    if remaining <= maximum + 2:
        # A final leaf may be up to two frames larger than the nominal size,
        # which is safer than silently disabling SigmaClip for a tiny tail.
        return remaining
    size = min(maximum, remaining)
    tail = remaining - size
    if tail in (1, 2):
        size -= 3 - tail
    return max(MIN_LEAF_FRAMES, size)


def _guard_before_leaf(required_bytes: int, budget_bytes: int) -> dict[str, int | bool]:
    state = _memory_state()
    if state is None:
        if required_bytes > budget_bytes:
            raise MemoryError("estimated RAM leaf exceeds configured budget")
        return {"guarded": False, "available": -1, "rss": -1}
    available, total, rss = state
    # Keep a generous OS reserve.  The extra 10% margin covers allocator and
    # FITS/OpenCV temporaries that are not represented by the leaf estimate.
    reserve = max(MIN_SYSTEM_HEADROOM, int(total * 0.20))
    if available - reserve < int(required_bytes * 1.10):
        raise MemoryError(
            f"RAM guard: need about {required_bytes / GIB:.2f} GiB, "
            f"only {(available - reserve) / GIB:.2f} GiB is safely available"
        )
    if required_bytes > int(budget_bytes * 0.90):
        raise MemoryError("estimated RAM leaf exceeds the safety budget")
    return {"guarded": True, "available": int(available), "rss": int(rss)}


def _load_corrected(
    path: Path,
    rgb_mode: str = "similarity",
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    data, mask, _header = _read_rgb_and_mask(path)
    diagnostics: dict[str, object] = {}
    corrected = warp_frame(
        data,
        np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32),
        "bilinear",
        rgb_registration=True,
        engine_profile="Fast",
        diagnostics=diagnostics,
        rgb_registration_mode=rgb_mode,
    )
    valid = _apply_rgb_validity(mask, corrected, diagnostics)
    models = diagnostics.get("rgb_models", {})
    summary = {
        "full_model": isinstance(models, dict) and len(models) == 2,
        "model_count": len(models) if isinstance(models, dict) else 0,
        "valid_fraction": float(np.mean(valid)),
    }
    # ``corrected`` is HWC for the alignment API; leaves use C,H,W so each
    # channel can be reduced without another full-size transpose later.
    return np.ascontiguousarray(np.moveaxis(corrected, -1, 0)), np.ascontiguousarray(valid), summary


def _nanmedian_axis0_fast(values: np.ndarray) -> np.ndarray:
    """NaN median for a small leaf without NumPy's masked-array wrapper."""

    values = np.asarray(values, dtype=np.float32)
    valid_count = np.sum(~np.isnan(values), axis=0, dtype=np.intp)
    filled = np.where(np.isnan(values), np.float32(np.inf), values)
    ordered = np.sort(filled, axis=0)
    lower_index = np.maximum(valid_count - 1, 0) // 2
    upper_index = np.maximum(valid_count, 1) // 2
    lower = np.take_along_axis(ordered, lower_index[None, ...], axis=0)[0]
    upper = np.take_along_axis(ordered, upper_index[None, ...], axis=0)[0]
    result = np.where(
        valid_count <= 0,
        np.float32(np.nan),
        np.where(lower == upper, lower, (lower + upper) * np.float32(0.5)),
    )
    return np.asarray(result, dtype=np.float32)


def _stable_sigma_clip_rgb(values: np.ndarray, masks: np.ndarray) -> np.ndarray:
    """Run the Stable SigmaClip equations for all RGB planes in one pass.

    ``stacking_logic`` keeps the public reducer channel-oriented for mono and
    RGB compatibility.  The RAM diagnostic already owns a contiguous
    ``[N, C, Y, X]`` leaf, so doing the same NaN/median/std sequence over all
    channels avoids three independent Python-to-NumPy dispatches.  The
    all-valid finite case deliberately falls back to the canonical reducer;
    this keeps its fast path and arithmetic order untouched.
    """

    values = np.asarray(values, dtype=np.float32)
    masks = np.asarray(masks, dtype=bool)
    if values.ndim != 4 or values.shape[0] != masks.shape[0] or values.shape[2:] != masks.shape[1:]:
        raise ValueError("invalid RGB SigmaClip leaf")
    if np.all(masks) and np.isfinite(values).all():
        result = np.empty(values.shape[1:], dtype=np.float32)
        for channel in range(values.shape[1]):
            result[channel] = stacking.reject_and_combine_block(
                values[:, channel],
                masks,
                "Mean",
                "SigmaClip",
                3.0,
                3.0,
                engine_profile="Stable",
                kernel_parallel=False,
            )
        return result

    masked = np.where(masks[:, None, :, :], values, np.float32(np.nan))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        center = _nanmedian_axis0_fast(masked)
        std = np.nanstd(masked, axis=0)
    lower = center - np.float32(3.0) * std
    upper = center + np.float32(3.0) * std
    valid = (masked >= lower) & (masked <= upper)
    stable = np.isfinite(std) & (std > np.float32(1e-10))
    valid |= ~stable
    masked[~valid] = np.nan
    result = _nanmedian_axis0_fast(masked)
    return result


def _build_leaf(
    paths: list[Path],
    shape: tuple[int, int, int],
    budget_bytes: int,
    rgb_mode: str,
) -> tuple[RamNode, dict[str, object]]:
    channels, height, width = shape
    pixels = height * width
    n_frames = len(paths)
    estimate = _bytes_per_frame(pixels) * n_frames
    guard = _guard_before_leaf(estimate, budget_bytes)
    values = np.empty((n_frames, channels, height, width), dtype=np.float32)
    masks = np.empty((n_frames, height, width), dtype=bool)
    full_models = 0
    fallback_models = 0
    valid_fractions: list[float] = []
    for index, path in enumerate(paths):
        corrected, valid, summary = _load_corrected(path, rgb_mode)
        if corrected.shape != (channels, height, width) or valid.shape != (height, width):
            raise RuntimeError(f"incompatible RGB shape: {path} {corrected.shape} / {valid.shape}")
        values[index] = corrected
        masks[index] = valid
        full_models += int(bool(summary["full_model"]))
        fallback_models += int(not bool(summary["full_model"]))
        valid_fractions.append(float(summary["valid_fraction"]))
        del corrected, valid

    coverage = masks.sum(axis=0, dtype=np.uint32)
    result = _stable_sigma_clip_rgb(values, masks)
    del values, masks
    gc.collect()
    post_state = _memory_state()
    node = RamNode(
        data=np.asarray(result, dtype=np.float32),
        valid_mask=coverage > 0,
        coverage=coverage,
        frame_count=n_frames,
        leaf=True,
    )
    return node, {
        "full_models": full_models,
        "fallback_models": fallback_models,
        "valid_min": min(valid_fractions) if valid_fractions else None,
        "valid_median": float(np.median(valid_fractions)) if valid_fractions else None,
        "valid_max": max(valid_fractions) if valid_fractions else None,
        "valid_fractions": valid_fractions,
        "guard_available": guard["available"],
        "guard_rss": guard["rss"],
        "post_available": post_state[0] if post_state is not None else -1,
        "post_rss": post_state[2] if post_state is not None else -1,
        "estimated_working_bytes": estimate,
    }


def _combine_nodes(left: RamNode, right: RamNode) -> RamNode:
    """Mirror the production binary child combine without a FITS leaf."""

    values = np.stack((left.data, right.data), axis=0)
    masks = np.stack((left.valid_mask, right.valid_mask), axis=0)
    channels = values.shape[1]
    result = np.empty_like(left.data, dtype=np.float32)
    for channel in range(channels):
        # Two children intentionally trigger the Stable ``None`` rejection
        # fast path, matching ``_combine_substacks`` in stacking_logic.py.
        result[channel] = stacking.reject_and_combine_block(
            values[:, channel],
            masks,
            "Mean",
            "SigmaClip",
            3.0,
            3.0,
            engine_profile="Stable",
            kernel_parallel=False,
        )
    coverage = left.coverage + right.coverage
    total = left.frame_count + right.frame_count
    valid_mask = coverage >= max(1, math.ceil(total * 0.70))
    del values, masks
    return RamNode(
        data=np.asarray(result, dtype=np.float32),
        valid_mask=valid_mask,
        coverage=coverage,
        frame_count=total,
        leaf=False,
    )


def _reduce_tree(nodes: list[RamNode]) -> RamNode:
    current = nodes
    while len(current) > 1:
        next_level: list[RamNode] = []
        for index in range(0, len(current), 2):
            left = current[index]
            if index + 1 < len(current):
                right = current[index + 1]
                merged = _combine_nodes(left, right)
                del left, right
            else:
                merged = left
            next_level.append(merged)
        current = next_level
        gc.collect()
    return current[0]


def _metric_average(paths: list[Path], flow: dict[tuple[str, str], dict], name: str) -> float:
    values: list[float] = []
    for path in paths:
        value = flow.get((path.parent.name, path.name), {}).get(name)
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            values.append(number)
    return float(np.mean(values)) if values else 0.0


def main() -> int:
    args = _parse_args()
    source = args.source.resolve()
    flow_root = args.flow_root.resolve()
    output_dir = args.output_dir.resolve()
    if not source.is_dir() or not flow_root.is_dir():
        raise SystemExit("source/flow-root must be existing directories")
    _outside(output_dir, (source, flow_root))
    if args.max_frames < 0 or args.chunk_frames < 0:
        raise SystemExit("max-frames and chunk-frames must be >= 0")
    if args.chunk_frames and not MIN_LEAF_FRAMES <= args.chunk_frames <= MAX_LEAF_FRAMES:
        raise SystemExit(f"chunk-frames must be between {MIN_LEAF_FRAMES} and {MAX_LEAF_FRAMES}")
    if args.memory_budget_mb < 1024:
        raise SystemExit("memory-budget-mb must be at least 1024")
    threshold = float(args.roundness)
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise SystemExit("roundness must be finite and between 0 and 1")

    source_before = {"aligned": _source_snapshot(source), "flow": _source_snapshot(flow_root)}
    paths, total_frames, flow = _records(source, flow_root, threshold)
    if args.max_frames:
        paths = paths[: args.max_frames]
    if len(paths) < MIN_LEAF_FRAMES:
        raise SystemExit("fewer than three frames pass the diagnostic selection")

    effective_budget, initial_available, total_memory = _effective_budget(args.memory_budget_mb)
    configure_opencv_threads(1)
    configure_numba_threads(1)
    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    first_data, first_mask, _first_summary = _load_corrected(paths[0], args.rgb_mode)
    shape = tuple(int(value) for value in first_data.shape)
    if len(shape) != 3 or shape[0] != 3 or first_mask.shape != shape[1:]:
        raise SystemExit(f"Lagoon RAM reducer expects RGB frames, got {shape} / {first_mask.shape}")
    del first_data, first_mask
    gc.collect()

    pixels = shape[1] * shape[2]
    if effective_budget <= 0 or _bytes_per_frame(pixels) * MIN_LEAF_FRAMES > int(effective_budget * 0.90):
        raise SystemExit("RAM guard refused the run: insufficient safe working memory for three frames")
    current_chunk = _choose_leaf_size(len(paths), pixels, effective_budget, args.chunk_frames)
    nodes: list[RamNode] = []
    # The first frame is probed once to establish geometry and is then read by
    # its normal leaf below.  Do not count that probe twice in the report.
    full_models = 0
    fallback_models = 0
    valid_fractions: list[float] = []
    rss_peak = 0
    available_min = initial_available or 0
    offset = 0
    leaf_sizes: list[int] = []
    while offset < len(paths):
        n_frames = _next_leaf_size(len(paths) - offset, current_chunk)
        batch_paths = paths[offset : offset + n_frames]
        try:
            node, stats = _build_leaf(batch_paths, shape, effective_budget, args.rgb_mode)
        except MemoryError as exc:
            gc.collect()
            if current_chunk <= MIN_LEAF_FRAMES:
                raise SystemExit(f"RAM guard stopped safely: {exc}") from exc
            current_chunk = max(MIN_LEAF_FRAMES, current_chunk // 2)
            print(f"RAM guard reduced leaf to {current_chunk} frames: {exc}", flush=True)
            continue
        nodes.append(node)
        leaf_sizes.append(n_frames)
        offset += n_frames
        full_models += int(stats["full_models"])
        fallback_models += int(stats["fallback_models"])
        valid_fractions.extend(float(value) for value in stats["valid_fractions"])
        for available in (int(stats["guard_available"]), int(stats["post_available"])):
            if available >= 0:
                available_min = available if not available_min else min(available_min, available)
        for rss in (int(stats["guard_rss"]), int(stats["post_rss"])):
            if rss >= 0:
                rss_peak = max(rss_peak, rss)
        print(f"RAM leaf {len(nodes)}: {offset}/{len(paths)} frames | {n_frames} in memory", flush=True)

    root = _reduce_tree(nodes)
    # A single leaf has no binary branch to apply the production 70% coverage
    # rule.  Apply it explicitly so a small smoke run follows the same final
    # validity contract as Stack._combine_substacks.
    root.valid_mask = root.coverage >= max(1, math.ceil(len(paths) * 0.70))
    root.data[:, ~root.valid_mask] = 0.0
    root.data = np.nan_to_num(root.data, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)

    config = stacking.StackingConfig(
        base_dir=output_dir,
        input_dir=source,
        output_dir=output_dir,
        selection_mode="All",
        trail_filter_enabled=False,
        method="Mean",
        rejection_method="SigmaClip",
        rejection_low=3.0,
        rejection_high=3.0,
        normalize=False,
        apply_dither_correction=False,
        remove_background=False,
        output_name="lagoon_similarity_ram_sigma_stack.fits",
        output_bit_depth="16-bit",
        compress_output=False,
        workers=1,
        memory_budget_mb=int(args.memory_budget_mb),
        cache_decompressed_fits=False,
        engine_profile="Stable",
    )
    header = stacking.prepare_output_header(
        stacking.load_source_header(paths[0]),
        config,
        len(paths),
        total_frames,
        len({path.parent.name for path in paths}),
        _metric_average(paths, flow, "quality"),
        _metric_average(paths, flow, "star_count"),
        _metric_average(paths, flow, "fwhm"),
    )
    header["RGBMODE"] = args.rgb_mode
    header["STACK_RAM"] = True
    header["STACK_RBMB"] = int(effective_budget / MIB)
    header["STACK_LFV"] = int(min(leaf_sizes)) if leaf_sizes else 0
    data_to_save = stacking.convert_output_dtype(root.data, config)
    output_path = output_dir / config.output_name
    stacking.write_stack_output(output_path, data_to_save, root.valid_mask, header, config)

    files_unchanged = source_before == {"aligned": _source_snapshot(source), "flow": _source_snapshot(flow_root)}
    final_state = _memory_state()
    if final_state is not None:
        available_min = min(available_min or final_state[0], final_state[0])
        rss_peak = max(rss_peak, final_state[2])
    report = {
        "schema_version": 1,
        "source": str(source),
        "flow_root": str(flow_root),
        "output": str(output_path),
        "files_unchanged": files_unchanged,
        "roundness_threshold": threshold,
        "input_frame_count": total_frames,
        "selected_frame_count": len(paths),
        "rgb_similarity_full_model_frames": full_models,
        "rgb_similarity_fallback_frames": fallback_models,
        "valid_fraction_min": min(valid_fractions) if valid_fractions else None,
        "valid_fraction_median": float(np.median(valid_fractions)) if valid_fractions else None,
        "valid_fraction_max": max(valid_fractions) if valid_fractions else None,
        "ram_only": True,
        "memory_budget_requested_mb": int(args.memory_budget_mb),
        "memory_budget_effective_mb": int(effective_budget / MIB),
        "initial_available_mb": int(initial_available / MIB) if initial_available else None,
        "total_memory_mb": int(total_memory / MIB) if total_memory else None,
        "minimum_available_mb": int(available_min / MIB) if available_min else None,
        "peak_process_rss_mb": int(rss_peak / MIB) if rss_peak else None,
        "estimated_peak_working_mb": int(
            math.ceil(max((_bytes_per_frame(pixels) * size for size in leaf_sizes), default=0) / MIB)
        ),
        "leaf_sizes": leaf_sizes,
        "leaf_count": len(leaf_sizes),
        "elapsed_seconds": time.perf_counter() - started,
        "method": "bounded-RAM Stable SigmaClip diagnostic; RGB correction opt-in",
        "rgb_mode": args.rgb_mode,
    }
    _atomic_json(output_dir / "lagoon_similarity_ram_report.json", report)
    if not files_unchanged:
        raise RuntimeError("a source file changed during the read-only RAM reduction")
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
