"""Versioned end-to-end CPU benchmark for Flow -> Align -> Stack.

The benchmark creates a deterministic mono/RGB FITS corpus in a temporary
directory, runs the same scientific pipeline in Stable and opt-in Fast
profiles, and reports stage timings, process I/O counters, peak RSS, peak
threads and SIMD capabilities.  Fast uses the existing Numba reducers; the
output digest check prevents a faster mode from silently changing the selected
pixels on this corpus.

Examples::

    .venv/Scripts/python.exe benchmarks/pipeline_benchmark.py
    .venv/Scripts/python.exe benchmarks/pipeline_benchmark.py --repeats 7 --json-out reports/pipeline.json
    .venv/Scripts/python.exe benchmarks/io_concurrency_benchmark.py --workers 1,2,4,8
"""

from __future__ import annotations

import argparse
import cProfile
import hashlib
import json
import os
import platform
from pathlib import Path
import shutil
import statistics
import sys
import tempfile
import time
from threading import Event

import numpy as np
from astropy.io import fits
from numba import njit, prange

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.engines.execution import ExecutionBudget
from app.engines.telemetry import ProcessTelemetry
from cpu_kernels import warm_cpu_kernels
from cpu_runtime import inspect_numba_assembly, numpy_cpu_features, runtime_info


CORPUS_SCHEMA_VERSION = 2
# 512x512 models a dedicated-camera frame while the uncached assembly probes
# exercise the large AVX2/prange reducer without making the end-to-end gate
# unreasonably long on a developer workstation.
HEIGHT = 512
WIDTH = 512
FRAMES_PER_BATCH = 8
SMALL_HEIGHT = 96
SMALL_WIDTH = 96
SMALL_FRAMES = 4

CORPUS_LAYOUT = (
    ("batch_00_small_mono", False, SMALL_HEIGHT, SMALL_WIDTH, SMALL_FRAMES),
    ("batch_01_dedicated_mono", False, HEIGHT, WIDTH, FRAMES_PER_BATCH),
    ("batch_02_dedicated_rgb", True, HEIGHT, WIDTH, FRAMES_PER_BATCH),
)


@njit(cache=False, parallel=True, nogil=True)
def _assembly_probe(values: np.ndarray) -> np.ndarray:
    """Uncached inspection probe; production kernels remain cache=True."""

    height, width = values.shape
    output = np.empty_like(values)
    for y in prange(height):
        for x in range(width):
            output[y, x] = values[y, x] * np.float32(1.25) + np.float32(2.0)
    return output


@njit(cache=False, parallel=True, nogil=True)
def _assembly_extrema_probe(values: np.ndarray, masks: np.ndarray) -> np.ndarray:
    """Uncached assembly probe mirroring the production masked extrema loop."""

    frames, height, width = values.shape
    output = np.empty((height, width), dtype=np.float32)
    for y in prange(height):
        for x in range(width):
            found = False
            candidate = np.float32(0.0)
            for frame in range(frames):
                value = values[frame, y, x]
                if masks[frame, y, x] != 0 and not np.isnan(value):
                    if not found or value > candidate:
                        candidate = value
                        found = True
            output[y, x] = candidate if found else np.nan
    return output


@njit(cache=False, parallel=True, nogil=True)
def _assembly_sum_count_probe(
    values: np.ndarray, masks: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Uncached assembly probe mirroring the production masked sum loop."""

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


def _star_field(
    seed: int,
    index: int,
    rgb: bool,
    height: int = HEIGHT,
    width: int = WIDTH,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    margin = min(12, max(3, min(height, width) // 8))
    yy, xx = np.mgrid[:height, :width].astype(np.float32)
    base_x = rng.uniform(margin, width - margin, 32)
    base_y = rng.uniform(margin, height - margin, 32)
    image = rng.normal(100.0, 2.0, (height, width)).astype(np.float32)
    shift_x = 0.65 * index
    shift_y = -0.35 * index
    for x, y in zip(base_x, base_y, strict=True):
        sigma = 1.45 + 0.05 * ((int(x) + int(y)) % 3)
        image += (rng.uniform(700, 1800) * np.exp(
            -((xx - x - shift_x) ** 2 + (yy - y - shift_y) ** 2)
            / (2.0 * sigma * sigma)
        )).astype(np.float32)
    if not rgb:
        return image
    channels = np.stack((image * 1.00, image * 0.92, image * 1.08), axis=0)
    return np.asarray(channels, dtype=np.float32)


def create_corpus(root: Path) -> None:
    """Write small and dedicated-camera mono/RGB loads with valid masks."""

    for batch_index, (name, rgb, height, width, frame_count) in enumerate(CORPUS_LAYOUT, start=1):
        batch = root / name
        batch.mkdir(parents=True, exist_ok=True)
        for index in range(frame_count):
            data = _star_field(20260914 + batch_index * 100, index, rgb, height, width)
            # Insert a deterministic >15-minute pause halfway through each
            # load so the temporal sidecar exercises logical grouping too.
            minute = batch_index + (18 if index >= frame_count // 2 else 0)
            timestamp = f"2026-01-15T02:{minute:02d}:{index:02d}.000Z"
            header = fits.Header()
            header["DATE-OBS"] = timestamp
            header["EXPTIME"] = 2.0
            header["GAIN"] = 100.0
            header["BUNIT"] = "adu"
            valid = np.ones((height, width), dtype=np.uint8)
            valid[:2, :] = 0
            valid[:, :2] = 0
            fits.HDUList([
                fits.PrimaryHDU(data=data, header=header),
                fits.ImageHDU(data=valid, name="VALID_MASK"),
            ]).writeto(batch / f"frame_{index:03d}.fits", overwrite=True)


def _run_pipeline(
    root: Path,
    profile: str,
    workers: int = 1,
    compress_align: bool = False,
    compress_stack: bool = False,
    writer_workers: int = 0,
) -> dict[str, object]:
    from astroalign_logic import process_all_alignments
    from astroflow_logic import process_all_flows
    import stacking_logic
    from stacking_logic import process_all_stacking

    cancel = Event()
    messages: list[str] = []
    progress: list[tuple[int, int, str]] = []
    log = messages.append
    report_progress = lambda current, total, message="": progress.append((current, total, message))
    workers = max(1, int(workers))
    flow_config = {
        "global_master": "Auto",
        "fwhm": 3.0,
        "sigma": 4.0,
        "matching_radius": 14.0,
        "ransac": 3.0,
        "min_stars": 4,
        "min_inliers": 4,
        "min_ratio": 0.15,
        "max_stars": 150,
        "engine": "DAO",
        "engine_profile": "Stable",
        "detector_engine": "",
        "transform_fallback": "Disabled",
        "flow_workers": workers,
        "memory_budget_mb": 512,
        "temporal_analysis_enabled": True,
        "temporal_gap_minutes": 15.0,
        "temporal_seeing_sigma": 3.0,
    }
    flow_probe = ProcessTelemetry()
    with flow_probe:
        flow_result = process_all_flows(root, flow_config, log, report_progress, cancel)
    flow_telemetry = flow_probe.result
    flow_seconds = flow_telemetry.wall_seconds if flow_telemetry else 0.0
    if not isinstance(flow_result, dict) or flow_result.get("status") not in {"success", "partial"}:
        raise RuntimeError(f"Flow benchmark failed: {flow_result!r}; log={messages[-5:]}")
    temporal_group_count = 0
    temporal_unknown_count = 0
    temporal_path = root / "temporal_analysis.json"
    if temporal_path.exists():
        temporal = json.loads(temporal_path.read_text(encoding="utf-8"))
        temporal_group_count = len(temporal.get("groups", []) or [])
        temporal_unknown_count = len(temporal.get("unknown_timestamp_frames", []) or [])

    aligned = root / f"aligned_{profile.casefold()}"
    align_probe = ProcessTelemetry()
    with align_probe:
        aligned_count = process_all_alignments(
            root,
            aligned,
            {
            "debayer_pattern": "Auto",
            "debayer_method": "Bilinear",
            "interpolation": "Bilinear",
            "rgb_registration": False,
            "overwrite": True,
            "dry_run": False,
            "keep_header": True,
            "delete_intermediates": False,
            "compress_output": bool(compress_align),
            "engine_profile": profile,
            "quality_gate": False,
            "workers": workers,
            "memory_budget_mb": 512,
            "writer_workers": max(0, int(writer_workers)),
            },
            log,
            report_progress,
            cancel,
        )
    align_telemetry = align_probe.result
    align_seconds = align_telemetry.wall_seconds if align_telemetry else 0.0
    if not isinstance(aligned_count, tuple) or aligned_count[0] <= 0:
        raise RuntimeError(f"Align benchmark failed: {aligned_count!r}; log={messages[-5:]}")

    output = root / f"stacked_{profile.casefold()}"
    stack_probe = ProcessTelemetry()
    stack_config = {
            "base_dir": str(root),
            "input_dir": str(aligned),
            "output_dir": str(output),
            "selection_mode": "All",
            "selection_percentage": 100.0,
            "selection_metric": "quality",
            "trail_filter_enabled": False,
            "method": "Maximum",
            "rejection_method": "SigmaClip",
            "rejection_low": 3.0,
            "rejection_high": 3.0,
            "normalize": False,
            "output_name": "stacked_image.fits",
            "output_bit_depth": "16-bit",
            "compress_output": bool(compress_stack),
            "workers": workers,
            "memory_budget_mb": 512,
            "engine_profile": profile,
            "reducer_engine": "" if profile == "Stable" else "fast-numba",
    }
    profiler_enabled = stacking_logic.HAS_PYINSTRUMENT
    stacking_logic.HAS_PYINSTRUMENT = False
    try:
        with stack_probe:
            stack_result = process_all_stacking(
                aligned,
                stack_config,
                report_progress,
                log,
                cancel,
            )
    finally:
        stacking_logic.HAS_PYINSTRUMENT = profiler_enabled
    stack_telemetry = stack_probe.result
    stack_seconds = stack_telemetry.wall_seconds if stack_telemetry else 0.0
    if not isinstance(stack_result, dict) or stack_result.get("status") != "success":
        raise RuntimeError(f"Stack benchmark failed: {stack_result!r}; log={messages[-5:]}")

    output_path = Path(stack_result["output_path"])
    with fits.open(output_path, memmap=False) as hdul:
        data = np.asarray(hdul[0].data)
        mask = np.asarray(hdul["VALID_MASK"].data) if "VALID_MASK" in hdul else np.empty(0, dtype=np.uint8)
        count = np.asarray(hdul["SUB_COUNT"].data) if "SUB_COUNT" in hdul else np.empty(0, dtype=np.uint32)
        digest = hashlib.sha256(data.tobytes() + mask.tobytes() + count.tobytes()).hexdigest()
        metadata = {
            key: hdul[0].header.get(key)
            for key in ("BITPIX", "NAXIS", "NAXIS1", "NAXIS2", "NAXIS3", "CALNORM", "CALMIN", "CALMAX")
            if key in hdul[0].header
        }
        rss = int(getattr(__import__("psutil").Process().memory_info(), "rss", 0))
    stage_telemetry = {
        "flow": flow_telemetry.as_dict() if flow_telemetry else None,
        "align": align_telemetry.as_dict() if align_telemetry else None,
        "stack": stack_telemetry.as_dict() if stack_telemetry else None,
    }
    peak_rss = max(
        int((item or {}).get("peak_rss_bytes", 0))
        for item in stage_telemetry.values()
    )
    peak_threads = max(
        int((item or {}).get("peak_thread_count", 0))
        for item in stage_telemetry.values()
    )
    return {
        "profile": profile,
        "flow_seconds": flow_seconds,
        "anchor_detection_cache_hits": int(flow_result.get("anchor_detection_cache_hits", 0))
        if isinstance(flow_result, dict) else 0,
        "temporal_group_count": temporal_group_count,
        "temporal_unknown_count": temporal_unknown_count,
        "align_seconds": align_seconds,
        "stack_seconds": stack_seconds,
        "total_seconds": flow_seconds + align_seconds + stack_seconds,
        "output_digest": digest,
        "metadata": metadata,
        "output_dtype": str(data.dtype),
        "valid_mask_dtype": str(mask.dtype),
        "sub_count_dtype": str(count.dtype),
        "aligned_frames": aligned_count[0],
        "rss_bytes": rss,
        "peak_rss_bytes": peak_rss,
        "peak_thread_count": peak_threads,
        "stage_telemetry": stage_telemetry,
        "workers_requested": workers,
        "compress_align": bool(compress_align),
        "compress_stack": bool(compress_stack),
        "messages": messages[-5:],
    }


def _median_runs(runs: list[dict[str, object]]) -> dict[str, object]:
    if not runs:
        return {}
    result: dict[str, object] = {"runs": runs}
    for key in (
        "flow_seconds",
        "align_seconds",
        "stack_seconds",
        "total_seconds",
        "rss_bytes",
        "peak_rss_bytes",
        "peak_thread_count",
        "temporal_group_count",
        "temporal_unknown_count",
        "anchor_detection_cache_hits",
    ):
        values = [float(run[key]) for run in runs]
        result[f"{key}_median"] = statistics.median(values)
        result[f"{key}_stdev"] = statistics.stdev(values) if len(values) > 1 else 0.0
    stages: dict[str, dict[str, float]] = {}
    for stage in ("flow", "align", "stack"):
        stage_runs = [run.get("stage_telemetry", {}).get(stage) for run in runs]
        stage_runs = [item for item in stage_runs if isinstance(item, dict)]
        if not stage_runs:
            continue
        stage_summary: dict[str, float] = {}
        for field in (
            "wall_seconds",
            "cpu_seconds",
            "cpu_to_wall",
            "read_bytes",
            "write_bytes",
            "read_count",
            "write_count",
            "peak_rss_bytes",
            "peak_thread_count",
        ):
            values = [float(item.get(field, 0.0)) for item in stage_runs]
            stage_summary[f"{field}_median"] = statistics.median(values)
            stage_summary[f"{field}_stdev"] = statistics.stdev(values) if len(values) > 1 else 0.0
        stages[stage] = stage_summary
    result["stage_telemetry"] = stages
    return result


def _pin_logical_affinity() -> tuple[list[int] | None, list[int] | None]:
    """Pin this process to a deterministic eight-logical-CPU slice when able."""

    try:
        process = __import__("psutil").Process()
        original = list(process.cpu_affinity())
        logical = int(os.cpu_count() or len(original) or 1)
        target = list(range(min(8, logical)))
        if target and set(target) != set(original):
            process.cpu_affinity(target)
        return original, target
    except Exception:
        return None, None


def _restore_affinity(original: list[int] | None) -> None:
    if original is None:
        return
    try:
        __import__("psutil").Process().cpu_affinity(original)
    except Exception:
        pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--keep-corpus", action="store_true")
    parser.add_argument("--profile-out", type=Path, help="Write cProfile stats for the first Stable run")
    parser.add_argument(
        "--baseline-json",
        type=Path,
        help="Compare the Stable median with a previous benchmark JSON",
    )
    parser.add_argument("--workers", type=int, default=1,
                        help="Frame/leaf worker request used for every stage (default: 1)")
    parser.add_argument("--compress-align", action="store_true",
                        help="Compress aligned FITS outputs while measuring the writer queue")
    parser.add_argument("--compress-stack", action="store_true",
                        help="Compress final Stack FITS output")
    parser.add_argument("--writer-workers", type=int, default=0,
                        help="Bounded Align writer count (0 keeps the default one)")
    args = parser.parse_args(argv)
    if args.repeats < 1:
        parser.error("--repeats must be >= 1")

    original_affinity, applied_affinity = _pin_logical_affinity()
    warm_started = time.perf_counter()
    warm_cpu_kernels()
    runtime = runtime_info()
    cpu_features = numpy_cpu_features()
    warm_seconds = time.perf_counter() - warm_started
    print(f"runtime={json.dumps(runtime.as_dict(), sort_keys=True)}")
    print(f"numpy_cpu_features={json.dumps(cpu_features, sort_keys=True)}")
    print(f"numba_warm_seconds={warm_seconds:.6f}")
    budget = ExecutionBudget.for_pipeline(1)
    print(f"execution_budget={json.dumps(budget.as_dict(), sort_keys=True)}")
    import_started = time.perf_counter()
    # Import the complete pipeline before the timed loop so the seven hot
    # repetitions do not mix module-import cost with throughput.
    import astroalign_logic  # noqa: F401
    import astroflow_logic  # noqa: F401
    import stacking_logic  # noqa: F401
    import_seconds = time.perf_counter() - import_started
    print(f"pipeline_import_seconds={import_seconds:.6f}")
    probe_values = np.ones((16, 512, 512), dtype=np.float32)
    probe_masks = np.ones(probe_values.shape, dtype=np.uint8)
    _assembly_probe(np.ones((64, 64), dtype=np.float32))
    _assembly_extrema_probe(probe_values, probe_masks)
    _assembly_sum_count_probe(probe_values, probe_masks)
    assembly = {
        "masked_extrema_parallel": inspect_numba_assembly(_assembly_extrema_probe),
        "masked_sum_count_parallel": inspect_numba_assembly(_assembly_sum_count_probe),
        "uncached_vector_probe": inspect_numba_assembly(_assembly_probe),
    }
    print(f"numba_assembly={json.dumps(assembly, sort_keys=True)}")
    del probe_values, probe_masks

    if args.keep_corpus:
        temp_context = None
        corpus_root = Path(tempfile.mkdtemp(prefix="astrobatch-pipeline-"))
    else:
        temp_context = tempfile.TemporaryDirectory(prefix="astrobatch-pipeline-")
        corpus_root = Path(temp_context.name)
    create_corpus(corpus_root)
    all_runs: dict[str, list[dict[str, object]]] = {"Stable": [], "Fast": []}
    try:
        # Warm once, then alternate profiles to avoid assigning a thermal drift
        # to one profile.  Each run gets a fresh output directory and corpus.
        for index in range(args.repeats):
            for profile in ("Stable", "Fast"):
                run_root = corpus_root / f"run_{index:02d}_{profile.casefold()}"
                shutil.copytree(corpus_root, run_root, ignore=shutil.ignore_patterns("run_*"))
                profiler = cProfile.Profile() if args.profile_out and index == 0 and profile == "Stable" else None
                if profiler:
                    profiler.enable()
                try:
                    metrics = _run_pipeline(
                        run_root,
                        profile,
                        workers=args.workers,
                        compress_align=args.compress_align,
                        compress_stack=args.compress_stack,
                        writer_workers=args.writer_workers,
                    )
                finally:
                    if profiler:
                        profiler.disable()
                        args.profile_out.parent.mkdir(parents=True, exist_ok=True)
                        profiler.dump_stats(str(args.profile_out))
                all_runs[profile].append(metrics)
                print(f"{profile} run {index + 1}/{args.repeats}: total={float(metrics['total_seconds']):.3f}s")
                if not args.keep_corpus:
                    shutil.rmtree(run_root, ignore_errors=True)
        summary = {
            "schema_version": CORPUS_SCHEMA_VERSION,
            "corpus": {
                "batches": len(CORPUS_LAYOUT),
                "loads": [
                    {
                        "name": name,
                        "rgb": rgb,
                        "shape": [height, width],
                        "frames": frame_count,
                    }
                    for name, rgb, height, width, frame_count in CORPUS_LAYOUT
                ],
                "mono_and_rgb": True,
            },
            "runtime": runtime.as_dict(),
            "numpy_cpu_features": cpu_features,
            "host": {
                "processor": platform.processor(),
                "platform": platform.platform(),
                "python": platform.python_version(),
            },
            "numba_assembly": assembly,
            "warm_seconds": warm_seconds,
            "pipeline_import_seconds": import_seconds,
            "profiles": {profile: _median_runs(runs) for profile, runs in all_runs.items()},
            "comparison": {
                "baseline": "Stable",
                "candidate": "Fast",
                "metric": "total_seconds_median",
            },
            "pipeline": {
                "flow": "neighbor_bfs + DATE-OBS sidecar",
                "align": "Bilinear",
                "stack_method": "Maximum",
                "rejection_method": "SigmaClip",
                "output_dtype": "uint16",
            },
            "concurrency": {
                "workers_requested": max(1, int(args.workers)),
                "compress_align": bool(args.compress_align),
                "compress_stack": bool(args.compress_stack),
                "writer_workers_requested": max(0, int(args.writer_workers)),
                "telemetry_includes_observer_thread": True,
            },
        }
        stable_digest = {run["output_digest"] for run in all_runs["Stable"]}
        fast_digest = {run["output_digest"] for run in all_runs["Fast"]}
        stable_metadata = {json.dumps(run["metadata"], sort_keys=True) for run in all_runs["Stable"]}
        fast_metadata = {json.dumps(run["metadata"], sort_keys=True) for run in all_runs["Fast"]}
        stable_dtype = {run["output_dtype"] for run in all_runs["Stable"]}
        fast_dtype = {run["output_dtype"] for run in all_runs["Fast"]}
        stable_mask_dtype = {run["valid_mask_dtype"] for run in all_runs["Stable"]}
        fast_mask_dtype = {run["valid_mask_dtype"] for run in all_runs["Fast"]}
        stable_count_dtype = {run["sub_count_dtype"] for run in all_runs["Stable"]}
        fast_count_dtype = {run["sub_count_dtype"] for run in all_runs["Fast"]}
        summary["bitwise_output_match"] = bool(
            stable_digest == fast_digest
            and len(stable_digest) == 1
            and stable_metadata == fast_metadata
            and stable_dtype == fast_dtype == {"uint16"}
            and stable_mask_dtype == fast_mask_dtype == {"uint8"}
            and stable_count_dtype == fast_count_dtype == {"uint32"}
        )
        stable_total = float(summary["profiles"]["Stable"]["total_seconds_median"])
        fast_total = float(summary["profiles"]["Fast"]["total_seconds_median"])
        summary["fast_speedup"] = stable_total / fast_total if fast_total > 0 else None

        baseline_total = None
        if args.baseline_json is not None:
            try:
                baseline_doc = json.loads(args.baseline_json.read_text(encoding="utf-8"))
                baseline_total = float(
                    baseline_doc["profiles"]["Stable"]["total_seconds_median"]
                )
            except (OSError, ValueError, KeyError, TypeError) as exc:
                parser.error(f"could not read Stable baseline JSON: {exc}")

        gate_speedup = (
            baseline_total / stable_total
            if baseline_total is not None and stable_total > 0
            else summary["fast_speedup"]
        )
        gate_reduction = (
            1.0 - stable_total / baseline_total
            if baseline_total is not None and baseline_total > 0
            else None
        )
        summary["gate"] = {
            # A 25% time reduction means new_time <= .75 * baseline,
            # equivalent to a speedup of at least 4/3.
            "target_speedup": 4 / 3,
            "comparison": "historical_baseline" if baseline_total is not None else "Stable_vs_Fast",
            "baseline_seconds": baseline_total,
            "candidate_seconds": stable_total if baseline_total is not None else fast_total,
            "speedup": gate_speedup,
            "reduction": gate_reduction,
            "speedup_reached": bool(gate_speedup is not None and gate_speedup >= 4 / 3),
            "bitwise_output_match": summary["bitwise_output_match"],
            "status": "passed" if summary["bitwise_output_match"] and gate_speedup is not None and gate_speedup >= 4 / 3 else "not_reached",
        }
        summary["affinity"] = {
            "requested_logical_cpus": applied_affinity,
            "pinned": bool(applied_affinity and original_affinity and set(applied_affinity) != set(original_affinity)),
            "restored_after_run": bool(original_affinity),
        }
        print(json.dumps(summary["gate"], sort_keys=True))
        if args.json_out:
            args.json_out.parent.mkdir(parents=True, exist_ok=True)
            args.json_out.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
        return 0
    finally:
        _restore_affinity(original_affinity)
        if temp_context is not None:
            temp_context.cleanup()
        elif args.keep_corpus:
            print(f"corpus_kept={corpus_root}")


if __name__ == "__main__":
    raise SystemExit(main())
