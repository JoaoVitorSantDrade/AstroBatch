"""Build a production-style SigmaClip Lagoon stack outside the source tree.

This command is deliberately read-only with respect to the capture.  It
materializes corrected aligned FITS under an explicitly supplied output
directory, then invokes the normal AstroStack streaming reducer on those
temporary inputs.  The source and Flow trees are inventoried before/after so
an external write cannot be mistaken for a successful diagnostic.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from astropy.io import fits

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_DIR = PROJECT_ROOT / "benchmarks"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(BENCHMARK_DIR) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_DIR))

import stacking_logic as stacking
from astroalign_logic import save_aligned_fits, warp_frame
from cpu_runtime import configure_opencv_threads
from lagoon_similarity_stack import (
    _apply_rgb_validity,
    _atomic_json,
    _read_flow,
    _read_rgb_and_mask,
    _source_snapshot,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True, help="aligned FITS tree (read-only)")
    parser.add_argument("--flow-root", type=Path, required=True, help="Flow tree (read-only)")
    parser.add_argument("--output-dir", type=Path, required=True, help="temporary output outside both sources")
    parser.add_argument("--roundness", type=float, default=0.5953124412305257)
    parser.add_argument("--max-frames", type=int, default=0, help="smoke-test limit; zero means all selected")
    parser.add_argument("--workers", type=int, default=2, help="bounded RGB materialization workers")
    parser.add_argument("--stack-workers", type=int, default=4)
    parser.add_argument("--memory-budget-mb", type=int, default=4096)
    parser.add_argument(
        "--allow-disk-materialization",
        action="store_true",
        help="explicitly re-enable the deprecated per-frame FITS path",
    )
    return parser.parse_args()


def _records(source: Path, flow_root: Path, threshold: float) -> tuple[list[Path], int]:
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
    return selected, len(paths)


def _outside(path: Path, roots: tuple[Path, ...]) -> None:
    for root in roots:
        try:
            path.relative_to(root)
        except ValueError:
            continue
        raise ValueError(f"output-dir must be outside read-only source: {path}")


def _thread_init() -> None:
    # RGB feature fitting is native OpenCV work. Frame workers own the CPU;
    # do not multiply them by an OpenCV pool.
    configure_opencv_threads(1)


def _materialize_one(
    path: Path,
    source: Path,
    corrected_root: Path,
) -> dict[str, object]:
    data, mask, header = _read_rgb_and_mask(path)
    diagnostics: dict = {}
    corrected = warp_frame(
        data,
        np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32),
        "bilinear",
        rgb_registration=True,
        engine_profile="Fast",
        diagnostics=diagnostics,
        rgb_registration_mode="similarity",
    )
    valid = _apply_rgb_validity(mask, corrected, diagnostics)
    relative = path.relative_to(source)
    output_path = corrected_root / relative
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_aligned_fits(
        corrected,
        valid.astype(np.uint8),
        header,
        output_path,
        compress_output=False,
        metadata={"RGBMODE": "similarity", "ALNSTAT": "diagnostic"},
    )
    models = diagnostics.get("rgb_models", {})
    return {
        "source": str(path),
        "output": str(output_path),
        "full_model": len(models) == 2,
        "model_count": len(models),
        "valid_fraction": float(np.mean(valid)),
    }


def _status(message: str) -> None:
    text = message.rstrip()
    encoding = getattr(sys.stdout, "encoding", None) or "utf-8"
    # Windows terminals may expose cp1252 while Stack status strings contain
    # emoji. Keep the diagnostic usable without changing the library logger.
    safe = text.encode(encoding, errors="replace").decode(encoding, errors="replace")
    print(safe, flush=True)


def _progress(current: int, total: int, message: str = "") -> None:
    if current == 1 or current == total or current % 25 == 0:
        print(f"{message} {current}/{total}", flush=True)


def main() -> int:
    args = _parse_args()
    if not args.allow_disk_materialization:
        raise SystemExit(
            "disk materialization is disabled; use "
            "benchmarks/lagoon_similarity_ram_stack.py for the bounded-RAM path "
            "(or pass --allow-disk-materialization only for an explicit legacy diagnostic)"
        )
    source = args.source.resolve()
    flow_root = args.flow_root.resolve()
    output_dir = args.output_dir.resolve()
    if not source.is_dir() or not flow_root.is_dir():
        raise SystemExit("source/flow-root must be existing directories")
    _outside(output_dir, (source, flow_root))
    if args.workers < 1 or args.workers > 4:
        raise SystemExit("workers must be between 1 and 4")
    if args.stack_workers < 1 or args.stack_workers > 8:
        raise SystemExit("stack-workers must be between 1 and 8")
    if args.memory_budget_mb < 1024:
        raise SystemExit("memory-budget-mb must be at least 1024")
    threshold = float(args.roundness)
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise SystemExit("roundness must be finite and between 0 and 1")

    source_before = {
        "aligned": _source_snapshot(source),
        "flow": _source_snapshot(flow_root),
    }
    paths, total_frames = _records(source, flow_root, threshold)
    if args.max_frames < 0:
        raise SystemExit("max-frames must be >= 0")
    if args.max_frames > 0:
        paths = paths[: args.max_frames]
    if len(paths) < 3:
        raise SystemExit("fewer than three frames pass the diagnostic selection")

    output_dir.mkdir(parents=True, exist_ok=True)
    corrected_root = output_dir / "aligned_similarity"
    stack_root = output_dir / "stack"
    started = time.perf_counter()
    full_models = 0
    fallback_models = 0
    valid_fractions: list[float] = []
    print(f"Materializing {len(paths)}/{total_frames} frames into {corrected_root}", flush=True)
    with ThreadPoolExecutor(
        max_workers=int(args.workers),
        thread_name_prefix="lagoon-rgb",
        initializer=_thread_init,
    ) as executor:
        futures = [executor.submit(_materialize_one, path, source, corrected_root) for path in paths]
        for index, future in enumerate(futures, start=1):
            result = future.result()
            full_models += int(bool(result["full_model"]))
            fallback_models += int(not bool(result["full_model"]))
            valid_fractions.append(float(result["valid_fraction"]))
            _progress(index, len(paths), "RGB materialization")

    source_unchanged_before_stack = source_before == {
        "aligned": _source_snapshot(source),
        "flow": _source_snapshot(flow_root),
    }
    if not source_unchanged_before_stack:
        raise RuntimeError("a read-only source changed during RGB materialization")

    config = stacking.StackingConfig(
        base_dir=output_dir,
        input_dir=corrected_root,
        output_dir=stack_root,
        selection_mode="All",
        trail_filter_enabled=False,
        method="Mean",
        rejection_method="SigmaClip",
        rejection_low=3.0,
        rejection_high=3.0,
        normalize=False,
        apply_dither_correction=False,
        remove_background=False,
        output_name="lagoon_similarity_sigma_stack.fits",
        output_bit_depth="16-bit",
        compress_output=False,
        workers=int(args.stack_workers),
        memory_budget_mb=int(args.memory_budget_mb),
        cache_decompressed_fits=False,
        engine_profile="Stable",
    )
    print("Running production Stack Mean + SigmaClip", flush=True)
    result = stacking.process_stacking(
        config,
        progress_callback=_progress,
        status_callback=_status,
        cancel_event=threading.Event(),
    )
    files_unchanged = source_unchanged_before_stack and source_before == {
        "aligned": _source_snapshot(source),
        "flow": _source_snapshot(flow_root),
    }
    report = {
        "schema_version": 1,
        "source": str(source),
        "flow_root": str(flow_root),
        "output_dir": str(output_dir),
        "corrected_input": str(corrected_root),
        "stack_result": result,
        "roundness_threshold": threshold,
        "input_frame_count": total_frames,
        "selected_frame_count": len(paths),
        "rgb_similarity_full_model_frames": full_models,
        "rgb_similarity_fallback_frames": fallback_models,
        "valid_fraction_min": min(valid_fractions) if valid_fractions else None,
        "valid_fraction_median": float(np.median(valid_fractions)) if valid_fractions else None,
        "valid_fraction_max": max(valid_fractions) if valid_fractions else None,
        "files_unchanged": files_unchanged,
        "elapsed_seconds": time.perf_counter() - started,
        "method": "production Stack Mean + SigmaClip; Stable; RGB similarity opt-in",
    }
    _atomic_json(output_dir / "lagoon_similarity_sigma_report.json", report)
    print(json.dumps(report, indent=2, ensure_ascii=False, default=str), flush=True)
    if not files_unchanged:
        raise RuntimeError("a source file changed during the production reduction")
    if result.get("status") != "success":
        raise RuntimeError(f"production Stack failed: {result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
