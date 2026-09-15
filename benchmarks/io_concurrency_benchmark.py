"""Diagnose I/O and executor scaling for the AstroBatch CPU pipeline.

This benchmark is intentionally separate from the Stable/Fast throughput gate.
It runs the same deterministic corpus with several executor widths and records
process-level file counters, CPU/wall ratio, peak RSS and peak thread count.
The result is a prioritization aid: bytes moved are not treated as proof of a
physical-disk bottleneck when the operating-system cache is warm.

Example::

    .venv/Scripts/python.exe benchmarks/io_concurrency_benchmark.py \
        --workers 1,2,4,8 --repeats 3 --compress-align \
        --json-out Docs/io_concurrency.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import statistics
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.pipeline_benchmark import CORPUS_LAYOUT, create_corpus, _run_pipeline
from cpu_kernels import warm_cpu_kernels
from cpu_runtime import numpy_cpu_features, runtime_info


def _parse_workers(value: str) -> list[int]:
    values: list[int] = []
    for token in value.split(","):
        try:
            worker = int(token.strip())
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"invalid worker count: {token!r}") from exc
        if worker < 1:
            raise argparse.ArgumentTypeError("worker counts must be >= 1")
        if worker not in values:
            values.append(worker)
    if not values:
        raise argparse.ArgumentTypeError("at least one worker count is required")
    return values


def _median_summary(runs: list[dict[str, object]]) -> dict[str, object]:
    if not runs:
        return {}
    summary: dict[str, object] = {"runs": runs}
    for key in (
        "total_seconds",
        "flow_seconds",
        "align_seconds",
        "stack_seconds",
        "peak_rss_bytes",
        "peak_thread_count",
        "aligned_frames",
        "anchor_detection_cache_hits",
    ):
        values = [float(run[key]) for run in runs]
        summary[f"{key}_median"] = statistics.median(values)
        summary[f"{key}_stdev"] = statistics.stdev(values) if len(values) > 1 else 0.0
    stages: dict[str, dict[str, float]] = {}
    for stage in ("flow", "align", "stack"):
        stage_runs = [
            run.get("stage_telemetry", {}).get(stage)
            for run in runs
            if isinstance(run.get("stage_telemetry"), dict)
        ]
        stage_runs = [item for item in stage_runs if isinstance(item, dict)]
        if not stage_runs:
            continue
        stage_summary: dict[str, float] = {}
        for key in (
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
            values = [float(item.get(key, 0.0)) for item in stage_runs]
            stage_summary[f"{key}_median"] = statistics.median(values)
            stage_summary[f"{key}_stdev"] = statistics.stdev(values) if len(values) > 1 else 0.0
        stages[stage] = stage_summary
    summary["stage_telemetry"] = stages
    summary["digests"] = sorted({str(run.get("output_digest")) for run in runs})
    summary["bitwise_stable_across_runs"] = len(summary["digests"]) == 1
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=_parse_workers, default=[1, 2, 4, 8])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--compress-align", action="store_true")
    parser.add_argument("--compress-stack", action="store_true")
    parser.add_argument("--writer-workers", type=int, default=0,
                        help="Bounded Align writer count (0 keeps the default one)")
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args(argv)
    if args.repeats < 1:
        parser.error("--repeats must be >= 1")

    warm_cpu_kernels()
    runtime = runtime_info()
    corpus_context = tempfile.TemporaryDirectory(prefix="astrobatch-io-compare-")
    corpus_root = Path(corpus_context.name)
    create_corpus(corpus_root)
    results: dict[str, dict[str, object]] = {}
    try:
        for workers in args.workers:
            runs: list[dict[str, object]] = []
            for index in range(args.repeats):
                run_root = corpus_root / f"run_{workers:02d}_{index:02d}"
                shutil.copytree(corpus_root, run_root, ignore=shutil.ignore_patterns("run_*"))
                try:
                    runs.append(_run_pipeline(
                        run_root,
                        "Stable",
                        workers=workers,
                        compress_align=args.compress_align,
                        compress_stack=args.compress_stack,
                        writer_workers=args.writer_workers,
                    ))
                finally:
                    shutil.rmtree(run_root, ignore_errors=True)
            results[str(workers)] = _median_summary(runs)
            summary = results[str(workers)]
            print(
                f"workers={workers} total={summary.get('total_seconds_median', 0):.3f}s "
                f"flow={summary.get('flow_seconds_median', 0):.3f}s "
                f"align={summary.get('align_seconds_median', 0):.3f}s "
                f"stack={summary.get('stack_seconds_median', 0):.3f}s "
                f"peak_rss={summary.get('peak_rss_bytes_median', 0) / 1024**2:.1f}MiB "
                f"peak_threads={summary.get('peak_thread_count_median', 0):.0f}"
            )

        baseline = results.get(str(args.workers[0]), {})
        baseline_time = float(baseline.get("total_seconds_median", 0.0) or 0.0)
        for worker, summary in results.items():
            total = float(summary.get("total_seconds_median", 0.0) or 0.0)
            summary["relative_speedup_vs_first"] = baseline_time / total if total > 0 else None

        worker_digests = {
            digest
            for summary in results.values()
            for digest in summary.get("digests", [])
        }

        report = {
            "schema_version": 1,
            "runtime": runtime.as_dict(),
            "numpy_cpu_features": numpy_cpu_features(),
            "corpus": {
                "loads": [
                    {"name": name, "rgb": rgb, "shape": [height, width], "frames": frames}
                    for name, rgb, height, width, frames in CORPUS_LAYOUT
                ],
            },
            "workers": args.workers,
            "repeats": args.repeats,
            "compress_align": bool(args.compress_align),
            "compress_stack": bool(args.compress_stack),
            "writer_workers_requested": max(0, int(args.writer_workers)),
            "telemetry_includes_observer_thread": True,
            "results": results,
            "bitwise_stable_across_workers": len(worker_digests) == 1,
            "interpretation": {
                "read_write_bytes": "process logical counters; warm-cache runs do not prove physical disk throughput",
                "cpu_to_wall": "values near or above 1 indicate CPU work/parallel overlap; values below 1 indicate wait or I/O overlap",
                "bitwise_gate": "Stable digest must remain constant across worker widths",
            },
        }
        if args.json_out:
            args.json_out.parent.mkdir(parents=True, exist_ok=True)
            args.json_out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        return 0
    finally:
        corpus_context.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
