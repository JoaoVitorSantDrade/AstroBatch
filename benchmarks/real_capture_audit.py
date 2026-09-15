"""Read-only audit for a real AstroBatch capture.

The command intentionally separates inspection from processing.  It never
creates a cache, report, profile or temporary FITS file below ``--input-dir``;
the only write is the requested JSON report outside that tree.  A bounded
sample can be sent through the current Stack leaf reader, with all temporary
outputs created by :mod:`tempfile` in the system temporary directory.

Example::

    .venv/Scripts/python.exe benchmarks/real_capture_audit.py \
        --input-dir B:\\AstroImages\\Notebook\\Lagoon\\21_23_00_align \
        --flow-dir B:\\AstroImages\\Notebook\\Lagoon\\21_23_00_batch \
        --profile-dir B:\\AstroImages\\Notebook\\Lagoon\\21_23_00_stack_2 \
        --output-json Docs/lagoon_audit_20260915.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys
import tempfile
import time

import numpy as np
from astropy.io import fits

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import stacking_logic as stacking
from temporal_analysis import build_session_temporal_report


def _natural_key(path: Path) -> tuple[object, ...]:
    import re

    return tuple(
        (0, int(piece)) if piece.isdigit() else (1, piece.casefold())
        for piece in re.split(r"(\d+)", path.name)
    )


def _outside(path: Path, root: Path) -> None:
    """Reject a report path that would mutate the user's source tree."""

    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return
    raise ValueError(f"output must be outside the read-only source: {path}")


def _sample_paths(paths: list[Path], count: int) -> list[Path]:
    if count <= 0 or len(paths) <= count:
        return paths
    # Include both ends of a long session while keeping the choice stable.
    indices = np.linspace(0, len(paths) - 1, count, dtype=int)
    return [paths[int(index)] for index in dict.fromkeys(indices)]


def _inventory(input_dir: Path, paths: list[Path]) -> dict[str, object]:
    sizes = [path.stat().st_size for path in paths]
    by_suffix: dict[str, int] = {}
    by_parent: dict[str, int] = {}
    for path in paths:
        by_suffix[path.suffix.casefold()] = by_suffix.get(path.suffix.casefold(), 0) + 1
        parent = str(path.parent.relative_to(input_dir)) if path.parent != input_dir else "."
        by_parent[parent] = by_parent.get(parent, 0) + 1
    return {
        "fits_count": len(paths),
        "bytes": int(sum(sizes)),
        "size_median_bytes": statistics.median(sizes) if sizes else 0,
        "size_min_bytes": min(sizes) if sizes else 0,
        "size_max_bytes": max(sizes) if sizes else 0,
        "by_suffix": by_suffix,
        "by_parent": by_parent,
    }


def _sample_geometry(paths: list[Path]) -> dict[str, object]:
    geometries: list[stacking.FrameGeometry] = []
    errors: list[dict[str, str]] = []
    for path in paths:
        try:
            geometries.append(stacking.inspect_fits(path))
        except Exception as exc:  # pragma: no cover - depends on user files
            errors.append({"path": str(path), "error": f"{type(exc).__name__}: {exc}"})
    shapes = sorted({
        (geometry.height, geometry.width, geometry.image_kind, geometry.channels)
        for geometry in geometries
    })
    return {
        "sample_count": len(paths),
        "valid_count": len(geometries),
        "errors": errors,
        "shapes": [
            {"height": h, "width": w, "kind": kind, "channels": channels}
            for h, w, kind, channels in shapes
        ],
        "compressed_science": sum(g.science_compressed for g in geometries),
        "scaled_science": sum(g.requires_scaling for g in geometries),
        "raw_mmap_candidate": sum(
            (not g.science_compressed) for g in geometries
        ),
        "bzero_values": sorted({g.source_bzero for g in geometries}),
        "bscale_values": sorted({g.source_bscale for g in geometries}),
        "valid_mask_count": sum(g.mask_hdu_index is not None for g in geometries),
    }


def _sample_stack(paths: list[Path], output_dir: Path) -> dict[str, object]:
    """Run only a bounded Stack leaf; all writes stay in a temp directory."""

    geometries = {path: stacking.inspect_fits(path) for path in paths}
    frames: list[stacking.FrameInfo] = []
    for path in paths:
        geometry = geometries[path]
        frame = stacking.FrameInfo(
            path=path,
            name=path.name,
            batch=path.parent.name,
            metrics={},
            quality=1.0,
            star_count=1.0,
            fwhm=1.0,
            snr=1.0,
            rms=1.0,
            has_valid_mask=geometry.mask_hdu_index is not None,
            shape=(geometry.height, geometry.width),
            image_kind=geometry.image_kind,
            channels=geometry.channels,
        )
        if frame.has_valid_mask:
            frame.valid_mask = stacking._load_full_valid_mask(frame, geometry)
        frames.append(frame)

    reference = geometries[paths[0]]
    config = stacking.StackingConfig(
        method="Median",
        rejection_method="SigmaClip",
        normalize=False,
        # Keep the same per-leaf memory split used by the production four
        # worker configuration, while this bounded audit still executes one
        # leaf in the current process.
        workers=4,
        memory_budget_mb=4096,
        output_dir=output_dir,
    )
    output_path = output_dir / "leaf.fits"
    started = time.perf_counter()
    stacking._process_substack(
        frames,
        geometries,
        [1.0] * len(frames),
        [None] * len(frames),
        config,
        output_path,
        reference.height,
        reference.width,
        reference.channels,
        None,
    )
    elapsed = time.perf_counter() - started
    with fits.open(output_path, memmap=False) as hdul:
        data = np.asarray(hdul[0].data)
        mask = np.asarray(hdul["VALID_MASK"].data)
        counts = np.asarray(hdul["SUB_COUNT"].data)
        digest = hashlib.sha256(
            data.tobytes() + mask.tobytes() + counts.tobytes()
        ).hexdigest()
    return {
        "frames": len(paths),
        "seconds": elapsed,
        "band_rows": stacking._leaf_band_rows(
            config,
            len(paths),
            reference.width,
            reference.channels,
            reference.height,
        ),
        "output_digest": digest,
        "output_dtype": str(data.dtype),
        "mask_dtype": str(mask.dtype),
        "count_dtype": str(counts.dtype),
    }


def _profile_summary(profile_dir: Path | None) -> list[dict[str, object]]:
    if profile_dir is None or not profile_dir.is_dir():
        return []
    summaries: list[dict[str, object]] = []
    decoder = json.JSONDecoder()
    for path in sorted(profile_dir.glob("*.html"), key=_natural_key):
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
            marker = "const sessionData = "
            start = text.index(marker) + len(marker)
            session, _ = decoder.raw_decode(text[start:])
        except (OSError, ValueError, json.JSONDecodeError):
            continue
        duration = session.get("session", {}).get("duration")
        attributes: dict[str, float] = {}

        def walk(node: object) -> None:
            if isinstance(node, dict):
                identifier = str(node.get("identifier", ""))
                if identifier.startswith("process_stacking"):
                    for key, value in (node.get("attributes") or {}).items():
                        try:
                            attributes[str(key)] = float(value)
                        except (TypeError, ValueError):
                            pass
                for value in node.values():
                    walk(value)
            elif isinstance(node, list):
                for value in node:
                    walk(value)

        walk(session.get("frame_tree", []))
        summaries.append({
            "path": str(path),
            "duration_seconds": duration,
            "process_stacking_attributes": attributes,
        })
    return summaries


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--flow-dir", type=Path,
                        help="optional Flow/session directory for temporal report")
    parser.add_argument("--profile-dir", type=Path,
                        help="optional directory containing old profiler HTML")
    parser.add_argument("--sample-frames", type=int, default=8)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args(argv)

    input_dir = args.input_dir.expanduser().resolve()
    output_json = args.output_json.expanduser().resolve()
    if not input_dir.is_dir():
        parser.error(f"input directory does not exist: {input_dir}")
    if args.sample_frames < 1:
        parser.error("--sample-frames must be >= 1")
    source_roots = [input_dir]
    for optional_root in (args.flow_dir, args.profile_dir):
        if optional_root is not None:
            source_roots.append(optional_root.expanduser().resolve())
    try:
        for source_root in source_roots:
            _outside(output_json, source_root)
    except ValueError as exc:
        parser.error(str(exc))

    paths = sorted(
        [path for path in input_dir.rglob("*") if path.is_file() and path.suffix.casefold() in stacking.FITS_SUFFIXES],
        key=lambda path: (_natural_key(path.parent), _natural_key(path)),
    )
    sample = _sample_paths(paths, args.sample_frames)

    report: dict[str, object] = {
        "schema_version": 1,
        "kind": "read_only_real_capture_audit",
        "source": {"input_dir": str(input_dir), "files_unchanged": True},
        "inventory": _inventory(input_dir, paths),
        "sample": {"paths": [str(path) for path in sample], **_sample_geometry(sample)},
        "profiles": _profile_summary(args.profile_dir.expanduser().resolve() if args.profile_dir else None),
        "interpretation": {
            "raw_mmap": "uncompressed FITS with BZERO/BSCALE is read as raw storage and restored to float32; no source-side cache is written",
            "stack_sample": "bounded leaf timing, not a full-session throughput claim",
            "temporal": "review-only; no group is excluded or reweighted",
        },
    }
    if args.flow_dir:
        flow_dir = args.flow_dir.expanduser().resolve()
        report["temporal"] = build_session_temporal_report(flow_dir)

    with tempfile.TemporaryDirectory(prefix="astrobatch-real-audit-") as temporary:
        if sample:
            try:
                report["sample_stack"] = _sample_stack(sample, Path(temporary))
            except Exception as exc:  # pragma: no cover - depends on user files
                report["sample_stack"] = {
                    "error": f"{type(exc).__name__}: {exc}",
                }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps({
        "output_json": str(output_json),
        "fits_count": report["inventory"]["fits_count"],
        "sample_frames": len(sample),
        "source_unchanged": True,
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
