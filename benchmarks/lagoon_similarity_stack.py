"""Build a read-only Lagoon diagnostic stack with opt-in RGB similarity correction.

The source tree is never used as an output directory.  This is intentionally a
small, reproducible diagnostic: it applies the existing Align RGB correction
in memory, then computes a masked mean for frames passing a roundness review
cut.  The production Stack still owns the scientific SigmaClip reduction.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
from astropy.io import fits

# Allow direct execution from the ``benchmarks`` directory without installing
# AstroBatch as a package.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from astroalign_logic import warp_frame
from cpu_runtime import configure_opencv_threads


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True, help="aligned FITS tree (read-only)")
    parser.add_argument("--flow-root", type=Path, required=True, help="Flow batch tree (read-only)")
    parser.add_argument("--output-dir", type=Path, required=True, help="output directory outside source")
    parser.add_argument("--baseline", type=Path, default=None, help="optional existing stack for a preview comparison")
    parser.add_argument("--roundness", type=float, default=0.5953124412305257)
    parser.add_argument("--max-frames", type=int, default=0, help="diagnostic limit; zero means all")
    parser.add_argument("--rgb-mode", choices=("translation", "similarity", "hybrid"), default="similarity")
    return parser.parse_args()


def _read_flow(flow_root: Path) -> dict[tuple[str, str], dict]:
    result: dict[tuple[str, str], dict] = {}
    for batch in sorted(flow_root.glob("batch_*"), key=lambda p: p.name.casefold()):
        path = batch / "flow_local.json"
        if not path.is_file():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        for name, frame in (payload.get("frames", {}) or {}).items():
            if isinstance(frame, dict):
                result[(batch.name, str(name))] = frame
    return result


def _source_snapshot(source: Path) -> dict[str, tuple[int, int]]:
    """Capture a cheap read-only inventory to detect writes during a run."""

    snapshot: dict[str, tuple[int, int]] = {}
    for path in source.rglob("*"):
        if not path.is_file():
            continue
        stat = path.stat()
        snapshot[str(path.relative_to(source))] = (int(stat.st_size), int(stat.st_mtime_ns))
    return snapshot


def _read_rgb_and_mask(path: Path) -> tuple[np.ndarray, np.ndarray, fits.Header]:
    with fits.open(path, memmap=False, ignore_missing_end=True) as hdul:
        image_hdu = None
        mask_hdu = None
        for hdu in hdul:
            if hdu.name == "VALID_MASK":
                mask_hdu = hdu
            elif hdu.is_image and hdu.data is not None and getattr(hdu.data, "ndim", 0) == 3:
                image_hdu = hdu
        if image_hdu is None:
            raise ValueError(f"RGB science HDU ausente: {path}")
        data = np.asarray(image_hdu.data, dtype=np.float32)
        if data.shape[0] in (3, 4):
            data = np.moveaxis(data[:3], 0, -1)
        elif data.shape[-1] in (3, 4):
            data = data[..., :3]
        else:
            raise ValueError(f"Geometria RGB incompatível: {path} {data.shape}")
        if mask_hdu is None or mask_hdu.data is None:
            mask = np.ones(data.shape[:2], dtype=np.uint8)
        else:
            mask = np.asarray(mask_hdu.data, dtype=np.uint8)
        return np.ascontiguousarray(data), np.ascontiguousarray(mask), image_hdu.header.copy()


def _apply_rgb_validity(
    mask: np.ndarray,
    corrected: np.ndarray,
    diagnostics: dict,
) -> np.ndarray:
    """Intersect the original footprint with every RGB correction footprint."""

    valid = np.asarray(mask, dtype=np.float32) > 0.5
    original_mask = valid.copy()
    rgb_shifts = diagnostics.get("rgb_shifts", {})
    rgb_models = diagnostics.get("rgb_models", {})
    for channel, shift in rgb_shifts.items():
        dx, dy, _confidence = shift
        matrix = None
        model = rgb_models.get(channel)
        if isinstance(model, dict):
            candidate = np.asarray(model.get("matrix"), dtype=np.float32)
            if candidate.shape == (2, 3) and np.all(np.isfinite(candidate)):
                matrix = candidate
        if matrix is None:
            matrix = np.asarray([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)
        shifted = cv2.warpAffine(
            original_mask.astype(np.float32),
            matrix,
            (valid.shape[1], valid.shape[0]),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0.0,
        )
        valid &= shifted > 0.5
    valid &= np.all(np.isfinite(corrected), axis=2)
    return valid


def _atomic_fits(path: Path, data: np.ndarray, mask: np.ndarray, header: fits.Header) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        data_uint16 = np.clip(np.nan_to_num(data, nan=0.0, posinf=65535.0, neginf=0.0), 0.0, 65535.0).astype(np.uint16)
        data_uint16 = np.moveaxis(data_uint16, -1, 0)
        clean = header.copy()
        clean["BITPIX"] = 16
        clean["BZERO"] = 32768
        clean["BSCALE"] = 1
        primary = fits.PrimaryHDU(data=data_uint16, header=clean)
        valid = fits.ImageHDU(data=np.asarray(mask, dtype=np.uint8), name="VALID_MASK")
        with fits.HDUList([primary, valid]) as hdul:
            hdul.writeto(temporary, overwrite=True, output_verify="ignore")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _stretch(data: np.ndarray) -> np.ndarray:
    output = np.empty_like(data, dtype=np.float32)
    for channel in range(3):
        plane = data[:, :, channel]
        finite = plane[np.isfinite(plane)]
        low, high = np.percentile(finite, [1.0, 99.7]) if finite.size else (0.0, 1.0)
        normalized = np.clip((plane - low) / max(float(high - low), 1.0), 0.0, 1.0)
        output[:, :, channel] = np.arcsinh(10.0 * normalized) / np.arcsinh(10.0)
    return np.clip(output, 0.0, 1.0)


def _write_preview(path: Path, stacked: np.ndarray, baseline: np.ndarray | None) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    x, y = 2034, 977
    half = 180
    images = [("similarity / roundness", stacked)]
    if baseline is not None and baseline.shape == stacked.shape:
        images.insert(0, ("existing stack", baseline))
    figure, axes = plt.subplots(1, len(images), figsize=(6 * len(images), 5), dpi=130, squeeze=False)
    for axis, (title, image) in zip(axes[0], images):
        y1, y2 = max(0, y - half), min(image.shape[0], y + half)
        x1, x2 = max(0, x - half), min(image.shape[1], x + half)
        axis.imshow(_stretch(image[y1:y2, x1:x2]), origin="upper", interpolation="nearest")
        axis.set_title(title)
        axis.axis("off")
    figure.suptitle("Lagoon diagnostic stack — chromatic similarity")
    figure.tight_layout()
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def _load_baseline(path: Path | None) -> np.ndarray | None:
    if path is None or not path.is_file():
        return None
    with fits.open(path, memmap=False, ignore_missing_end=True) as hdul:
        hdu = next((h for h in hdul if h.is_image and h.data is not None and getattr(h.data, "ndim", 0) == 3), None)
        if hdu is None:
            return None
        data = np.asarray(hdu.data, dtype=np.float32)
        return np.moveaxis(data[:3], 0, -1) if data.shape[0] in (3, 4) else data[..., :3]


def main() -> int:
    args = _parse_args()
    source = args.source.resolve()
    flow_root = args.flow_root.resolve()
    output_dir = args.output_dir.resolve()
    if not source.is_dir() or not flow_root.is_dir():
        raise SystemExit("source/flow-root must be existing directories")
    if output_dir == source or output_dir.is_relative_to(source):
        raise SystemExit("output-dir must be outside the read-only source tree")
    threshold = float(args.roundness)
    if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise SystemExit("roundness must be finite and between 0 and 1")

    source_before = {
        "aligned": _source_snapshot(source),
        "flow": _source_snapshot(flow_root),
    }
    flow = _read_flow(flow_root)
    paths = sorted(source.rglob("*.fits"), key=lambda p: (str(p.parent).casefold(), p.name.casefold()))
    records = []
    for path in paths:
        frame = flow.get((path.parent.name, path.name), {})
        status = str(frame.get("status", "unknown"))
        roundness = frame.get("roundness")
        try:
            roundness_value = float(roundness)
        except (TypeError, ValueError):
            roundness_value = float("nan")
        if status not in {"accepted", "master", "unknown"}:
            continue
        if math.isfinite(roundness_value) and roundness_value < threshold:
            continue
        records.append((path, roundness_value))
    if args.max_frames > 0:
        records = records[: args.max_frames]
    if len(records) < 3:
        raise SystemExit("fewer than three frames pass the diagnostic selection")

    configure_opencv_threads(1)
    sum_rgb = None
    count = None
    shape = None
    model_frames = 0
    fallback_frames = 0
    correction_magnitudes: list[float] = []
    started = time.perf_counter()
    for index, (path, _roundness) in enumerate(records, start=1):
        data, mask, header = _read_rgb_and_mask(path)
        if shape is None:
            shape = data.shape
            sum_rgb = np.zeros(shape, dtype=np.float64)
            count = np.zeros(shape[:2], dtype=np.float64)
        if data.shape != shape:
            raise RuntimeError(f"incompatible RGB shape: {path} {data.shape} != {shape}")
        diagnostics: dict = {}
        corrected = warp_frame(
            data,
            np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32),
            "bilinear",
            rgb_registration=True,
            engine_profile="Fast",
            diagnostics=diagnostics,
            rgb_registration_mode=args.rgb_mode,
        )
        models = diagnostics.get("rgb_models", {})
        if len(models) == 2:
            model_frames += 1
        else:
            fallback_frames += 1
        for model in models.values():
            matrix = np.asarray(model.get("matrix"), dtype=np.float64)
            if matrix.shape == (2, 3):
                correction_magnitudes.append(float(np.hypot(matrix[0, 2], matrix[1, 2])))
        valid = _apply_rgb_validity(mask, corrected, diagnostics)
        sum_rgb += np.asarray(corrected, dtype=np.float64) * valid[:, :, None]
        count += valid
        if index == 1 or index % 25 == 0 or index == len(records):
            elapsed = time.perf_counter() - started
            print(f"{index}/{len(records)} frames | models={model_frames} | {elapsed:.1f}s", flush=True)

    assert sum_rgb is not None and count is not None and shape is not None
    stacked = np.divide(sum_rgb, np.maximum(count[:, :, None], 1.0), dtype=np.float64).astype(np.float32)
    stacked[count <= 0.0] = 0.0
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "lagoon_similarity_roundness_stack.fits"
    header["RGBMODE"] = args.rgb_mode
    header["STACKING"] = "Mean"
    header["STACK_NFR"] = len(records)
    header["STACK_TOT"] = len(paths)
    header["TRAILFLT"] = True
    header["MINROUND"] = threshold
    header["STACK_DIAG"] = True
    _atomic_fits(output_path, stacked, count > 0.0, header)
    baseline = _load_baseline(args.baseline)
    preview_path = output_dir / "lagoon_similarity_roundness_preview.png"
    _write_preview(preview_path, stacked, baseline)
    files_unchanged = source_before == {
        "aligned": _source_snapshot(source),
        "flow": _source_snapshot(flow_root),
    }
    report = {
        "schema_version": 1,
        "source": str(source),
        "flow_root": str(flow_root),
        "output": str(output_path),
        "preview": str(preview_path),
        "files_unchanged": files_unchanged,
        "roundness_threshold": threshold,
        "input_frame_count": len(paths),
        "selected_frame_count": len(records),
        "excluded_frame_count": len(paths) - len(records),
        "rgb_similarity_full_model_frames": model_frames,
        "rgb_similarity_fallback_frames": fallback_frames,
        "median_channel_translation_px": float(np.median(correction_magnitudes)) if correction_magnitudes else None,
        "coverage_frames_min": int(np.min(count)) if count.size else 0,
        "coverage_frames_median": float(np.median(count)) if count.size else 0.0,
        "coverage_frames_max": int(np.max(count)) if count.size else 0,
        "coverage_zero_pixels": int(np.count_nonzero(count <= 0.0)),
        "elapsed_seconds": time.perf_counter() - started,
        "method": f"masked_mean_diagnostic; RGB mode={args.rgb_mode}; production Stack SigmaClip not replaced",
    }
    _atomic_json(output_dir / "lagoon_similarity_roundness_report.json", report)
    if not files_unchanged:
        raise RuntimeError("a source file changed during the read-only diagnostic")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
