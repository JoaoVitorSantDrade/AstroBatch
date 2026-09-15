"""Read-only quality comparison for two RGB Lagoon stacks.

The comparison uses the same star coordinates detected in the baseline and
measures each stack independently.  It is deliberately a review aid, not a
scientific recalibration: a robust local background and second moments provide
consistent indicators for trailing/FWHM and per-channel centroid offsets for
the Uranus-C/SV48P chromatic check.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from photutils.detection import DAOStarFinder
from scipy.ndimage import gaussian_filter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lagoon_similarity_stack import _atomic_json, _source_snapshot


FWHM_FACTOR = 2.354820045


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-stars", type=int, default=40)
    return parser.parse_args()


def _read_rgb(path: Path) -> np.ndarray:
    with fits.open(path, memmap=False, ignore_missing_end=True) as hdul:
        hdu = next(
            (
                item
                for item in hdul
                if item.is_image and item.data is not None and getattr(item.data, "ndim", 0) == 3
            ),
            None,
        )
        if hdu is None:
            raise ValueError(f"RGB science HDU ausente: {path}")
        data = np.asarray(hdu.data, dtype=np.float32)
        if data.shape[0] in (3, 4):
            data = np.moveaxis(data[:3], 0, -1)
        elif data.shape[-1] in (3, 4):
            data = data[..., :3]
        else:
            raise ValueError(f"Geometria RGB incompatível: {path} {data.shape}")
        return np.ascontiguousarray(data)


def _detect_candidates(data: np.ndarray, limit: int) -> np.ndarray:
    green = data[:, :, 1]
    finite = np.isfinite(green)
    if not np.any(finite):
        return np.empty((0, 2), dtype=np.float64)
    # Remove the large-scale nebular gradient before DAO detection.  The
    # smoothing radius is much larger than the measured 5--8 px stellar FWHM.
    smooth = gaussian_filter(np.nan_to_num(green, nan=0.0), sigma=20.0)
    high_pass = green - smooth
    high_pass[~finite] = np.nan
    _mean, median, sigma = sigma_clipped_stats(high_pass, sigma=3.0, maxiters=3)
    sigma = float(sigma) if math.isfinite(float(sigma)) and sigma > 0 else 1.0
    finder = DAOStarFinder(
        fwhm=7.5,
        threshold=max(5.0 * sigma, 1.0),
        exclude_border=True,
        peakmax=None,
    )
    table = finder(np.nan_to_num(high_pass - median, nan=0.0))
    if table is None or len(table) == 0:
        return np.empty((0, 2), dtype=np.float64)
    x_key = "x_centroid" if "x_centroid" in table.colnames else "xcentroid"
    y_key = "y_centroid" if "y_centroid" in table.colnames else "ycentroid"
    flux = np.asarray(table["flux"], dtype=np.float64)
    order = np.argsort(flux)[::-1]
    height, width = green.shape
    selected: list[tuple[float, float]] = []
    # Non-maximum suppression prevents a bright PSF halo from becoming several
    # measurements while retaining stars across the full field.
    for index in order:
        x = float(table[x_key][index])
        y = float(table[y_key][index])
        if not (20.0 <= x < width - 20.0 and 20.0 <= y < height - 20.0):
            continue
        if any((x - px) ** 2 + (y - py) ** 2 < 24.0**2 for px, py in selected):
            continue
        selected.append((x, y))
        if len(selected) >= max(1, int(limit)):
            break
    return np.asarray(selected, dtype=np.float64)


def _star_measure(data: np.ndarray, x: float, y: float, radius: int = 10) -> dict[str, object] | None:
    height, width, channels = data.shape
    cx, cy = int(round(x)), int(round(y))
    if channels != 3 or cx - radius < 0 or cy - radius < 0 or cx + radius >= width or cy + radius >= height:
        return None
    yy, xx = np.indices((2 * radius + 1, 2 * radius + 1), dtype=np.float64)
    border = np.zeros_like(yy, dtype=bool)
    border[0, :] = border[-1, :] = True
    border[:, 0] = border[:, -1] = True
    centroids: list[tuple[float, float]] = []
    shapes: list[tuple[float, float, float]] = []
    for channel in range(3):
        cutout = data[cy - radius : cy + radius + 1, cx - radius : cx + radius + 1, channel]
        finite = np.isfinite(cutout)
        border_values = cutout[border & finite]
        if border_values.size < 8:
            return None
        background = float(np.median(border_values))
        residual = np.asarray(cutout, dtype=np.float64) - background
        border_residual = residual[border & finite]
        border_center = float(np.median(border_residual))
        mad = float(np.median(np.abs(border_residual - border_center)))
        threshold = max(0.0, 3.0 * 1.4826 * mad)
        weights = np.where(finite & (residual > threshold), residual, 0.0)
        total = float(np.sum(weights, dtype=np.float64))
        if not math.isfinite(total) or total <= 0:
            return None
        sx = float(np.sum(weights * xx, dtype=np.float64) / total)
        sy = float(np.sum(weights * yy, dtype=np.float64) / total)
        dx = xx - sx
        dy = yy - sy
        var_x = float(np.sum(weights * dx * dx, dtype=np.float64) / total)
        var_y = float(np.sum(weights * dy * dy, dtype=np.float64) / total)
        cov_xy = float(np.sum(weights * dx * dy, dtype=np.float64) / total)
        eigenvalues = np.linalg.eigvalsh(np.asarray([[var_x, cov_xy], [cov_xy, var_y]]))
        minor = max(float(eigenvalues[0]), 0.0)
        major = max(float(eigenvalues[1]), minor)
        major_sigma = math.sqrt(major)
        minor_sigma = math.sqrt(minor)
        if major_sigma <= 1.0e-3:
            return None
        centroids.append((x - radius + sx, y - radius + sy))
        shapes.append(
            (
                float(minor_sigma / major_sigma),
                float(FWHM_FACTOR * math.sqrt(max(major_sigma * minor_sigma, 0.0))),
                float(major_sigma / max(minor_sigma, 1.0e-6)),
            )
        )
    return {
        "centroids": centroids,
        "roundness": float(np.median([shape[0] for shape in shapes])),
        "fwhm": float(np.median([shape[1] for shape in shapes])),
        "elongation": float(np.median([shape[2] for shape in shapes])),
    }


def _summarize(data: np.ndarray, coordinates: np.ndarray) -> dict[str, object]:
    records: list[dict[str, object]] = []
    for x, y in coordinates:
        measurement = _star_measure(data, float(x), float(y))
        if measurement is not None:
            records.append(measurement)
    if not records:
        return {"star_count": 0}
    offsets = np.asarray(
        [
            [
                record["centroids"][0][0] - record["centroids"][1][0],
                record["centroids"][0][1] - record["centroids"][1][1],
                record["centroids"][2][0] - record["centroids"][1][0],
                record["centroids"][2][1] - record["centroids"][1][1],
            ]
            for record in records
        ],
        dtype=np.float64,
    )
    return {
        "star_count": len(records),
        "roundness_median": float(np.median([record["roundness"] for record in records])),
        "fwhm_median_px": float(np.median([record["fwhm"] for record in records])),
        "elongation_median": float(np.median([record["elongation"] for record in records])),
        "rg_offset_median_px": [float(value) for value in np.median(offsets[:, :2], axis=0)],
        "bg_offset_median_px": [float(value) for value in np.median(offsets[:, 2:], axis=0)],
        "rg_offset_abs_median_px": float(np.median(np.linalg.norm(offsets[:, :2], axis=1))),
        "bg_offset_abs_median_px": float(np.median(np.linalg.norm(offsets[:, 2:], axis=1))),
    }


def _stretch(data: np.ndarray) -> np.ndarray:
    result = np.empty_like(data, dtype=np.float32)
    for channel in range(3):
        finite = data[:, :, channel][np.isfinite(data[:, :, channel])]
        low, high = np.percentile(finite, [1.0, 99.7]) if finite.size else (0.0, 1.0)
        normalized = np.clip((data[:, :, channel] - low) / max(float(high - low), 1.0), 0.0, 1.0)
        result[:, :, channel] = np.arcsinh(10.0 * normalized) / np.arcsinh(10.0)
    return np.clip(result, 0.0, 1.0)


def _write_preview(path: Path, baseline: np.ndarray, candidate: np.ndarray, coordinates: np.ndarray) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    x, y, half = 2034, 977, 220
    images = [("baseline", baseline), ("candidate", candidate)]
    figure, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=130, squeeze=False)
    for axis, (title, image) in zip(axes[0], images):
        y1, y2 = max(0, y - half), min(image.shape[0], y + half)
        x1, x2 = max(0, x - half), min(image.shape[1], x + half)
        axis.imshow(_stretch(image[y1:y2, x1:x2]), origin="upper", interpolation="nearest")
        axis.set_title(title)
        axis.axis("off")
    figure.suptitle("Lagoon stack quality review")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, bbox_inches="tight")
    plt.close(figure)


def main() -> int:
    args = _parse_args()
    baseline = args.baseline.resolve()
    candidate = args.candidate.resolve()
    output_dir = args.output_dir.resolve()
    if not baseline.is_file() or not candidate.is_file():
        raise SystemExit("baseline/candidate must be existing FITS files")
    if args.max_stars < 1 or args.max_stars > 128:
        raise SystemExit("max-stars must be between 1 and 128")
    baseline_before = _source_snapshot(baseline.parent)
    baseline_data = _read_rgb(baseline)
    candidate_data = _read_rgb(candidate)
    if baseline_data.shape != candidate_data.shape:
        raise SystemExit(f"incompatible stack shapes: {baseline_data.shape} != {candidate_data.shape}")
    coordinates = _detect_candidates(baseline_data, args.max_stars)
    baseline_metrics = _summarize(baseline_data, coordinates)
    candidate_metrics = _summarize(candidate_data, coordinates)
    output_dir.mkdir(parents=True, exist_ok=True)
    preview = output_dir / "lagoon_stack_quality_preview.png"
    _write_preview(preview, baseline_data, candidate_data, coordinates)
    report = {
        "schema_version": 1,
        "baseline": str(baseline),
        "candidate": str(candidate),
        "output_dir": str(output_dir),
        "detected_star_candidates": int(len(coordinates)),
        "baseline_metrics": baseline_metrics,
        "candidate_metrics": candidate_metrics,
        "preview": str(preview),
        "baseline_unchanged": baseline_before == _source_snapshot(baseline.parent),
        "method": "common baseline DAO candidates; local RGB centroids and second moments",
    }
    _atomic_json(output_dir / "lagoon_stack_quality_report.json", report)
    print(json.dumps(report, indent=2, ensure_ascii=False), flush=True)
    if not report["baseline_unchanged"]:
        raise RuntimeError("baseline tree changed during the read-only comparison")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
