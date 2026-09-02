import itertools
import json
import os
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.utils.exceptions import AstropyWarning
from photutils.detection import DAOStarFinder
from scipy.spatial import KDTree

warnings.simplefilter("ignore", category=AstropyWarning)


# ============================================================
# Bayer / CFA (Contrato 100% compatível com [source: 17])
# ============================================================

BAYER_PATTERNS = {"RGGB", "BGGR", "GRBG", "GBRG"}


def get_bayer_pattern(header: fits.Header) -> str | None:
    for key in ["BAYERPAT", "BAYERPATTERN", "COLORTYP"]:
        if key in header:
            val = str(header[key]).strip().upper().strip("'")
            if val in BAYER_PATTERNS:
                return val
    return None


def split_cfa(data: np.ndarray, pattern: str):
    if pattern == "RGGB":
        return data[0::2, 0::2], data[0::2, 1::2], data[1::2, 0::2], data[1::2, 1::2]
    if pattern == "BGGR":
        return data[1::2, 1::2], data[0::2, 1::2], data[1::2, 0::2], data[0::2, 0::2]
    if pattern == "GRBG":
        return data[0::2, 1::2], data[0::2, 0::2], data[1::2, 1::2], data[1::2, 0::2]
    if pattern == "GBRG":
        return data[1::2, 0::2], data[1::2, 1::2], data[0::2, 0::2], data[0::2, 1::2]
    return data[0::2, 0::2], data[0::2, 1::2], data[1::2, 0::2], data[1::2, 1::2]


def extract_luminance(data: np.ndarray, header: fits.Header) -> np.ndarray:
    """Extrai a luminância retornando rigorosamente um array 2D float32 (essencial para o Card Preview)."""
    if data.ndim == 3:
        if data.shape[0] in (3, 4):
            img_hwc = np.moveaxis(data, 0, -1)
        else:
            img_hwc = data
        return (
            0.2126 * img_hwc[:, :, 0]
            + 0.7152 * img_hwc[:, :, 1]
            + 0.0722 * img_hwc[:, :, 2]
        ).astype(np.float32)

    if data.ndim == 2:
        pattern = get_bayer_pattern(header)
        if pattern:
            r, g1, g2, b = split_cfa(data, pattern)
            l_sub = 0.2126 * r + 0.3576 * g1 + 0.3576 * g2 + 0.0722 * b
            l_full = np.repeat(np.repeat(l_sub, 2, axis=0), 2, axis=1)
            return l_full.astype(np.float32)
        return data.astype(np.float32)

    raise ValueError(f"Dimensões não suportadas: {data.shape}")


# ============================================================
# FITS (Contrato compatível)
# ============================================================


def load_fits_data(filepath: Path) -> tuple[np.ndarray, fits.Header]:
    with fits.open(filepath, memmap=False) as hdul:
        for hdu in hdul:
            if hdu.is_image and hdu.data is not None and hdu.data.ndim in (2, 3):
                return np.asarray(hdu.data, dtype=np.float32), hdu.header.copy()
    raise ValueError(f"Imagem inválida: {filepath.name}")


# ============================================================
# Detecção de Estrelas
# ============================================================


def prepare_for_phase_correlation(data: np.ndarray) -> np.ndarray:
    finite_mask = np.isfinite(data)
    if not np.any(finite_mask):
        return np.zeros_like(data, dtype=np.float32)
    finite_data = data[finite_mask]
    d_min, d_max = float(np.min(finite_data)), float(np.max(finite_data))
    if d_max > d_min:
        norm = (data - d_min) / (d_max - d_min)
    else:
        norm = np.zeros_like(data, dtype=np.float32)
    return np.nan_to_num(norm, nan=0.0, posinf=1.0, neginf=0.0).astype(
        np.float32, copy=False
    )


def calculate_anchor_quality(star_count: int, fwhm: float) -> float:
    if fwhm <= 0:
        return float(star_count)
    return float(star_count) / float(fwhm)


def _greedy_spatial_filter(coords: np.ndarray, min_dist: float, max_count: int) -> list:
    """
    Non-max suppression espacial usando grid hashing.
    Substitui a versão anterior O(N^2) (que recriava um np.array a cada
    iteração e comparava contra todos os pontos já aceitos) por uma
    versão O(N) médio, checando apenas as células vizinhas na grade.
    Assume que `coords` já está ordenado por prioridade (ex.: flux decrescente).
    """
    n = len(coords)
    if n == 0:
        return []

    cell_size = max(float(min_dist), 1e-3)
    min_dist_sq = float(min_dist) ** 2
    grid: dict[tuple[int, int], list[int]] = {}
    accepted: list[int] = []

    for idx in range(n):
        x, y = float(coords[idx][0]), float(coords[idx][1])
        cell_x, cell_y = int(x // cell_size), int(y // cell_size)

        too_close = False
        for gx in (cell_x - 1, cell_x, cell_x + 1):
            for gy in (cell_y - 1, cell_y, cell_y + 1):
                bucket = grid.get((gx, gy))
                if not bucket:
                    continue
                for other_idx in bucket:
                    ox, oy = coords[other_idx]
                    if (ox - x) ** 2 + (oy - y) ** 2 < min_dist_sq:
                        too_close = True
                        break
                if too_close:
                    break
            if too_close:
                break

        if not too_close:
            grid.setdefault((cell_x, cell_y), []).append(idx)
            accepted.append(idx)
            if len(accepted) >= max_count:
                break

    return accepted


def detect_stars_dao(
    data: np.ndarray, fwhm: float, sigma: float, max_stars: int
) -> tuple[np.ndarray, float, dict]:
    mean_val, median_val, std_val = (
        float(np.mean(data)),
        float(np.median(data)),
        float(np.std(data)),
    )
    _, bkg_median, bkg_std = sigma_clipped_stats(data, sigma=3.0)
    bkg_median, bkg_std = float(bkg_median), float(bkg_std)

    if not np.isfinite(bkg_std) or bkg_std <= 0:
        metrics = {
            "star_count": 0,
            "fwhm": 0.0,
            "mean": round(mean_val, 2),
            "median": round(median_val, 2),
            "std": round(std_val, 2),
            "background": round(bkg_median, 2),
            "snr": 0.0,
            "min_flux": 0.0,
            "max_flux": 0.0,
            "valid": False,
        }
        return (np.empty((0, 2), dtype=np.float32), 0.0, metrics)

    daofind = DAOStarFinder(fwhm=fwhm, threshold=sigma * bkg_std)
    sources = daofind(data - bkg_median)

    if sources is not None and len(sources) > 0:
        sources.sort("flux")
        sources.reverse()

        raw_coords = np.transpose((sources["xcentroid"], sources["ycentroid"])).astype(
            np.float32
        )
        raw_fluxes = np.asarray(sources["flux"], dtype=np.float32)
        has_sharpness = "sharpness" in sources.colnames
        raw_sharpness = (
            np.asarray(sources["sharpness"], dtype=np.float32)
            if has_sharpness
            else None
        )

        min_dist = fwhm * 1.5
        accepted_idx = _greedy_spatial_filter(raw_coords, min_dist, max_stars)

        coords = (
            raw_coords[accepted_idx]
            if accepted_idx
            else np.empty((0, 2), dtype=np.float32)
        )
        fluxes = (
            raw_fluxes[accepted_idx]
            if accepted_idx
            else np.empty((0,), dtype=np.float32)
        )
        star_count = len(coords)

        current_fwhm = (
            float(np.median(raw_sharpness[accepted_idx]) * fwhm)
            if has_sharpness and star_count > 0
            else float(fwhm)
        )
        min_flux = float(np.min(fluxes)) if star_count > 0 else 0.0
        max_flux = float(np.max(fluxes)) if star_count > 0 else 0.0
        snr = float(np.mean(fluxes) / bkg_std) if star_count > 0 else 0.0
    else:
        coords = np.empty((0, 2), dtype=np.float32)
        current_fwhm, star_count, min_flux, max_flux, snr = 0.0, 0, 0.0, 0.0, 0.0

    valid = bool(star_count > 10 and current_fwhm > 0 and current_fwhm < (fwhm * 2.0))
    metrics = {
        "star_count": star_count,
        "fwhm": round(current_fwhm, 2),
        "mean": round(mean_val, 2),
        "median": round(median_val, 2),
        "std": round(std_val, 2),
        "background": round(bkg_median, 2),
        "snr": round(snr, 2),
        "min_flux": round(min_flux, 2),
        "max_flux": round(max_flux, 2),
        "valid": valid,
    }
    return (coords, current_fwhm, metrics)


def detect_stars_opencv(
    data: np.ndarray, fwhm: float, sigma: float, max_stars: int
) -> tuple[np.ndarray, float, dict]:
    mean_val, median_val, std_val = (
        float(np.mean(data)),
        float(np.median(data)),
        float(np.std(data)),
    )
    norm_img = cv2.normalize(data, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    ksize = max(3, int(fwhm) | 1)
    blurred = cv2.GaussianBlur(norm_img, (ksize, ksize), 0)
    threshold_val = min(
        255.0, float(np.median(blurred)) + (sigma * float(np.std(blurred)))
    )

    _, thresh = cv2.threshold(blurred, threshold_val, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    valid_contours = sorted(
        (c for c in contours if 2 < cv2.contourArea(c) < 1000),
        key=cv2.contourArea,
        reverse=True,
    )[:max_stars]

    coords_list, areas, fluxes = [], [], []
    height, width = data.shape[:2]

    for cnt in valid_contours:
        moments = cv2.moments(cnt)
        if moments["m00"] == 0:
            continue
        cX = moments["m10"] / moments["m00"]
        cY = moments["m01"] / moments["m00"]
        px = int(np.clip(round(cX), 0, width - 1))
        py = int(np.clip(round(cY), 0, height - 1))
        coords_list.append([cX, cY])
        areas.append(cv2.contourArea(cnt))
        fluxes.append(float(data[py, px]))

    coords = (
        np.asarray(coords_list, dtype=np.float32)
        if coords_list
        else np.empty((0, 2), dtype=np.float32)
    )
    star_count = len(coords)
    current_fwhm = (
        float(np.mean([np.sqrt(area / np.pi) * 2.0 for area in areas]))
        if areas
        else 0.0
    )
    min_flux = float(np.min(fluxes)) if fluxes else 0.0
    max_flux = float(np.max(fluxes)) if fluxes else 0.0
    snr = float(np.mean(fluxes) / std_val) if std_val > 0 and fluxes else 0.0

    valid = bool(star_count > 10 and current_fwhm > 0 and current_fwhm < (fwhm * 2.5))
    metrics = {
        "star_count": star_count,
        "fwhm": round(current_fwhm, 2),
        "mean": round(mean_val, 2),
        "median": round(median_val, 2),
        "std": round(std_val, 2),
        "background": round(median_val, 2),
        "snr": round(snr, 2),
        "min_flux": round(min_flux, 2),
        "max_flux": round(max_flux, 2),
        "valid": valid,
    }
    return (coords, current_fwhm, metrics)


def detect_stars(
    data: np.ndarray, fwhm: float, sigma: float, max_stars: int, engine: str = "DAO"
) -> tuple[np.ndarray, float, dict]:
    if str(engine).upper() == "OPENCV":
        return detect_stars_opencv(data, fwhm, sigma, max_stars)
    return detect_stars_dao(data, fwhm, sigma, max_stars)


# ============================================================
# Geometria
# ============================================================


def extract_geometric_properties(
    matrix_2x3: np.ndarray,
) -> tuple[float, float, float, float]:
    a, b, tx = matrix_2x3[0]
    c, d, ty = matrix_2x3[1]
    scale = float(np.sqrt(a**2 + c**2))
    rotation_deg = float(np.degrees(np.arctan2(c, a)))
    return (float(tx), float(ty), rotation_deg, scale)


def validate_transform(
    matrix_2x3: np.ndarray | None, metrics: dict, limits: dict
) -> tuple[bool, str]:
    if matrix_2x3 is None:
        if metrics.get("matches", 0) == 0:
            return (False, "phase_correlation_failed")
        return (False, "insufficient_matches")

    if metrics.get("inliers", 0) < limits["min_inliers"]:
        return (False, "insufficient_inliers")
    if metrics.get("inlier_ratio", 0.0) < limits["min_ratio"]:
        return (False, "low_inlier_ratio")
    if metrics.get("rms", 999.0) > limits["max_rms"]:
        return (False, "high_rms")

    tx, ty, rotation, scale = extract_geometric_properties(matrix_2x3)
    metrics["translation"] = [round(tx, 3), round(ty, 3)]
    metrics["translation_magnitude"] = round(float(np.hypot(tx, ty)), 3)
    metrics["rotation_deg"] = round(rotation, 4)
    metrics["scale"] = round(scale, 6)

    if np.hypot(tx, ty) > limits["max_translation"]:
        return (False, "high_translation")
    if abs(rotation) > limits["max_rotation"]:
        return (False, "high_rotation")
    if not (limits["min_scale"] <= scale <= limits["max_scale"]):
        return (False, "invalid_scale")

    return (True, "accepted")


def make_homogeneous(matrix_2x3: np.ndarray) -> np.ndarray:
    hom = np.eye(3, dtype=np.float64)
    hom[:2, :] = matrix_2x3
    return hom


def _match_incremental_stars(
    previous_stars: np.ndarray,
    current_stars: np.ndarray,
    shift: tuple[float, float],
    matching_radius: float,
) -> tuple[list, list]:
    if len(previous_stars) == 0 or len(current_stars) == 0:
        return [], []
    dx, dy = shift
    shifted_current = current_stars + np.array([dx, dy], dtype=np.float32)
    tree = KDTree(previous_stars)
    distances, indices = tree.query(
        shifted_current, distance_upper_bound=matching_radius
    )

    candidates = [
        (float(distance), int(previous_idx), int(current_idx))
        for current_idx, (distance, previous_idx) in enumerate(zip(distances, indices))
        if np.isfinite(distance) and previous_idx < len(previous_stars)
    ]
    candidates.sort(key=lambda item: item[0])

    used_previous = set()
    previous_points, current_points = [], []
    for distance, previous_idx, current_idx in candidates:
        if previous_idx in used_previous:
            continue
        used_previous.add(previous_idx)
        previous_points.append(previous_stars[previous_idx])
        current_points.append(current_stars[current_idx])

    return previous_points, current_points


# ============================================================
# Estimativa de Transformação via USAC_MAGSAC
# ============================================================


def _estimate_incremental_transform(
    previous_stars: list | np.ndarray,
    current_stars: list | np.ndarray,
    ransac_thresh: float,
    min_stars: int,
):
    matches = min(len(previous_stars), len(current_stars))
    if matches < min_stars:
        return (
            None,
            {"matches": matches, "inliers": 0, "inlier_ratio": 0.0, "rms": 999.0},
        )

    previous = np.asarray(previous_stars, dtype=np.float32)
    current = np.asarray(current_stars, dtype=np.float32)

    matrix_2x3, inliers = cv2.estimateAffinePartial2D(
        current,
        previous,
        method=cv2.RANSAC,
        ransacReprojThreshold=ransac_thresh,
        maxIters=4000,
        confidence=0.9999,
        refineIters=25,
    )

    if matrix_2x3 is None or inliers is None:
        return (
            None,
            {"matches": len(previous), "inliers": 0, "inlier_ratio": 0.0, "rms": 999.0},
        )

    mask = inliers.ravel().astype(bool)
    inlier_count = int(mask.sum())
    if inlier_count == 0:
        return (
            None,
            {"matches": len(previous), "inliers": 0, "inlier_ratio": 0.0, "rms": 999.0},
        )

    transformed = cv2.transform(current.reshape(-1, 1, 2), matrix_2x3).reshape(-1, 2)
    errors = transformed[mask] - previous[mask]
    distances = np.linalg.norm(errors, axis=1)
    rms = float(np.sqrt(np.mean(distances**2)))
    inlier_ratio = inlier_count / len(previous)

    return (
        matrix_2x3,
        {
            "matches": len(previous),
            "inliers": inlier_count,
            "inlier_ratio": float(inlier_ratio),
            "rms": rms,
        },
    )


# ============================================================
# Local Flow
# ============================================================


def _process_single_frame(
    filepath: Path,
    fwhm_val: float,
    sigma_val: float,
    max_stars_val: int,
    min_stars: int,
    engine_val: str,
) -> tuple[str, dict | None]:
    try:
        data, header = load_fits_data(filepath)
        working_data = extract_luminance(data, header)

        current_sigma = sigma_val
        best_stars, best_fwhm, best_metrics = [], 0.0, {}
        target_stars = max(20, min_stars * 2)

        while current_sigma >= 3.0:
            stars, measured_fwhm, metrics = detect_stars(
                working_data, fwhm_val, current_sigma, max_stars_val, engine_val
            )
            best_stars, best_fwhm, best_metrics = stars, measured_fwhm, metrics
            if len(stars) >= target_stars:
                break
            current_sigma -= 0.5

        if len(best_stars) < min_stars:
            return (
                filepath.name,
                {
                    "path": filepath,
                    "data": data,
                    "phase_data": None,
                    "stars": best_stars,
                    "fwhm": best_fwhm,
                    "metrics": best_metrics,
                    "status": "rejected",
                    "reason": "insufficient_stars_in_detection",
                },
            )

        phase_data = prepare_for_phase_correlation(working_data)
        return (
            filepath.name,
            {
                "path": filepath,
                "data": data,
                "phase_data": phase_data,
                "stars": best_stars,
                "fwhm": best_fwhm,
                "metrics": best_metrics,
                "status": "prepared",
            },
        )
    except Exception as exc:
        return (
            filepath.name,
            {
                "status": "error",
                "reason": str(exc),
                "path": filepath,
                "data": None,
                "phase_data": None,
                "stars": np.empty((0, 2), dtype=np.float32),
                "fwhm": 0.0,
                "metrics": {},
            },
        )


def process_local_flow(batch_dir: Path, config: dict, app_print) -> dict:
    files = sorted(
        [
            p
            for p in batch_dir.iterdir()
            if p.is_file() and p.suffix.lower() in {".fit", ".fits", ".fts"}
        ]
    )
    if not files:
        return {}
    if not isinstance(config, dict):
        config = {}

    chosen_anchor_name = config.get("custom_anchors", {}).get(batch_dir.name)
    anchor_file = files[0]
    if chosen_anchor_name and any(p.name == chosen_anchor_name for p in files):
        anchor_file = next(p for p in files if p.name == chosen_anchor_name)
        app_print(
            f"[{batch_dir.name}] Frame Central (Batch Reference): {chosen_anchor_name}\n"
        )
    else:
        chosen_anchor_name = anchor_file.name
        app_print(
            f"[{batch_dir.name}] Nenhuma referência manual válida. Usando o 1º frame.\n"
        )

    fwhm_val = float(config.get("fwhm", 4.0))
    sigma_val = float(config.get("sigma", 5.0))
    max_stars_val = int(config.get("max_stars", 250))
    matching_radius = float(config.get("matching_radius", 25.0))
    ransac_thresh = float(config.get("ransac", 4.0))
    engine_val = config.get("engine", "DAO")

    limits = {
        "min_stars": int(config.get("min_stars", 4)),
        "min_inliers": int(config.get("min_inliers", 4)),
        "min_ratio": float(config.get("min_ratio", 0.15)),
        "max_rms": float(config.get("max_rms", 4.0)),
        "max_translation": float(config.get("max_translation", 1500.0)),
        "max_rotation": float(config.get("max_rotation", 10.0)),
        "min_scale": float(config.get("min_scale", 0.95)),
        "max_scale": float(config.get("max_scale", 1.05)),
    }

    try:
        cpu_count = getattr(os, "process_cpu_count", os.cpu_count)() or 1
    except Exception:
        cpu_count = os.cpu_count() or 1
    worker_count = max(1, min(2, cpu_count / 4))

    prepared_frames = {}
    with ThreadPoolExecutor(
        max_workers=worker_count, thread_name_prefix="astroflow"
    ) as executor:
        futures = {
            executor.submit(
                _process_single_frame,
                filepath,
                fwhm_val,
                sigma_val,
                max_stars_val,
                limits["min_stars"],
                engine_val,
            ): filepath
            for filepath in files
        }
        for future in as_completed(futures):
            filepath = futures[future]
            try:
                fname, result = future.result()
                prepared_frames[fname] = result
            except Exception as exc:
                app_print(f"[{filepath.name}] Erro no worker: {exc}\n")

    anchor = prepared_frames.get(anchor_file.name)
    if (
        anchor is None
        or anchor.get("status") == "error"
        or len(anchor["stars"]) < limits["min_stars"]
    ):
        app_print(f"[{batch_dir.name}] ERRO: Falha na âncora.\n")
        return {}

    anchor_quality = calculate_anchor_quality(len(anchor["stars"]), anchor["fwhm"])

    flow_data = {
        "schema_version": 2,
        "batch_anchor": anchor_file.name,
        "selected_reference": chosen_anchor_name,
        "mode": "incremental_chain",
        "workers": worker_count,
        "engine": engine_val,
        "parameters": {
            "fwhm": fwhm_val,
            "sigma": sigma_val,
            "max_stars": max_stars_val,
            "matching_radius": matching_radius,
            "ransac": ransac_thresh,
            **limits,
        },
        "anchor_metrics": {
            "star_count": len(anchor["stars"]),
            "fwhm": anchor["fwhm"],
            "quality": anchor_quality,
            **anchor["metrics"],
        },
        "frames": {},
    }

    flow_data["frames"][anchor_file.name] = {
        "status": "accepted",
        "matrix": np.eye(3).tolist(),
        "relative_to": None,
        "matches": len(anchor["stars"]),
        "inliers": len(anchor["stars"]),
        "inlier_ratio": 1.0,
        "rms": 0.0,
        "translation": [0.0, 0.0],
        "translation_magnitude": 0.0,
        "rotation_deg": 0.0,
        "scale": 1.0,
        "cumulative_rms": 0.0,
        "star_count": len(anchor["stars"]),
        "fwhm": anchor["fwhm"],
    }

    previous_name = anchor_file.name
    previous_frame = anchor
    temporal_history: list[tuple[float, float, float]] = []

    for index in range(1, len(files)):
        current_file = files[index]
        current_name = current_file.name
        current_frame = prepared_frames.get(current_name)

        if (
            current_frame is None
            or current_frame.get("status") == "error"
            or len(current_frame["stars"]) < limits["min_stars"]
        ):
            flow_data["frames"][current_name] = {
                "status": "rejected",
                "reason": "insufficient_stars_or_error",
            }
            continue

        attempts = [
            ("normal", previous_frame, matching_radius, previous_name),
            ("relaxed_radius", previous_frame, matching_radius * 2.0, previous_name),
        ]
        accepted = False
        best_metrics = {
            "reason": "phase_correlation_failed",
            "matches": 0,
            "inliers": 0,
            "inlier_ratio": 0.0,
            "rms": 999.0,
        }

        for attempt_name, ref_frame, radius, ref_name in attempts:
            if (
                ref_frame.get("phase_data") is None
                or current_frame.get("phase_data") is None
            ):
                continue

            shift, response = cv2.phaseCorrelate(
                ref_frame["phase_data"], current_frame["phase_data"]
            )
            dx, dy = shift
            m_ref, m_cur = _match_incremental_stars(
                ref_frame["stars"], current_frame["stars"], (dx, dy), radius
            )
            rel_matrix, metrics = _estimate_incremental_transform(
                m_ref, m_cur, ransac_thresh, limits["min_stars"]
            )

            metrics["phase_shift"] = [round(float(dx), 3), round(float(dy), 3)]
            metrics["phase_response"] = round(float(response), 5)
            valid, reason = validate_transform(rel_matrix, metrics, limits)

            # Validação Temporal por Inércia
            if valid and rel_matrix is not None:
                tx, ty, rot, sc = extract_geometric_properties(rel_matrix)
                if len(temporal_history) >= 4:
                    med_dx = np.median([h[0] for h in temporal_history[:8]])
                    med_dy = np.median([h[1] for h in temporal_history[:8]])
                    med_rot = np.median([h[2] for h in temporal_history[:8]])
                    if (
                        abs(tx - med_dx) > 120.0
                        or abs(ty - med_dy) > 120.0
                        or abs(rot - med_rot) > 2.0
                    ):
                        valid = False
                        reason = "temporal_validation_failed"

            if valid:
                accepted = True
                relative_homogeneous = make_homogeneous(rel_matrix)
                ref_matrix = np.asarray(
                    flow_data["frames"][ref_name]["matrix"], dtype=np.float64
                )
                cumulative_matrix = ref_matrix @ relative_homogeneous

                prev_cum_rms = float(
                    flow_data["frames"][ref_name].get("cumulative_rms", 0.0)
                )
                curr_rms = float(metrics.get("rms", 0.0))
                cumulative_rms = float(np.sqrt(prev_cum_rms**2 + curr_rms**2))

                flow_data["frames"][current_name] = {
                    "status": "accepted",
                    "matrix": cumulative_matrix.tolist(),
                    "relative_matrix": relative_homogeneous.tolist(),
                    "relative_to": ref_name,
                    "recovery_method": attempt_name,
                    "cumulative_rms": cumulative_rms,
                    "fwhm": current_frame["fwhm"],
                    "star_count": len(current_frame["stars"]),
                    **metrics,
                }

                if len(temporal_history) < 8:
                    tx, ty, rot, _ = extract_geometric_properties(rel_matrix)
                    temporal_history.append((tx, ty, rot))

                app_print(
                    f"[{current_name}] OK ({attempt_name}) <- {ref_name} | "
                    f"{metrics['inliers']}/{metrics['matches']} inliers | "
                    f"ratio={metrics['inlier_ratio']:.1%} | RMS={metrics['rms']:.3f}px\n"
                )
                break

            metrics["reason"] = reason
            if metrics.get("inliers", 0) > best_metrics.get("inliers", 0) or (
                metrics.get("inliers", 0) == best_metrics.get("inliers", 0)
                and metrics.get("rms", 999.0) < best_metrics.get("rms", 999.0)
            ):
                best_metrics = metrics

        if accepted:
            previous_name = current_name
            previous_frame = current_frame
        else:
            flow_data["frames"][current_name] = {
                "status": "rejected",
                "reason": best_metrics.get("reason", "unknown"),
                "matches": best_metrics.get("matches", 0),
                "inliers": best_metrics.get("inliers", 0),
                "inlier_ratio": best_metrics.get("inlier_ratio", 0.0),
                "rms": best_metrics.get("rms", 999.0),
                "phase_shift": best_metrics.get("phase_shift", [0.0, 0.0]),
                "phase_response": best_metrics.get("phase_response", 0.0),
                "fwhm": current_frame["fwhm"],
                "star_count": len(current_frame["stars"]),
            }
            app_print(
                f"[{current_name}] REJEITADO: {best_metrics.get('reason', 'unknown')} | "
                f"inliers={best_metrics.get('inliers', 0)} | RMS={best_metrics.get('rms', 999.0):.3f}px\n"
            )

    # Recentralização se a âncora manual não for o 1º frame
    selected_reference_info = flow_data["frames"].get(chosen_anchor_name)
    if selected_reference_info and selected_reference_info.get("status") == "accepted":
        reference_matrix = np.asarray(
            selected_reference_info["matrix"], dtype=np.float64
        )
        try:
            inverse_reference = np.linalg.inv(reference_matrix)
            for frame_data in flow_data["frames"].values():
                if frame_data.get("status") == "accepted":
                    old_matrix = np.asarray(frame_data["matrix"], dtype=np.float64)
                    frame_data["matrix"] = (inverse_reference @ old_matrix).tolist()
            flow_data["batch_anchor"] = chosen_anchor_name
        except np.linalg.LinAlgError:
            pass

    accepted_frames = [
        f for f in flow_data["frames"].values() if f.get("status") == "accepted"
    ]
    rejected_frames = [
        f for f in flow_data["frames"].values() if f.get("status") == "rejected"
    ]
    valid_count = len(accepted_frames)
    total_count = len(files)
    coverage = valid_count / total_count if total_count else 0.0

    flow_data["statistics"] = {
        "total_frames": total_count,
        "accepted_frames": valid_count,
        "rejected_frames": len(rejected_frames),
        "coverage": coverage,
        "first_frame": files[0].name,
        "last_frame": files[-1].name,
        "first_frame_valid": (
            flow_data["frames"].get(files[0].name, {}).get("status") == "accepted"
        ),
        "last_frame_valid": (
            flow_data["frames"].get(files[-1].name, {}).get("status") == "accepted"
        ),
        "chain_segments": _count_chain_segments(flow_data),
    }

    output_path = batch_dir / "flow_local.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(flow_data, f, indent=4, ensure_ascii=False)

    app_print(
        f"[{batch_dir.name}] Flow salvo: {valid_count}/{total_count} ({coverage:.1%})\n"
    )

    return {
        "batch_name": batch_dir.name,
        "anchor_path": anchor_file,
        "anchor_data": anchor["phase_data"],
        "anchor_stars": anchor["stars"],
        "star_count": len(anchor["stars"]),
        "fwhm": anchor["fwhm"],
        "anchor_quality": anchor_quality,
        "anchor_metrics": anchor["metrics"],
        "valid_frames": valid_count,
        "total_frames": total_count,
        "coverage": coverage,
    }


def _count_chain_segments(flow_data: dict) -> int:
    segments, previous_accepted = 0, False
    for frame_data in flow_data.get("frames", {}).values():
        accepted = frame_data.get("status") == "accepted"
        if accepted and not previous_accepted:
            segments += 1
        previous_accepted = accepted
    return segments


# ============================================================
# Geometric Quad Asterism Hashing
# ============================================================


def _build_quad_hash(
    points_4: np.ndarray,
) -> tuple[tuple[float, float, float, float], list[int]] | None:
    """
    Normaliza 4 pontos em um sistema de coordenadas invariante a escala e rotação:
    A e B tornam-se (0,0) e (1,1). C e D geram o hash 4D.
    """
    # Encontra os dois pontos mais distantes para servir de eixo de base
    best_dist = -1.0
    best_pair = (0, 1)
    for i in range(4):
        for j in range(i + 1, 4):
            d = np.hypot(
                points_4[i, 0] - points_4[j, 0], points_4[i, 1] - points_4[j, 1]
            )
            if d > best_dist:
                best_dist = d
                best_pair = (i, j)

    if best_dist < 1e-4:
        return None

    i_a, i_b = best_pair
    pt_a, pt_b = points_4[i_a], points_4[i_b]

    # Ordena para garantir orientação consistente
    if pt_a[0] > pt_b[0] or (pt_a[0] == pt_b[0] and pt_a[1] > pt_b[1]):
        i_a, i_b = i_b, i_a
        pt_a, pt_b = points_4[i_a], points_4[i_b]

    # Vetores de transformação para base normalizada
    dx = float(pt_b[0] - pt_a[0])
    dy = float(pt_b[1] - pt_a[1])
    scale_sq = dx * dx + dy * dy

    # Os outros dois pontos restantes
    others = [idx for idx in range(4) if idx != i_a and idx != i_b]
    pt_c = points_4[others[0]]
    pt_d = points_4[others[1]]

    # Projeção Afim 2D
    def normalize_pt(pt):
        px = float(pt[0] - pt_a[0])
        py = float(pt[1] - pt_a[1])
        nx = (px * dx + py * dy) / scale_sq
        ny = (py * dx - px * dy) / scale_sq
        return nx, ny

    cx, cy = normalize_pt(pt_c)
    dx_, dy_ = normalize_pt(pt_d)

    # Ordenação canônica entre C e D
    if cx > dx_ or (cx == dx_ and cy > dy_):
        cx, dx_ = dx_, cx
        cy, dy_ = dy_, cy
        others[0], others[1] = others[1], others[0]

    hash_key = (float(cx), float(cy), float(dx_), float(dy_))
    ordering = [i_a, i_b, others[0], others[1]]
    return hash_key, ordering


def _extract_asterism_database(
    stars: np.ndarray, max_stars: int = 50
) -> tuple[list[tuple], list[list[int]]]:
    """Extrai quads locais usando K-Nearest Neighbors para manter complexidade O(N)."""
    if len(stars) < 4:
        return [], []

    subset = stars[:max_stars]
    tree = KDTree(subset)
    hashes = []
    star_quads = []
    seen_quads = set()

    k_neighbors = min(8, len(subset))
    for i in range(len(subset)):
        _, neighbors = tree.query(subset[i], k=k_neighbors)
        # Gera combinações de 4 estrelas dentro do grupo de vizinhos
        for combo in itertools.combinations(neighbors, 4):
            quad_key = tuple(sorted(combo))
            if quad_key in seen_quads:
                continue
            seen_quads.add(quad_key)

            quad_pts = subset[list(combo)]
            result = _build_quad_hash(quad_pts)
            if result is not None:
                h_key, ordering = result
                hashes.append(h_key)
                star_quads.append([combo[idx] for idx in ordering])

    return hashes, star_quads


def _match_quad_asterisms(
    ref_stars: np.ndarray, tgt_stars: np.ndarray, tolerance: float = 0.02
) -> tuple[np.ndarray, np.ndarray]:
    """Pareia estrelas comparando as distâncias euclidianas no espaço de hash 4D."""
    ref_hashes, ref_quads = _extract_asterism_database(ref_stars, max_stars=60)
    tgt_hashes, tgt_quads = _extract_asterism_database(tgt_stars, max_stars=60)

    if not ref_hashes or not tgt_hashes:
        return np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32)

    ref_tree = KDTree(ref_hashes)
    match_pairs = {}

    for t_idx, t_hash in enumerate(tgt_hashes):
        dist, r_idx = ref_tree.query(t_hash)
        if dist <= tolerance:
            # Associa os 4 vértices correspondentes
            r_star_indices = ref_quads[r_idx]
            t_star_indices = tgt_quads[t_idx]

            for r_s, t_s in zip(r_star_indices, t_star_indices):
                match_pairs[r_s] = match_pairs.get(r_s, set())
                match_pairs[r_s].add(t_s)

    pts_ref = []
    pts_tgt = []
    for r_s, t_set in match_pairs.items():
        # Filtra pareamentos consistentes (1 para 1)
        if len(t_set) == 1:
            t_s = list(t_set)[0]
            pts_ref.append(ref_stars[r_s])
            pts_tgt.append(tgt_stars[t_s])

    if not pts_ref:
        return np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32)

    return np.asarray(pts_ref, dtype=np.float32), np.asarray(pts_tgt, dtype=np.float32)


# ============================================================
# Global Flow
# ============================================================


def _global_limits(config: dict) -> dict:
    user_inliers = int(config.get("global_min_inliers", config.get("min_inliers", 4)))
    return {
        "min_inliers": max(3, user_inliers),
        "min_ratio": 0.1,
        "max_rms": float(config.get("global_max_rms", 4.0)),
        "max_translation": float(config.get("global_max_translation", 3000.0)),
        "max_rotation": float(config.get("global_max_rotation", 20.0)),
        "min_scale": max(0.96, float(config.get("global_min_scale", 0.95))),
        "max_scale": min(1.04, float(config.get("global_max_scale", 1.05))),
    }


def _estimate_global_pair(
    ref_info: dict,
    target_info: dict,
    matching_radius: float,
    ransac_thresh: float,
    limits: dict,
):
    stars_ref = ref_info.get("anchor_stars")
    stars_tgt = target_info.get("anchor_stars")

    if (
        stars_ref is None
        or stars_tgt is None
        or len(stars_ref) < 4
        or len(stars_tgt) < 4
    ):
        return (
            None,
            {
                "status": "rejected",
                "reason": "missing_data",
                "matches": 0,
                "inliers": 0,
                "inlier_ratio": 0.0,
                "rms": 999.0,
            },
        )

    # 1. Pareamento por Hashing de Quads Geométricos (Invariante a Rotação e Translação)
    pts_ref, pts_tgt = _match_quad_asterisms(stars_ref, stars_tgt, tolerance=0.02)

    # 2. Resolução Robusta com MAGSAC++
    matrix_2x3, metrics = _estimate_incremental_transform(
        pts_ref, pts_tgt, ransac_thresh, limits["min_inliers"]
    )

    metrics["phase_shift"] = [0.0, 0.0]
    metrics["phase_response"] = 1.0

    # 3. Validação Limítrofe dos Parâmetros Geométricos
    valid, reason = validate_transform(matrix_2x3, metrics, limits)
    if not valid:
        metrics["status"] = "rejected"
        metrics["reason"] = reason
        return (None, metrics)

    metrics["status"] = "accepted"
    return (make_homogeneous(matrix_2x3), metrics)


def _matrix_difference_score(first: np.ndarray, second: np.ndarray) -> float:
    return float(np.sqrt(np.mean((first - second) ** 2)))


def _detect_anchor_stars_task(
    info: dict, fwhm_val: float, base_sigma: float, engine_val: str
):
    """Worker independente por batch: usado para paralelizar a fase de detecção de
    estrelas-âncora do Global Flow (anteriormente sequencial)."""
    data, header = load_fits_data(info["anchor_path"])
    working_data = extract_luminance(data, header)

    current_sigma = base_sigma
    best_stars, best_fwhm = [], fwhm_val

    while current_sigma >= 2.8:
        g_stars, g_fwhm, _ = detect_stars(
            working_data, fwhm_val, current_sigma, 250, engine_val
        )
        best_stars, best_fwhm = g_stars, g_fwhm
        if len(g_stars) >= 35:
            break
        current_sigma -= 0.2

    phase_data = prepare_for_phase_correlation(working_data)
    return info, working_data.shape, best_stars, best_fwhm, current_sigma, phase_data


def process_all_flows(
    base_dir: Path, config: dict, app_print, app_progress, cancel_event
):
    if not isinstance(config, dict):
        config = {}

    batch_folders = sorted(
        [d for d in base_dir.iterdir() if d.is_dir() and "batch" in d.name.lower()],
        key=lambda p: p.name.lower(),
    )
    total_batches = len(batch_folders)

    if not batch_folders:
        app_print(f"Nenhuma subpasta de Batch encontrada em {base_dir}\n")
        return

    app_progress(0, total_batches, "Iniciando AstroFlow...")
    anchors_info = []
    skip_local = bool(config.get("skip_local_flow", False))

    if skip_local:
        app_print("\n[GLOBAL] Recarregando Flows Locais existentes...\n")
        for batch_folder in batch_folders:
            flow_path = batch_folder / "flow_local.json"
            if not flow_path.exists():
                app_print(
                    f"[{batch_folder.name}] AVISO: flow_local.json ausente. Ignorando.\n"
                )
                continue

            try:
                with flow_path.open("r", encoding="utf-8") as f:
                    local_data = json.load(f)

                anchor_name = local_data.get("batch_anchor")
                if not anchor_name:
                    continue

                anchor_metrics = local_data.get("anchor_metrics", {})
                info = {
                    "batch_name": batch_folder.name,
                    "anchor_path": batch_folder / anchor_name,
                    "anchor_data": None,
                    "anchor_stars": [],
                    "shape": None,
                    "star_count": anchor_metrics.get("star_count", 0),
                    "fwhm": anchor_metrics.get("fwhm", 0.0),
                    "anchor_quality": anchor_metrics.get("quality", 0.0),
                    "anchor_metrics": anchor_metrics,
                    "valid_frames": local_data.get("statistics", {}).get(
                        "accepted_frames", 0
                    ),
                    "total_frames": local_data.get("statistics", {}).get(
                        "total_frames", 0
                    ),
                    "coverage": local_data.get("statistics", {}).get("coverage", 0.0),
                }
                anchors_info.append(info)
            except Exception as exc:
                app_print(f"[{batch_folder.name}] Erro ao ler json: {exc}\n")

    else:
        try:
            cpu_count = getattr(os, "process_cpu_count", os.cpu_count)() or 1
        except Exception:
            cpu_count = os.cpu_count() or 1

        local_workers = max(1, min(4, cpu_count))
        with ThreadPoolExecutor(
            max_workers=local_workers, thread_name_prefix="astroflow-batch"
        ) as executor:
            futures = {
                executor.submit(
                    process_local_flow,
                    batch_folder,
                    config,
                    lambda message: app_print(message),
                ): batch_folder
                for batch_folder in batch_folders
            }
            for future in as_completed(futures):
                if cancel_event.is_set():
                    return
                batch_folder = futures[future]
                try:
                    info = future.result()
                    if info:
                        anchors_info.append(info)
                    app_print(f"Flow Local Finalizado: {batch_folder.name}\n")
                except Exception as exc:
                    app_print(f"Erro em {batch_folder.name}: {exc}\n")

    if not anchors_info:
        app_print("Nenhum Flow Local válido foi produzido ou encontrado.\n")
        return

    anchors_info.sort(key=lambda x: x["batch_name"].lower())

    base_sigma = float(config.get("sigma", 5.0))
    engine_val = config.get("engine", "DAO")
    fwhm_val = float(config.get("fwhm", 4.0))

    app_print("\n[GLOBAL] Gerando imagens sintéticas para Pareamento Global...\n")

    try:
        cpu_count = getattr(os, "process_cpu_count", os.cpu_count)() or 1
    except Exception:
        cpu_count = os.cpu_count() or 1
    global_workers = max(1, min(8, cpu_count))

    # Detecção de estrelas-âncora por batch é independente entre si: paraleliza
    # (antes era um loop sequencial de I/O + CPU, um dos principais gargalos).
    with ThreadPoolExecutor(
        max_workers=global_workers, thread_name_prefix="astroflow-anchor"
    ) as executor:
        futures = {
            executor.submit(
                _detect_anchor_stars_task, info, fwhm_val, base_sigma, engine_val
            ): info
            for info in anchors_info
        }
        for future in as_completed(futures):
            if cancel_event.is_set():
                return
            info, shape, best_stars, best_fwhm, stopped_sigma, phase_data = (
                future.result()
            )
            info["shape"] = shape
            info["anchor_stars"] = best_stars
            info["star_count"] = len(best_stars)
            info["fwhm"] = best_fwhm
            info["anchor_data"] = phase_data
            app_print(
                f"  -> {info['batch_name']}: {len(best_stars):02d} estrelas base | (Sigma parou em {stopped_sigma:.1f})\n"
            )

    global_master_cfg = config.get("global_master", "Auto")
    if str(global_master_cfg).lower() == "auto":
        master_info = max(
            anchors_info,
            key=lambda item: (
                item.get("anchor_quality", 0.0),
                item.get("star_count", 0),
            ),
        )
        app_print(
            f"\n[GLOBAL] Master Automático eleito: {master_info['batch_name']} | estrelas={master_info['star_count']} | FWHM={master_info['fwhm']:.2f}\n"
        )
    else:
        master_info = next(
            (item for item in anchors_info if item["batch_name"] == global_master_cfg),
            None,
        )
        if master_info is None:
            master_info = max(
                anchors_info,
                key=lambda item: (
                    item.get("anchor_quality", 0.0),
                    item.get("star_count", 0),
                ),
            )
            app_print(
                f"\n[GLOBAL] Master especificado não encontrado. Fallback: {master_info['batch_name']}\n"
            )

    matching_radius = float(
        config.get("global_matching_radius", config.get("matching_radius", 100.0))
    )
    ransac_thresh = float(config.get("global_ransac", config.get("ransac", 5.0)))
    limits = _global_limits(config)

    global_flow = {
        "schema_version": 2,
        "mode": "global_hybrid",
        "global_master_batch": master_info["batch_name"],
        "parameters": {
            "matching_radius": matching_radius,
            "ransac": ransac_thresh,
            **limits,
        },
        "batches": {},
        "quality": {},
        "cross_checks": [],
    }

    global_flow["batches"][master_info["batch_name"]] = {
        "status": "accepted",
        "matrix": np.eye(3).tolist(),
        "relative_to": None,
        "strategy": "master",
        "matches": master_info["star_count"],
        "inliers": master_info["star_count"],
        "inlier_ratio": 1.0,
        "rms": 0.0,
        "translation": [0.0, 0.0],
        "translation_magnitude": 0.0,
        "rotation_deg": 0.0,
        "scale": 1.0,
        "phase_response": 1.0,
    }

    # O pareamento direto de cada batch contra o Master é independente entre
    # batches: paraleliza (antes era um loop sequencial).
    direct_results = {}
    targets_for_direct = [
        info for info in anchors_info if info["batch_name"] != master_info["batch_name"]
    ]

    with ThreadPoolExecutor(
        max_workers=global_workers, thread_name_prefix="astroflow-direct"
    ) as executor:
        futures = {
            executor.submit(
                _estimate_global_pair,
                master_info,
                target_info,
                matching_radius,
                ransac_thresh,
                limits,
            ): target_info
            for target_info in targets_for_direct
        }
        completed = 0
        for future in as_completed(futures):
            if cancel_event.is_set():
                return
            target_info = futures[future]
            batch_name = target_info["batch_name"]
            matrix, metrics = future.result()
            direct_results[batch_name] = (matrix, metrics)
            completed += 1
            app_progress(
                completed, total_batches, f"Alinhando {batch_name} ao Master Global..."
            )

    ordered_global_matrices = {master_info["batch_name"]: np.eye(3, dtype=np.float64)}

    for distance in range(1, len(anchors_info)):
        progress_made = False
        for index, target_info in enumerate(anchors_info):
            target_name = target_info["batch_name"]
            if target_name in ordered_global_matrices:
                continue

            candidate_indices = []
            if index > 0:
                candidate_indices.append(index - 1)
            if index + 1 < len(anchors_info):
                candidate_indices.append(index + 1)

            for ref_index in candidate_indices:
                ref_info = anchors_info[ref_index]
                ref_name = ref_info["batch_name"]
                if ref_name not in ordered_global_matrices:
                    continue

                matrix, metrics = _estimate_global_pair(
                    ref_info, target_info, matching_radius, ransac_thresh, limits
                )
                if matrix is not None and metrics.get("status") == "accepted":
                    reference_matrix = ordered_global_matrices[ref_name]
                    absolute_matrix = reference_matrix @ matrix
                    ordered_global_matrices[target_name] = absolute_matrix

                    global_flow["batches"][target_name] = {
                        "status": "accepted",
                        "matrix": absolute_matrix.tolist(),
                        "relative_matrix": matrix.tolist(),
                        "relative_to": ref_name,
                        "strategy": (
                            "master_direct"
                            if ref_name == master_info["batch_name"]
                            else "neighbor_chain"
                        ),
                        **metrics,
                    }
                    progress_made = True
                    break
        if not progress_made:
            break

    for target_info in anchors_info:
        batch_name = target_info["batch_name"]
        if batch_name == master_info["batch_name"]:
            continue
        if batch_name in global_flow["batches"]:
            continue

        direct_matrix, direct_metrics = direct_results.get(batch_name, (None, {}))
        if direct_matrix is not None and direct_metrics.get("status") == "accepted":
            global_flow["batches"][batch_name] = {
                "status": "accepted",
                "matrix": direct_matrix.tolist(),
                "relative_matrix": direct_matrix.tolist(),
                "relative_to": master_info["batch_name"],
                "strategy": "master_direct",
                **direct_metrics,
            }
            ordered_global_matrices[batch_name] = direct_matrix
        else:
            target_index = anchors_info.index(target_info)
            candidate_indices = []
            if target_index > 0:
                candidate_indices.append(target_index - 1)
            if target_index + 1 < len(anchors_info):
                candidate_indices.append(target_index + 1)

            progress_made = False
            for ref_index in candidate_indices:
                ref_info = anchors_info[ref_index]
                ref_name = ref_info["batch_name"]
                if ref_name not in ordered_global_matrices:
                    continue

                matrix, metrics = _estimate_global_pair(
                    ref_info, target_info, matching_radius, ransac_thresh, limits
                )
                if matrix is not None and metrics.get("status") == "accepted":
                    reference_matrix = ordered_global_matrices[ref_name]
                    absolute_matrix = reference_matrix @ matrix
                    ordered_global_matrices[batch_name] = absolute_matrix

                    global_flow["batches"][batch_name] = {
                        "status": "accepted",
                        "matrix": absolute_matrix.tolist(),
                        "relative_matrix": matrix.tolist(),
                        "relative_to": ref_name,
                        "strategy": "neighbor_chain",
                        **metrics,
                    }
                    progress_made = True
                    break
            if not progress_made:
                global_flow["batches"][batch_name] = {
                    "status": "rejected",
                    "matrix": None,
                    "relative_to": None,
                    "strategy": None,
                    **{key: value for key, value in direct_metrics.items()},
                    "reason": direct_metrics.get("reason", "global_alignment_failed"),
                }

    for index, target_info in enumerate(anchors_info):
        batch_name = target_info["batch_name"]
        if batch_name == master_info["batch_name"]:
            continue
        entry = global_flow["batches"].get(batch_name)
        if not entry:
            continue

        direct_matrix, direct_metrics = direct_results.get(batch_name, (None, {}))
        final_matrix = entry.get("matrix")
        if direct_matrix is None or final_matrix is None:
            continue

        closure_error = _matrix_difference_score(
            direct_matrix, np.asarray(final_matrix, dtype=np.float64)
        )
        cross_check = {
            "target": batch_name,
            "direct_status": direct_metrics.get("status", "unknown"),
            "closure_error": round(closure_error, 6),
            "consistent": bool(
                closure_error <= float(config.get("global_closure_threshold", 0.05))
            ),
        }
        global_flow["cross_checks"].append(cross_check)
        entry["closure_error"] = cross_check["closure_error"]
        if not cross_check["consistent"]:
            entry["warning"] = "global_cross_check_inconsistent"

    accepted_batches = [
        e for e in global_flow["batches"].values() if e.get("status") == "accepted"
    ]
    rejected_batches = [
        e for e in global_flow["batches"].values() if e.get("status") == "rejected"
    ]

    global_flow["quality"] = {
        "total_batches": len(anchors_info),
        "accepted_batches": len(accepted_batches),
        "rejected_batches": len(rejected_batches),
        "coverage": (
            len(accepted_batches) / len(anchors_info) if anchors_info else 0.0
        ),
        "cross_checks": len(global_flow["cross_checks"]),
        "cross_check_failures": sum(
            1 for item in global_flow["cross_checks"] if not item["consistent"]
        ),
    }
    global_flow["master_metrics"] = {
        "batch": master_info["batch_name"],
        "star_count": master_info["star_count"],
        "fwhm": master_info["fwhm"],
        "anchor_quality": master_info.get("anchor_quality", 0.0),
    }

    global_path = base_dir / "global_flow.json"
    with global_path.open("w", encoding="utf-8") as f:
        json.dump(global_flow, f, indent=4, ensure_ascii=False)

    app_progress(total_batches, total_batches, "AstroFlow Finalizado.")
    app_print(
        f"\n>>> AstroFlow Finalizado. {len(accepted_batches)}/{len(anchors_info)} Batches aceitas no Global Flow. <<<\n"
    )


# ============================================================
# Preview (Contrato 100% compatível com [source: 17])
# ============================================================


def preview_star_detection(
    batch_dir: Path, config: dict
) -> tuple[np.ndarray | None, int, float]:
    """Gera o preview com as marcações de estrelas em conformidade com show_astroflow_preview em main.py."""
    files = sorted(
        [
            p
            for p in batch_dir.iterdir()
            if p.is_file() and p.suffix.lower() in {".fit", ".fits", ".fts"}
        ]
    )
    if not files:
        return (None, 0, 0.0)

    anchor_name = config.get("custom_anchors", {}).get(batch_dir.name)
    anchor_file = next((p for p in files if p.name == anchor_name), files[0])

    data, header = load_fits_data(anchor_file)
    working_data = extract_luminance(data, header)

    fwhm_val = float(config.get("fwhm", 4.0))
    sigma_val = float(config.get("sigma", 5.0))
    max_stars_val = int(config.get("max_stars", 250))
    engine_val = config.get("engine", "DAO")

    stars, measured_fwhm, _ = detect_stars(
        working_data, fwhm_val, sigma_val, max_stars_val, engine_val
    )
    _, median, std = sigma_clipped_stats(working_data, sigma=3.0)
    median, std = float(median), float(std)

    denominator = max((8.0 * std), 1e-5)
    vmin, vmax = median, median + denominator
    norm_data = np.clip((working_data - vmin) / max(vmax - vmin, 1e-5), 0, 1) * 255.0

    img_color = cv2.cvtColor(norm_data.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    radius = max(3, int(max(measured_fwhm, fwhm_val) * 1.5))

    for x, y in stars:
        cv2.circle(img_color, (int(round(x)), int(round(y))), radius, (0, 0, 255), 1)

    return (img_color, len(stars), float(measured_fwhm))


def save_frame_metrics(image_path: Path | str, metrics: dict):
    base_path, _ = os.path.splitext(str(image_path))
    json_path = f"{base_path}_metrics.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False)
    return json_path
