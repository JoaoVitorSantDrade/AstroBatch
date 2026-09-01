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
# Bayer / CFA
# ============================================================

BAYER_CV2_MAPPING = {
    "RGGB": cv2.COLOR_BayerBG2RGB,
    "BGGR": cv2.COLOR_BayerRG2RGB,
    "GRBG": cv2.COLOR_BayerGB2RGB,
    "GBRG": cv2.COLOR_BayerGR2RGB,
}


def get_bayer_pattern(header: fits.Header) -> str | None:
    for key in ["BAYERPAT", "BAYERPATTERN", "COLORTYP"]:
        if key in header:
            val = str(header[key]).strip().upper().strip("'")
            if val in BAYER_CV2_MAPPING:
                return val
    return None


def split_cfa(data: np.ndarray, pattern: str):
    if pattern == "RGGB":
        return (
            data[0::2, 0::2],
            data[0::2, 1::2],
            data[1::2, 0::2],
            data[1::2, 1::2],
        )

    if pattern == "BGGR":
        return (
            data[1::2, 1::2],
            data[0::2, 1::2],
            data[1::2, 0::2],
            data[0::2, 0::2],
        )

    if pattern == "GRBG":
        return (
            data[0::2, 1::2],
            data[0::2, 0::2],
            data[1::2, 1::2],
            data[1::2, 0::2],
        )

    if pattern == "GBRG":
        return (
            data[1::2, 0::2],
            data[1::2, 1::2],
            data[0::2, 0::2],
            data[0::2, 1::2],
        )

    return (
        data[0::2, 0::2],
        data[0::2, 1::2],
        data[1::2, 0::2],
        data[1::2, 1::2],
    )


def extract_luminance(
    data: np.ndarray,
    header: fits.Header,
) -> np.ndarray:
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

            l_full = np.repeat(
                np.repeat(
                    l_sub,
                    2,
                    axis=0,
                ),
                2,
                axis=1,
            )

            return l_full.astype(np.float32)

        return data.astype(np.float32)

    raise ValueError(f"Dimensões não suportadas: {data.shape}")


# ============================================================
# FITS
# ============================================================


def load_fits_data(
    filepath: Path,
) -> tuple[np.ndarray, fits.Header]:
    with fits.open(
        filepath,
        memmap=False,
    ) as hdul:
        for hdu in hdul:
            if hdu.is_image and hdu.data is not None and hdu.data.ndim in (2, 3):
                return (
                    np.asarray(
                        hdu.data,
                        dtype=np.float32,
                    ),
                    hdu.header.copy(),
                )

    raise ValueError(f"Imagem inválida: {filepath.name}")


# ============================================================
# Processamento / detecção
# ============================================================


def prepare_for_phase_correlation(
    data: np.ndarray,
) -> np.ndarray:
    finite_mask = np.isfinite(data)

    if not np.any(finite_mask):
        return np.zeros_like(
            data,
            dtype=np.float32,
        )

    finite_data = data[finite_mask]

    d_min = float(np.min(finite_data))
    d_max = float(np.max(finite_data))

    if d_max > d_min:
        norm = (data - d_min) / (d_max - d_min)
    else:
        norm = np.zeros_like(
            data,
            dtype=np.float32,
        )

    norm = np.nan_to_num(
        norm,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    )

    return norm.astype(
        np.float32,
        copy=False,
    )


def calculate_anchor_quality(
    star_count: int,
    fwhm: float,
) -> float:
    if fwhm <= 0:
        return float(star_count)

    return float(star_count) / float(fwhm)


def detect_stars_dao(
    data: np.ndarray,
    fwhm: float,
    sigma: float,
    max_stars: int,
) -> tuple[np.ndarray, float, dict]:

    mean_val = float(np.mean(data))
    median_val = float(np.median(data))
    std_val = float(np.std(data))

    _, bkg_median, bkg_std = sigma_clipped_stats(
        data,
        sigma=3.0,
    )

    bkg_median = float(bkg_median)
    bkg_std = float(bkg_std)

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

    daofind = DAOStarFinder(
        fwhm=fwhm,
        threshold=sigma * bkg_std,
    )

    sources = daofind(data - bkg_median)

    if sources is not None and len(sources) > 0:
        # Ordena do mais brilhante para o mais fraco
        sources.sort("flux")
        sources.reverse()

        # Extrai os arrays brutos
        raw_coords = np.transpose((sources["xcentroid"], sources["ycentroid"])).astype(
            np.float32
        )
        raw_fluxes = np.asarray(sources["flux"], dtype=np.float32)
        has_sharpness = "sharpness" in sources.colnames

        if has_sharpness:
            raw_sharpness = np.asarray(sources["sharpness"], dtype=np.float32)

        filtered_coords = []
        filtered_fluxes = []
        filtered_sharpness = []

        # Cria uma zona de exclusão baseada no FWHM (elimina múltiplos círculos na mesma estrela)
        min_dist_sq = (fwhm * 1.5) ** 2

        # Non-Maximum Suppression (NMS)
        for i in range(len(raw_coords)):
            pt = raw_coords[i]

            if not filtered_coords:
                filtered_coords.append(pt)
                filtered_fluxes.append(raw_fluxes[i])
                if has_sharpness:
                    filtered_sharpness.append(raw_sharpness[i])
                continue

            # Mede a distância ao quadrado para as estrelas mais brilhantes já aceitas
            dists_sq = np.sum((np.array(filtered_coords) - pt) ** 2, axis=1)

            # Só aceita se estiver fora do raio de exclusão de TODAS as estrelas já aceitas
            if np.min(dists_sq) >= min_dist_sq:
                filtered_coords.append(pt)
                filtered_fluxes.append(raw_fluxes[i])
                if has_sharpness:
                    filtered_sharpness.append(raw_sharpness[i])

            # Para ao atingir o max_stars
            if len(filtered_coords) >= max_stars:
                break

        coords = np.asarray(filtered_coords, dtype=np.float32)
        fluxes = np.asarray(filtered_fluxes, dtype=np.float32)

        star_count = len(coords)

        if has_sharpness and star_count > 0:
            current_fwhm = float(np.median(filtered_sharpness) * fwhm)
        else:
            current_fwhm = float(fwhm)

        min_flux = float(np.min(fluxes)) if star_count > 0 else 0.0
        max_flux = float(np.max(fluxes)) if star_count > 0 else 0.0
        snr = float(np.mean(fluxes) / bkg_std) if star_count > 0 else 0.0

    else:
        coords = np.empty((0, 2), dtype=np.float32)
        current_fwhm = 0.0
        star_count = 0
        min_flux = 0.0
        max_flux = 0.0
        snr = 0.0

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
    data: np.ndarray,
    fwhm: float,
    sigma: float,
    max_stars: int,
) -> tuple[np.ndarray, float, dict]:

    mean_val = float(np.mean(data))

    median_val = float(np.median(data))

    std_val = float(np.std(data))

    norm_img = cv2.normalize(
        data,
        None,
        0,
        255,
        cv2.NORM_MINMAX,
        dtype=cv2.CV_8U,
    )

    ksize = max(
        3,
        int(fwhm) | 1,
    )

    blurred = cv2.GaussianBlur(
        norm_img,
        (ksize, ksize),
        0,
    )

    threshold_val = min(
        255.0,
        float(np.median(blurred)) + (sigma * float(np.std(blurred))),
    )

    _, thresh = cv2.threshold(
        blurred,
        threshold_val,
        255,
        cv2.THRESH_BINARY,
    )

    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    valid_contours = sorted(
        (c for c in contours if 2 < cv2.contourArea(c) < 1000),
        key=cv2.contourArea,
        reverse=True,
    )[:max_stars]

    coords_list = []
    areas = []
    fluxes = []

    height, width = data.shape[:2]

    for cnt in valid_contours:
        moments = cv2.moments(cnt)

        if moments["m00"] == 0:
            continue

        cX = moments["m10"] / moments["m00"]

        cY = moments["m01"] / moments["m00"]

        px = int(
            np.clip(
                round(cX),
                0,
                width - 1,
            )
        )

        py = int(
            np.clip(
                round(cY),
                0,
                height - 1,
            )
        )

        coords_list.append(
            [
                cX,
                cY,
            ]
        )

        areas.append(cv2.contourArea(cnt))

        fluxes.append(float(data[py, px]))

    if coords_list:
        coords = np.asarray(
            coords_list,
            dtype=np.float32,
        )
    else:
        coords = np.empty(
            (0, 2),
            dtype=np.float32,
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
        "fwhm": round(
            current_fwhm,
            2,
        ),
        "mean": round(
            mean_val,
            2,
        ),
        "median": round(
            median_val,
            2,
        ),
        "std": round(
            std_val,
            2,
        ),
        "background": round(
            median_val,
            2,
        ),
        "snr": round(
            snr,
            2,
        ),
        "min_flux": round(
            min_flux,
            2,
        ),
        "max_flux": round(
            max_flux,
            2,
        ),
        "valid": valid,
    }

    return (
        coords,
        current_fwhm,
        metrics,
    )


def detect_stars(
    data: np.ndarray,
    fwhm: float,
    sigma: float,
    max_stars: int,
    engine: str = "DAO",
) -> tuple[np.ndarray, float, dict]:

    if str(engine).upper() == "OPENCV":
        return detect_stars_opencv(
            data,
            fwhm,
            sigma,
            max_stars,
        )

    return detect_stars_dao(
        data,
        fwhm,
        sigma,
        max_stars,
    )


# ============================================================
# Transformações
# ============================================================


def extract_geometric_properties(
    matrix_2x3: np.ndarray,
) -> tuple[
    float,
    float,
    float,
    float,
]:
    a, b, tx = matrix_2x3[0]
    c, d, ty = matrix_2x3[1]

    scale = float(np.sqrt(a**2 + c**2))

    rotation_deg = float(
        np.degrees(
            np.arctan2(
                c,
                a,
            )
        )
    )

    return (
        float(tx),
        float(ty),
        rotation_deg,
        scale,
    )


def validate_transform(
    matrix_2x3,
    metrics,
    limits,
) -> tuple[bool, str]:

    if matrix_2x3 is None:
        if (
            metrics.get(
                "matches",
                0,
            )
            == 0
        ):
            return (
                False,
                "phase_correlation_failed",
            )

        return (
            False,
            "insufficient_matches",
        )

    if metrics["inliers"] < limits["min_inliers"]:
        return (
            False,
            "insufficient_inliers",
        )

    if metrics["inlier_ratio"] < limits["min_ratio"]:
        return (
            False,
            "low_inlier_ratio",
        )

    if (
        metrics.get(
            "rms",
            999.0,
        )
        > limits["max_rms"]
    ):
        return (
            False,
            "high_rms",
        )

    tx, ty, rotation, scale = extract_geometric_properties(matrix_2x3)

    metrics["translation"] = [
        round(tx, 3),
        round(ty, 3),
    ]

    metrics["translation_magnitude"] = round(
        float(np.hypot(tx, ty)),
        3,
    )

    metrics["rotation_deg"] = round(
        rotation,
        4,
    )

    metrics["scale"] = round(
        scale,
        6,
    )

    if np.hypot(tx, ty) > limits["max_translation"]:
        return (
            False,
            "high_translation",
        )

    if abs(rotation) > limits["max_rotation"]:
        return (
            False,
            "high_rotation",
        )

    if not (limits["min_scale"] <= scale <= limits["max_scale"]):
        return (
            False,
            "invalid_scale",
        )

    return (
        True,
        "accepted",
    )


def make_homogeneous(
    matrix_2x3: np.ndarray,
) -> np.ndarray:
    hom = np.eye(
        3,
        dtype=np.float64,
    )

    hom[:2, :] = matrix_2x3

    return hom


# ============================================================
# Frame worker
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

        working_data = extract_luminance(
            data,
            header,
        )

        current_sigma = sigma_val
        best_stars = []
        best_fwhm = 0.0
        best_metrics = {}

        # Alvo de estrelas para considerar o frame "saudável" e parar a busca.
        # Usa 20 estrelas ou o dobro do mínimo exigido (o que for maior).
        target_stars = max(28, min_stars * 2)

        # Busca Adaptativa Local: desce o sigma em passos de 0.2 até o limite seguro de 3.0
        while current_sigma >= 3.0:
            stars, measured_fwhm, metrics = detect_stars(
                working_data,
                fwhm_val,
                current_sigma,
                max_stars_val,
                engine_val,
            )

            best_stars = stars
            best_fwhm = measured_fwhm
            best_metrics = metrics

            # Se achou estrelas suficientes para um bom RANSAC, interrompe a busca
            if len(stars) >= target_stars:
                break

            current_sigma -= 0.2

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
                "stars": np.empty(
                    (0, 2),
                    dtype=np.float32,
                ),
                "fwhm": 0.0,
                "metrics": {},
            },
        )


# ============================================================
# Matching
# ============================================================


def _match_incremental_stars(
    previous_stars: np.ndarray,
    current_stars: np.ndarray,
    shift: tuple[float, float],
    matching_radius: float,
) -> tuple[list, list]:

    if len(previous_stars) == 0 or len(current_stars) == 0:
        return [], []

    dx, dy = shift

    shifted_current = current_stars + np.array(
        [dx, dy],
        dtype=np.float32,
    )

    tree = KDTree(previous_stars)

    distances, indices = tree.query(
        shifted_current,
        distance_upper_bound=matching_radius,
    )

    candidates = [
        (
            float(distance),
            int(previous_idx),
            int(current_idx),
        )
        for current_idx, (
            distance,
            previous_idx,
        ) in enumerate(
            zip(
                distances,
                indices,
            )
        )
        if np.isfinite(distance) and previous_idx < len(previous_stars)
    ]

    candidates.sort(key=lambda item: item[0])

    used_previous = set()

    previous_points = []
    current_points = []

    for (
        distance,
        previous_idx,
        current_idx,
    ) in candidates:
        if previous_idx in used_previous:
            continue

        used_previous.add(previous_idx)

        previous_points.append(previous_stars[previous_idx])

        current_points.append(current_stars[current_idx])

    return (
        previous_points,
        current_points,
    )


# ============================================================
# Estimativa incremental
# ============================================================


def _estimate_incremental_transform(
    previous_stars: list,
    current_stars: list,
    ransac_thresh: float,
    min_stars: int,
):
    matches = min(
        len(previous_stars),
        len(current_stars),
    )

    if matches < min_stars:
        return (
            None,
            {
                "matches": matches,
                "inliers": 0,
                "inlier_ratio": 0.0,
                "rms": 999.0,
            },
        )

    previous = np.asarray(
        previous_stars,
        dtype=np.float32,
    )

    current = np.asarray(
        current_stars,
        dtype=np.float32,
    )

    matrix_2x3, inliers = cv2.estimateAffinePartial2D(
        current,
        previous,
        method=cv2.RANSAC,
        ransacReprojThreshold=ransac_thresh,
        maxIters=5000,
        confidence=0.995,
        refineIters=10,
    )

    if matrix_2x3 is None or inliers is None:
        return (
            None,
            {
                "matches": len(previous),
                "inliers": 0,
                "inlier_ratio": 0.0,
                "rms": 999.0,
            },
        )

    mask = inliers.ravel().astype(bool)

    inlier_count = int(mask.sum())

    if inlier_count == 0:
        return (
            None,
            {
                "matches": len(previous),
                "inliers": 0,
                "inlier_ratio": 0.0,
                "rms": 999.0,
            },
        )

    transformed = cv2.transform(
        current.reshape(
            -1,
            1,
            2,
        ),
        matrix_2x3,
    ).reshape(
        -1,
        2,
    )

    errors = transformed[mask] - previous[mask]

    distances = np.linalg.norm(
        errors,
        axis=1,
    )

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


def process_local_flow(
    batch_dir: Path,
    config: dict,
    app_print,
) -> dict:

    files = sorted(
        [
            p
            for p in batch_dir.iterdir()
            if (
                p.is_file()
                and p.suffix.lower()
                in {
                    ".fit",
                    ".fits",
                    ".fts",
                }
            )
        ]
    )

    if not files:
        return {}

    if not isinstance(
        config,
        dict,
    ):
        config = {}

    custom_anchors = config.get(
        "custom_anchors",
        {},
    )

    chosen_anchor_name = custom_anchors.get(batch_dir.name)

    anchor_file = files[0]

    if chosen_anchor_name and any(p.name == chosen_anchor_name for p in files):
        app_print(
            f"[{batch_dir.name}] "
            f"Frame Central (Batch Reference): "
            f"{chosen_anchor_name}\n"
        )
    else:
        chosen_anchor_name = anchor_file.name

        app_print(
            f"[{batch_dir.name}] Nenhuma referência manual válida. Usando o 1º frame.\n"
        )

    fwhm_val = float(
        config.get(
            "fwhm",
            4.0,
        )
    )

    sigma_val = float(
        config.get(
            "sigma",
            5.0,
        )
    )

    max_stars_val = int(
        config.get(
            "max_stars",
            250,
        )
    )

    matching_radius = float(
        config.get(
            "matching_radius",
            25.0,
        )
    )

    ransac_thresh = float(
        config.get(
            "ransac",
            4.0,
        )
    )

    engine_val = config.get(
        "engine",
        "DAO",
    )

    limits = {
        "min_stars": int(
            config.get(
                "min_stars",
                4,
            )
        ),
        "min_inliers": int(
            config.get(
                "min_inliers",
                4,
            )
        ),
        "min_ratio": float(
            config.get(
                "min_ratio",
                0.15,
            )
        ),
        "max_rms": float(
            config.get(
                "max_rms",
                4.0,
            )
        ),
        "max_translation": float(
            config.get(
                "max_translation",
                1500.0,
            )
        ),
        "max_rotation": float(
            config.get(
                "max_rotation",
                10.0,
            )
        ),
        "min_scale": float(
            config.get(
                "min_scale",
                0.95,
            )
        ),
        "max_scale": float(
            config.get(
                "max_scale",
                1.05,
            )
        ),
    }

    try:
        cpu_count = (
            getattr(
                os,
                "process_cpu_count",
                os.cpu_count,
            )()
            or 1
        )
    except Exception:
        cpu_count = os.cpu_count() or 1

    worker_count = max(
        1,
        min(
            16,
            cpu_count,
        ),
    )

    app_print(
        f"[{batch_dir.name}] Pré-processamento paralelo: {worker_count} workers\n"
    )

    prepared_frames = {}

    with ThreadPoolExecutor(
        max_workers=worker_count,
        thread_name_prefix="astroflow",
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

            except Exception as exc:
                app_print(f"[{filepath.name}] Erro no worker: {exc}\n")
                continue

            prepared_frames[fname] = result

    anchor = prepared_frames.get(anchor_file.name)

    if anchor is None or anchor.get("status") == "error":
        app_print(f"[{batch_dir.name}] ERRO: Não foi possível processar a âncora.\n")
        return {}

    if len(anchor["stars"]) < limits["min_stars"]:
        app_print(
            f"[{batch_dir.name}] "
            f"ERRO: Estrelas insuficientes "
            f"na âncora "
            f"({len(anchor['stars'])}).\n"
        )
        return {}

    anchor_quality = calculate_anchor_quality(
        len(anchor["stars"]),
        anchor["fwhm"],
    )

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
            "min_stars": limits["min_stars"],
            "min_inliers": limits["min_inliers"],
            "min_ratio": limits["min_ratio"],
            "max_rms": limits["max_rms"],
            "max_translation": limits["max_translation"],
            "max_rotation": limits["max_rotation"],
            "min_scale": limits["min_scale"],
            "max_scale": limits["max_scale"],
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
        "translation": [
            0.0,
            0.0,
        ],
        "translation_magnitude": 0.0,
        "rotation_deg": 0.0,
        "scale": 1.0,
        "cumulative_rms": 0.0,
        "star_count": len(anchor["stars"]),
        "fwhm": anchor["fwhm"],
    }

    cumulative_matrix = np.eye(
        3,
        dtype=np.float64,
    )

    previous_name = anchor_file.name
    previous_frame = anchor

    for index in range(
        1,
        len(files),
    ):
        current_file = files[index]

        current_name = current_file.name

        current_frame = prepared_frames.get(current_name)

        if current_frame is None or current_frame.get("status") == "error":
            flow_data["frames"][current_name] = {
                "status": "rejected",
                "reason": (
                    current_frame.get(
                        "reason",
                        "processing_error",
                    )
                    if current_frame
                    else "processing_error"
                ),
            }

            app_print(f"[{current_name}] REJEITADO: falha no pré-processamento.\n")

            continue

        if len(current_frame["stars"]) < limits["min_stars"]:
            flow_data["frames"][current_name] = {
                "status": "rejected",
                "reason": "insufficient_stars",
                "star_count": len(current_frame["stars"]),
            }

            app_print(f"[{current_name}] REJEITADO: estrelas insuficientes.\n")

            continue

        attempts = [
            (
                "normal",
                previous_frame,
                matching_radius,
                previous_name,
            ),
            (
                "relaxed_radius",
                previous_frame,
                matching_radius * 2.0,
                previous_name,
            ),
        ]

        accepted = False
        best_metrics = {
            "reason": "phase_correlation_failed",
            "matches": 0,
            "inliers": 0,
            "inlier_ratio": 0.0,
            "rms": 999.0,
        }

        for (
            attempt_name,
            ref_frame,
            radius,
            ref_name,
        ) in attempts:
            if (
                ref_frame.get("phase_data") is None
                or current_frame.get("phase_data") is None
            ):
                continue

            shift, response = cv2.phaseCorrelate(
                ref_frame["phase_data"],
                current_frame["phase_data"],
            )

            dx, dy = shift

            m_ref, m_cur = _match_incremental_stars(
                ref_frame["stars"],
                current_frame["stars"],
                (
                    dx,
                    dy,
                ),
                radius,
            )

            rel_matrix, metrics = _estimate_incremental_transform(
                m_ref,
                m_cur,
                ransac_thresh,
                limits["min_stars"],
            )

            metrics["phase_shift"] = [
                round(
                    float(dx),
                    3,
                ),
                round(
                    float(dy),
                    3,
                ),
            ]

            metrics["phase_response"] = round(
                float(response),
                5,
            )

            valid, reason = validate_transform(
                rel_matrix,
                metrics,
                limits,
            )

            if valid:
                accepted = True

                relative_homogeneous = make_homogeneous(rel_matrix)

                reference_cumulative = flow_data["frames"][ref_name]["matrix"]

                ref_matrix = np.asarray(
                    reference_cumulative,
                    dtype=np.float64,
                )

                cumulative_matrix = ref_matrix @ relative_homogeneous

                previous_cumulative_rms = float(
                    flow_data["frames"][ref_name].get(
                        "cumulative_rms",
                        0.0,
                    )
                )

                current_rms = float(
                    metrics.get(
                        "rms",
                        0.0,
                    )
                )

                cumulative_rms = float(
                    np.sqrt(previous_cumulative_rms**2 + current_rms**2)
                )

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

                app_print(
                    f"[{current_name}] "
                    f"OK ({attempt_name}) "
                    f"<- {ref_name} | "
                    f"{metrics['inliers']}/"
                    f"{metrics['matches']} inliers | "
                    f"ratio={metrics['inlier_ratio']:.1%} | "
                    f"RMS={metrics['rms']:.3f}px\n"
                )

                break

            metrics["reason"] = reason

            if metrics.get(
                "inliers",
                0,
            ) > best_metrics.get(
                "inliers",
                0,
            ) or (
                metrics.get(
                    "inliers",
                    0,
                )
                == best_metrics.get(
                    "inliers",
                    0,
                )
                and metrics.get(
                    "rms",
                    999.0,
                )
                < best_metrics.get(
                    "rms",
                    999.0,
                )
            ):
                best_metrics = metrics

        if accepted:
            previous_name = current_name
            previous_frame = current_frame

        else:
            flow_data["frames"][current_name] = {
                "status": "rejected",
                "reason": best_metrics.get(
                    "reason",
                    "unknown",
                ),
                "matches": best_metrics.get(
                    "matches",
                    0,
                ),
                "inliers": best_metrics.get(
                    "inliers",
                    0,
                ),
                "inlier_ratio": best_metrics.get(
                    "inlier_ratio",
                    0.0,
                ),
                "rms": best_metrics.get(
                    "rms",
                    999.0,
                ),
                "phase_shift": best_metrics.get(
                    "phase_shift",
                    [0.0, 0.0],
                ),
                "phase_response": best_metrics.get(
                    "phase_response",
                    0.0,
                ),
                "fwhm": current_frame["fwhm"],
                "star_count": len(current_frame["stars"]),
            }

            app_print(
                f"[{current_name}] "
                f"REJEITADO: "
                f"{best_metrics.get('reason', 'unknown')} | "
                f"inliers="
                f"{best_metrics.get('inliers', 0)} | "
                f"RMS="
                f"{best_metrics.get('rms', 999.0):.3f}px\n"
            )

    # ========================================================
    # Recentralização
    # ========================================================

    selected_reference_info = flow_data["frames"].get(chosen_anchor_name)

    if selected_reference_info and selected_reference_info.get("status") == "accepted":
        reference_matrix = np.asarray(
            selected_reference_info["matrix"],
            dtype=np.float64,
        )

        try:
            inverse_reference = np.linalg.inv(reference_matrix)

            for frame_data in flow_data["frames"].values():
                if frame_data.get("status") != "accepted":
                    continue

                old_matrix = np.asarray(
                    frame_data["matrix"],
                    dtype=np.float64,
                )

                frame_data["matrix"] = (inverse_reference @ old_matrix).tolist()

            flow_data["batch_anchor"] = chosen_anchor_name

            app_print(
                f"[{batch_dir.name}] "
                f"Matrizes recentralizadas na referência "
                f"{chosen_anchor_name}.\n"
            )

        except np.linalg.LinAlgError:
            app_print(
                f"[{batch_dir.name}] "
                f"ERRO: referência possui matriz singular. "
                f"Recentralização ignorada.\n"
            )

    else:
        app_print(
            f"[{batch_dir.name}] "
            f"AVISO: referência escolhida não possui "
            f"matriz aceita. Origem preservada.\n"
        )

    # ========================================================
    # Estatísticas finais
    # ========================================================

    accepted_frames = [
        frame
        for frame in flow_data["frames"].values()
        if frame.get("status") == "accepted"
    ]

    rejected_frames = [
        frame
        for frame in flow_data["frames"].values()
        if frame.get("status") == "rejected"
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
            flow_data["frames"]
            .get(
                files[0].name,
                {},
            )
            .get("status")
            == "accepted"
        ),
        "last_frame_valid": (
            flow_data["frames"]
            .get(
                files[-1].name,
                {},
            )
            .get("status")
            == "accepted"
        ),
        "chain_segments": _count_chain_segments(flow_data),
    }

    output_path = batch_dir / "flow_local.json"

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            flow_data,
            f,
            indent=4,
            ensure_ascii=False,
        )

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


def _count_chain_segments(
    flow_data: dict,
) -> int:
    segments = 0
    previous_accepted = False

    for frame_data in flow_data.get("frames", {}).values():
        accepted = frame_data.get("status") == "accepted"

        if accepted and not previous_accepted:
            segments += 1

        previous_accepted = accepted

    return segments


# ============================================================
# Global Flow
# ============================================================


def _global_limits(config: dict) -> dict:
    return {
        "min_inliers": int(
            config.get(
                "global_min_inliers",
                config.get(
                    "min_inliers",
                    4,
                ),
            )
        ),
        "min_ratio": float(
            config.get(
                "global_min_ratio",
                config.get(
                    "min_ratio",
                    0.20,
                ),
            )
        ),
        "max_rms": float(
            config.get(
                "global_max_rms",
                4.0,
            )
        ),
        "max_translation": float(
            config.get(
                "global_max_translation",
                3000.0,
            )
        ),
        "max_rotation": float(
            config.get(
                "global_max_rotation",
                20.0,
            )
        ),
        "min_scale": float(
            config.get(
                "global_min_scale",
                0.85,
            )
        ),
        "max_scale": float(
            config.get(
                "global_max_scale",
                1.15,
            )
        ),
    }


def _estimate_global_pair(
    ref_info: dict,
    target_info: dict,
    matching_radius: float,
    ransac_thresh: float,
    limits: dict,
):
    if ref_info.get("anchor_data") is None or target_info.get("anchor_data") is None:
        return (
            None,
            {
                "status": "rejected",
                "reason": "missing_anchor_data",
                "matches": 0,
                "inliers": 0,
                "inlier_ratio": 0.0,
                "rms": 999.0,
            },
        )

    shift, response = cv2.phaseCorrelate(
        ref_info["anchor_data"],
        target_info["anchor_data"],
    )

    dx, dy = shift

    m_ref, m_target = _match_incremental_stars(
        ref_info["anchor_stars"],
        target_info["anchor_stars"],
        (
            dx,
            dy,
        ),
        matching_radius,
    )

    matrix_2x3, metrics = _estimate_incremental_transform(
        m_ref,
        m_target,
        ransac_thresh,
        limits["min_inliers"],
    )

    metrics["phase_shift"] = [
        round(
            float(dx),
            3,
        ),
        round(
            float(dy),
            3,
        ),
    ]

    metrics["phase_response"] = round(
        float(response),
        5,
    )

    valid, reason = validate_transform(
        matrix_2x3,
        metrics,
        limits,
    )

    if not valid:
        metrics["status"] = "rejected"
        metrics["reason"] = reason
        return (
            None,
            metrics,
        )

    metrics["status"] = "accepted"

    return (
        make_homogeneous(matrix_2x3),
        metrics,
    )


def _matrix_difference_score(
    first: np.ndarray,
    second: np.ndarray,
) -> float:
    """
    Métrica simples para comparar duas transformações 3x3.
    """
    diff = first - second

    return float(np.sqrt(np.mean(diff**2)))


def process_all_flows(
    base_dir: Path,
    config: dict,
    app_print,
    app_progress,
    cancel_event,
):
    if not isinstance(
        config,
        dict,
    ):
        config = {}

    batch_folders = sorted(
        [d for d in base_dir.iterdir() if d.is_dir() and "batch" in d.name.lower()],
        key=lambda p: p.name.lower(),
    )

    total_batches = len(batch_folders)

    if not batch_folders:
        app_print(f"Nenhuma subpasta de Batch encontrada em {base_dir}\n")
        return

    app_progress(
        0,
        total_batches,
        "Iniciando AstroFlow...",
    )

    anchors_info = []

    # --------------------------------------------------------
    # Verificação da Flag de Skip Local Flow
    # --------------------------------------------------------
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

                # Reconstrói a estrutura esperada pelo anchors_info
                info = {
                    "batch_name": batch_folder.name,
                    "anchor_path": batch_folder / anchor_name,
                    "anchor_data": None,  # Será gerado na etapa da Ação 1
                    "anchor_stars": [],  # Será gerado na etapa da Ação 1
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
            cpu_count = (
                getattr(
                    os,
                    "process_cpu_count",
                    os.cpu_count,
                )()
                or 1
            )
        except Exception:
            cpu_count = os.cpu_count() or 1

        local_workers = max(
            1,
            min(
                4,
                cpu_count,
            ),
        )

        with ThreadPoolExecutor(
            max_workers=local_workers,
            thread_name_prefix="astroflow-batch",
        ) as executor:
            futures = {
                executor.submit(
                    process_local_flow,
                    batch_folder,
                    config,
                    lambda message: None,
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

    # --- AÇÃO 1 E 2: RE-DETECÇÃO ADAPTATIVA PARA O GLOBAL FLOW ---
    # Em vez de forçar um Sigma que pesca ruído, fazemos uma busca dinâmica.
    # Começamos com o sigma do usuário e descemos até achar um número
    # saudável de estrelas (ex: 35), respeitando um limite de segurança (Sigma=3.0).
    base_sigma = float(config.get("sigma", 5.0))
    engine_val = config.get("engine", "DAO")
    fwhm_val = float(config.get("fwhm", 4.0))

    app_print(
        "\n[GLOBAL] Otimizando extração de estrelas nos Masters (Busca Adaptativa)...\n"
    )

    for info in anchors_info:
        data, header = load_fits_data(info["anchor_path"])
        working_data = extract_luminance(data, header)

        current_sigma = base_sigma
        best_stars = []
        best_fwhm = fwhm_val

        # Busca descendo o sigma de 0.2 em 0.2, parando no limite seguro de 3.0
        while current_sigma >= 2.8:
            g_stars, g_fwhm, _ = detect_stars(
                working_data, fwhm_val, current_sigma, 150, engine_val
            )
            best_stars = g_stars
            best_fwhm = g_fwhm

            # Se achamos pelo menos 35 estrelas, temos dados suficientes e paramos
            if len(g_stars) >= 35:
                break

            current_sigma -= 0.2

        # Sobrescreve os dados do frame Master com a melhor detecção encontrada
        info["anchor_stars"] = best_stars
        info["star_count"] = len(best_stars)
        info["fwhm"] = best_fwhm
        info["anchor_data"] = prepare_for_phase_correlation(working_data)

        app_print(
            f"  -> {info['batch_name']}: {len(best_stars):02d} estrelas pescadas "
            f"(Sigma parou em {current_sigma:.1f})\n"
        )

    # (O código segue normalmente com a eleição do global_master_cfg...)
    global_master_cfg = config.get("global_master", "Auto")

    if str(global_master_cfg).lower() == "auto":
        master_info = max(
            anchors_info,
            key=lambda item: (
                item.get(
                    "anchor_quality",
                    0.0,
                ),
                item.get(
                    "star_count",
                    0,
                ),
            ),
        )

        app_print(
            f"\n[GLOBAL] "
            f"Master Automático eleito: "
            f"{master_info['batch_name']} | "
            f"estrelas={master_info['star_count']} | "
            f"FWHM={master_info['fwhm']:.2f}\n"
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
                    item.get(
                        "anchor_quality",
                        0.0,
                    ),
                    item.get(
                        "star_count",
                        0,
                    ),
                ),
            )

            app_print(
                f"\n[GLOBAL] "
                f"Master especificado não encontrado. "
                f"Fallback: {master_info['batch_name']}\n"
            )

    matching_radius = float(
        config.get(
            "global_matching_radius",
            config.get(
                "matching_radius",
                100.0,  # Modificação 2
            ),
        )
    )

    ransac_thresh = float(
        config.get(
            "global_ransac",
            config.get(
                "ransac",
                5.0,
            ),
        )
    )

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

    # Master = identidade.
    global_flow["batches"][master_info["batch_name"]] = {
        "status": "accepted",
        "matrix": np.eye(3).tolist(),
        "relative_to": None,
        "strategy": "master",
        "matches": master_info["star_count"],
        "inliers": master_info["star_count"],
        "inlier_ratio": 1.0,
        "rms": 0.0,
        "translation": [
            0.0,
            0.0,
        ],
        "translation_magnitude": 0.0,
        "rotation_deg": 0.0,
        "scale": 1.0,
        "phase_response": 1.0,
    }

    master_index = anchors_info.index(master_info)

    # --------------------------------------------------------
    # Primeira passagem:
    # Master Direct
    # --------------------------------------------------------

    direct_results = {}

    for index, target_info in enumerate(anchors_info):
        if cancel_event.is_set():
            return

        batch_name = target_info["batch_name"]

        if batch_name == master_info["batch_name"]:
            continue

        app_progress(
            index + 1,
            total_batches,
            f"Alinhando {batch_name} ao Master Global...",
        )

        matrix, metrics = _estimate_global_pair(
            master_info,
            target_info,
            matching_radius,
            ransac_thresh,
            limits,
        )

        direct_results[batch_name] = (
            matrix,
            metrics,
        )

    # --------------------------------------------------------
    # Segunda passagem:
    # construção por vizinhança para falhas diretas
    # --------------------------------------------------------

    ordered_global_matrices = {
        master_info["batch_name"]: np.eye(
            3,
            dtype=np.float64,
        )
    }

    # Tenta processar batches adjacentes em ambas as direções
    # a partir das matrizes que já estão disponíveis.
    for distance in range(
        1,
        len(anchors_info),
    ):
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
                    ref_info,
                    target_info,
                    matching_radius,
                    ransac_thresh,
                    limits,
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

    # --------------------------------------------------------
    # Insere resultados diretos que ainda não entraram
    # --------------------------------------------------------

    for target_info in anchors_info:
        batch_name = target_info["batch_name"]

        if batch_name == master_info["batch_name"]:
            continue

        if batch_name in global_flow["batches"]:
            continue

        direct_matrix, direct_metrics = direct_results.get(
            batch_name,
            (None, {}),
        )

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
            # Modificação 3
            # Se não conseguiu alinhamento direto, tenta neighbor_chain
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
                    ref_info,
                    target_info,
                    matching_radius,
                    ransac_thresh,
                    limits,
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
                    "reason": direct_metrics.get(
                        "reason",
                        "global_alignment_failed",
                    ),
                }

    # --------------------------------------------------------
    # Recentraliza pelo Master
    #
    # O Master já é identidade. Aqui só garantimos que todas
    # as matrizes finais são expressas no mesmo referencial.
    # --------------------------------------------------------

    # --------------------------------------------------------
    # Cross-checks:
    # compara direto Master vs matriz já obtida por cadeia.
    # --------------------------------------------------------

    for index, target_info in enumerate(anchors_info):
        batch_name = target_info["batch_name"]

        if batch_name == master_info["batch_name"]:
            continue

        entry = global_flow["batches"].get(batch_name)

        if not entry:
            continue

        direct_matrix, direct_metrics = direct_results.get(
            batch_name,
            (None, {}),
        )

        final_matrix = entry.get("matrix")

        if direct_matrix is None or final_matrix is None:
            continue

        final_matrix_np = np.asarray(
            final_matrix,
            dtype=np.float64,
        )

        closure_error = _matrix_difference_score(
            direct_matrix,
            final_matrix_np,
        )

        cross_check = {
            "target": batch_name,
            "direct_status": direct_metrics.get(
                "status",
                "unknown",
            ),
            "closure_error": round(
                closure_error,
                6,
            ),
            "consistent": bool(
                closure_error
                <= float(
                    config.get(
                        "global_closure_threshold",
                        0.05,
                    )
                )
            ),
        }

        global_flow["cross_checks"].append(cross_check)

        entry["closure_error"] = cross_check["closure_error"]

        if not cross_check["consistent"]:
            entry["warning"] = "global_cross_check_inconsistent"

    # --------------------------------------------------------
    # Quality summary
    # --------------------------------------------------------

    accepted_batches = [
        entry
        for entry in global_flow["batches"].values()
        if entry.get("status") == "accepted"
    ]

    rejected_batches = [
        entry
        for entry in global_flow["batches"].values()
        if entry.get("status") == "rejected"
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
        "anchor_quality": master_info.get(
            "anchor_quality",
            0.0,
        ),
    }

    global_path = base_dir / "global_flow.json"

    with global_path.open(
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            global_flow,
            f,
            indent=4,
            ensure_ascii=False,
        )

    app_progress(
        total_batches,
        total_batches,
        "AstroFlow Finalizado.",
    )

    app_print(
        "\n>>> AstroFlow Finalizado. "
        f"{len(accepted_batches)}/"
        f"{len(anchors_info)} Batches aceitas no Global Flow. <<<\n"
    )


# ============================================================
# Preview
# ============================================================


def preview_star_detection(
    batch_dir: Path,
    config: dict,
) -> tuple[
    np.ndarray | None,
    int,
    float,
]:

    files = sorted(
        [
            p
            for p in batch_dir.iterdir()
            if (
                p.is_file()
                and p.suffix.lower()
                in {
                    ".fit",
                    ".fits",
                    ".fts",
                }
            )
        ]
    )

    if not files:
        return (
            None,
            0,
            0.0,
        )

    custom_anchors = config.get(
        "custom_anchors",
        {},
    )

    anchor_name = custom_anchors.get(batch_dir.name)

    anchor_file = next(
        (p for p in files if p.name == anchor_name),
        files[0],
    )

    data, header = load_fits_data(anchor_file)

    working_data = extract_luminance(
        data,
        header,
    )

    fwhm_val = float(
        config.get(
            "fwhm",
            4.0,
        )
    )

    sigma_val = float(
        config.get(
            "sigma",
            5.0,
        )
    )

    max_stars_val = int(
        config.get(
            "max_stars",
            250,
        )
    )

    engine_val = config.get(
        "engine",
        "DAO",
    )

    stars, measured_fwhm, metrics = detect_stars(
        working_data,
        fwhm_val,
        sigma_val,
        max_stars_val,
        engine_val,
    )

    _, median, std = sigma_clipped_stats(
        working_data,
        sigma=3.0,
    )

    median = float(median)

    std = float(std)

    denominator = max(
        (8.0 * std),
        1e-5,
    )

    vmin = median

    vmax = median + denominator

    norm_data = (
        np.clip(
            (working_data - vmin)
            / max(
                vmax - vmin,
                1e-5,
            ),
            0,
            1,
        )
        * 255.0
    )

    img_8u = norm_data.astype(np.uint8)

    img_color = cv2.cvtColor(
        img_8u,
        cv2.COLOR_GRAY2BGR,
    )

    radius = max(
        3,
        int(
            max(
                measured_fwhm,
                fwhm_val,
            )
            * 1.5
        ),
    )

    for x, y in stars:
        cv2.circle(
            img_color,
            (
                int(round(x)),
                int(round(y)),
            ),
            radius,
            (0, 0, 255),
            1,
        )

    return (
        img_color,
        len(stars),
        float(measured_fwhm),
    )


# ============================================================
# Frame metrics
# ============================================================


def save_frame_metrics(
    image_path: Path | str,
    metrics: dict,
):
    base_path, _ = os.path.splitext(str(image_path))

    json_path = f"{base_path}_metrics.json"

    with open(
        json_path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            metrics,
            f,
            indent=4,
            ensure_ascii=False,
        )

    return json_path
