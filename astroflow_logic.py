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

# Suprime todos os avisos de verificação de cabeçalho do Astropy
warnings.simplefilter("ignore", category=AstropyWarning)


def get_bayer_pattern(header: fits.Header) -> str | None:
    for key in ["BAYERPAT", "BAYERPATTERN", "COLORTYP"]:
        if key in header:
            val = str(header[key]).strip().upper().strip("'")
            if val in ["RGGB", "BGGR", "GRBG", "GBRG"]:
                return val
    return None


def split_cfa(data: np.ndarray, pattern: str):
    if pattern == "RGGB":
        return data[0::2, 0::2], data[0::2, 1::2], data[1::2, 0::2], data[1::2, 1::2]
    elif pattern == "BGGR":
        return data[1::2, 1::2], data[0::2, 1::2], data[1::2, 0::2], data[0::2, 0::2]
    elif pattern == "GRBG":
        return data[0::2, 1::2], data[0::2, 0::2], data[1::2, 1::2], data[1::2, 0::2]
    elif pattern == "GBRG":
        return data[1::2, 0::2], data[1::2, 1::2], data[0::2, 0::2], data[0::2, 1::2]
    return data[0::2, 0::2], data[0::2, 1::2], data[1::2, 0::2], data[1::2, 1::2]


def extract_luminance(data: np.ndarray, header: fits.Header) -> np.ndarray:
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

    elif data.ndim == 2:
        pattern = get_bayer_pattern(header)
        if pattern:
            r, g1, g2, b = split_cfa(data, pattern)
            l_sub = 0.2126 * r + 0.3576 * g1 + 0.3576 * g2 + 0.0722 * b
            l_full = np.repeat(np.repeat(l_sub, 2, axis=0), 2, axis=1)
            return l_full.astype(np.float32)
        else:
            return data.astype(np.float32)
    raise ValueError(f"Dimensões não suportadas: {data.shape}")


def load_fits_data(filepath: Path) -> tuple[np.ndarray, fits.Header]:
    with fits.open(filepath, memmap=False) as hdul:
        for hdu in hdul:
            if hdu.is_image and hdu.data is not None and hdu.data.ndim in (2, 3):
                return np.asarray(hdu.data, dtype=np.float32), hdu.header.copy()
    raise ValueError(f"Imagem inválida: {filepath.name}")


def prepare_for_phase_correlation(data: np.ndarray) -> np.ndarray:
    d_min, d_max = np.min(data), np.max(data)
    if d_max > d_min:
        return ((data - d_min) / (d_max - d_min)).astype(np.float32)
    return np.zeros_like(data).astype(np.float32)


def calculate_anchor_quality(star_count: int, fwhm: float) -> float:
    if fwhm <= 0:
        return float(star_count)
    return float(star_count) / float(fwhm)


def detect_stars_dao(
    data: np.ndarray, fwhm: float, sigma: float, max_stars: int
) -> tuple[np.ndarray, float, dict]:
    mean_val, median_val, std_val = (
        float(np.mean(data)),
        float(np.median(data)),
        float(np.std(data)),
    )
    _, bkg_median, bkg_std = sigma_clipped_stats(data, sigma=3.0)

    daofind = DAOStarFinder(fwhm=fwhm, threshold=sigma * bkg_std)
    sources = daofind(data - bkg_median)

    if sources is not None and len(sources) > 0:
        sources.sort("flux")
        sources.reverse()
        sources = sources[:max_stars]
        coords = np.transpose((sources["xcentroid"], sources["ycentroid"]))
        fluxes = sources["flux"]
        current_fwhm = (
            float(np.mean(sources["sharpness"]) * fwhm)
            if "sharpness" in sources.colnames
            else fwhm
        )
        star_count, min_flux, max_flux = (
            len(sources),
            float(np.min(fluxes)),
            float(np.max(fluxes)),
        )
        snr = float(np.mean(fluxes) / bkg_std) if bkg_std > 0 else 0.0
    else:
        coords, current_fwhm, star_count, min_flux, max_flux, snr = (
            np.empty((0, 2)),
            0.0,
            0,
            0.0,
            0.0,
            0.0,
        )

    valid = bool(star_count > 10 and current_fwhm < (fwhm * 2.0))
    metrics = {
        "star_count": star_count,
        "fwhm": round(current_fwhm, 2),
        "mean": round(mean_val, 2),
        "median": round(float(bkg_median), 2),
        "std": round(std_val, 2),
        "background": round(float(bkg_median), 2),
        "snr": round(snr, 2),
        "min_flux": round(min_flux, 2),
        "max_flux": round(max_flux, 2),
        "valid": valid,
    }
    return coords, current_fwhm, metrics


def detect_stars_opencv(
    data: np.ndarray, fwhm: float, sigma: float, max_stars: int
) -> tuple[np.ndarray, float, dict]:
    mean_val, median_val, std_val = (
        float(np.mean(data)),
        float(np.median(data)),
        float(np.std(data)),
    )
    norm_img = cv2.normalize(data, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    ksize = int(fwhm) | 1
    blurred = cv2.GaussianBlur(norm_img, (ksize, ksize), 0)
    threshold_val = min(255, np.median(blurred) + (sigma * np.std(blurred)))
    _, thresh = cv2.threshold(blurred, threshold_val, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    valid_contours = sorted(
        [c for c in contours if 2 < cv2.contourArea(c) < 1000],
        key=cv2.contourArea,
        reverse=True,
    )[:max_stars]

    coords_list, areas, fluxes = [], [], []
    for cnt in valid_contours:
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cX, cY = M["m10"] / M["m00"], M["m01"] / M["m00"]
            coords_list.append([cX, cY])
            areas.append(cv2.contourArea(cnt))
            fluxes.append(float(data[int(cY), int(cX)]))

    coords = np.array(coords_list) if coords_list else np.empty((0, 2))
    star_count = len(coords)
    current_fwhm = (
        float(np.mean([np.sqrt(a / np.pi) * 2 for a in areas])) if areas else 0.0
    )
    min_flux, max_flux = (
        (float(np.min(fluxes)), float(np.max(fluxes))) if fluxes else (0.0, 0.0)
    )
    snr = float(np.mean(fluxes) / std_val) if std_val > 0 and fluxes else 0.0

    valid = bool(star_count > 10 and current_fwhm < (fwhm * 2.5))
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
    return coords, current_fwhm, metrics


def detect_stars(
    data: np.ndarray, fwhm: float, sigma: float, max_stars: int, engine: str = "DAO"
) -> tuple[np.ndarray, float, dict]:
    if engine.upper() == "OPENCV":
        return detect_stars_opencv(data, fwhm, sigma, max_stars)
    return detect_stars_dao(data, fwhm, sigma, max_stars)


def extract_geometric_properties(
    matrix_2x3: np.ndarray,
) -> tuple[float, float, float, float]:
    """Extrai translação, rotação e escala de uma matriz afim."""
    a, b, tx = matrix_2x3[0]
    c, d, ty = matrix_2x3[1]
    scale = float(np.sqrt(a**2 + c**2))
    rotation_deg = float(np.degrees(np.arctan2(c, a)))
    return float(tx), float(ty), rotation_deg, scale


def validate_transform(matrix_2x3, metrics, limits) -> tuple[bool, str]:
    """Valida geometricamente a confiabilidade do matching calculado."""
    if matrix_2x3 is None:
        return False, "phase_correlation_failed" if metrics.get(
            "matches"
        ) == 0 else "insufficient_matches"

    if metrics["inliers"] < limits["min_inliers"]:
        return False, "insufficient_inliers"
    if metrics["inlier_ratio"] < limits["min_ratio"]:
        return False, "low_inlier_ratio"
    if metrics.get("rms", 999.0) > limits["max_rms"]:
        return False, "high_rms"

    tx, ty, rot, scale = extract_geometric_properties(matrix_2x3)
    metrics["translation"] = [round(tx, 3), round(ty, 3)]
    metrics["rotation_deg"] = round(rot, 4)
    metrics["scale"] = round(scale, 4)

    if np.hypot(tx, ty) > limits["max_translation"]:
        return False, "high_translation"
    if abs(rot) > limits["max_rotation"]:
        return False, "high_rotation"
    if not (limits["min_scale"] <= scale <= limits["max_scale"]):
        return False, "invalid_scale"

    return True, "accepted"


def make_homogeneous(matrix_2x3: np.ndarray) -> np.ndarray:
    hom = np.eye(3, dtype=np.float64)
    hom[:2, :] = matrix_2x3
    return hom


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
        stars, measured_fwhm, metrics = detect_stars(
            working_data, fwhm_val, sigma_val, max_stars_val, engine_val
        )

        phase_data = prepare_for_phase_correlation(working_data)

        if len(stars) < min_stars:
            return filepath.name, None

        return filepath.name, {
            "path": filepath,
            "data": data,
            "phase_data": phase_data,
            "stars": stars,
            "fwhm": measured_fwhm,
            "metrics": metrics,
        }
    except Exception:
        return filepath.name, None


def _match_incremental_stars(
    previous_stars: np.ndarray,
    current_stars: np.ndarray,
    shift: tuple[float, float],
    matching_radius: float,
) -> tuple[list, list]:
    dx, dy = shift
    shifted_current = current_stars + np.array([dx, dy], dtype=np.float32)
    tree = KDTree(previous_stars)
    distances, indices = tree.query(
        shifted_current, distance_upper_bound=matching_radius
    )

    candidates = [
        (float(d), int(p_idx), int(c_idx))
        for c_idx, (d, p_idx) in enumerate(zip(distances, indices))
        if np.isfinite(d) and p_idx < len(previous_stars)
    ]
    candidates.sort(key=lambda x: x[0])

    used_previous = set()
    previous_points, current_points = [], []

    for dist, previous_idx, current_idx in candidates:
        if previous_idx not in used_previous:
            used_previous.add(previous_idx)
            previous_points.append(previous_stars[previous_idx])
            current_points.append(current_stars[current_idx])

    return previous_points, current_points


def _estimate_incremental_transform(
    previous_stars: list, current_stars: list, ransac_thresh: float, min_stars: int
):
    if len(previous_stars) < min_stars:
        return None, {
            "matches": len(previous_stars),
            "inliers": 0,
            "inlier_ratio": 0.0,
            "rms": 999.0,
        }

    previous = np.asarray(previous_stars, dtype=np.float32)
    current = np.asarray(current_stars, dtype=np.float32)

    matrix_2x3, inliers = cv2.estimateAffinePartial2D(
        current,
        previous,
        method=cv2.RANSAC,
        ransacReprojThreshold=ransac_thresh,
        maxIters=5000,
        confidence=0.995,
    )

    if matrix_2x3 is None or inliers is None:
        return None, {
            "matches": len(previous),
            "inliers": 0,
            "inlier_ratio": 0.0,
            "rms": 999.0,
        }

    mask = inliers.ravel().astype(bool)
    inlier_count = int(mask.sum())

    rms = 999.0
    if inlier_count > 0:
        transformed = cv2.transform(current.reshape(-1, 1, 2), matrix_2x3).reshape(
            -1, 2
        )
        errors = transformed[mask] - previous[mask]
        rms = float(np.sqrt(np.mean(np.linalg.norm(errors, axis=1) ** 2)))

    return matrix_2x3, {
        "matches": len(previous),
        "inliers": inlier_count,
        "inlier_ratio": float(inlier_count / len(previous)),
        "rms": rms,
    }


def process_local_flow(batch_dir: Path, config: dict, app_print) -> dict:
    files = sorted(
        [
            p
            for p in batch_dir.iterdir()
            if p.is_file() and p.suffix.lower() in {".fit", ".fits"}
        ]
    )
    if not files:
        return {}
    if not isinstance(config, dict):
        config = {}

    custom_anchors = config.get("custom_anchors", {})
    chosen_anchor_name = custom_anchors.get(batch_dir.name)
    anchor_file = files[0]

    if chosen_anchor_name:
        app_print(
            f"[{batch_dir.name}] Frame Central (Batch Reference): {chosen_anchor_name}\n"
        )
    else:
        chosen_anchor_name = anchor_file.name
        app_print(
            f"[{batch_dir.name}] Nenhuma referência manual. Centralizando no 1º frame.\n"
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
    worker_count = max(1, min(4, cpu_count))

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
                if result is not None:
                    prepared_frames[fname] = result
            except Exception as exc:
                app_print(f"[{filepath.name}] Erro: {exc}\n")

    anchor = prepared_frames.get(anchor_file.name)
    if anchor is None:
        app_print(f"[{batch_dir.name}] ERRO: Não foi possível processar a âncora.\n")
        return {}

    anchor_quality = calculate_anchor_quality(len(anchor["stars"]), anchor["fwhm"])
    app_print(
        f"[{batch_dir.name}] Âncora: {len(anchor['stars'])} estrelas | FWHM={anchor['fwhm']:.2f}px\n"
    )

    flow_data = {
        "batch_anchor": anchor_file.name,
        "frames": {
            anchor_file.name: {
                "status": "accepted",
                "matrix": np.eye(3).tolist(),
                "relative_to": None,
                "matches": len(anchor["stars"]),
                "inliers": len(anchor["stars"]),
                "inlier_ratio": 1.0,
                "rms": 0.0,
                "translation": [0.0, 0.0],
                "rotation_deg": 0.0,
                "scale": 1.0,
            }
        },
    }

    cumulative_matrix = np.eye(3, dtype=np.float64)
    previous_name = anchor_file.name
    previous_frame = anchor

    for index in range(1, len(files)):
        current_name = files[index].name
        current_frame = prepared_frames.get(current_name)

        if current_frame is None:
            flow_data["frames"][current_name] = {
                "status": "rejected",
                "reason": "insufficient_stars_in_detection",
            }
            app_print(f"[{current_name}] REJEITADO: Falha na detecção de fontes.\n")
            continue

        attempts = [
            ("normal", previous_frame, matching_radius, previous_name),
            ("relaxed_radius", previous_frame, matching_radius * 2.0, previous_name),
            ("anchor_fallback", anchor, matching_radius * 1.5, anchor_file.name),
        ]

        accepted = False
        best_metrics = {
            "reason": "phase_correlation_failed",
            "inliers": 0,
            "matches": 0,
            "inlier_ratio": 0.0,
        }
        best_rel_matrix = None

        for attempt_name, ref_frame, radius, ref_name in attempts:
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
            metrics["phase_response"] = round(float(response), 4)

            valid, reason = validate_transform(rel_matrix, metrics, limits)

            if valid:
                accepted = True
                best_rel_matrix = make_homogeneous(rel_matrix)

                if attempt_name == "anchor_fallback":
                    cumulative_matrix = best_rel_matrix
                else:
                    prev_cum = np.array(
                        flow_data["frames"][ref_name]["matrix"], dtype=np.float64
                    )
                    cumulative_matrix = prev_cum @ best_rel_matrix

                flow_data["frames"][current_name] = {
                    "status": "accepted",
                    "matrix": cumulative_matrix.tolist(),
                    "relative_to": ref_name,
                    "recovery_method": attempt_name,
                    **metrics,
                }
                app_print(
                    f"[{current_name}] OK ({attempt_name}) ← {ref_name} | {metrics['inliers']}/{metrics['matches']} inls | RMS={metrics['rms']:.2f}\n"
                )
                break
            else:
                metrics["reason"] = reason
                if metrics.get("inliers", 0) >= best_metrics.get("inliers", 0):
                    best_metrics = metrics

        if accepted:
            previous_frame = current_frame
            previous_name = current_name
        else:
            flow_data["frames"][current_name] = {
                "status": "rejected",
                "reason": best_metrics.get("reason", "unknown"),
                "matches": best_metrics.get("matches", 0),
                "inliers": best_metrics.get("inliers", 0),
                "inlier_ratio": round(best_metrics.get("inlier_ratio", 0.0), 3),
            }
            app_print(
                f"[{current_name}] REJEITADO: {best_metrics.get('reason')} (Inliers: {best_metrics.get('inliers')})\n"
            )

    if (
        chosen_anchor_name in flow_data["frames"]
        and flow_data["frames"][chosen_anchor_name].get("status") == "accepted"
    ):
        anchor_matrix = np.array(
            flow_data["frames"][chosen_anchor_name]["matrix"], dtype=np.float64
        )
        try:
            inv_anchor = np.linalg.inv(anchor_matrix)
            for fname, fdata in flow_data["frames"].items():
                if fdata.get("status") == "accepted":
                    old_mat = np.array(fdata["matrix"], dtype=np.float64)
                    fdata["matrix"] = (inv_anchor @ old_mat).tolist()

            flow_data["batch_anchor"] = chosen_anchor_name
        except np.linalg.LinAlgError:
            pass

    with open(batch_dir / "flow_local.json", "w", encoding="utf-8") as f:
        json.dump(flow_data, f, indent=4)

    return {
        "batch_name": batch_dir.name,
        "anchor_data": anchor["phase_data"],
        "anchor_stars": anchor["stars"],
        "star_count": len(anchor["stars"]),
        "fwhm": anchor["fwhm"],
    }


def process_all_flows(
    base_dir: Path, config: dict, app_print, app_progress, cancel_event
):
    batch_folders = sorted(
        [d for d in base_dir.iterdir() if d.is_dir() and "batch" in d.name.lower()]
    )
    if not batch_folders:
        return

    anchors_info = []
    worker_count = max(1, min(4, (os.cpu_count() or 1)))

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        future_to_batch = {
            executor.submit(
                process_local_flow, b_folder, config, lambda msg: None
            ): b_folder
            for b_folder in batch_folders
        }
        for future in as_completed(future_to_batch):
            if cancel_event.is_set():
                return
            b_folder = future_to_batch[future]
            try:
                info = future.result()
                if info:
                    anchors_info.append(info)
                app_print(f"Flow Local Finalizado: {b_folder.name}\n")
            except Exception as exc:
                app_print(f"Erro em {b_folder.name}: {exc}\n")

    if not anchors_info:
        return
    anchors_info.sort(key=lambda x: x["batch_name"])

    global_master_cfg = config.get("global_master", "Auto")
    if str(global_master_cfg).lower() == "auto":
        master_info = min(
            anchors_info, key=lambda x: x["fwhm"] / max(1, x["star_count"])
        )
        app_print(f"\n[GLOBAL] Master Automático eleito: {master_info['batch_name']}\n")
    else:
        master_info = next(
            (a for a in anchors_info if a["batch_name"] == global_master_cfg),
            anchors_info[0],
        )

    global_flow = {"global_master_batch": master_info["batch_name"], "batches": {}}

    global_flow["batches"][master_info["batch_name"]] = {
        "status": "accepted",
        "matrix": np.eye(3).tolist(),
        "relative_to": None,
    }

    limits = {
        "min_stars": 4,
        "min_inliers": 4,
        "min_ratio": 0.10,
        "max_rms": 6.0,
        "max_translation": 3000.0,
        "max_rotation": 20.0,
        "min_scale": 0.85,
        "max_scale": 1.15,
    }

    total_global = len(anchors_info)
    master_idx = anchors_info.index(master_info)

    for i in range(total_global):
        if i == master_idx:
            continue
        tgt_info = anchors_info[i]

        attempts = [
            ("master_direct", anchors_info[master_idx]),
            ("neighbor_chain", anchors_info[i - 1] if i > 0 else anchors_info[i + 1]),
        ]

        accepted = False
        for attempt_name, ref_info in attempts:
            if ref_info["batch_name"] == tgt_info["batch_name"]:
                continue

            shift, response = cv2.phaseCorrelate(
                ref_info["anchor_data"], tgt_info["anchor_data"]
            )
            m_ref, m_tgt = _match_incremental_stars(
                ref_info["anchor_stars"], tgt_info["anchor_stars"], shift, 35.0
            )

            rel_matrix, metrics = _estimate_incremental_transform(m_ref, m_tgt, 5.0, 4)
            valid, reason = validate_transform(rel_matrix, metrics, limits)

            if valid:
                hom_matrix = make_homogeneous(rel_matrix)

                if (
                    attempt_name == "neighbor_chain"
                    and ref_info["batch_name"] in global_flow["batches"]
                ):
                    ref_matrix = np.array(
                        global_flow["batches"][ref_info["batch_name"]]["matrix"],
                        dtype=np.float64,
                    )
                    final_matrix = ref_matrix @ hom_matrix
                else:
                    final_matrix = hom_matrix

                global_flow["batches"][tgt_info["batch_name"]] = {
                    "status": "accepted",
                    "matrix": final_matrix.tolist(),
                    "relative_to": ref_info["batch_name"],
                    "strategy": attempt_name,
                    "phase_shift": [
                        round(float(shift[0]), 3),
                        round(float(shift[1]), 3),
                    ],
                    "phase_response": round(float(response), 4),
                    **metrics,
                }
                accepted = True
                break

        if not accepted:
            app_print(
                f"  -> Falha Crítica Global: {tgt_info['batch_name']}. Assumindo tracking estático.\n"
            )
            global_flow["batches"][tgt_info["batch_name"]] = {
                "status": "rejected_fallback_static",
                "matrix": np.eye(3).tolist(),
            }

    with open(base_dir / "global_flow.json", "w") as f:
        json.dump(global_flow, f, indent=4)


def preview_star_detection(
    batch_dir: Path, config: dict
) -> tuple[np.ndarray | None, int, float]:
    """
    Carrega a âncora de uma batch, detecta as estrelas e desenha os círculos
    proporcionais ao FWHM detectado para pré-visualização em tempo real.
    """
    files = sorted(
        [
            p
            for p in batch_dir.iterdir()
            if p.is_file() and p.suffix.lower() in {".fit", ".fits"}
        ]
    )
    if not files:
        return None, 0, 0.0

    anchor_file = files[0]

    data, header = load_fits_data(anchor_file)
    working_data = extract_luminance(data, header)

    fwhm_val = float(config.get("fwhm", 4.0))
    sigma_val = float(config.get("sigma", 5.0))
    max_stars_val = int(config.get("max_stars", 250))

    engine_val = config.get("engine", "DAO")
    stars, measured_fwhm, metrics = detect_stars(
        working_data, fwhm_val, sigma_val, max_stars_val, engine_val
    )
    star_count = len(stars)

    # Normaliza a imagem para 8-bits para exibir no OpenCV/Matplotlib
    mean, median, std = sigma_clipped_stats(working_data, sigma=3.0)
    vmin, vmax = median, median + (8 * std)

    # Proteção de divisão por zero caso a imagem seja completamente preta
    denominator = max(vmax - vmin, 1e-5)
    norm_data = np.clip((working_data - vmin) / denominator, 0, 1) * 255
    img_8u = norm_data.astype(np.uint8)

    # Converte para BGR para desenhar os círculos coloridos
    if img_8u.ndim == 2:
        img_color = cv2.cvtColor(img_8u, cv2.COLOR_GRAY2BGR)
    else:
        img_color = img_8u.copy()

    # Desenha os círculos dimensionados proporcionalmente ao FWHM real da estrela
    radius = max(3, int(fwhm_val * 1.5))
    for x, y in stars:
        cv2.circle(img_color, (int(x), int(y)), radius, (0, 0, 255), 1)

    return img_color, star_count, measured_fwhm


def save_frame_metrics(image_path: Path | str, metrics: dict):
    """Salva as métricas fotométricas em um arquivo .json ao lado do frame original."""
    import os

    # Permite compatibilidade com Pathlib ou Strings de caminho
    base_path, _ = os.path.splitext(str(image_path))
    json_path = f"{base_path}_metrics.json"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    return json_path
