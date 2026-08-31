from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import json
import cv2
import numpy as np
from pathlib import Path
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from photutils.detection import DAOStarFinder
from scipy.spatial import KDTree


# Insira estas funções no início de astroflow_logic.py

def get_bayer_pattern(header: fits.Header) -> str | None:
    for key in ['BAYERPAT', 'BAYERPATTERN', 'COLORTYP']:
        if key in header:
            val = str(header[key]).strip().upper().strip("'")
            if val in ['RGGB', 'BGGR', 'GRBG', 'GBRG']:
                return val
    return None

def split_cfa(data: np.ndarray, pattern: str):
    if pattern == 'RGGB':
        return data[0::2, 0::2], data[0::2, 1::2], data[1::2, 0::2], data[1::2, 1::2]
    elif pattern == 'BGGR':
        return data[1::2, 1::2], data[0::2, 1::2], data[1::2, 0::2], data[0::2, 0::2]
    elif pattern == 'GRBG':
        return data[0::2, 1::2], data[0::2, 0::2], data[1::2, 1::2], data[1::2, 0::2]
    elif pattern == 'GBRG':
        return data[1::2, 0::2], data[1::2, 1::2], data[0::2, 0::2], data[0::2, 1::2]
    return data[0::2, 0::2], data[0::2, 1::2], data[1::2, 0::2], data[1::2, 1::2]

def extract_luminance(data: np.ndarray, header: fits.Header) -> np.ndarray:
    pattern = get_bayer_pattern(header)
    if not pattern:
        return data  # Mantém monocromático ou canais únicos puros
        
    r, g1, g2, b = split_cfa(data, pattern)
    
    # Aplica a ponderação visual humana padrão sem misturar pixels de posições diferentes
    l_sub = 0.2126 * r + 0.3576 * g1 + 0.3576 * g2 + 0.0722 * b
    
    # Realiza up-scale com "Nearest Neighbor" purista (np.repeat) para 
    # manter as coordenadas x,y das estrelas rigorosamente fiéis à escala do sensor original
    l_full = np.repeat(np.repeat(l_sub, 2, axis=0), 2, axis=1)
    
    return l_full.astype(np.float32)

def load_fits_data(filepath: Path) -> tuple[np.ndarray, fits.Header]:
    with fits.open(filepath, memmap=False) as hdul:
        for hdu in hdul:
            if hdu.is_image and hdu.data is not None and hdu.data.ndim == 2:
                header = hdu.header.copy()
                data = np.asarray(hdu.data, dtype=np.float32)
                return data, header
    raise ValueError(f"Imagem 2D não encontrada em {filepath.name}")

def prepare_for_phase_correlation(data: np.ndarray) -> np.ndarray:
    """Prepara a imagem normalizada para o cv2.phaseCorrelate."""
    d_min, d_max = np.min(data), np.max(data)
    if d_max > d_min:
        norm = (data - d_min) / (d_max - d_min)
    else:
        norm = np.zeros_like(data)
    return norm.astype(np.float32)

def calculate_anchor_quality(star_count: int, fwhm: float) -> float:
    """Métrica simples para avaliar a qualidade da âncora (mais estrelas e menor FWHM = melhor)."""
    if fwhm <= 0:
        return float(star_count)
    return float(star_count) / float(fwhm)

def detect_stars_opencv(data: np.ndarray, fwhm: float, sigma: float, max_stars: int) -> tuple[np.ndarray, float, dict]:
    """Motor ultrarrápido utilizando Visão Computacional bruta do OpenCV e Momentos de Imagem."""
    mean, median, std = sigma_clipped_stats(data, sigma=3.0)
    
    if not np.isfinite(std) or std <= 0:
        return np.empty((0, 2), dtype=np.float32), 0.0, {"star_count": 0, "mean": 0.0, "median": 0.0, "std": 0.0, "fwhm": 0.0}

    # Limiar rigoroso para isolar as fontes de luz
    thresh_val = median + (sigma * std)
    binary_mask = (data > thresh_val).astype(np.uint8)

    # Identifica componentes conectados (muito mais rápido que findContours)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)

    pts = []
    brightness = []
    
    # Raio da caixa limitadora para cálculo do centroide sub-pixel
    r = int(max(2, fwhm))
    h, w = data.shape

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        
        # Filtra ruídos de 1 pixel e bolhas massivas
        if 4 < area < (fwhm * fwhm * 15):
            cx, cy = centroids[i]
            ix, iy = int(cx), int(cy)
            
            # Recorta a vizinhança da estrela para calcular o centro de massa exato (precisão sub-pixel)
            y0, y1 = max(0, iy - r), min(h, iy + r + 1)
            x0, x1 = max(0, ix - r), min(w, ix + r + 1)
            
            patch = data[y0:y1, x0:x1] - median
            patch[patch < 0] = 0
            
            M = cv2.moments(patch)
            if M["m00"] > 0:
                px = M["m10"] / M["m00"]
                py = M["m01"] / M["m00"]
                pts.append([x0 + px, y0 + py])
                brightness.append(M["m00"]) # Usa o fluxo de massa como brilho
            else:
                pts.append([cx, cy])
                brightness.append(area)

    if len(pts) == 0:
        return np.empty((0, 2), dtype=np.float32), float(fwhm), {"star_count": 0, "mean": float(mean), "median": float(median), "std": float(std), "fwhm": float(fwhm)}

    pts = np.array(pts, dtype=np.float32)
    brightness = np.array(brightness, dtype=np.float32)

    # Ordena mantendo apenas as N estrelas mais brilhantes
    sort_idx = np.argsort(brightness)[::-1][:max_stars]
    pts = pts[sort_idx]

    metrics = {"star_count": len(pts), "mean": float(mean), "median": float(median), "std": float(std), "fwhm": float(fwhm)}
    return pts, float(fwhm), metrics


def detect_stars_dao(data: np.ndarray, fwhm: float, sigma: float, max_stars: int) -> tuple[np.ndarray, float, dict]:
    """Motor original altamente rigoroso utilizando Astropy/Photutils."""
    mean, median, std = sigma_clipped_stats(data, sigma=3.0)

    if not np.isfinite(std) or std <= 0:
        return np.empty((0, 2), dtype=np.float32), 0.0, {"star_count": 0, "mean": float(mean) if np.isfinite(mean) else None, "median": float(median) if np.isfinite(median) else None, "std": 0.0, "fwhm": 0.0}

    daofind = DAOStarFinder(fwhm=fwhm, threshold=sigma * std)
    sources = daofind(data - median)

    if sources is None or len(sources) < 3:
        return np.empty((0, 2), dtype=np.float32), float(fwhm), {"star_count": 0, "mean": float(mean), "median": float(median), "std": float(std), "fwhm": float(fwhm)}

    sources.sort("flux", reverse=True)
    sources = sources[:max_stars]

    pts = np.column_stack((np.asarray(sources["x_centroid"], dtype=np.float32), np.asarray(sources["y_centroid"], dtype=np.float32)))

    metrics = {"star_count": int(len(pts)), "mean": float(mean), "median": float(median), "std": float(std), "fwhm": float(fwhm)}
    return pts, float(fwhm), metrics


def detect_stars(data: np.ndarray, fwhm: float, sigma: float, max_stars: int, engine: str = "DAO") -> tuple[np.ndarray, float, dict]:
    """Roteador principal que decide qual motor utilizar."""
    if engine.upper() == "OPENCV":
        return detect_stars_opencv(data, fwhm, sigma, max_stars)
    return detect_stars_dao(data, fwhm, sigma, max_stars)

def generate_debug_image(data: np.ndarray, stars: np.ndarray, output_path: Path):
    mean, median, std = sigma_clipped_stats(data, sigma=3.0)
    vmin, vmax = median, median + (10 * std)
    
    norm_data = np.clip((data - vmin) / (vmax - vmin), 0, 1) * 255
    img_8u = norm_data.astype(np.uint8)
    img_color = cv2.cvtColor(img_8u, cv2.COLOR_GRAY2BGR)
    
    for x, y in stars:
        cv2.circle(img_color, (int(x), int(y)), 10, (0, 0, 255), 1)
        
    cv2.imwrite(str(output_path), img_color)

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
    engine_val: str
) -> tuple[str, dict | None]:
    """
    Executada em paralelo.

    Responsabilidade:
      1. Carregar FITS
      2. Detectar estrelas
      3. Medir FWHM
      4. Preparar imagem para phase correlation

    NÃO faz alinhamento.

    O alinhamento é feito posteriormente de forma sequencial,
    pois Frame N depende do Frame N-1.
    """

    try:
        data, header = load_fits_data(filepath) 
        
        # A imagem que o algoritmo vai investigar passa a ser a representação pseudo-Luminância isolada:
        working_data = extract_luminance(data, header)
        stars, measured_fwhm, metrics = detect_stars(working_data, fwhm_val, sigma_val, max_stars_val, engine_val)

        if len(stars) < min_stars:
            return filepath.name, None

        phase_data = prepare_for_phase_correlation(data)
        return filepath.name, {
            "path": filepath, "data": data, "phase_data": phase_data,
            "stars": stars, "fwhm": measured_fwhm, "metrics": metrics
        }
    except Exception:
        return filepath.name, None



def _match_incremental_stars(
    previous_stars: np.ndarray,
    current_stars: np.ndarray,
    shift: tuple[float, float],
    matching_radius: float
) -> tuple[list, list]:
    """
    Faz o matching entre dois frames consecutivos.

    current_stars é deslocado pela estimativa de phase correlation
    antes do KDTree.

    Também garante correspondência 1:1.
    """

    dx, dy = shift

    shifted_current = (
        current_stars
        + np.array(
            [dx, dy],
            dtype=np.float32
        )
    )

    tree = KDTree(previous_stars)

    distances, indices = tree.query(
        shifted_current,
        distance_upper_bound=matching_radius
    )

    candidates = []

    for current_idx, (dist, previous_idx) in enumerate(
        zip(distances, indices)
    ):

        if (
            np.isfinite(dist)
            and previous_idx < len(previous_stars)
        ):
            candidates.append(
                (
                    float(dist),
                    int(previous_idx),
                    int(current_idx)
                )
            )

    # Melhor correspondência primeiro.
    candidates.sort(
        key=lambda x: x[0]
    )

    used_previous = set()

    previous_points = []
    current_points = []

    for dist, previous_idx, current_idx in candidates:

        if previous_idx in used_previous:
            continue

        used_previous.add(previous_idx)

        previous_points.append(
            previous_stars[previous_idx]
        )

        current_points.append(
            current_stars[current_idx]
        )

    return previous_points, current_points


def _estimate_incremental_transform(
    previous_stars: list,
    current_stars: list,
    ransac_thresh: float,
    min_stars: int,
    min_inliers: int,
    min_ratio: float
):
    """
    Calcula a transformação:

        Current -> Previous

    Retorna matriz homogênea e métricas.
    """

    if len(previous_stars) < min_stars:
        return None, {"matches": len(previous_stars), "inliers": 0, "inlier_ratio": 0.0, "rms": None}


    previous = np.asarray(
        previous_stars,
        dtype=np.float32
    )

    current = np.asarray(
        current_stars,
        dtype=np.float32
    )

    matrix_2x3, inliers = cv2.estimateAffinePartial2D(
        current,
        previous,
        method=cv2.RANSAC,
        ransacReprojThreshold=ransac_thresh,
        maxIters=5000,
        confidence=0.995,
        refineIters=10
    )

    if matrix_2x3 is None or inliers is None:
        return None, {
            "matches": len(previous),
            "inliers": 0,
            "inlier_ratio": 0.0,
            "rms": None
        }

    mask = inliers.ravel().astype(bool)

    inlier_count = int(mask.sum())

    if inlier_count == 0:
        return None, {
            "matches": len(previous),
            "inliers": 0,
            "inlier_ratio": 0.0,
            "rms": None
        }

    transformed = cv2.transform(
        current.reshape(-1, 1, 2),
        matrix_2x3
    ).reshape(-1, 2)

    errors = (
        transformed[mask]
        - previous[mask]
    )

    distances = np.linalg.norm(
        errors,
        axis=1
    )

    rms = float(
        np.sqrt(np.mean(distances ** 2))
    )

    inlier_ratio = (
        inlier_count / len(previous)
    )

    metrics = {
        "matches": len(previous),
        "inliers": inlier_count,
        "inlier_ratio": float(inlier_ratio),
        "rms": rms
    }

    if inlier_count < min_inliers:
        return None, metrics

    if inlier_ratio < min_ratio:
        return None, metrics

    return make_homogeneous(matrix_2x3), metrics


def process_local_flow(
    batch_dir: Path,
    config: dict,
    app_print
) -> dict:

    files = sorted([p for p in batch_dir.iterdir() if p.is_file() and p.suffix.lower() in {".fit", ".fits"}])
    if not files: return {}

    if not isinstance(config, dict): config = {}

    anchor_mode = config.get("anchor_mode", "first") # Pode ser "first", "middle", "last"

    # Extrai o dicionário de âncoras personalizadas injetado pela interface
    custom_anchors = config.get("custom_anchors", {})
    chosen_anchor_name = custom_anchors.get(batch_dir.name)

    anchor_file = files[0]
    
    if chosen_anchor_name:
        app_print(f"[{batch_dir.name}] Frame Central (Batch Reference): {chosen_anchor_name}\n")
    else:
        chosen_anchor_name = anchor_file.name
        app_print(f"[{batch_dir.name}] Nenhuma referência manual. Centralizando no 1º frame.\n")

    if not isinstance(config, dict):
        config = {}

    fwhm_val = float(config.get("fwhm", 4.0))
    sigma_val = float(config.get("sigma", 5.0))
    max_stars_val = int(config.get("max_stars", 250))
    matching_radius = float(config.get("matching_radius", 25.0))
    ransac_thresh = float(config.get("ransac", 4.0))
    
    # Parâmetros que antes eram hardcoded agora são dinâmicos da UI:
    min_stars = int(config.get("min_stars", 4))
    min_inliers = int(config.get("min_inliers", 4))
    min_ratio = float(config.get("min_ratio", 0.15))
    debug_flag = bool(config.get("debug_images", False))

    # ========================================================
    # 1. PRÉ-CARREGAMENTO / DETECÇÃO PARALELA
    # ========================================================

    try:
        cpu_count = (getattr(os,"process_cpu_count",os.cpu_count)()or 1)
    except Exception:
        cpu_count = os.cpu_count() or 1

    worker_count = max(1, min(4, cpu_count)
    )

    app_print(f"[{batch_dir.name}] Pré-processamento paralelo: {worker_count} workers\n")

    prepared_frames = {}
    
    engine_val = config.get("engine", "DAO")
    
    with ThreadPoolExecutor(
        max_workers=worker_count,
        thread_name_prefix="astroflow"
    ) as executor:

        futures = { executor.submit(
                    _process_single_frame,
                    filepath,
                    fwhm_val,
                    sigma_val,
                    max_stars_val,
                    min_stars,
                    engine_val): filepath
                        for filepath in files}

        for future in as_completed(futures):

            filepath = futures[future]
            try:
                fname, result = future.result()

            except Exception as exc:

                app_print(f"[{filepath.name}] Erro no worker: {exc}\n")
                continue

            if result is None:
                app_print(f"[{fname}] Estrelas insuficientes.\n")
                continue

            prepared_frames[fname] = result

    # ========================================================
    # 2. VALIDAR ÂNCORA
    # ========================================================

    anchor = prepared_frames.get(anchor_file.name)
    if anchor is None:
        app_print(f"[{batch_dir.name}] ERRO: Não foi possível processar a âncora.\n")
        return {}

    anchor_stars = anchor["stars"]
    anchor_data = anchor["data"]
    anchor_fwhm = anchor["fwhm"]

    if len(anchor_stars) < min_stars:

        app_print(f"[{batch_dir.name}] ERRO: Estrelas insuficientes na âncora ({len(anchor_stars)}).\n")
        return {}

    anchor_quality = calculate_anchor_quality(
        len(anchor_stars),
        anchor_fwhm
    )

    app_print(f"[{batch_dir.name}] Âncora: {len(anchor_stars)} estrelas | FWHM={anchor_fwhm:.2f}px | qualidade={anchor_quality:.4f}\n")
    if debug_flag:
        generate_debug_image(
            anchor_data,
            anchor_stars,
            batch_dir / f"debug_{anchor_file.stem}.jpg"
        )

    # ========================================================
    # 3. ALINHAMENTO INCREMENTAL
    #
    # F0 → F1 → F2 → F3 → ... → FN
    #
    # Cada transformação é local.
    # ========================================================

    flow_data = {
        "batch_anchor": anchor_file.name,
        "mode": "incremental_chain",
        "workers": worker_count,
        "anchor_metrics": {
            "star_count": len(anchor_stars),
            "fwhm": anchor_fwhm,
            "quality": anchor_quality,
            **anchor["metrics"]
        },
        "frames": {}
    }

    # --------------------------------------------------------
    # Âncora = identidade
    # --------------------------------------------------------

    flow_data["frames"][
        anchor_file.name
    ] = {
        "matrix": np.eye(3).tolist(),
        "relative_to": None,
        "matches": len(anchor_stars),
        "inliers": len(anchor_stars),
        "inlier_ratio": 1.0,
        "rms": 0.0,
        "cumulative_rms": 0.0,
        "fwhm": anchor_fwhm
    }

    # --------------------------------------------------------
    # Matriz acumulada:
    #
    # M(F0)
    # M(F1) = M(F0) @ T(F1→F0)
    # M(F2) = M(F1) @ T(F2→F1)
    #
    # ...
    # --------------------------------------------------------

    cumulative_matrix = np.eye(
        3,
        dtype=np.float64
    )

    previous_name = anchor_file.name
    previous_frame = anchor

    # ========================================================
    # 4. CADEIA SEQUENCIAL
    # ========================================================

    for index in range(1, len(files)):

        current_file = files[index]
        current_name = current_file.name

        current_frame = prepared_frames.get(
            current_name
        )

        # ----------------------------------------------------
        # Frame não conseguiu ser detectado.
        #
        # Aqui está uma decisão importante:
        #
        # Não "pula" silenciosamente para o próximo.
        # Tentamos preservar a cadeia.
        # ----------------------------------------------------

        if current_frame is None:

            app_print(f"[{current_name}] REJEITADO na detecção. Cadeia interrompida neste ponto.\n")
            continue

        previous_stars = previous_frame[
            "stars"
        ]

        current_stars = current_frame[
            "stars"
        ]

        # ----------------------------------------------------
        # Phase correlation entre frames ADJACENTES
        # ----------------------------------------------------

        shift, response = cv2.phaseCorrelate(
            previous_frame["phase_data"],
            current_frame["phase_data"]
        )

        dx, dy = shift

        # ----------------------------------------------------
        # Matching
        # ----------------------------------------------------

        m_previous, m_current = (
            _match_incremental_stars(
                previous_stars,
                current_stars,
                (dx, dy),
                matching_radius
            )
        )

        if len(m_previous) < min_stars:

            app_print(f"[{current_name}] REJEITADO: apenas {len(m_previous)} matches com {previous_name}.\n")
            # Mantemos previous_frame como está.
            # Isso permite que o próximo frame ainda seja
            # comparado ao último frame VÁLIDO.
            continue

        # ----------------------------------------------------
        # Transformação Current → Previous
        # ----------------------------------------------------

        relative_matrix, metrics = (
            _estimate_incremental_transform(
                m_previous,
                m_current,
                ransac_thresh,
                min_stars,
                min_inliers,
                min_ratio
            )
        )

        if relative_matrix is None:

            app_print(
                f"  [{current_name}] "
                f"RANSAC REJEITADO: "
                f"{metrics['inliers']}/"
                f"{metrics['matches']} "
                f"inliers "
                f"({metrics['inlier_ratio']:.1%}).\n"
            )

            continue

        # ----------------------------------------------------
        # ACUMULAÇÃO
        #
        # Current → Previous → Anchor
        #
        # T_current_anchor =
        #     T_previous_anchor @
        #     T_current_previous
        # ----------------------------------------------------

        cumulative_matrix = (cumulative_matrix @ relative_matrix)

        # ----------------------------------------------------
        # RMS acumulado aproximado
        # ----------------------------------------------------

        previous_cumulative_rms = (flow_data["frames"][previous_name].get("cumulative_rms",0.0))
        
        current_rms = metrics["rms"] or 0.0

        cumulative_rms = float(np.sqrt(previous_cumulative_rms ** 2 + current_rms ** 2))

        # ----------------------------------------------------
        # Salva frame
        # ----------------------------------------------------

        flow_data["frames"][
            current_name
        ] = {
            "matrix": cumulative_matrix.tolist(),
            "relative_matrix": (
                relative_matrix.tolist()
            ),
            "relative_to": previous_name,
            "matches": metrics[
                "matches"
            ],
            "inliers": metrics[
                "inliers"
            ],
            "inlier_ratio": metrics[
                "inlier_ratio"
            ],
            "rms": metrics[
                "rms"
            ],
            "cumulative_rms": cumulative_rms,
            "phase_correlation_response": float(
                response
            ),
            "phase_shift": [
                float(dx),
                float(dy)
            ],
            "fwhm": current_frame[
                "fwhm"
            ],
            "star_count": len(
                current_stars
            )
        }
        app_print(f"[{current_name}] OK ← {previous_name} | {metrics['inliers']}/{metrics['matches']} inliers "
            f"({metrics['inlier_ratio']:.1%}) | "
            f"RMS={metrics['rms']:.3f}px\n"
        )

        # ----------------------------------------------------
        # MUITO IMPORTANTE:
        #
        # O próximo frame será comparado ao frame atual.
        # ----------------------------------------------------

        previous_name = current_name
        previous_frame = current_frame

    # ========================================================
    # 5. RE-CENTRALIZAÇÃO NA BATCH REFERENCE (Álgebra Reversa)
    # ========================================================
    if chosen_anchor_name in flow_data["frames"]:
        # Extrai a matriz cumulativa que a referência sofreu desde a origem F0
        anchor_matrix = np.array(flow_data["frames"][chosen_anchor_name]["matrix"], dtype=np.float64)
        try:
            # Calcula o inverso da trajetória da referência
            inv_anchor = np.linalg.inv(anchor_matrix)
            
            # Desloca a cinemática do "universo da Batch" para que a referência vire [0,0,0]
            for fname, fdata in flow_data["frames"].items():
                old_mat = np.array(fdata["matrix"], dtype=np.float64)
                new_mat = inv_anchor @ old_mat
                fdata["matrix"] = new_mat.tolist()
            
            # Atualiza o cabeçalho do JSON informando quem governa a centralização final
            flow_data["batch_anchor"] = chosen_anchor_name
            app_print(f"[{batch_dir.name}] Todas as matrizes foram re-centralizadas na Referência: {chosen_anchor_name}\n")
            
        except np.linalg.LinAlgError:
            app_print(f"[{batch_dir.name}] ERRO: Matriz da referência é singular. Mantendo origem no frame F0.\n")
    else:
        app_print(f"[{batch_dir.name}] AVISO: A Referência escolhida ({chosen_anchor_name}) não possui matriz. Mantendo origem em F0.\n")
        
    # ========================================================
    # 6. GARANTIA / DIAGNÓSTICO DO ÚLTIMO FRAME
    # ========================================================

    last_file = files[-1]

    if last_file.name not in flow_data["frames"]:
        app_print(f"\n[{batch_dir.name}] AVISO: último frame {last_file.name} NÃO entrou no Flow.\n")

    else:
        app_print(f"\n[{batch_dir.name}] Último frame garantido no Flow: {last_file.name}\n")

    # ========================================================
    # 7. SALVA
    # ========================================================

    with open(
        batch_dir / "flow_local.json", "w", encoding="utf-8") as f:
            json.dump(flow_data,f,indent=4)

    valid_frames = len(
        flow_data["frames"]
    )

    return {
        "batch_name": batch_dir.name,
        "anchor_path": anchor_file,
        "anchor_stars": anchor_stars,
        "star_count": len(anchor_stars),
        "fwhm": anchor_fwhm,
        "anchor_quality": anchor_quality,
        "anchor_data": anchor_data,
        "anchor_metrics": anchor["metrics"],
        "valid_frames": valid_frames,
        "total_frames": len(files),
        "coverage": (
            valid_frames / len(files)
            if files
            else 0.0
        )
    }

def process_all_flows(base_dir: Path, config: dict, app_print, app_progress, cancel_event):
    batch_folders = sorted([d for d in base_dir.iterdir() if d.is_dir() and "batch" in d.name.lower()])
    total_batches = len(batch_folders)
    
    if not batch_folders:
        app_print(f"Nenhuma subpasta de Batch encontrada em {base_dir}\n")
        return
    
    anchors_info = []
    app_progress(0, total_batches, "Iniciando AstroFlow (0%)...")
    
    matching_radius = config.get('matching_radius', 25.0)
    ransac_thresh = config.get('ransac', 4.0)
    global_master_cfg = config.get('global_master', 'Auto')
    
    # =========================================================================
    # 1. FLOW LOCAL (Executado em Paralelo para todas as Batches)
    # =========================================================================
    try:
        cpu_count = getattr(os, "process_cpu_count", os.cpu_count)() or 1
    except Exception:
        cpu_count = os.cpu_count() or 1
    worker_count = max(1, min(4, cpu_count))

    completed_count = 0
    app_progress(0, total_batches, "Iniciando AstroFlow em paralelo...")

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        future_to_batch = {
            executor.submit(process_local_flow, b_folder, config, lambda msg: None): b_folder 
            for b_folder in batch_folders
        }
        
        for future in as_completed(future_to_batch):
            if cancel_event.is_set():
                executor.shutdown(wait=False, cancel_futures=True)
                return
                
            b_folder = future_to_batch[future]
            completed_count += 1
            app_progress(completed_count, total_batches, f"Processado ({completed_count}/{total_batches})")
            app_print(f"Processado Flow Local: {b_folder.name}\n")
            
            try:
                info = future.result()
                if info:
                    anchors_info.append(info)
            except Exception as exc:
                app_print(f"Erro ao processar batch {b_folder.name}: {exc}\n")

    if not anchors_info:
        return
        
    # Ordena as informações de âncora pela ordem nominal das batches
    anchors_info.sort(key=lambda x: x['batch_name'])
        
    # =========================================================================
    # 2. DEFINIR MASTER GLOBAL
    # =========================================================================
    master_info = None
    global_master_cfg = config.get('global_master', 'Auto') if isinstance(config, dict) else 'Auto'
    
    if str(global_master_cfg).lower() == "auto":
        master_info = min(anchors_info, key=lambda x: (x['fwhm'] / max(1, x['star_count'])))
        app_print(f"\n[GLOBAL] Master Automático eleito: {master_info['batch_name']} (FWHM: {master_info['fwhm']:.2f})\n")
    else:
        master_info = next((a for a in anchors_info if a['batch_name'] == global_master_cfg), None)
        if not master_info:
            master_info = anchors_info[0]
            app_print(f"\n[GLOBAL] Master especificado não encontrado. Usando fallback: {master_info['batch_name']}\n")

    # =========================================================================
    # 3. FLOW GLOBAL INCREMENTAL EM CADEIA (Batch N -> Batch N-1)
    # =========================================================================
    global_flow = {"global_master_batch": master_info['batch_name'], "batches": {}}
    
    total_global = len(anchors_info)
    app_progress(0, total_global, "Calculando Cadeia Global das Batches...")
    
    # Inicia a matriz da primeira Batch como Identidade (Origem Temporária)
    temp_matrices = [np.eye(3, dtype=np.float64)]
    
    for i in range(1, total_global):
        if cancel_event.is_set(): return
        
        ref_info = anchors_info[i-1]
        tgt_info = anchors_info[i]
        app_progress(i, total_global, f"Alinhando {tgt_info['batch_name']} ao vizinho...")
        
        shift, _ = cv2.phaseCorrelate(ref_info['anchor_data'], tgt_info['anchor_data'])
        dx, dy = shift
        
        shifted_target = tgt_info['anchor_stars'] + np.array([dx, dy])
        tree = KDTree(ref_info['anchor_stars'])
        
        m_ref, m_tgt = [], []
        distances, indices = tree.query(shifted_target, distance_upper_bound=matching_radius)
        
        for idx, dist in enumerate(distances):
            if dist != float('inf'):
                m_ref.append(ref_info['anchor_stars'][indices[idx]])
                m_tgt.append(tgt_info['anchor_stars'][idx])
                
        if len(m_ref) >= 4:
            matrix_2x3, inliers = cv2.estimateAffinePartial2D(
                np.asarray(m_tgt, dtype=np.float32), 
                np.asarray(m_ref, dtype=np.float32), 
                method=cv2.RANSAC, 
                ransacReprojThreshold=ransac_thresh
            )
            
            if matrix_2x3 is not None and inliers is not None and inliers.sum() >= 3:
                step_matrix = make_homogeneous(matrix_2x3)
                # Acumula a matriz: M_i = M_i-1 @ M_passo
                accumulated = temp_matrices[i-1] @ step_matrix
                temp_matrices.append(accumulated)
            else:
                app_print(f"  -> Falha de RANSAC Global entre {tgt_info['batch_name']} e {ref_info['batch_name']}. Assumindo drift estático.\n")
                temp_matrices.append(temp_matrices[i-1])
        else:
            app_print(f"  -> Pareamento insuficiente entre {tgt_info['batch_name']} e {ref_info['batch_name']}. Assumindo drift estático.\n")
            temp_matrices.append(temp_matrices[i-1])

    # =========================================================================
    # 4. RE-CENTRALIZAÇÃO ABSOLUTA PELO MASTER
    # =========================================================================
    # Identifica o índice da Batch eleita como Master na lista sequencial
    master_idx = anchors_info.index(master_info)
    
    # Pega a matriz temporária do Master e calcula sua inversa
    master_matrix_inv = np.linalg.inv(temp_matrices[master_idx])
    
    for i, info in enumerate(anchors_info):
        # A matriz final de cada Batch é deslocada para que o Master se torne a Identidade pura
        final_matrix = master_matrix_inv @ temp_matrices[i]
        global_flow["batches"][info['batch_name']] = {"matrix": final_matrix.tolist()}
                
    with open(base_dir / "global_flow.json", "w") as f:
        json.dump(global_flow, f, indent=4)
        
    app_progress(total_global, total_global, "AstroFlow Finalizado.")
    app_print("\n>>> Processamento Cinemático (AstroFlow) Concluído com Cadeia Total! <<<\n")
    
    
def preview_star_detection(batch_dir: Path, config: dict) -> tuple[np.ndarray | None, int, float]:
    """
    Carrega a âncora de uma batch, detecta as estrelas e desenha os círculos 
    proporcionais ao FWHM detectado para pré-visualização em tempo real.
    """
    files = sorted([p for p in batch_dir.iterdir() if p.is_file() and p.suffix.lower() in {'.fit', '.fits'}])
    if not files:
        return None, 0, 0.0
        
    anchor_file = files[0]
    
    data, header = load_fits_data(anchor_file)
    working_data = extract_luminance(data, header)
    
    fwhm_val = float(config.get('fwhm', 4.0))
    sigma_val = float(config.get('sigma', 5.0))
    max_stars_val = int(config.get('max_stars', 250))
    
    engine_val = config.get("engine", "DAO")
    stars, measured_fwhm, metrics = detect_stars(working_data, fwhm_val, sigma_val, max_stars_val, engine_val)
    star_count = len(stars)
    
    # Normaliza a imagem para 8-bits para exibir no OpenCV/Matplotlib
    mean, median, std = sigma_clipped_stats(data, sigma=3.0)
    vmin, vmax = median, median + (8 * std)
    norm_data = np.clip((data - vmin) / (vmax - vmin), 0, 1) * 255
    img_8u = norm_data.astype(np.uint8)
    img_color = cv2.cvtColor(img_8u, cv2.COLOR_GRAY2BGR)
    
    # Desenha os círculos dimensionados proporcionalmente ao FWHM real da estrela
    radius = max(3, int(fwhm_val * 1.5))
    for x, y in stars:
        cv2.circle(img_color, (int(x), int(y)), radius, (0, 0, 255), 1)
        
    return img_color, star_count, measured_fwhm