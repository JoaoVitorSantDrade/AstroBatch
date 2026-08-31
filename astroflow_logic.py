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

def load_fits_data(filepath: Path) -> np.ndarray:
    with fits.open(filepath, memmap=False) as hdul:
        for hdu in hdul:
            if hdu.is_image and hdu.data is not None and hdu.data.ndim == 2:
                return np.asarray(hdu.data, dtype=np.float32)
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

def detect_stars(
    data: np.ndarray,
    fwhm: float,
    sigma: float,
    max_stars: int
) -> tuple[np.ndarray, float, dict]:

    mean, median, std = sigma_clipped_stats(
        data,
        sigma=3.0
    )

    if not np.isfinite(std) or std <= 0:
        return (
            np.empty((0, 2), dtype=np.float32),
            0.0,
            {
                "star_count": 0,
                "mean": float(mean) if np.isfinite(mean) else None,
                "median": float(median) if np.isfinite(median) else None,
                "std": 0.0,
                "fwhm": 0.0
            }
        )

    daofind = DAOStarFinder(
        fwhm=fwhm,
        threshold=sigma * std
    )

    sources = daofind(data - median)

    if sources is None or len(sources) < 3:
        return (
            np.empty((0, 2), dtype=np.float32),
            float(fwhm),
            {
                "star_count": 0,
                "mean": float(mean),
                "median": float(median),
                "std": float(std),
                "fwhm": float(fwhm)
            }
        )

    # Ordena pelas estrelas mais brilhantes
    sources.sort("flux", reverse=True)
    sources = sources[:max_stars]

    pts = np.column_stack((
        np.asarray(
            sources["x_centroid"],
            dtype=np.float32
        ),
        np.asarray(
            sources["y_centroid"],
            dtype=np.float32
        )
    ))

    # DAOStarFinder não fornece FWHM medido.
    # O FWHM configurado no detector é usado
    # como estimativa operacional.
    measured_fwhm = float(fwhm)

    metrics = {
        "star_count": int(len(pts)),
        "mean": float(mean),
        "median": float(median),
        "std": float(std),
        "fwhm": measured_fwhm
    }

    return pts, measured_fwhm, metrics

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
    min_stars: int
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
        data = load_fits_data(filepath)

        stars, measured_fwhm, metrics = detect_stars(data, fwhm_val, sigma_val, max_stars_val)

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

    files = sorted([
        p
        for p in batch_dir.iterdir()
        if (
            p.is_file()
            and p.suffix.lower() in {
                ".fit",
                ".fits"
            }
        )
    ])

    if not files:
        return {}

    anchor_file = files[0]

    app_print(
        f"[{batch_dir.name}] "
        f"Âncora definida: "
        f"{anchor_file.name}\n"
    )

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
        cpu_count = (
            getattr(
                os,
                "process_cpu_count",
                os.cpu_count
            )()
            or 1
        )
    except Exception:
        cpu_count = os.cpu_count() or 1

    worker_count = max(
        1,
        min(16, cpu_count)
    )

    app_print(
        f"[{batch_dir.name}] "
        f"Pré-processamento paralelo: "
        f"{worker_count} workers\n"
    )

    prepared_frames = {}

    with ThreadPoolExecutor(
        max_workers=worker_count,
        thread_name_prefix="astroflow"
    ) as executor:

        futures = {
            executor.submit(
                _process_single_frame,
                filepath,
                fwhm_val,
                sigma_val,
                max_stars_val,
                min_stars
            ): filepath
            for filepath in files
        }

        for future in as_completed(futures):

            filepath = futures[future]

            try:
                fname, result = future.result()

            except Exception as exc:

                app_print(
                    f"  [{filepath.name}] "
                    f"Erro no worker: "
                    f"{exc}\n"
                )

                continue

            if result is None:

                app_print(
                    f"  [{fname}] "
                    f"Estrelas insuficientes.\n"
                )

                continue

            prepared_frames[fname] = result

    # ========================================================
    # 2. VALIDAR ÂNCORA
    # ========================================================

    anchor = prepared_frames.get(
        anchor_file.name
    )

    if anchor is None:

        app_print(
            f"[{batch_dir.name}] "
            f"ERRO: Não foi possível "
            f"processar a âncora.\n"
        )

        return {}

    anchor_stars = anchor["stars"]
    anchor_data = anchor["data"]

    anchor_fwhm = anchor["fwhm"]

    if len(anchor_stars) < min_stars:

        app_print(
            f"[{batch_dir.name}] "
            f"ERRO: Estrelas insuficientes "
            f"na âncora "
            f"({len(anchor_stars)}).\n"
        )

        return {}

    anchor_quality = calculate_anchor_quality(
        len(anchor_stars),
        anchor_fwhm
    )

    app_print(
        f"[{batch_dir.name}] "
        f"Âncora: {len(anchor_stars)} estrelas | "
        f"FWHM={anchor_fwhm:.2f}px | "
        f"qualidade={anchor_quality:.4f}\n"
    )

    if debug_flag:

        generate_debug_image(
            anchor_data,
            anchor_stars,
            batch_dir
            / f"debug_{anchor_file.stem}.jpg"
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

            app_print(
                f"  [{current_name}] "
                f"REJEITADO na detecção. "
                f"Cadeia interrompida neste ponto.\n"
            )

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

            app_print(
                f"  [{current_name}] "
                f"REJEITADO: apenas "
                f"{len(m_previous)} matches "
                f"com {previous_name}.\n"
            )

            # Mantemos previous_frame como está.
            #
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

        cumulative_matrix = (
            cumulative_matrix
            @ relative_matrix
        )

        # ----------------------------------------------------
        # RMS acumulado aproximado
        # ----------------------------------------------------

        previous_cumulative_rms = (
            flow_data["frames"][
                previous_name
            ].get(
                "cumulative_rms",
                0.0
            )
        )
        
        current_rms = metrics[
            "rms"
        ] or 0.0

        cumulative_rms = float(
            np.sqrt(
                previous_cumulative_rms ** 2
                + current_rms ** 2
            )
        )

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
        app_print(
            f"  [{current_name}] "
            f"OK ← {previous_name} | "
            f"{metrics['inliers']}/"
            f"{metrics['matches']} inliers "
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
    # 5. GARANTIA / DIAGNÓSTICO DO ÚLTIMO FRAME
    # ========================================================

    last_file = files[-1]

    if last_file.name not in flow_data["frames"]:

        app_print(
            f"\n[{batch_dir.name}] "
            f"AVISO: último frame "
            f"{last_file.name} "
            f"NÃO entrou no Flow.\n"
        )

    else:

        app_print(
            f"\n[{batch_dir.name}] "
            f"Último frame garantido no Flow: "
            f"{last_file.name}\n"
        )

    # ========================================================
    # 6. SALVA
    # ========================================================

    with open(
        batch_dir / "flow_local.json",
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            flow_data,
            f,
            indent=4
        )

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
    
    matching_radius = config.get('matching_radius', 15)
    ransac_thresh = config.get('ransac', 3.0)
    global_master_cfg = config.get('global_master', 'Auto')
    
    # 1. FLOW LOCAL (Executado em Paralelo para todas as Batches)
    
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
        
    # 2. DEFINIR MASTER GLOBAL
    master_info = None
    global_master_cfg = config.get('global_master', 'Auto') if isinstance(config, dict) else 'Auto'
    
    if str(global_master_cfg).lower() == "auto":
        master_info = min(anchors_info, key=lambda x: (x['fwhm'] / max(1, x['star_count'])))
        app_print(f"\n[GLOBAL] Master Automático eleito: {master_info['batch_name']} (FWHM: {master_info['fwhm']:.2f})\n")
    else:
        master_info = next((a for a in anchors_info if a['batch_name'] == global_master_cfg), None)
        if not master_info:
            master_info = anchors_info[0]
            app_print(f"\n[GLOBAL] Master especificado não encontrado. Usando: {master_info['batch_name']}\n")

    # 3. FLOW GLOBAL
    global_flow = {"global_master_batch": master_info['batch_name'], "batches": {}}
    master_tree = KDTree(master_info['anchor_stars'])
    
    total_global = len(anchors_info)
    app_progress(0, total_global, "Iniciando alinhamento global das Batches...")
    
    for i, target_info in enumerate(anchors_info):
        if cancel_event.is_set(): 
            return
            
        app_progress(i, total_global, f"Alinhando {target_info['batch_name']} ao Master Global ({i+1}/{total_global})...")
        
        if target_info['batch_name'] == master_info['batch_name']:
            global_flow["batches"][target_info['batch_name']] = {"matrix": np.eye(3).tolist()}
            continue
            
        app_print(f"Alinhando {target_info['batch_name']} ao Master Global...\n")
        shift, _ = cv2.phaseCorrelate(master_info['anchor_data'], target_info['anchor_data'])
        dx, dy = shift
        
        shifted_target = target_info['anchor_stars'] + np.array([dx, dy])
        m_master, m_target = [], []
        
        for idx, pt in enumerate(shifted_target):
            dist, m_idx = master_tree.query(pt, distance_upper_bound=matching_radius)
            if dist != float('inf'):
                m_master.append(master_info['anchor_stars'][m_idx])
                m_target.append(target_info['anchor_stars'][idx])
                
        if len(m_master) >= 3:
            matrix_2x3, _ = cv2.estimateAffinePartial2D(
                np.array(m_target), np.array(m_master), 
                method=cv2.RANSAC, ransacReprojThreshold=ransac_thresh
            )
            if matrix_2x3 is not None:
                global_flow["batches"][target_info['batch_name']] = {"matrix": make_homogeneous(matrix_2x3).tolist()}
            else:
                app_print(f"  -> Falha de RANSAC no Flow Global para {target_info['batch_name']}\n")
        else:
            app_print(f"  -> Aviso: Pareamento insuficiente (<3 estrelas) entre {target_info['batch_name']} e o Master Global.\n")
                
    with open(base_dir / "global_flow.json", "w") as f:
        json.dump(global_flow, f, indent=4)
        
    app_progress(total_global, total_global, "AstroFlow Finalizado.")
    app_print("\n>>> Processamento Cinemático (AstroFlow) Concluído! <<<\n")