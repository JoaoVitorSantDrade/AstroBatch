import json
import cv2
import numpy as np
from pathlib import Path
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from photutils.detection import DAOStarFinder
from scipy.spatial import cKDTree

def load_fits_data(filepath: Path) -> np.ndarray:
    """Lê a imagem FITS e retorna a matriz 2D de pixels."""
    with fits.open(filepath, memmap=False) as hdul:
        for hdu in hdul:
            if hdu.is_image and hdu.data is not None and hdu.data.ndim == 2:
                return np.asarray(hdu.data, dtype=np.float32)
    raise ValueError(f"Imagem 2D não encontrada em {filepath.name}")

def detect_stars(data: np.ndarray, fwhm: float, sigma: float, max_stars: int) -> tuple[np.ndarray, float]:
    """Encontra os centroides das estrelas mais brilhantes usando DAOStarFinder."""
    mean, median, std = sigma_clipped_stats(data, sigma=3.0)
    daofind = DAOStarFinder(fwhm=fwhm, threshold=median + (sigma * std))
    sources = daofind(data - median)
    
    if sources is None or len(sources) < 3:
        return np.array([]), 0.0
        
    sources.sort('flux', reverse=True)
    sources = sources[:max_stars]
    pts = np.transpose((sources['xcentroid'], sources['ycentroid']))
    return pts, float(np.mean(sources['fwhm']))

def generate_debug_image(data: np.ndarray, stars: np.ndarray, output_path: Path):
    """Gera um JPG da imagem com círculos nas estrelas detectadas para calibração."""
    mean, median, std = sigma_clipped_stats(data, sigma=3.0)
    vmin, vmax = median, median + (10 * std)
    
    norm_data = np.clip((data - vmin) / (vmax - vmin), 0, 1) * 255
    img_8u = norm_data.astype(np.uint8)
    img_color = cv2.cvtColor(img_8u, cv2.COLOR_GRAY2BGR)
    
    for x, y in stars:
        cv2.circle(img_color, (int(x), int(y)), 10, (0, 0, 255), 1)
        
    cv2.imwrite(str(output_path), img_color)

def make_homogeneous(matrix_2x3: np.ndarray) -> np.ndarray:
    """Converte matriz 2x3 para 3x3 para permitir multiplicação."""
    hom = np.eye(3, dtype=np.float64)
    hom[:2, :] = matrix_2x3
    return hom

def process_local_flow(batch_dir: Path, config: dict, app_print) -> dict:
    """Calcula a cinemática intra-batch e retorna metadados da âncora."""
    files = sorted([p for p in batch_dir.iterdir() if p.is_file() and p.suffix.lower() in {'.fit', '.fits'}])
    if not files:
        return {}
        
    anchor_file = files[0]
    app_print(f"[{batch_dir.name}] Âncora definida: {anchor_file.name}\n")
    
    anchor_data = load_fits_data(anchor_file)
    anchor_stars, avg_fwhm = detect_stars(anchor_data, config['fwhm'], config['sigma'], config['max_stars'])
    
    if len(anchor_stars) < 3:
        app_print(f"[{batch_dir.name}] ERRO: Estrelas insuficientes na âncora.\n")
        return {}
        
    if config['debug_images']:
        generate_debug_image(anchor_data, anchor_stars, batch_dir / f"debug_{anchor_file.stem}.jpg")

    anchor_tree = cKDTree(anchor_stars)
    flow_data = {"batch_anchor": anchor_file.name, "frames": {}}
    
    for i, filepath in enumerate(files):
        if filepath == anchor_file:
            flow_data["frames"][filepath.name] = {"matrix": np.eye(3).tolist()}
            continue
            
        target_data = load_fits_data(filepath)
        shift, _ = cv2.phaseCorrelate(anchor_data, target_data)
        dx, dy = shift
        
        target_stars, _ = detect_stars(target_data, config['fwhm'], config['sigma'], config['max_stars'])
        if len(target_stars) < 3:
            continue
            
        shifted_target = target_stars + np.array([dx, dy])
        m_anchor, m_target = [], []
        
        for idx, pt in enumerate(shifted_target):
            dist, a_idx = anchor_tree.query(pt, distance_upper_bound=config['matching_radius'])
            if dist != float('inf'):
                m_anchor.append(anchor_stars[a_idx])
                m_target.append(target_stars[idx])
                
        if len(m_anchor) < 3:
            continue
            
        matrix_2x3, _ = cv2.estimateAffinePartial2D(
            np.array(m_target), np.array(m_anchor), 
            method=cv2.RANSAC, ransacReprojThreshold=config['ransac']
        )
        
        if matrix_2x3 is not None:
            flow_data["frames"][filepath.name] = {"matrix": make_homogeneous(matrix_2x3).tolist()}
            
    with open(batch_dir / "flow_local.json", "w") as f:
        json.dump(flow_data, f, indent=4)
        
    return {
        "batch_name": batch_dir.name,
        "anchor_path": anchor_file,
        "anchor_stars": anchor_stars,
        "star_count": len(anchor_stars),
        "fwhm": avg_fwhm,
        "anchor_data": anchor_data
    }

def process_all_flows(base_dir: Path, config: dict, app_print, app_progress, cancel_event):
    """Orquestra o Flow Local de cada Batch e depois calcula o Flow Global."""
    batch_folders = sorted([d for d in base_dir.iterdir() if d.is_dir() and "batch" in d.name.lower()])
    total_batches = len(batch_folders)
    
    if not batch_folders:
        app_print(f"Nenhuma subpasta de Batch encontrada em {base_dir}\n")
        return
        
    anchors_info = []
    app_progress(0, total_batches, "Iniciando AstroFlow (0%)...")
    
    # 1. FLOW LOCAL
    for b_folder in batch_folders:
        if cancel_event.is_set(): return
        app_progress(i, total_batches, f"Calculando Flow Local ({i+1}/{total_batches}): {b_folder.name}...")
        
        info = process_local_flow(b_folder, config, app_print)
        if info:
            anchors_info.append(info)
            
    if not anchors_info:
        return
        
    # 2. DEFINIR MASTER GLOBAL
    master_info = None
    if config['global_master'].lower() == "auto":
        # Elege a âncora com mais estrelas pontuais
        master_info = min(anchors_info, key=lambda x: (x['fwhm'] / max(1, x['star_count'])))
        app_print(f"\n[GLOBAL] Master Automático eleito: {master_info['batch_name']} (FWHM: {master_info['fwhm']:.2f})\n")
    else:
        master_info = next((a for a in anchors_info if a['batch_name'] == config['global_master']), None)
        if not master_info:
            master_info = anchors_info[0]
            app_print(f"\n[GLOBAL] Master especificado não encontrado. Usando: {master_info['batch_name']}\n")

    # 3. FLOW GLOBAL
    global_flow = {"global_master_batch": master_info['batch_name'], "batches": {}}
    master_tree = cKDTree(master_info['anchor_stars'])
    
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
            dist, m_idx = master_tree.query(pt, distance_upper_bound=config['matching_radius'])
            if dist != float('inf'):
                m_master.append(master_info['anchor_stars'][m_idx])
                m_target.append(target_info['anchor_stars'][idx])
                
        if len(m_master) >= 3:
            matrix_2x3, _ = cv2.estimateAffinePartial2D(
                np.array(m_target), np.array(m_master), 
                method=cv2.RANSAC, ransacReprojThreshold=config['ransac']
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