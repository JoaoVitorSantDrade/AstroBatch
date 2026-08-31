import math
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

def detect_stars(data: np.ndarray, fwhm: float = 4.0, sigma: float = 5.0, max_stars: int = 150) -> np.ndarray:
    """Encontra os centroides das estrelas mais brilhantes usando DAOStarFinder."""
    # Calcula o fundo da imagem para ignorar ruído
    mean, median, std = sigma_clipped_stats(data, sigma=3.0)
    
    # Procura estrelas que se destaquem X sigmas acima do fundo
    daofind = DAOStarFinder(fwhm=fwhm, threshold=median + (sigma * std))
    sources = daofind(data - median)
    
    if sources is None or len(sources) < 3:
        return np.array([])
        
    # Ordena pelo fluxo (brilho) e pega as 'max_stars' mais brilhantes
    sources.sort('flux', reverse=True)
    sources = sources[:max_stars]
    
    # Retorna uma matriz Nx2 com [x, y]
    return np.transpose((sources['xcentroid'], sources['ycentroid']))

def estimate_global_shift(anchor_data: np.ndarray, target_data: np.ndarray) -> tuple[float, float]:
    """Usa Transformada de Fourier para achar o deslocamento X, Y bruto."""
    # cv2.phaseCorrelate exige float32 ou float64
    shift, response = cv2.phaseCorrelate(anchor_data, target_data)
    dx, dy = shift
    return dx, dy

def calculate_affine_matrix(anchor_pts: np.ndarray, target_pts: np.ndarray) -> list[list[float]]:
    """Calcula matriz de Translação + Rotação ignorando falsos positivos (RANSAC)."""
    # cv2.estimateAffinePartial2D descobre Translação, Rotação e Escala Uniforme
    matrix, inliers = cv2.estimateAffinePartial2D(
        target_pts, anchor_pts, method=cv2.RANSAC, ransacReprojThreshold=3.0
    )
    
    if matrix is None:
        raise ValueError("Falha ao calcular matriz afim (RANSAC falhou).")
        
    return matrix.tolist()

def process_batch_flow(batch_dir: Path, fwhm: float = 4.0, max_stars: int = 150) -> dict:
    """Gera o arquivo flow_data.json contendo a cinemática da Batch."""
    files = sorted([p for p in batch_dir.iterdir() if p.is_file() and p.suffix.lower() in {'.fit', '.fits'}])
    
    if not files:
        raise FileNotFoundError(f"Nenhum FITS encontrado em {batch_dir}")
        
    anchor_file = files[0]
    print(f"[{batch_dir.name}] Âncora definida: {anchor_file.name}")
    
    anchor_data = load_fits_data(anchor_file)
    anchor_stars = detect_stars(anchor_data, fwhm=fwhm, max_stars=max_stars)
    
    if len(anchor_stars) < 3:
        raise ValueError(f"Estrelas insuficientes na âncora {anchor_file.name}")
        
    # Árvore KD para busca ultra-rápida de vizinhos mais próximos
    anchor_tree = cKDTree(anchor_stars)
    
    flow_data = {
        "batch_anchor": anchor_file.name,
        "frames": {}
    }
    
    for i, filepath in enumerate(files):
        if filepath == anchor_file:
            # A matriz identidade significa que a âncora não se move em relação a ela mesma
            flow_data["frames"][filepath.name] = {"matrix": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]}
            continue
            
        print(f"[{batch_dir.name}] Calculando flow: {filepath.name} ({i+1}/{len(files)})")
        target_data = load_fits_data(filepath)
        
        # 1. Deslocamento Bruto (Fourier)
        dx, dy = estimate_global_shift(anchor_data, target_data)
        
        # 2. Detecção de Estrelas no alvo
        target_stars = detect_stars(target_data, fwhm=fwhm, max_stars=max_stars)
        if len(target_stars) < 3:
            print(f"  -> Aviso: Poucas estrelas em {filepath.name}. Posição pode ser imprecisa.")
            continue
            
        # 3. Pareamento de Estrelas (Matching)
        # Deslocamos os pontos alvo temporariamente pelo dx, dy bruto para que se alinhem com a âncora
        shifted_target_stars = target_stars + np.array([dx, dy])
        
        matched_anchor = []
        matched_target = []
        
        for idx, target_pt in enumerate(shifted_target_stars):
            # Procura a estrela âncora mais próxima dentro de um raio de 10 pixels
            dist, anchor_idx = anchor_tree.query(target_pt, distance_upper_bound=10.0)
            if dist != float('inf'):
                matched_anchor.append(anchor_stars[anchor_idx])
                matched_target.append(target_stars[idx]) # Salva o alvo original (sem o shift)
                
        matched_anchor = np.array(matched_anchor)
        matched_target = np.array(matched_target)
        
        if len(matched_anchor) < 3:
            print(f"  -> Aviso: Falha ao parear estrelas suficientes em {filepath.name}")
            continue
            
        # 4. Cálculo da Matriz Afim Final (Sub-pixel, Rotação e Translação)
        try:
            matrix = calculate_affine_matrix(matched_anchor, matched_target)
            flow_data["frames"][filepath.name] = {"matrix": matrix}
        except ValueError as e:
            print(f"  -> {e}")
            
    # Salva o resultado
    output_json = batch_dir / "flow_data.json"
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(flow_data, f, indent=4)
        
    print(f"[{batch_dir.name}] Flow calculado com sucesso e salvo em {output_json.name}\n")
    return flow_data

def process_all_batches_flow(base_dir: Path, fwhm: float, max_stars: int, app_print, cancel_event):
    """
    Varre a pasta principal procurando por subpastas (ex: batch_001).
    Roda o process_batch_flow em cada uma.
    """
    batch_folders = sorted([d for d in base_dir.iterdir() if d.is_dir() and "batch" in d.name.lower()])
    
    if not batch_folders:
        app_print(f"Nenhuma subpasta de Batch encontrada em {base_dir}\n")
        return
        
    for i, b_folder in enumerate(batch_folders, start=1):
        if cancel_event.is_set():
            app_print("\nCancelamento solicitado pelo usuário.\n")
            break
            
        app_print(f"=== Processando Batch {i}/{len(batch_folders)}: {b_folder.name} ===\n")
        
        try:
            # Aqui você deve alterar os 'prints' do process_batch_flow original 
            # para utilizar o 'app_print' fornecido pela interface.
            process_batch_flow(b_folder, fwhm=fwhm, max_stars=max_stars) 
        except Exception as e:
            app_print(f"Erro na {b_folder.name}: {e}\n")

# Bloco para testar o script isoladamente
if __name__ == "__main__":
    pasta_batch = Path(input("Caminho da pasta da Batch: ").strip())
    process_batch_flow(pasta_batch)