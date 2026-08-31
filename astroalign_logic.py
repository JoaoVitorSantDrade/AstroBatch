import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from astropy.io import fits

FITS_SUFFIXES = {".fit", ".fits", ".fts"}

INTERPOLATION_MODES = {
    "Nearest": cv2.INTER_NEAREST,
    "Bilinear": cv2.INTER_LINEAR,
    "Bicubic": cv2.INTER_CUBIC,
    "Lanczos": cv2.INTER_LANCZOS4,
}

@dataclass(frozen=True)
class AlignConfig:
    base_dir: Path
    output_dir: Path
    interpolation: str
    overwrite: bool
    dry_run: bool
    keep_header: bool

def get_optimal_worker_count() -> int:
    try:
        cpu_count = getattr(os, "process_cpu_count", os.cpu_count)() or 1
    except Exception:
        cpu_count = os.cpu_count() or 1
        
    try:
        import psutil
        available_ram = psutil.virtual_memory().available
        ram_workers = max(1, available_ram // (400 * 1024 * 1024))
    except ImportError:
        ram_workers = cpu_count # Fallback caso a biblioteca psutil não esteja instalada
        
    return max(1, min(16, cpu_count, ram_workers))

def find_batch_folders(base_dir: Path) -> list[Path]:
    return sorted(d for d in base_dir.iterdir() if d.is_dir() and "batch" in d.name.lower())

def load_json(filepath: Path) -> dict:
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

def load_local_flow(batch_dir: Path) -> dict | None:
    flow_path = batch_dir / "flow_local.json"
    if not flow_path.exists(): return None
    return load_json(flow_path)

def load_global_flow(base_dir: Path) -> dict | None:
    flow_path = base_dir / "global_flow.json"
    if not flow_path.exists(): return None
    return load_json(flow_path)

def compute_final_matrix(local_matrix: list, global_matrix: list) -> np.ndarray:
    local = np.asarray(local_matrix, dtype=np.float64)
    offset = np.asarray(global_matrix, dtype=np.float64)
    return offset @ local

def load_fits_data(filepath: Path) -> tuple[np.ndarray, fits.Header]:
    with fits.open(filepath, memmap=False) as hdul:
        for hdu in hdul:
            if hdu.is_image and hdu.data is not None and hdu.data.ndim == 2:
                header = hdu.header.copy(strip=False)
                data = np.asarray(hdu.data, dtype=np.float32)
                return data, header
    raise ValueError(f"Imagem 2D não encontrada em {filepath.name}")

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
    # Fallback
    return data[0::2, 0::2], data[0::2, 1::2], data[1::2, 0::2], data[1::2, 1::2]

def merge_cfa(r: np.ndarray, g1: np.ndarray, g2: np.ndarray, b: np.ndarray, pattern: str, shape: tuple):
    out = np.zeros(shape, dtype=np.float32)
    if pattern == 'RGGB':
        out[0::2, 0::2], out[0::2, 1::2] = r, g1
        out[1::2, 0::2], out[1::2, 1::2] = g2, b
    elif pattern == 'BGGR':
        out[1::2, 1::2], out[0::2, 1::2] = r, g1
        out[1::2, 0::2], out[0::2, 0::2] = g2, b
    elif pattern == 'GRBG':
        out[0::2, 1::2], out[0::2, 0::2] = r, g1
        out[1::2, 1::2], out[1::2, 0::2] = g2, b
    elif pattern == 'GBRG':
        out[1::2, 0::2], out[1::2, 1::2] = r, g1
        out[0::2, 0::2], out[0::2, 1::2] = g2, b
    return out

def warp_cfa(data: np.ndarray, matrix_3x3: np.ndarray, pattern: str, interpolation_flag: int) -> np.ndarray:
    r, g1, g2, b = split_cfa(data, pattern)
    
    # Reduz o deslocamento pela metade porque a matriz do canal físico possui metade da resolução
    m_sub = matrix_3x3.copy()
    m_sub[0, 2] /= 2.0
    m_sub[1, 2] /= 2.0
    m_2x3 = m_sub[:2, :].astype(np.float64)
    
    sub_h, sub_w = r.shape
    kwargs = {"dsize": (sub_w, sub_h), "flags": interpolation_flag, "borderMode": cv2.BORDER_CONSTANT, "borderValue": 0.0}
    
    r_w  = cv2.warpAffine(r, m_2x3, **kwargs)
    g1_w = cv2.warpAffine(g1, m_2x3, **kwargs)
    g2_w = cv2.warpAffine(g2, m_2x3, **kwargs)
    b_w  = cv2.warpAffine(b, m_2x3, **kwargs)
    
    return merge_cfa(r_w, g1_w, g2_w, b_w, pattern, data.shape)

def warp_frame(data: np.ndarray, final_matrix: np.ndarray, interpolation_flag: int, pattern: str | None) -> np.ndarray:
    if pattern is not None:
        # 5. Artefatos de Ringing: Força interpolação Linear estritamente para matrizes CFA separadas
        return warp_cfa(data, final_matrix, pattern, cv2.INTER_LINEAR)
    
    # Tratamento direto para Monocromático ou RGB pré-debayerizado (respeita a escolha do usuário)
    h, w = data.shape[:2]
    matrix_2x3 = final_matrix[:2, :].astype(np.float64)
    return cv2.warpAffine(data, matrix_2x3, (w, h), flags=interpolation_flag, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)

def generate_valid_mask(shape: tuple, final_matrix: np.ndarray) -> np.ndarray:
    mask = np.ones(shape, dtype=np.uint8)
    matrix_2x3 = final_matrix[:2, :].astype(np.float64)
    return cv2.warpAffine(mask, matrix_2x3, (shape[1], shape[0]), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

def save_fits(data: np.ndarray, mask: np.ndarray, header: fits.Header | None, output_path: Path):
    if header is not None:
        header.remove('BZERO', ignore_missing=True)
        header.remove('BSCALE', ignore_missing=True)
        header['BITPIX'] = 16 # 1. Otimização de armazenamento: Restaura tag 16-bits

    # 1. Otimização de Armazenamento: Retorna o array para uint16 cortando o tamanho do arquivo pela metade
    data_uint16 = np.clip(data, 0, 65535).astype(np.uint16)

    hdu_data = fits.PrimaryHDU(data=data_uint16, header=header)
    hdu_mask = fits.ImageHDU(data=mask, name='VALID_MASK')
    
    hdul = fits.HDUList([hdu_data, hdu_mask])
    hdul.writeto(str(output_path), overwrite=True, output_verify='ignore')

def _process_single_alignment(
    frame_name: str,
    frame_info: dict,
    batch_dir: Path,
    output_dir: Path,
    global_matrix: list,
    interpolation_flag: int,
    config: AlignConfig,
) -> tuple[str, str | None]:
    try:
        filepath = batch_dir / frame_name
        if not filepath.exists():
            return frame_name, f"Aviso: arquivo original não encontrado: {filepath}"

        data, header = load_fits_data(filepath)
        pattern = get_bayer_pattern(header)

        final_matrix = compute_final_matrix(frame_info["matrix"], global_matrix)
        
        warped = warp_frame(data, final_matrix, interpolation_flag, pattern)
        mask = generate_valid_mask(data.shape[:2], final_matrix)

        output_path = output_dir / frame_name
        if output_path.exists() and not config.overwrite:
            return frame_name, f"ERRO: destino já existe, arquivo ignorado: {output_path}"

        if not config.dry_run:
            output_dir.mkdir(parents=True, exist_ok=True)
            save_fits(warped, mask, header if config.keep_header else None, output_path)

        return frame_name, None
    except Exception as exc:
        return frame_name, f"Erro ao alinhar {frame_name}: {exc}"

def process_batch_alignment(
    batch_dir: Path,
    global_flow: dict,
    config: AlignConfig,
    app_print,
    app_progress,
    cancel_event: threading.Event,
    progress_state: dict,
) -> tuple[int, int]:
    local_flow = load_local_flow(batch_dir)
    if local_flow is None: return 0, 0

    batch_entry = global_flow["batches"].get(batch_dir.name)
    if batch_entry is None: return 0, 0

    global_matrix = batch_entry["matrix"]
    frames = local_flow.get("frames", {})
    total_frames = len(frames)

    if total_frames == 0: return 0, 0

    output_dir = config.output_dir / batch_dir.name
    interpolation_flag = INTERPOLATION_MODES.get(config.interpolation, cv2.INTER_LANCZOS4)

    worker_count = get_optimal_worker_count()
    processed, failed = 0, 0

    with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="astroalign") as executor:
        futures = {
            executor.submit(_process_single_alignment, fname, finfo, batch_dir, output_dir, global_matrix, interpolation_flag, config): fname
            for fname, finfo in frames.items()
        }

        for future in as_completed(futures):
            if cancel_event.is_set():
                executor.shutdown(wait=False, cancel_futures=True)
                break

            frame_name, error = future.result()
            progress_state["done"] += 1

            if error:
                app_print(f"  [{frame_name}] {error}\n")
                failed += 1
            else:
                processed += 1

            if progress_state["done"] % 10 == 0 or progress_state["done"] == progress_state["total"]:
                app_progress(progress_state["done"], progress_state["total"], f"Alinhando frames ({progress_state['done']}/{progress_state['total']})...")

    app_print(f"[{batch_dir.name}] Concluído: {processed} alinhados, {failed} falhas.\n")
    return processed, failed

def process_all_alignments(
    base_dir: Path, output_dir: Path, config_dict: dict, app_print, app_progress, cancel_event: threading.Event
) -> tuple[int, int]:
    if not isinstance(config_dict, dict): config_dict = {}

    align_config = AlignConfig(
        base_dir=base_dir,
        output_dir=output_dir,
        interpolation=config_dict.get("interpolation", "Lanczos"),
        overwrite=bool(config_dict.get("overwrite", False)),
        dry_run=bool(config_dict.get("dry_run", False)),
        keep_header=bool(config_dict.get("keep_header", True)),
    )

    global_flow = load_global_flow(base_dir)
    if global_flow is None:
        app_print("ERRO: global_flow.json não encontrado.\n")
        return 0, 0

    batch_folders = find_batch_folders(base_dir)
    if not batch_folders: return 0, 0

    total_frames = 0
    batches_with_flow = []
    for b_folder in batch_folders:
        local_flow = load_local_flow(b_folder)
        if local_flow:
            batches_with_flow.append(b_folder)
            total_frames += len(local_flow.get("frames", {}))

    if total_frames == 0: return 0, 0

    progress_state = {"done": 0, "total": total_frames}
    if not align_config.dry_run: align_config.output_dir.mkdir(parents=True, exist_ok=True)

    total_processed, total_failed = 0, 0

    for b_folder in batches_with_flow:
        if cancel_event.is_set(): break
        app_print(f"\nAlinhando Batch: {b_folder.name}\n")
        processed, failed = process_batch_alignment(b_folder, global_flow, align_config, app_print, app_progress, cancel_event, progress_state)
        total_processed += processed
        total_failed += failed

    app_progress(total_frames, total_frames, "Concluído.")
    app_print(f"\n>>> AstroAlign Finalizado! {total_processed} frames alinhados, {total_failed} falhas. <<<\n")
    return total_processed, total_failed