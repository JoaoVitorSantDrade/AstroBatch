import os
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from astropy.io import fits

def load_raw_fits(filepath: Path) -> tuple[np.ndarray, fits.Header]:
    with fits.open(filepath, memmap=False) as hdul:
        for hdu in hdul:
            if hdu.is_image and hdu.data is not None and hdu.data.ndim == 2:
                return np.asarray(hdu.data, dtype=np.float32), hdu.header.copy()
    raise ValueError(f"Imagem 2D não encontrada em {filepath.name}")

def create_master_frame(folder: Path, output_path: Path, frame_type: str, app_print) -> np.ndarray | None:
    files = sorted([p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in {'.fit', '.fits'}])
    if not files:
        app_print(f"Nenhum arquivo encontrado para gerar Master {frame_type}.\n")
        return None
        
    app_print(f"Gerando Master {frame_type} a partir de {len(files)} arquivos...\n")
    data_stack = []
    header = None
    
    for f in files:
        d, h = load_raw_fits(f)
        data_stack.append(d)
        if header is None: header = h

    # Mediana robusta para remover raios cósmicos e hot pixels
    master = np.median(data_stack, axis=0).astype(np.float32)
    
    if frame_type.upper() == "FLAT":
        mean_flat = np.mean(master)
        if mean_flat > 0:
            master = master / mean_flat # Normaliza o Master Flat

    if header is not None:
        header['FRAMETYP'] = f'Master {frame_type}'
    
    fits.writeto(output_path, master, header, overwrite=True, output_verify='ignore')
    app_print(f"Master {frame_type} salvo em {output_path.name}\n")
    return master

def run_calibration_pipeline(config: dict, app_print, app_progress, cancel_event: threading.Event):
    input_dir = Path(config['input_dir'])
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    master_dark, master_flat = None, None
    
    # 1. Carrega ou Gera Master Dark
    if config['apply_dark']:
        dark_path = Path(config['dark_path'])
        if dark_path.is_file():
            master_dark, _ = load_raw_fits(dark_path)
            app_print(f"Master Dark carregado: {dark_path.name}\n")
        elif dark_path.is_dir():
            out_md = output_dir / "master_dark.fit"
            master_dark = create_master_frame(dark_path, out_md, "DARK", app_print)

    # 2. Carrega ou Gera Master Flat
    if config['apply_flat']:
        flat_path = Path(config['flat_path'])
        if flat_path.is_file():
            master_flat, _ = load_raw_fits(flat_path)
            app_print(f"Master Flat carregado: {flat_path.name}\n")
        elif flat_path.is_dir():
            out_mf = output_dir / "master_flat.fit"
            master_flat = create_master_frame(flat_path, out_mf, "FLAT", app_print)

    # 3. Calibra os Lights
    light_files = sorted([p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in {'.fit', '.fits'}])
    total = len(light_files)
    
    def process_calib(filepath: Path):
        data, header = load_raw_fits(filepath)
        
        # I_calibrated = (I_raw - Dark) / Flat
        if master_dark is not None:
            data = data - master_dark
        
        if master_flat is not None:
            data = data / np.maximum(master_flat, 1e-5) # Evita divisão por zero
            
        data = np.clip(data, 0, 65535).astype(np.float32)
        header['CALIBRAT'] = 'T'
        
        out_path = output_dir / filepath.name
        fits.writeto(out_path, data, header, overwrite=True, output_verify='ignore')

    app_progress(0, total, "Calibrando frames...")
    processed = 0
    with ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as executor:
        futures = [executor.submit(process_calib, f) for f in light_files]
        for future in futures:
            if cancel_event.is_set(): break
            future.result()
            processed += 1
            app_progress(processed, total, f"Calibrado ({processed}/{total})")
            
    app_print(f"\n>>> Calibração concluída: {processed} arquivos salvos em {output_dir.name} <<<\n")