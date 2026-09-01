import os
import threading
import cv2
import numpy as np
from pathlib import Path
from astropy.io import fits
from concurrent.futures import ThreadPoolExecutor

BAYER_CV2_MAPPING = {
    'RGGB': cv2.COLOR_BayerBG2RGB, 
    'BGGR': cv2.COLOR_BayerRG2RGB,
    'GRBG': cv2.COLOR_BayerGB2RGB,
    'GBRG': cv2.COLOR_BayerGR2RGB
}

def get_bayer_pattern(header: fits.Header) -> str | None:
    for key in ['BAYERPAT', 'BAYERPATTERN', 'COLORTYP']:
        if key in header:
            val = str(header[key]).strip().upper().strip("'")
            if val in BAYER_CV2_MAPPING: 
                return val
    return None

def process_debayer(config: dict, app_print, app_progress, cancel_event: threading.Event):
    input_dir = Path(config['input_dir'])
    output_dir = Path(config['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    files = sorted([p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in {'.fit', '.fits'}])
    total = len(files)
    if total == 0:
        app_print("Nenhum arquivo encontrado para Debayer.\n")
        return
        
    user_pattern = config.get('pattern', 'Auto')
    
    # Identifica o padrão usando o primeiro arquivo se estiver em "Auto"
    if user_pattern == 'Auto':
        with fits.open(files[0], memmap=False) as hdul:
            primeiro_header = next(hdu.header for hdu in hdul if hdu.is_image)
        detected_pattern = get_bayer_pattern(primeiro_header)
        if not detected_pattern:
            app_print("ERRO: Padrão Bayer não detectado no cabeçalho. Selecione manualmente.\n")
            return
        pattern = detected_pattern
        app_print(f"Padrão Bayer detectado automaticamente: {pattern}\n")
    else:
        pattern = user_pattern
        app_print(f"Padrão Bayer manual: {pattern}\n")

    cv2_conversion_code = BAYER_CV2_MAPPING[pattern]
    
    def debayer_file(filepath: Path):
        with fits.open(filepath, memmap=False) as hdul:
            hdu_img = next(hdu for hdu in hdul if hdu.is_image and hdu.data is not None)
            header = hdu_img.header.copy()
            data = np.asarray(hdu_img.data, dtype=np.float32)
            
        # O cv2.cvtColor para Bayer exige uint8 ou uint16.
        data_u16 = np.clip(data, 0, 65535).astype(np.uint16)
        
        # Debayering (transforma 2D CFA em 3D RGB)
        rgb_data = cv2.cvtColor(data_u16, cv2_conversion_code)
        
        # Limpa chaves do header que podem confundir visualizadores futuros
        for key in ['BAYERPAT', 'BAYERPATTERN', 'COLORTYP', 'BZERO', 'BSCALE']:
            header.remove(key, ignore_missing=True)
            
        header['DEBAYER'] = pattern
        header['CTYPE3'] = 'RGB'
        
        # No FITS/Astropy, matrizes 3D coloridas devem ser salvas no formato (Canais, Altura, Largura)
        rgb_fits_format = np.moveaxis(rgb_data, -1, 0)
        
        out_path = output_dir / filepath.name
        fits.writeto(out_path, rgb_fits_format, header, overwrite=True, output_verify='ignore')

    app_progress(0, total, "Debayerizando frames...")
    processed = 0
    
    with ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as executor:
        for future in [executor.submit(debayer_file, f) for f in files]:
            if cancel_event.is_set(): break
            try:
                future.result()
            except Exception as e:
                app_print(f"Erro no debayer: {e}\n")
            processed += 1
            app_progress(processed, total, f"Convertido para RGB ({processed}/{total})")
            
    app_print(f"\n>>> Debayer concluído: {processed} arquivos RGB gerados em {output_dir.name}. <<<\n")