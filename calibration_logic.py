import os
import threading
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.utils.exceptions import AstropyWarning

# Suprime todos os avisos de verificação de cabeçalho do Astropy
warnings.simplefilter("ignore", category=AstropyWarning)


def load_fits_data(filepath: Path) -> tuple[np.ndarray, fits.Header]:
    with fits.open(filepath, memmap=False, ignore_missing_end=True) as hdul:
        for hdu in hdul:
            if hdu.is_image and hdu.data is not None and hdu.data.ndim == 2:
                return np.asarray(hdu.data, dtype=np.float32), hdu.header.copy()
    raise ValueError(f"Imagem 2D não encontrada em {filepath.name}")


def _read_for_master(filepath: Path) -> tuple[np.ndarray, fits.Header]:
    """Isola a leitura dos FITS para permitir o despacho asíncrono no ThreadPoolExecutor."""
    return load_fits_data(filepath)


def save_compressed_fits(data: np.ndarray, header: fits.Header, output_path: Path):
    """
    Força o cast da imagem para uint16 e salva usando compressão RICE_1 sem perdas.
    Isso reduz drasticamente o tamanho do arquivo no disco.
    """
    # 1. Cast seguro para não estourar os limites de 16-bit
    data_uint16 = np.clip(data, 0, 65535).astype(np.uint16)

    # 2. Correção de Header obrigatória
    if header is not None:
        header["BITPIX"] = 16
        header["BZERO"] = 32768
        header["BSCALE"] = 1
        # Limpa resíduos de float32
        header.remove("BLANK", ignore_missing=True)

    # 3. Criação do arquivo com FITS Tile Compression
    hdu = fits.CompImageHDU(data=data_uint16, header=header, compression_type="RICE_1")
    hdul = fits.HDUList([fits.PrimaryHDU(), hdu])

    hdul.writeto(str(output_path), overwrite=True, output_verify="ignore")


def make_master(
    folder_path: Path, output_path: Path, app_print, cancel_event
) -> np.ndarray | None:
    files = sorted(
        [
            f
            for f in folder_path.iterdir()
            if f.is_file() and f.suffix.lower() in {".fit", ".fits", ".fts"}
        ]
    )
    if not files:
        app_print(f"Nenhum frame encontrado em {folder_path.name} para criar Master.\n")
        return None

    app_print(
        f"Lendo {len(files)} frames em paralelo para gerar Master ({folder_path.name})...\n"
    )

    data_stack = []
    base_header = None

    # Carregamento em Lote com Multithreading
    workers = min(16, (os.cpu_count() or 1) + 2)
    with ThreadPoolExecutor(
        max_workers=workers, thread_name_prefix="master_read"
    ) as executor:
        futures = {executor.submit(_read_for_master, f): f for f in files}

        for future in as_completed(futures):
            if cancel_event.is_set():
                executor.shutdown(wait=False, cancel_futures=True)
                return None
            try:
                d, h = future.result()
                data_stack.append(d)
                if base_header is None:
                    base_header = h
            except Exception as e:
                app_print(f"Erro ao ler {futures[future].name}: {e}\n")

    if not data_stack:
        return None

    app_print(f"Calculando a mediana dos {len(data_stack)} frames...\n")
    # Mediana robusta para eliminar raios cósmicos e satélites
    master_data = np.median(data_stack, axis=0)

    save_compressed_fits(master_data, base_header, output_path)
    app_print(f"Master salvo em: {output_path.name}\n")

    return master_data


def calibrate_single_frame(
    light_path: Path,
    output_dir: Path,
    master_dark: np.ndarray | None,
    master_flat: np.ndarray | None,
    overwrite: bool,
) -> str | None:
    out_path = output_dir / light_path.name
    if out_path.exists() and not overwrite:
        return f"Ignorado (já existe): {light_path.name}"

    try:
        data, header = load_fits_data(light_path)

        # Subtração do Dark
        if master_dark is not None:
            data = data - master_dark
            data[data < 0] = 0.0

        # Divisão pelo Flat
        if master_flat is not None:
            data = data / master_flat

        save_compressed_fits(data, header, out_path)
        return None
    except Exception as e:
        return f"Erro em {light_path.name}: {e}"


def run_calibration_pipeline(
    config: dict, app_print, app_progress, cancel_event: threading.Event
):
    input_dir = Path(config["input_dir"])
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Resolução dos Masters
    master_dark = None
    if config["apply_dark"] and config["dark_path"]:
        dark_p = Path(config["dark_path"])
        if dark_p.is_dir() and config["create_master"]:
            master_dark_path = input_dir.parent / "MasterDark.fits"
            master_dark = make_master(dark_p, master_dark_path, app_print, cancel_event)
        elif dark_p.is_file():
            master_dark, _ = load_fits_data(dark_p)
            app_print("Master Dark carregado.\n")

    master_flat = None
    if config["apply_flat"] and config["flat_path"]:
        flat_p = Path(config["flat_path"])
        if flat_p.is_dir() and config["create_master"]:
            master_flat_path = input_dir.parent / "MasterFlat.fits"
            m_flat = make_master(flat_p, master_flat_path, app_print, cancel_event)
            if m_flat is not None:
                # Normalização do Master Flat (divide pela própria mediana)
                median_val = np.median(m_flat)
                master_flat = m_flat / (median_val if median_val > 0 else 1.0)
                master_flat[master_flat <= 0.01] = 1.0  # Prevenção de divisão por zero
        elif flat_p.is_file():
            m_flat, _ = load_fits_data(flat_p)
            median_val = np.median(m_flat)
            master_flat = m_flat / (median_val if median_val > 0 else 1.0)
            master_flat[master_flat <= 0.01] = 1.0
            app_print("Master Flat carregado.\n")

    if cancel_event.is_set():
        return

    # 2. Calibração dos Lights
    lights = sorted(
        [
            f
            for f in input_dir.iterdir()
            if f.is_file() and f.suffix.lower() in {".fit", ".fits", ".fts"}
        ]
    )
    total = len(lights)
    if total == 0:
        app_print("Nenhum LIGHT encontrado na pasta de entrada.\n")
        return

    app_print(f"\nIniciando calibração paralela de {total} frames...\n")
    processed = 0
    failed = 0

    workers = min(16, (os.cpu_count() or 1) + 2)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                calibrate_single_frame,
                light,
                output_dir,
                master_dark,
                master_flat,
                config["overwrite"],
            ): light
            for light in lights
        }

        for future in as_completed(futures):
            if cancel_event.is_set():
                executor.shutdown(wait=False, cancel_futures=True)
                break

            error = future.result()
            processed += 1
            if error:
                app_print(f"{error}\n")
                failed += 1

            if processed % 10 == 0 or processed == total:
                app_progress(processed, total, f"Calibrando ({processed}/{total})...")

    app_print(
        f"\n>>> Calibração Finalizada: {processed - failed} arquivos salvos, {failed} erros. <<<\n"
    )
