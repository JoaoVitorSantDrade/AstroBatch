import json
import os
import shutil
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

BAYER_BASE_CODES = {
    "RGGB": cv2.COLOR_BayerBG2RGB,
    "BGGR": cv2.COLOR_BayerRG2RGB,
    "GRBG": cv2.COLOR_BayerGB2RGB,
    "GBRG": cv2.COLOR_BayerGR2RGB,
}


def get_debayer_conversion_code(pattern: str, method: str) -> int:
    """
    Retorna o código OpenCV correspondente ao padrão Bayer
    e ao método de debayerização escolhido.

    Observação:
    Os nomes dos padrões seguem a convenção FITS/header.
    O OpenCV utiliza uma nomenclatura própria para os códigos.
    """

    pattern_codes = {
        "RGGB": {
            "Bilinear": cv2.COLOR_BayerBG2RGB,
            "VNG": cv2.COLOR_BayerBG2RGB_VNG,
            "Edge-Aware": cv2.COLOR_BayerBG2RGB_EA,
        },
        "BGGR": {
            "Bilinear": cv2.COLOR_BayerRG2RGB,
            "VNG": cv2.COLOR_BayerRG2RGB_VNG,
            "Edge-Aware": cv2.COLOR_BayerRG2RGB_EA,
        },
        "GRBG": {
            "Bilinear": cv2.COLOR_BayerGB2RGB,
            "VNG": cv2.COLOR_BayerGB2RGB_VNG,
            "Edge-Aware": cv2.COLOR_BayerGB2RGB_EA,
        },
        "GBRG": {
            "Bilinear": cv2.COLOR_BayerGR2RGB,
            "VNG": cv2.COLOR_BayerGR2RGB_VNG,
            "Edge-Aware": cv2.COLOR_BayerGR2RGB_EA,
        },
    }

    try:
        return pattern_codes[pattern][method]
    except KeyError:
        raise ValueError(
            f"Método de debayer inválido: pattern={pattern!r}, method={method!r}"
        )


@dataclass(frozen=True)
class AlignConfig:
    base_dir: Path
    output_dir: Path

    # Debayer
    debayer_pattern: str
    debayer_method: str

    # Alignment
    interpolation: str

    # Storage / execution
    overwrite: bool
    dry_run: bool
    keep_header: bool
    delete_intermediates: bool


def get_optimal_worker_count() -> int:
    try:
        cpu_count = getattr(os, "process_cpu_count", os.cpu_count)() or 1
    except Exception:
        cpu_count = os.cpu_count() or 1

    try:
        import psutil

        available_ram = psutil.virtual_memory().available
        ram_workers = max(1, available_ram // (800 * 1024 * 1024))
    except ImportError:
        ram_workers = cpu_count

    return max(1, min(16, cpu_count, ram_workers))


def find_batch_folders(base_dir: Path) -> list[Path]:
    return sorted(
        d for d in base_dir.iterdir() if d.is_dir() and "batch" in d.name.lower()
    )


def load_json(filepath: Path) -> dict:
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def load_local_flow(batch_dir: Path) -> dict | None:
    flow_path = batch_dir / "flow_local.json"
    if not flow_path.exists():
        return None
    return load_json(flow_path)


def load_global_flow(base_dir: Path) -> dict | None:
    flow_path = base_dir / "global_flow.json"
    if not flow_path.exists():
        return None
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
    for key in ["BAYERPAT", "BAYERPATTERN", "COLORTYP"]:
        if key in header:
            val = str(header[key]).strip().upper().strip("'")
            if val in BAYER_CV2_MAPPING:
                return val
    return None


def process_in_memory_debayer(
    data: np.ndarray,
    header: fits.Header,
    pattern: str | None,
    method: str = "Bilinear",
) -> tuple[np.ndarray, fits.Header]:
    """
    Realiza o debayer em memória.

    pattern:
        RGGB, BGGR, GRBG ou GBRG.

    method:
        Bilinear
        VNG
        Edge-Aware

    Se pattern for None, os dados permanecem CFA/mono.
    """

    # Mono/CFA ou Debayer desativado
    if not pattern:
        return data, header

    if pattern not in BAYER_BASE_CODES:
        raise ValueError(f"Padrão Bayer inválido: {pattern}")

    cv2_conversion_code = get_debayer_conversion_code(
        pattern,
        method,
    )

    # OpenCV trabalha de forma mais previsível com uint16
    data_u16 = np.clip(data, 0, 65535).astype(np.uint16)

    rgb_data = cv2.cvtColor(
        data_u16,
        cv2_conversion_code,
    )

    # Remove informações CFA que não são mais válidas
    for key in [
        "BAYERPAT",
        "BAYERPATTERN",
        "COLORTYP",
        "BZERO",
        "BSCALE",
    ]:
        header.remove(key, ignore_missing=True)

    header["DEBAYER"] = pattern
    header["DEBMETHOD"] = method
    header["CTYPE3"] = "RGB"

    return rgb_data, heade


def warp_frame(
    data: np.ndarray, final_matrix: np.ndarray, interpolation_flag: int
) -> np.ndarray:
    h, w = data.shape[:2]
    matrix_2x3 = final_matrix[:2, :].astype(np.float64)
    return cv2.warpAffine(
        data,
        matrix_2x3,
        (w, h),
        flags=interpolation_flag,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )


def generate_valid_mask(shape: tuple, final_matrix: np.ndarray) -> np.ndarray:
    mask = np.ones(shape, dtype=np.uint8)
    matrix_2x3 = final_matrix[:2, :].astype(np.float64)
    return cv2.warpAffine(
        mask,
        matrix_2x3,
        (shape[1], shape[0]),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def save_compressed_fits(
    data: np.ndarray, mask: np.ndarray, header: fits.Header | None, output_path: Path
):
    if header is not None:
        header["BITPIX"] = 16
        header["BZERO"] = 32768
        header["BSCALE"] = 1

    data_uint16 = np.clip(data, 0, 65535).astype(np.uint16)

    if data_uint16.ndim == 3:
        data_uint16 = np.moveaxis(data_uint16, -1, 0)

    hdu_data = fits.CompImageHDU(
        data=data_uint16, header=header, compression_type="RICE_1"
    )
    hdu_mask = fits.CompImageHDU(
        data=mask, name="VALID_MASK", compression_type="PLIO_1"
    )

    hdul = fits.HDUList([fits.PrimaryHDU(), hdu_data, hdu_mask])
    hdul.writeto(str(output_path), overwrite=True, output_verify="ignore")


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

        output_path = output_dir / frame_name
        if output_path.exists() and not config.overwrite:
            return (
                frame_name,
                f"ERRO: destino já existe, arquivo ignorado: {output_path}",
            )

        raw_data, raw_header = load_fits_data(filepath)

        # ---> Lógica explícita de ByPass do Debayer <---
        if config.debayer_pattern == "Nenhum":
            pattern = None
        elif config.debayer_pattern == "Auto":
            pattern = get_bayer_pattern(raw_header)
        else:
            pattern = config.debayer_pattern

        final_matrix = compute_final_matrix(frame_info["matrix"], global_matrix)

        if not config.dry_run:
            rgb_data, updated_header = process_in_memory_debayer(
                raw_data,
                raw_header,
                pattern,
                config.debayer_method,
            )
            warped_data = warp_frame(rgb_data, final_matrix, interpolation_flag)
            mask = generate_valid_mask(raw_data.shape[:2], final_matrix)

            output_dir.mkdir(parents=True, exist_ok=True)
            save_compressed_fits(
                warped_data,
                mask,
                updated_header if config.keep_header else None,
                output_path,
            )

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
    if local_flow is None:
        return 0, 0

    batch_entry = global_flow["batches"].get(batch_dir.name)
    if batch_entry is None:
        return 0, 0

    global_matrix = batch_entry["matrix"]
    frames = local_flow.get("frames", {})
    total_frames = len(frames)

    if total_frames == 0:
        return 0, 0

    output_dir = config.output_dir / batch_dir.name
    interpolation_flag = INTERPOLATION_MODES.get(
        config.interpolation, cv2.INTER_LANCZOS4
    )

    worker_count = get_optimal_worker_count()
    processed, failed = 0, 0

    with ThreadPoolExecutor(
        max_workers=worker_count, thread_name_prefix="astroalign"
    ) as executor:
        futures = {
            executor.submit(
                _process_single_alignment,
                fname,
                finfo,
                batch_dir,
                output_dir,
                global_matrix,
                interpolation_flag,
                config,
            ): fname
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

            if (
                progress_state["done"] % 10 == 0
                or progress_state["done"] == progress_state["total"]
            ):
                app_progress(
                    progress_state["done"],
                    progress_state["total"],
                    f"Alinhando frames ({progress_state['done']}/{progress_state['total']})...",
                )

    if (
        config.delete_intermediates
        and failed == 0
        and not config.dry_run
        and not cancel_event.is_set()
    ):
        try:
            shutil.rmtree(batch_dir)
            app_print(f"[{batch_dir.name}] Batch original limpo com sucesso.\n")
        except Exception as e:
            app_print(f"[{batch_dir.name}] Erro ao apagar intermediários: {e}\n")

    app_print(
        f"[{batch_dir.name}] Concluído: {processed} alinhados, {failed} falhas.\n"
    )
    return processed, failed


def process_all_alignments(
    base_dir: Path,
    output_dir: Path,
    config_dict: dict,
    app_print,
    app_progress,
    cancel_event: threading.Event,
) -> tuple[int, int]:
    if not isinstance(config_dict, dict):
        config_dict = {}

    align_config = AlignConfig(
        base_dir=base_dir,
        output_dir=output_dir,
        debayer_pattern=config_dict.get(
            "debayer_pattern",
            "Auto",
        ),
        debayer_method=config_dict.get(
            "debayer_method",
            "Bilinear",
        ),
        interpolation=config_dict.get(
            "interpolation",
            "Lanczos",
        ),
        overwrite=bool(config_dict.get("overwrite", False)),
        dry_run=bool(config_dict.get("dry_run", False)),
        keep_header=bool(config_dict.get("keep_header", True)),
        delete_intermediates=bool(config_dict.get("delete_intermediates", False)),
    )

    global_flow = load_global_flow(base_dir)
    if global_flow is None:
        app_print("ERRO: global_flow.json não encontrado.\n")
        return 0, 0

    batch_folders = find_batch_folders(base_dir)
    if not batch_folders:
        return 0, 0

    total_frames = 0
    batches_with_flow = []
    for b_folder in batch_folders:
        local_flow = load_local_flow(b_folder)
        if local_flow:
            batches_with_flow.append(b_folder)
            total_frames += len(local_flow.get("frames", {}))

    if total_frames == 0:
        return 0, 0

    progress_state = {"done": 0, "total": total_frames}
    if not align_config.dry_run:
        align_config.output_dir.mkdir(parents=True, exist_ok=True)

    total_processed, total_failed = 0, 0

    for b_folder in batches_with_flow:
        if cancel_event.is_set():
            break
        app_print(f"\nAlinhando Batch: {b_folder.name}\n")
        processed, failed = process_batch_alignment(
            b_folder,
            global_flow,
            align_config,
            app_print,
            app_progress,
            cancel_event,
            progress_state,
        )
        total_processed += processed
        total_failed += failed

    app_progress(total_frames, total_frames, "Concluído.")
    app_print(
        f"\n>>> AstroAlign Finalizado! {total_processed} frames alinhados, {total_failed} falhas. <<<\n"
    )
    return total_processed, total_failed
