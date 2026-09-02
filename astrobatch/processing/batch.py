import math
import os
import queue
import re
import shutil
import threading
import warnings
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from astropy.io import fits
from astropy.utils.exceptions import AstropyWarning

# Suprime todos os avisos de verificação de cabeçalho do Astropy
warnings.simplefilter("ignore", category=AstropyWarning)

FITS_SUFFIXES = {".fit", ".fits", ".fts"}

# P0: cv2.resize no lugar de PIL
RESAMPLE_MODES = {
    "Nearest": cv2.INTER_NEAREST,
    "Bilinear": cv2.INTER_LINEAR,
    "Lanczos": cv2.INTER_LANCZOS4,
}


@dataclass(frozen=True)
class ProcessingConfig:
    input_dir: Path
    output_dir: Path
    threshold_factor: float
    crop_size: int
    dry_run: bool
    copy_files: bool
    overwrite: bool
    opt_method: str
    downsample_method: str
    downsample_scale: float


def get_sequence_number(filename: str) -> tuple[int, int | str]:
    match = re.search(r"_(\d+)\.(?:fit|fits|fts)$", filename, re.IGNORECASE)
    if match:
        return (0, int(match.group(1)))
    return (1, filename.casefold())


def find_fits_files(input_dir: Path) -> list[Path]:
    return sorted(
        (
            p
            for p in input_dir.iterdir()
            if p.is_file() and p.suffix.casefold() in FITS_SUFFIXES
        ),
        key=lambda p: get_sequence_number(p.name),
    )


def prepare_image(
    data: np.ndarray,
    opt_method: str,
    crop_size: int,
    downsample_method: str,
    downsample_scale: float,
) -> np.ndarray:
    if data.ndim != 2:
        raise ValueError(
            f"A imagem precisa ser 2D; dimensões encontradas: {data.shape}"
        )

    # P0: Crop ANTES de converter para float32 (Economiza imensa CPU e RAM)
    if opt_method == "Crop":
        h, w = data.shape
        size = min(crop_size, h, w)
        y1 = (h - size) // 2
        x1 = (w - size) // 2
        data = data[y1 : y1 + size, x1 : x1 + size]

    # Converte apenas a área útil para float32 para as análises matemáticas
    data_float = np.asarray(data, dtype=np.float32)

    if not np.isfinite(data_float).any():
        raise ValueError("A imagem não possui nenhum pixel finito.")

    if opt_method == "Crop":
        return data_float

    if opt_method == "Downsampling":
        # P0: Análise em resolução reduzida com cv2.resize
        new_w = max(1, round(data_float.shape[1] * downsample_scale))
        new_h = max(1, round(data_float.shape[0] * downsample_scale))
        interp = RESAMPLE_MODES.get(downsample_method, cv2.INTER_NEAREST)
        return cv2.resize(data_float, (new_w, new_h), interpolation=interp)

    raise ValueError(f"Método de otimização desconhecido: {opt_method}")


def comparison_score(current: np.ndarray, previous: np.ndarray) -> float:
    if current.shape != previous.shape:
        raise ValueError(
            f"Imagens incompatíveis para comparação: {current.shape} vs {previous.shape}"
        )

    # P1: Otimizar comparison_score usando rotinas subjacentes do OpenCV/Numpy
    diff = cv2.subtract(current, previous)

    # Verificação hiper-rápida de NaNs/Infs que só executa a máscara pesada se der positivo
    if np.isnan(diff).any() or np.isinf(diff).any():
        valid = np.isfinite(diff)
        if not valid.any():
            return math.nan
        diff = diff[valid]

    # In-place centralização e cálculo do RMS de forma direta (RMS vetorial)
    diff -= np.mean(diff)
    return float(np.linalg.norm(diff) / math.sqrt(diff.size))


def get_optimal_worker_count() -> int:
    try:
        cpu_count = getattr(os, "process_cpu_count", os.cpu_count)() or 1
    except Exception:
        cpu_count = os.cpu_count() or 1
    return max(1, min(16, cpu_count))


def prepare_fits_file(
    filepath: Path, config: ProcessingConfig
) -> tuple[Path, np.ndarray | None, str | None]:
    try:
        with fits.open(filepath, memmap=False) as hdul:
            data = None
            for hdu in hdul:
                if not hdu.is_image:
                    continue
                if hdu.data is None or hdu.data.ndim != 2:
                    continue
                data = hdu.data  # Lê o ponteiro bruto no formato nativo (ex: uint16)
                break

            if data is None:
                return (
                    filepath,
                    None,
                    f"Aviso: nenhum HDU de imagem 2D em {filepath.name}",
                )

            prepared = prepare_image(
                data,
                config.opt_method,
                config.crop_size,
                config.downsample_method,
                config.downsample_scale,
            )
            return filepath, prepared, None
    except Exception as exc:
        return filepath, None, f"Erro ao processar {filepath.name}: {exc}"


def file_mover_worker(
    move_queue: queue.Queue,
    app_print,
    cancel_event: threading.Event,
    app_progress,
    move_state: dict,
):
    while True:
        item = move_queue.get()
        if item is None:
            break

        src, dst, action, overwrite = item

        if cancel_event.is_set():
            move_queue.task_done()
            continue

        try:
            if dst.exists() and overwrite:
                dst.unlink()

            if action == "copy":
                shutil.copy2(str(src), str(dst))
            else:
                shutil.move(str(src), str(dst))
        except Exception as exc:
            verbo = "copiar" if action == "copy" else "mover"
            app_print(f"Erro ao {verbo} {src.name}: {exc}\n")
        finally:
            move_state["moved"] += 1

            if (
                move_state["moved"] % 20 == 0
                or move_state["moved"] == move_state["total"]
            ):
                if move_state["total"] > 0 and app_progress:
                    app_progress(
                        move_state["moved"],
                        move_state["total"],
                        f"Salvando no disco ({move_state['moved']}/{move_state['total']})...",
                    )

            move_queue.task_done()


def process_fits_logic(
    config: ProcessingConfig, app_print, app_progress, cancel_event: threading.Event
) -> tuple[int, int]:
    input_dir = config.input_dir
    output_dir = config.output_dir

    app_print(f"Lendo arquivos de: {input_dir}\n")
    files = find_fits_files(input_dir)
    total_files = len(files)
    if not files:
        app_print("Nenhum arquivo FITS encontrado no diretório.\n")
        return 0, 0

    app_progress(0, total_files, "Iniciando análise...")
    app_print(f"Total de arquivos encontrados: {total_files}\n")

    if not config.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    batch_num = 1
    current_batch_dir = output_dir / f"batch_{batch_num:03d}"
    if not config.dry_run:
        current_batch_dir.mkdir(parents=True, exist_ok=True)

    previous_data = None
    score_history = deque(maxlen=10)
    processed = 0
    queued_for_move = 0

    worker_count = get_optimal_worker_count()
    prefetch_count = min(len(files), max(worker_count * 2, worker_count + 2))

    move_state = {"moved": 0, "total": total_files}
    move_queue = queue.Queue()
    mover_thread = threading.Thread(
        target=file_mover_worker,
        args=(move_queue, app_print, cancel_event, app_progress, move_state),
        daemon=True,
    )
    if not config.dry_run:
        mover_thread.start()

    app_print(
        f"\nPipeline paralelo: {worker_count} workers leitura | 1 worker disco | prefetch: {prefetch_count}\n"
    )
    app_print(f"Iniciando Batch {batch_num:03d}...\n")

    # P0: Producer/Consumer ordenado e Gerenciamento de Memória P0
    def generate_prepared_frames():
        """Generator que fornece as imagens perfeitamente ordenadas e descarta assim que o loop principal o consome."""
        with ThreadPoolExecutor(
            max_workers=worker_count, thread_name_prefix="fits-prep"
        ) as executor:
            futures = {}
            next_submit = 0
            next_yield = 0

            while next_yield < len(files):
                if cancel_event.is_set():
                    break

                # Produz (Pre-fetch buffer)
                while next_submit < len(files) and len(futures) < prefetch_count:
                    futures[next_submit] = executor.submit(
                        prepare_fits_file, files[next_submit], config
                    )
                    next_submit += 1

                # Consome ordenadamente
                future = futures.pop(next_yield)
                yield next_yield, files[next_yield], future.result()
                next_yield += 1

    # O pipeline principal reage como consumidor rigorosamente assíncrono
    for idx, filepath, result in generate_prepared_frames():
        if cancel_event.is_set():
            app_print("\nCancelamento solicitado...\n")
            break

        try:
            _, prepared, error_message = result
        except Exception as exc:
            prepared = None
            error_message = f"Erro inesperado no worker para {filepath.name}: {exc}"

        next_process = idx + 1
        app_progress(
            next_process,
            total_files,
            f"Analisando deriva ({next_process}/{total_files})...",
        )

        if error_message:
            app_print(error_message + "\n")
            continue

        if prepared is None:
            continue

        if previous_data is not None:
            try:
                score = comparison_score(prepared, previous_data)
            except ValueError as exc:
                app_print(f"Aviso: {filepath.name}: {exc}\n")
                previous_data = prepared
                score_history.clear()
                continue

            if math.isfinite(score):
                if len(score_history) >= 5:
                    baseline = float(np.median(score_history))
                    if baseline > np.finfo(np.float32).eps:
                        limit = baseline * config.threshold_factor
                        if score > limit:
                            batch_num += 1
                            current_batch_dir = output_dir / f"batch_{batch_num:03d}"
                            if not config.dry_run:
                                current_batch_dir.mkdir(parents=True, exist_ok=True)

                            app_print("\n>>> MOVIMENTO BRUSCO DETECTADO <<<\n")
                            app_print(
                                f"Arquivo: {filepath.name} | Score: {score:.4f} | Baseline: {baseline:.4f} | Limite: {limit:.4f}\n"
                            )
                            app_print(f"Criando Batch {batch_num:03d}...\n")
                            score_history.clear()
                score_history.append(score)

        # P0: Não Manter Prepared Frames em Memória - Substitui o array liberando-o no Garbage Collector
        previous_data = prepared

        if not config.dry_run:
            destination = current_batch_dir / filepath.name
            if destination.exists() and not config.overwrite:
                app_print(f"ERRO: destino já existe, arquivo ignorado: {destination}\n")
            else:
                action = "copy" if config.copy_files else "move"
                move_queue.put((filepath, destination, action, config.overwrite))
                queued_for_move += 1

        processed += 1

    if not config.dry_run:
        move_state["total"] = queued_for_move
        app_print("\nAguardando finalização das movimentações de disco...\n")
        move_queue.put(None)
        mover_thread.join()

    app_progress(100, 100, "Concluído.")
    app_print(
        f"\nFinalizado! {processed} arquivos processados em {batch_num} batches.\n"
    )
    return processed, batch_num
