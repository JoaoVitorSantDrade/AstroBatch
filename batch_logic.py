# batch_logic.py
import math
import os
import queue
import re
import shutil
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from astropy.io import fits
from PIL import Image

FITS_SUFFIXES = {".fit", ".fits", ".fts"}
RESAMPLE_MODES = {
    "Nearest": Image.Resampling.NEAREST,
    "Bilinear": Image.Resampling.BILINEAR,
    "Lanczos": Image.Resampling.LANCZOS,
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
        (p for p in input_dir.iterdir() if p.is_file() and p.suffix.casefold() in FITS_SUFFIXES),
        key=lambda p: get_sequence_number(p.name),
    )

def prepare_image(data: np.ndarray, opt_method: str, crop_size: int, downsample_method: str, downsample_scale: float) -> np.ndarray:
    if data.ndim != 2:
        raise ValueError(f"A imagem precisa ser 2D; dimensões encontradas: {data.shape}")

    data_float = np.asarray(data, dtype=np.float32)
    if not np.isfinite(data_float).any():
        raise ValueError("A imagem não possui nenhum pixel finito.")

    if opt_method == "Crop":
        h, w = data_float.shape
        size = min(crop_size, h, w)
        y1 = (h - size) // 2
        x1 = (w - size) // 2
        return data_float[y1 : y1 + size, x1 : x1 + size].copy()

    if opt_method != "Downsampling":
        raise ValueError(f"Método de otimização desconhecido: {opt_method}")

    new_w = max(1, round(data_float.shape[1] * downsample_scale))
    new_h = max(1, round(data_float.shape[0] * downsample_scale))
    image = Image.fromarray(data_float, mode="F")
    resized = image.resize((new_w, new_h), resample=RESAMPLE_MODES[downsample_method])
    return np.asarray(resized, dtype=np.float32)

def comparison_score(current: np.ndarray, previous: np.ndarray) -> float:
    if current.shape != previous.shape:
        raise ValueError(f"Imagens incompatíveis para comparação: {current.shape} vs {previous.shape}")

    valid = np.isfinite(current) & np.isfinite(previous)
    if not np.any(valid):
        return math.nan

    diff = current[valid].astype(np.float32) - previous[valid].astype(np.float32)
    diff -= np.mean(diff)
    return float(np.sqrt(np.mean(np.square(diff), dtype=np.float64)))

def get_optimal_worker_count() -> int:
    try:
        cpu_count = getattr(os, "process_cpu_count", os.cpu_count)() or 1
    except Exception:
        cpu_count = os.cpu_count() or 1
    return max(1, min(16, cpu_count))

def prepare_fits_file(filepath: Path, config: ProcessingConfig) -> tuple[Path, np.ndarray | None, str | None]:
    try:
        with fits.open(filepath, memmap=False) as hdul:
            data = None
            for hdu in hdul:
                if not hdu.is_image: continue
                if hdu.data is None or hdu.data.ndim != 2: continue
                data = np.asarray(hdu.data, dtype=np.float32)
                break

            if data is None:
                return filepath, None, f"Aviso: nenhum HDU de imagem 2D em {filepath.name}"

            prepared = prepare_image(data, config.opt_method, config.crop_size, config.downsample_method, config.downsample_scale)
            return filepath, prepared, None
    except Exception as exc:
        return filepath, None, f"Erro ao processar {filepath.name}: {exc}"

def file_mover_worker(move_queue: queue.Queue, app_print, cancel_event: threading.Event, app_progress, move_state: dict):
    while True:
        item = move_queue.get()
        if item is None: break
            
        src, dst, action, overwrite = item  # Recebendo a flag overwrite
        
        if cancel_event.is_set():
            move_queue.task_done()
            continue
            
        try:
            # Se a flag estiver ativa e o arquivo já existir, remove o antigo primeiro
            if dst.exists() and overwrite:
                dst.unlink()
                
            if action == 'copy':
                shutil.copy2(str(src), str(dst))
            else:
                shutil.move(str(src), str(dst))
        except Exception as exc:
            verbo = "copiar" if action == 'copy' else "mover"
            app_print(f"Erro ao {verbo} {src.name}: {exc}\n")
        finally:
            move_state["moved"] += 1
            
            # Atualiza a UI apenas a cada 20 arquivos ou quando for o último
            if move_state["moved"] % 20 == 0 or move_state["moved"] == move_state['total']:
                if move_state["total"] > 0 and app_progress:
                    app_progress(move_state["moved"], move_state["total"], f"Salvando no disco ({move_state['moved']}/{move_state['total']})...")
            
            move_queue.task_done()

def process_fits_logic(config: ProcessingConfig, app_print, app_progress, cancel_event: threading.Event) -> tuple[int, int]:
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
    
    if not config.dry_run: output_dir.mkdir(parents=True, exist_ok=True)

    batch_num = 1
    current_batch_dir = output_dir / f"batch_{batch_num:03d}"
    if not config.dry_run: current_batch_dir.mkdir(parents=True, exist_ok=True)

    previous_data = None
    score_history = deque(maxlen=10)
    processed = 0
    queued_for_move = 0

    worker_count = get_optimal_worker_count()
    prefetch_count = min(len(files), max(worker_count * 2, worker_count + 2))

    move_state = {"moved": 0, "total": total_files}
    move_queue = queue.Queue()
    mover_thread = threading.Thread(target=file_mover_worker, args=(move_queue, app_print, cancel_event, app_progress, move_state), daemon=True)
    if not config.dry_run: mover_thread.start()

    app_print(f"\nPipeline paralelo: {worker_count} workers leitura | 1 worker disco | prefetch: {prefetch_count}\n")
    app_print(f"Iniciando Batch {batch_num:03d}...\n")

    with ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="fits-prep") as executor:
        next_submit = 0
        next_process = 0
        futures = {}

        def submit_until_window_full():
            nonlocal next_submit
            while next_submit < len(files) and len(futures) < prefetch_count and not cancel_event.is_set():
                futures[next_submit] = executor.submit(prepare_fits_file, files[next_submit], config)
                next_submit += 1

        submit_until_window_full()

        while next_process < len(files):
            if cancel_event.is_set():
                app_print("\nCancelamento solicitado...\n")
                break

            future = futures.pop(next_process)
            filepath = files[next_process]

            try:
                _, prepared, error_message = future.result()
            except Exception as exc:
                prepared = None
                error_message = f"Erro inesperado no worker para {filepath.name}: {exc}"

            next_process += 1
            app_progress(next_process, total_files, f"Analisando deriva ({next_process}/{total_files})...")

            if error_message:
                app_print(error_message + "\n")
                submit_until_window_full()
                continue

            if prepared is None:
                submit_until_window_full()
                continue

            if previous_data is not None:
                try:
                    score = comparison_score(prepared, previous_data)
                except ValueError as exc:
                    app_print(f"Aviso: {filepath.name}: {exc}\n")
                    previous_data = prepared
                    score_history.clear()
                    submit_until_window_full()
                    continue

                if math.isfinite(score):
                    if len(score_history) >= 5:
                        baseline = float(np.median(score_history))
                        if baseline > np.finfo(np.float32).eps:
                            limit = baseline * config.threshold_factor
                            if score > limit:
                                batch_num += 1
                                current_batch_dir = output_dir / f"batch_{batch_num:03d}"
                                if not config.dry_run: current_batch_dir.mkdir(parents=True, exist_ok=True)

                                app_print("\n>>> MOVIMENTO BRUSCO DETECTADO <<<\n")
                                app_print(f"Arquivo: {filepath.name} | Score: {score:.4f} | Baseline: {baseline:.4f} | Limite: {limit:.4f}\n")
                                app_print(f"Criando Batch {batch_num:03d}...\n")
                                score_history.clear()
                    score_history.append(score)

            previous_data = prepared

            if not config.dry_run:
                destination = current_batch_dir / filepath.name
                if destination.exists() and not config.overwrite:
                    app_print(f"ERRO: destino já existe, arquivo ignorado: {destination}\n")
                else:
                    action = 'copy' if config.copy_files else 'move'
                    move_queue.put((filepath, destination, action, config.overwrite))
                    queued_for_move += 1

            processed += 1
            submit_until_window_full()

    if not config.dry_run:
        move_state["total"] = queued_for_move
        app_print("\nAguardando finalização das movimentações de disco...\n")
        move_queue.put(None)
        mover_thread.join()

    app_progress(100, 100, "Concluído.")
    app_print(f"\nFinalizado! {processed} arquivos processados em {batch_num} batches.\n")
    return processed, batch_num