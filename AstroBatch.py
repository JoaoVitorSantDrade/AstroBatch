from __future__ import annotations

import math
import os
import queue
import re
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor
from collections import deque
from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
from astropy.io import fits
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk
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
    opt_method: str
    downsample_method: str
    downsample_scale: float


def get_sequence_number(filename: str) -> tuple[int, int | str]:
    """
    Returns a sortable key.

    Files containing '_123.fit' / '_123.fits' are ordered numerically.
    Files without a sequence number are ordered alphabetically after them.
    """
    match = re.search(r"_(\d+)\.(?:fit|fits|fts)$", filename, re.IGNORECASE)
    if match:
        return (0, int(match.group(1)))
    return (1, filename.casefold())


def find_fits_files(input_dir: Path) -> list[Path]:
    """Find FITS files without accidentally including the output directory."""
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
    """Prepare an image for comparison while preserving float precision."""
    if data.ndim != 2:
        raise ValueError(
            f"A imagem precisa ser 2D; dimensões encontradas: {data.shape}"
        )

    data_float = np.asarray(data, dtype=np.float32)

    if not np.isfinite(data_float).any():
        raise ValueError("A imagem não possui nenhum pixel finito.")

    if opt_method == "Crop":
        h, w = data_float.shape
        size = min(crop_size, h, w)

        # Garante uma região exatamente size x size.
        y1 = (h - size) // 2
        x1 = (w - size) // 2
        return data_float[y1 : y1 + size, x1 : x1 + size].copy()

    if opt_method != "Downsampling":
        raise ValueError(f"Método de otimização desconhecido: {opt_method}")

    new_w = max(1, round(data_float.shape[1] * downsample_scale))
    new_h = max(1, round(data_float.shape[0] * downsample_scale))

    # Pillow trabalha corretamente com arrays float32 no modo "F".
    image = Image.fromarray(data_float, mode="F")
    resized = image.resize(
        (new_w, new_h),
        resample=RESAMPLE_MODES[downsample_method],
    )
    return np.asarray(resized, dtype=np.float32)


def comparison_score(current: np.ndarray, previous: np.ndarray) -> float:
    """
    Robust global difference score.

    NaN/Inf pixels are ignored. A median offset is removed before computing
    the RMS difference, reducing false positives caused by small global
    brightness/background changes.
    """
    if current.shape != previous.shape:
        raise ValueError(
            f"Imagens incompatíveis para comparação: "
            f"{current.shape} vs {previous.shape}"
        )

    valid = np.isfinite(current) & np.isfinite(previous)
    if not np.any(valid):
        return math.nan

    diff = current[valid].astype(np.float32) - previous[valid].astype(np.float32)

    # Remove a variação global de nível de fundo/exposição.
    diff -= np.mean(diff)

    return float(np.sqrt(np.mean(np.square(diff), dtype=np.float64)))

def file_mover_worker(move_queue: queue.Queue, app_print, cancel_event: threading.Event):
    """Thread dedicada exclusivamente a operações de I/O de disco."""
    while True:
        item = move_queue.get()
        if item is None:
            break
            
        src, dst, action = item  # <-- Agora recebe a ação ('copy' ou 'move')
        
        if cancel_event.is_set():
            move_queue.task_done()
            continue
            
        try:
            if action == 'copy':
                shutil.copy2(str(src), str(dst))
            else:
                shutil.move(str(src), str(dst))
        except Exception as exc:
            verbo = "copiar" if action == 'copy' else "mover"
            app_print(f"Erro ao {verbo} {src.name}: {exc}\n")
        finally:
            move_queue.task_done()

def get_optimal_worker_count() -> int:
    """Escolhe automaticamente um número conservador de workers."""
    try:
        cpu_count = getattr(os, "process_cpu_count", os.cpu_count)() or 1
    except Exception:
        cpu_count = os.cpu_count() or 1

    # Até 16 workers: bom equilíbrio para FITS (I/O + NumPy/Pillow).
    # Em uma CPU com 16 threads lógicas, utiliza as 16.
    return max(1, min(16, cpu_count))


def prepare_fits_file(
    filepath: Path,
    config: ProcessingConfig,
) -> tuple[Path, np.ndarray | None, str | None]:
    """Lê e prepara um FITS. Não acessa Tkinter nem estado compartilhado."""
    try:
        with fits.open(filepath, memmap=False) as hdul:
            data = None

            for hdu in hdul:
                if not hdu.is_image:
                    continue

                hdu_data = hdu.data

                if hdu_data is None or hdu_data.ndim != 2:
                    continue

                data = np.asarray(hdu_data, dtype=np.float32)
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


def process_fits_logic(
    config: ProcessingConfig,
    app_print,
    cancel_event: threading.Event,
) -> tuple[int, int]:
    input_dir = config.input_dir
    output_dir = config.output_dir

    app_print(f"Lendo arquivos de: {input_dir}\n")

    files = find_fits_files(input_dir)
    if not files:
        app_print("Nenhum arquivo FITS encontrado no diretório.\n")
        return 0, 0

    app_print(f"Total de arquivos encontrados: {len(files)}\n")

    if not config.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    batch_num = 1
    current_batch_dir = output_dir / f"batch_{batch_num:03d}"

    if not config.dry_run:
        current_batch_dir.mkdir(parents=True, exist_ok=True)

    previous_data: np.ndarray | None = None
    score_history: deque[float] = deque(maxlen=10)
    processed = 0

    worker_count = get_optimal_worker_count()

    # Limita a quantidade de imagens preparadas simultaneamente.
    # Crop 1000x1000 float32 ~= 4 MB por frame; 2x workers mantém a fila
    # pequena sem deixar o produtor ocioso.
    prefetch_count = min(
        len(files),
        max(worker_count * 2, worker_count + 2),
    )

    app_print(
        f"\nPipeline paralelo: {worker_count} workers | "
        f"prefetch: {prefetch_count} frames\n"
    )
    
    # Inicialização da fila e thread de I/O Assíncrono
    move_queue = queue.Queue()
    mover_thread = threading.Thread(
        target=file_mover_worker,
        args=(move_queue, app_print, cancel_event),
        daemon=True
    )
    
    if not config.dry_run:
        mover_thread.start()
        
    app_print(f"Iniciando Batch {batch_num:03d}...\n")

    # Workers fazem somente I/O + preparação. A thread coordenadora mantém
    # comparação, detecção de batch e movimentação na ordem original.
    with ThreadPoolExecutor(
        max_workers=worker_count,
        thread_name_prefix="fits-prep",
    ) as executor:

        next_submit = 0
        next_process = 0
        futures = {}

        def submit_until_window_full() -> None:
            nonlocal next_submit

            while (
                next_submit < len(files)
                and len(futures) < prefetch_count
                and not cancel_event.is_set()
            ):
                filepath = files[next_submit]
                futures[next_submit] = executor.submit(
                    prepare_fits_file,
                    filepath,
                    config,
                )
                next_submit += 1

        submit_until_window_full()

        while next_process < len(files):
            if cancel_event.is_set():
                app_print("\nCancelamento solicitado...\n")
                break

            future = futures.pop(next_process)
            filepath = files[next_process]

            # Consome sempre na ordem. Enquanto este frame é aguardado,
            # os workers continuam preparando os próximos.
            try:
                _, prepared, error_message = future.result()
            except Exception as exc:
                prepared = None
                error_message = (
                    f"Erro inesperado no worker para {filepath.name}: {exc}"
                )

            next_process += 1

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
                                current_batch_dir = (
                                    output_dir / f"batch_{batch_num:03d}"
                                )

                                if not config.dry_run:
                                    current_batch_dir.mkdir(
                                        parents=True,
                                        exist_ok=True,
                                    )

                                app_print(
                                    "\n>>> MOVIMENTO BRUSCO DETECTADO <<<\n"
                                )
                                app_print(
                                    f"Arquivo: {filepath.name} | "
                                    f"Score: {score:.4f} | "
                                    f"Baseline: {baseline:.4f} | "
                                    f"Limite: {limit:.4f}\n"
                                )
                                app_print(
                                    f"Criando Batch {batch_num:03d}...\n"
                                )

                                score_history.clear()

                    score_history.append(score)

            previous_data = prepared

            if not config.dry_run:
                destination = current_batch_dir / filepath.name
                if destination.exists():
                    app_print(f"ERRO: destino já existe, arquivo não movido: {destination}\n")
                else:
                    action = 'copy' if config.copy_files else 'move'
                    move_queue.put((filepath, destination, action))

            processed += 1

            if next_process % 10 == 0 or next_process == len(files):
                app_print(
                    f"Processados {next_process}/{len(files)} arquivos "
                    f"(workers: {worker_count})...\n"
                )

            submit_until_window_full()
    if not config.dry_run:
        app_print("\nAguardando finalização das movimentações de disco...\n")
        move_queue.put(None) # Envia sinal de parada
        mover_thread.join()  # Aguarda a thread de I/O esvaziar a fila
        
    app_print(
        f"\nFinalizado! {processed} arquivos processados em "
        f"{batch_num} batches.\n"
    )

    return processed, batch_num


class AstroBatchApp(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Astro Batch Separator")
        self.geometry("820x720")
        self.minsize(720, 600)
        self.configure(padx=16, pady=16)

        self.input_dir_var = tk.StringVar()
        self.output_dir_var = tk.StringVar()
        self.opt_method_var = tk.StringVar(value="Crop")
        self.crop_size_var = tk.IntVar(value=1000)
        self.downsample_method_var = tk.StringVar(value="Nearest")
        self.downsample_scale_var = tk.DoubleVar(value=0.25)
        self.threshold_var = tk.DoubleVar(value=3.0)
        self.dry_run_var = tk.BooleanVar(value=True)
        self.status_var = tk.StringVar(value="Pronto.")
        self.copy_files_var = tk.BooleanVar(value=False)
        
        self.load_settings()

        # Comunicação worker -> GUI. A thread de processamento nunca chama Tk.
        self.log_queue: queue.Queue[str] = queue.Queue()
        self.worker: threading.Thread | None = None
        self.cancel_event = threading.Event()

        self.create_widgets()
        self.after(50, self._drain_log_queue)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def load_settings(self):
        """Carrega as configurações salvas em disco, se existirem."""
        config_file = Path("astrobatch_config.json")
        if config_file.exists():
            try:
                with open(config_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    
                if "input_dir" in data: self.input_dir_var.set(data["input_dir"])
                if "output_dir" in data: self.output_dir_var.set(data["output_dir"])
                if "opt_method" in data: self.opt_method_var.set(data["opt_method"])
                if "crop_size" in data: self.crop_size_var.set(data["crop_size"])
                if "downsample_method" in data: self.downsample_method_var.set(data["downsample_method"])
                if "downsample_scale" in data: self.downsample_scale_var.set(data["downsample_scale"])
                if "threshold_factor" in data: self.threshold_var.set(data["threshold_factor"])
                if "dry_run" in data: self.dry_run_var.set(data["dry_run"])
                if "copy_files" in data: self.copy_files_var.set(data["copy_files"])
            except Exception as exc:
                self.print_to_console(f"Aviso: Erro ao carregar configurações prévias: {exc}\n")

    def save_settings(self):
        """Salva as configurações atuais no disco."""
        config_file = Path("astrobatch_config.json")
        data = {
            "input_dir": self.input_dir_var.get(),
            "output_dir": self.output_dir_var.get(),
            "opt_method": self.opt_method_var.get(),
            "crop_size": self.crop_size_var.get(),
            "downsample_method": self.downsample_method_var.get(),
            "downsample_scale": self.downsample_scale_var.get(),
            "threshold_factor": self.threshold_var.get(),
            "dry_run": self.dry_run_var.get(),
            "copy_files": self.copy_files_var.get(),
        }
        try:
            with open(config_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4)
        except Exception as exc:
            self.print_to_console(f"Aviso: Erro ao salvar configurações: {exc}\n")
            
    def create_widgets(self):
        self.columnconfigure(0, weight=1)
        self.rowconfigure(4, weight=1)

        dir_frame = ttk.LabelFrame(self, text="Diretórios", padding=10)
        dir_frame.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        dir_frame.columnconfigure(1, weight=1)

        ttk.Label(dir_frame, text="Pasta Origem:").grid(
            row=0, column=0, sticky="w"
        )
        ttk.Entry(
            dir_frame, textvariable=self.input_dir_var
        ).grid(row=0, column=1, padx=6, sticky="ew")
        ttk.Button(
            dir_frame,
            text="Procurar...",
            command=lambda: self.browse_dir(self.input_dir_var),
        ).grid(row=0, column=2)

        ttk.Label(dir_frame, text="Pasta Destino:").grid(
            row=1, column=0, sticky="w", pady=(8, 0)
        )
        ttk.Entry(
            dir_frame, textvariable=self.output_dir_var
        ).grid(row=1, column=1, padx=6, pady=(8, 0), sticky="ew")
        ttk.Button(
            dir_frame,
            text="Procurar...",
            command=lambda: self.browse_dir(self.output_dir_var),
        ).grid(row=1, column=2, pady=(8, 0))

        opt_frame = ttk.LabelFrame(
            self, text="Otimização (Análise de Imagem)", padding=10
        )
        opt_frame.grid(row=1, column=0, sticky="ew", pady=(0, 10))
        opt_frame.columnconfigure(1, weight=1)

        ttk.Radiobutton(
            opt_frame,
            text="Recorte Central (Crop)",
            variable=self.opt_method_var,
            value="Crop",
            command=self.toggle_opt_options,
        ).grid(row=0, column=0, sticky="w")

        ttk.Radiobutton(
            opt_frame,
            text="Downsampling (Resize)",
            variable=self.opt_method_var,
            value="Downsampling",
            command=self.toggle_opt_options,
        ).grid(row=1, column=0, sticky="w")

        self.crop_frame = ttk.Frame(opt_frame)
        self.crop_frame.grid(row=0, column=1, sticky="ew", padx=(20, 0))
        self.crop_entry = ttk.Entry(
            self.crop_frame, textvariable=self.crop_size_var, width=10
        )
        ttk.Label(
            self.crop_frame, text="Tamanho do recorte (px):"
        ).pack(side="left")
        self.crop_entry.pack(side="left", padx=5)

        self.down_frame = ttk.Frame(opt_frame)
        self.down_frame.grid(row=1, column=1, sticky="ew", padx=(20, 0))

        ttk.Label(self.down_frame, text="Método:").pack(side="left")
        self.down_combo = ttk.Combobox(
            self.down_frame,
            textvariable=self.downsample_method_var,
            values=list(RESAMPLE_MODES),
            width=10,
            state="readonly",
        )
        self.down_combo.pack(side="left", padx=5)

        ttk.Label(
            self.down_frame, text="Escala (ex: 0.25 = 25%):"
        ).pack(side="left", padx=(10, 0))
        self.down_scale_entry = ttk.Entry(
            self.down_frame, textvariable=self.downsample_scale_var, width=8
        )
        self.down_scale_entry.pack(side="left", padx=5)

        param_frame = ttk.LabelFrame(
            self, text="Parâmetros de Detecção", padding=10
        )
        param_frame.grid(row=2, column=0, sticky="ew", pady=(0, 10))

        ttk.Label(
            param_frame, text="Threshold (fator):"
        ).grid(row=0, column=0, sticky="w")
        ttk.Entry(
            param_frame, textvariable=self.threshold_var, width=10
        ).grid(row=0, column=1, padx=6, sticky="w")

        ttk.Checkbutton(
            param_frame,
            text="Copiar arquivos em vez de mover",
            variable=self.copy_files_var,
        ).grid(row=1, column=0, columnspan=2, sticky="w", pady=(8, 0))

        ttk.Checkbutton(
            param_frame,
            text="Dry-Run (simular sem mover/copiar arquivos)",
            variable=self.dry_run_var,
        ).grid(row=2, column=0, columnspan=2, sticky="w", pady=(8, 0))

        action_frame = ttk.Frame(self)
        action_frame.grid(row=3, column=0, sticky="ew", pady=(0, 10))
        action_frame.columnconfigure(0, weight=1)

        self.run_btn = ttk.Button(
            action_frame,
            text="▶ INICIAR PROCESSAMENTO",
            command=self.start_processing,
        )
        self.run_btn.grid(row=0, column=0, sticky="ew", ipady=6)

        self.cancel_btn = ttk.Button(
            action_frame,
            text="■ CANCELAR",
            command=self.cancel_processing,
            state="disabled",
        )
        self.cancel_btn.grid(row=0, column=1, padx=(8, 0), ipady=6)

        ttk.Label(self, textvariable=self.status_var).grid(
            row=4, column=0, sticky="sw", pady=(0, 4)
        )

        console_frame = ttk.LabelFrame(self, text="Console de Saída", padding=5)
        console_frame.grid(row=5, column=0, sticky="nsew")
        console_frame.columnconfigure(0, weight=1)
        console_frame.rowconfigure(0, weight=1)

        self.rowconfigure(5, weight=1)

        self.console_text = scrolledtext.ScrolledText(
            console_frame,
            height=12,
            state=tk.DISABLED,
            font=("Consolas", 10),
        )
        self.console_text.grid(row=0, column=0, sticky="nsew")

        self.toggle_opt_options()

    def browse_dir(self, var: tk.StringVar):
        folder = filedialog.askdirectory()
        if folder:
            var.set(folder)

    def toggle_opt_options(self):
        is_crop = self.opt_method_var.get() == "Crop"

        self.crop_entry.configure(state="normal" if is_crop else "disabled")
        self.down_combo.configure(
            state="readonly" if not is_crop else "disabled"
        )
        self.down_scale_entry.configure(
            state="normal" if not is_crop else "disabled"
        )

    def print_to_console(self, text: str):
        """Thread-safe: only puts text into a queue."""
        self.log_queue.put(text)

    def _drain_log_queue(self):
        try:
            while True:
                text = self.log_queue.get_nowait()
                self.console_text.configure(state=tk.NORMAL)
                self.console_text.insert(tk.END, text)
                self.console_text.see(tk.END)
                self.console_text.configure(state=tk.DISABLED)
        except queue.Empty:
            pass
        finally:
            self.after(50, self._drain_log_queue)

    def _validate_config(self) -> ProcessingConfig | None:
        try:
            input_dir = Path(self.input_dir_var.get()).expanduser().resolve()
            output_dir = Path(self.output_dir_var.get()).expanduser().resolve()

            if not input_dir.is_dir():
                raise ValueError("A pasta de origem não existe.")

            if input_dir == output_dir:
                raise ValueError(
                    "A pasta de destino deve ser diferente da pasta de origem."
                )

            threshold = float(self.threshold_var.get())
            crop_size = int(self.crop_size_var.get())
            scale = float(self.downsample_scale_var.get())

            if not math.isfinite(threshold) or threshold <= 0:
                raise ValueError("O threshold deve ser maior que zero.")

            if crop_size <= 0:
                raise ValueError("O tamanho do crop deve ser maior que zero.")

            if not 0 < scale < 1:
                raise ValueError(
                    "A escala de downsampling deve estar entre 0 e 1."
                )

            return ProcessingConfig(
                input_dir=input_dir,
                output_dir=output_dir,
                threshold_factor=threshold,
                crop_size=crop_size,
                dry_run=self.dry_run_var.get(),
                copy_files=self.copy_files_var.get(),
                opt_method=self.opt_method_var.get(),
                downsample_method=self.downsample_method_var.get(),
                downsample_scale=scale,
            )

        except (TypeError, ValueError, tk.TclError) as exc:
            messagebox.showerror("Parâmetros inválidos", str(exc))
            return None

    def start_processing(self):
        if self.worker and self.worker.is_alive():
            return

        config = self._validate_config()
        if config is None:
            return

        self.save_settings()
        
        self.console_text.configure(state=tk.NORMAL)
        self.console_text.delete("1.0", tk.END)
        self.console_text.configure(state=tk.DISABLED)

        # Todos os valores Tk são lidos na thread principal ANTES de criar
        # a worker thread.
        self.cancel_event.clear()
        self.run_btn.configure(state="disabled")
        self.cancel_btn.configure(state="normal")
        self.status_var.set("Processando...")

        self.worker = threading.Thread(
            target=self.run_logic,
            args=(config,),
            name="fits-worker",
            daemon=True,
        )
        self.worker.start()

    def cancel_processing(self):
        if self.worker and self.worker.is_alive():
            self.cancel_event.set()
            self.status_var.set("Cancelando...")

    def run_logic(self, config: ProcessingConfig):
        try:
            processed, batches = process_fits_logic(
                config,
                self.print_to_console,
                self.cancel_event,
            )

            if self.cancel_event.is_set():
                status = f"Cancelado. {processed} arquivos processados."
            else:
                status = (
                    f"Concluído. {processed} arquivos processados, "
                    f"{batches} batches."
                )

            self.after(0, lambda: self.status_var.set(status))

        except Exception as exc:
            self.print_to_console(f"\nERRO FATAL: {exc}\n")
            self.after(0, lambda: self.status_var.set("Erro durante o processamento."))

        finally:
            self.after(0, self._processing_finished)

    def _processing_finished(self):
        self.run_btn.configure(state="normal")
        self.cancel_btn.configure(state="disabled")

    def _on_close(self):
        if self.worker and self.worker.is_alive():
            if not messagebox.askyesno(
                "Processamento em andamento",
                "O processamento ainda está em andamento. "
                "Deseja sair mesmo assim?",
            ):
                return

            self.cancel_event.set()
            
        self.save_settings()
        self.destroy()


if __name__ == "__main__":
    app = AstroBatchApp()
    app.mainloop()
