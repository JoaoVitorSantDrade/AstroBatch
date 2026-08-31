# main.py
import json
import math
import queue
import threading
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk

# Importa a lógica de negócio do arquivo separado
from batch_logic import ProcessingConfig, process_fits_logic, RESAMPLE_MODES

class AstroBatchApp(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Astro Process Manager")
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
        self.copy_files_var = tk.BooleanVar(value=False)
        self.status_var = tk.StringVar(value="Pronto.")

        self.load_settings()

        self.log_queue: queue.Queue[str] = queue.Queue()
        self.worker: threading.Thread | None = None
        self.cancel_event = threading.Event()

        self.create_widgets()
        self.after(50, self._drain_log_queue)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def load_settings(self):
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

        ttk.Label(dir_frame, text="Pasta Origem:").grid(row=0, column=0, sticky="w")
        ttk.Entry(dir_frame, textvariable=self.input_dir_var).grid(row=0, column=1, padx=6, sticky="ew")
        ttk.Button(dir_frame, text="Procurar...", command=lambda: self.browse_dir(self.input_dir_var)).grid(row=0, column=2)

        ttk.Label(dir_frame, text="Pasta Destino:").grid(row=1, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(dir_frame, textvariable=self.output_dir_var).grid(row=1, column=1, padx=6, pady=(8, 0), sticky="ew")
        ttk.Button(dir_frame, text="Procurar...", command=lambda: self.browse_dir(self.output_dir_var)).grid(row=1, column=2, pady=(8, 0))

        opt_frame = ttk.LabelFrame(self, text="Otimização (Análise de Imagem)", padding=10)
        opt_frame.grid(row=1, column=0, sticky="ew", pady=(0, 10))
        opt_frame.columnconfigure(1, weight=1)

        ttk.Radiobutton(opt_frame, text="Recorte Central (Crop)", variable=self.opt_method_var, value="Crop", command=self.toggle_opt_options).grid(row=0, column=0, sticky="w")
        ttk.Radiobutton(opt_frame, text="Downsampling (Resize)", variable=self.opt_method_var, value="Downsampling", command=self.toggle_opt_options).grid(row=1, column=0, sticky="w")

        self.crop_frame = ttk.Frame(opt_frame)
        self.crop_frame.grid(row=0, column=1, sticky="ew", padx=(20, 0))
        self.crop_entry = ttk.Entry(self.crop_frame, textvariable=self.crop_size_var, width=10)
        ttk.Label(self.crop_frame, text="Tamanho do recorte (px):").pack(side="left")
        self.crop_entry.pack(side="left", padx=5)

        self.down_frame = ttk.Frame(opt_frame)
        self.down_frame.grid(row=1, column=1, sticky="ew", padx=(20, 0))
        ttk.Label(self.down_frame, text="Método:").pack(side="left")
        self.down_combo = ttk.Combobox(self.down_frame, textvariable=self.downsample_method_var, values=list(RESAMPLE_MODES), width=10, state="readonly")
        self.down_combo.pack(side="left", padx=5)
        ttk.Label(self.down_frame, text="Escala (ex: 0.25 = 25%):").pack(side="left", padx=(10, 0))
        self.down_scale_entry = ttk.Entry(self.down_frame, textvariable=self.downsample_scale_var, width=8)
        self.down_scale_entry.pack(side="left", padx=5)

        param_frame = ttk.LabelFrame(self, text="Parâmetros de Detecção", padding=10)
        param_frame.grid(row=2, column=0, sticky="ew", pady=(0, 10))

        ttk.Label(param_frame, text="Threshold (fator):").grid(row=0, column=0, sticky="w")
        ttk.Entry(param_frame, textvariable=self.threshold_var, width=10).grid(row=0, column=1, padx=6, sticky="w")

        ttk.Checkbutton(param_frame, text="Copiar arquivos em vez de mover", variable=self.copy_files_var).grid(row=1, column=0, columnspan=2, sticky="w", pady=(8, 0))
        ttk.Checkbutton(param_frame, text="Dry-Run (simular sem mover/copiar arquivos)", variable=self.dry_run_var).grid(row=2, column=0, columnspan=2, sticky="w", pady=(8, 0))

        action_frame = ttk.Frame(self)
        action_frame.grid(row=3, column=0, sticky="ew", pady=(0, 10))
        action_frame.columnconfigure(0, weight=1)

        self.run_btn = ttk.Button(action_frame, text="▶ INICIAR PROCESSAMENTO", command=self.start_processing)
        self.run_btn.grid(row=0, column=0, sticky="ew", ipady=6)
        self.cancel_btn = ttk.Button(action_frame, text="■ CANCELAR", command=self.cancel_processing, state="disabled")
        self.cancel_btn.grid(row=0, column=1, padx=(8, 0), ipady=6)

        ttk.Label(self, textvariable=self.status_var).grid(row=4, column=0, sticky="sw", pady=(0, 4))

        console_frame = ttk.LabelFrame(self, text="Console de Saída", padding=5)
        console_frame.grid(row=5, column=0, sticky="nsew")
        console_frame.columnconfigure(0, weight=1)
        console_frame.rowconfigure(0, weight=1)
        self.rowconfigure(5, weight=1)

        self.console_text = scrolledtext.ScrolledText(console_frame, height=12, state=tk.DISABLED, font=("Consolas", 10))
        self.console_text.grid(row=0, column=0, sticky="nsew")

        self.toggle_opt_options()

    def browse_dir(self, var: tk.StringVar):
        folder = filedialog.askdirectory()
        if folder: var.set(folder)

    def toggle_opt_options(self):
        is_crop = self.opt_method_var.get() == "Crop"
        self.crop_entry.configure(state="normal" if is_crop else "disabled")
        self.down_combo.configure(state="readonly" if not is_crop else "disabled")
        self.down_scale_entry.configure(state="normal" if not is_crop else "disabled")

    def print_to_console(self, text: str):
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

            if not input_dir.is_dir(): raise ValueError("A pasta de origem não existe.")
            if input_dir == output_dir: raise ValueError("A pasta de destino deve ser diferente.")

            threshold = float(self.threshold_var.get())
            crop_size = int(self.crop_size_var.get())
            scale = float(self.downsample_scale_var.get())

            if not math.isfinite(threshold) or threshold <= 0: raise ValueError("Threshold > 0.")
            if crop_size <= 0: raise ValueError("Crop size > 0.")
            if not 0 < scale < 1: raise ValueError("Scale entre 0 e 1.")

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
        if self.worker and self.worker.is_alive(): return
        config = self._validate_config()
        if config is None: return
        
        self.save_settings()
        self.console_text.configure(state=tk.NORMAL)
        self.console_text.delete("1.0", tk.END)
        self.console_text.configure(state=tk.DISABLED)

        self.cancel_event.clear()
        self.run_btn.configure(state="disabled")
        self.cancel_btn.configure(state="normal")
        self.status_var.set("Processando...")

        self.worker = threading.Thread(target=self.run_logic, args=(config,), daemon=True)
        self.worker.start()

    def cancel_processing(self):
        if self.worker and self.worker.is_alive():
            self.cancel_event.set()
            self.status_var.set("Cancelando...")

    def run_logic(self, config: ProcessingConfig):
        try:
            processed, batches = process_fits_logic(config, self.print_to_console, self.cancel_event)
            status = f"Cancelado. {processed} arquivos processados." if self.cancel_event.is_set() else f"Concluído. {processed} processados, {batches} batches."
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
            if not messagebox.askyesno("Processamento em andamento", "Processamento ativo. Sair mesmo assim?"):
                return
            self.cancel_event.set()
        self.save_settings()
        self.destroy()

if __name__ == "__main__":
    app = AstroBatchApp()
    app.mainloop()