import json
import queue
import threading
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk
import cv2


# Importações dos módulos lógicos
from batch_logic import ProcessingConfig, RESAMPLE_MODES
from astroalign_logic import process_all_alignments, INTERPOLATION_MODES

class AstroProcessManager(tk.Tk):
    def __init__(self):
        super().__init__()
        self.batch_dir_var = tk.StringVar()
        self.batch_dir_var.trace_add("write", self._update_global_master_options) # Adicione esta linha
        self.status_var = tk.StringVar(value="Pronto.")
        self.title("Astro Process Manager")
        self.geometry("850x750")
        self.minsize(750, 650)
        self.configure(padx=16, pady=16)

        self._init_variables()
        self.load_settings()

        self.log_queue: queue.Queue[str] = queue.Queue()
        self.worker: threading.Thread | None = None
        self.cancel_event = threading.Event()

        self.create_widgets()
        self.after(50, self._drain_log_queue)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _init_variables(self):
        self.batch_dir_var = tk.StringVar()
        self.status_var = tk.StringVar(value="Pronto.")
        self.progress_var = tk.DoubleVar(value=0.0)
        
        # Parâmetros AstroBatch
        self.batch_input_dir_var = tk.StringVar()
        self.opt_method_var = tk.StringVar(value="Crop")
        self.crop_size_var = tk.IntVar(value=1000)
        self.downsample_method_var = tk.StringVar(value="Nearest")
        self.downsample_scale_var = tk.DoubleVar(value=0.25)
        self.threshold_var = tk.DoubleVar(value=3.0)
        self.dry_run_var = tk.BooleanVar(value=True)
        self.copy_files_var = tk.BooleanVar(value=False)
        self.batch_overwrite_var = tk.BooleanVar(value=False)

        # Parâmetros AstroFlow
        self.flow_global_master_var = tk.StringVar(value="Auto")
        self.flow_fwhm_var = tk.DoubleVar(value=4.0)
        self.flow_sigma_var = tk.DoubleVar(value=5.0)
        self.flow_matching_radius_var = tk.IntVar(value=15)
        self.flow_ransac_var = tk.DoubleVar(value=3.0)
        self.flow_debug_var = tk.BooleanVar(value=False)
        self.flow_min_stars_var = tk.IntVar(value=4)
        self.flow_min_inliers_var = tk.IntVar(value=4)
        self.flow_min_ratio_var = tk.DoubleVar(value=0.15)
        self.flow_engine_var = tk.StringVar(value="DAO")

        # Parâmetros AstroAlign
        self.align_output_dir_var = tk.StringVar()
        self.align_interpolation_var = tk.StringVar(value="Lanczos")
        self.align_overwrite_var = tk.BooleanVar(value=False)
        self.align_dry_run_var = tk.BooleanVar(value=False)
        self.align_keep_header_var = tk.BooleanVar(value=True)
        
        # Parâmetros AstroAlign
        self.align_output_dir_var = tk.StringVar()
        self.align_interpolation_var = tk.StringVar(value="Lanczos")
        self.align_overwrite_var = tk.BooleanVar(value=False)
        self.align_dry_run_var = tk.BooleanVar(value=False)
        self.align_keep_header_var = tk.BooleanVar(value=True)

        self.config_registry = {
            "Global": {"batch_dir": self.batch_dir_var},
            "AstroBatch": {
                "input_dir": self.batch_input_dir_var,
                "opt_method": self.opt_method_var,
                "crop_size": self.crop_size_var,
                "downsample_method": self.downsample_method_var,
                "downsample_scale": self.downsample_scale_var,
                "threshold_factor": self.threshold_var,
                "dry_run": self.dry_run_var,
                "copy_files": self.copy_files_var,
                "overwrite": self.batch_overwrite_var
            },
            "AstroFlow": {
                "global_master": self.flow_global_master_var,
                "fwhm": self.flow_fwhm_var,
                "sigma": self.flow_sigma_var,
                "matching_radius": self.flow_matching_radius_var,
                "ransac": self.flow_ransac_var,
                "min_stars": self.flow_min_stars_var,
                "min_inliers": self.flow_min_inliers_var,
                "min_ratio": self.flow_min_ratio_var,
                "debug_images": self.flow_debug_var,
                "engine": self.flow_engine_var
            },
            "AstroAlign": {
                "output_dir": self.align_output_dir_var,
                "interpolation": self.align_interpolation_var,
                "overwrite": self.align_overwrite_var,
                "dry_run": self.align_dry_run_var,
                "keep_header": self.align_keep_header_var
            }
        }

    def load_settings(self):
        config_file = Path("astro_config.json")
        if not config_file.exists(): return
        try:
            with open(config_file, "r") as f: data = json.load(f)
            for mod, variables in self.config_registry.items():
                if mod in data:
                    for key, var in variables.items():
                        if key in data[mod]: var.set(data[mod][key])
        except Exception: pass

    def save_settings(self):
        config_file = Path("astro_config.json")
        data = {mod: {k: v.get() for k, v in vars_.items()} for mod, vars_ in self.config_registry.items()}
        try:
            with open(config_file, "w") as f: json.dump(data, f, indent=4)
        except Exception: pass

    def create_widgets(self):
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        self.notebook = ttk.Notebook(self)
        self.notebook.grid(row=0, column=0, sticky="nsew", pady=(0, 10))

        self.tab_batch = ttk.Frame(self.notebook, padding=10)
        self.tab_flow = ttk.Frame(self.notebook, padding=10)
        self.tab_align = ttk.Frame(self.notebook, padding=10)
        
        self.notebook.add(self.tab_batch, text="1. AstroBatch")
        self.notebook.add(self.tab_flow, text="2. AstroFlow")
        self.notebook.add(self.tab_align, text="3. AstroAlign")

        self._build_tab_batch()
        self._build_tab_flow()
        self._build_tab_align()

        # --- ÁREA COMUM INFERIOR ---
        bottom_frame = ttk.Frame(self)
        bottom_frame.grid(row=1, column=0, sticky="nsew")
        bottom_frame.columnconfigure(0, weight=1)
        bottom_frame.rowconfigure(1, weight=1)

        status_frame = ttk.Frame(bottom_frame)
        status_frame.grid(row=0, column=0, sticky="ew", pady=(0, 4))
        status_frame.columnconfigure(1, weight=1)

        ttk.Label(status_frame, textvariable=self.status_var, font=("Segoe UI", 9, "bold")).grid(row=0, column=0, sticky="w")
        
        self.progress_bar = ttk.Progressbar(status_frame, variable=self.progress_var, maximum=100.0)
        self.progress_bar.grid(row=0, column=1, sticky="ew", padx=(10, 0))

        console_frame = ttk.LabelFrame(bottom_frame, text="Console de Saída Compartilhado", padding=5)
        console_frame.grid(row=1, column=0, sticky="nsew")
        console_frame.columnconfigure(0, weight=1)
        console_frame.rowconfigure(0, weight=1)

        self.console_text = scrolledtext.ScrolledText(console_frame, height=12, state=tk.DISABLED, font=("Consolas", 10), bg="#1e1e1e", fg="#d4d4d4")
        self.console_text.grid(row=0, column=0, sticky="nsew")
        self.rowconfigure(1, weight=1)

    def update_progress(self, current: int, total: int, phase_text: str = ""):
        """Callback genérico e thread-safe para atualizar a barra de progresso e o status."""
        if total > 0:
            pct = (current / total) * 100.0
            self.after(0, lambda: self.progress_var.set(pct))
        else:
            self.after(0, lambda: self.progress_var.set(0.0))
            
        if phase_text:
            self.after(0, lambda: self.status_var.set(phase_text))
            
    def _build_tab_batch(self):
        frame = self.tab_batch
        frame.columnconfigure(0, weight=1)

        dir_frame = ttk.LabelFrame(frame, text="Diretórios", padding=10)
        dir_frame.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        dir_frame.columnconfigure(1, weight=1)

        ttk.Label(dir_frame, text="Pasta Origem (Imagens Brutas):").grid(row=0, column=0, sticky="w")
        ttk.Entry(dir_frame, textvariable=self.batch_input_dir_var).grid(row=0, column=1, padx=6, sticky="ew")
        ttk.Button(dir_frame, text="Procurar...", command=lambda: self.browse_dir(self.batch_input_dir_var)).grid(row=0, column=2)

        ttk.Label(dir_frame, text="Pasta Destino (Pasta de Batches):").grid(row=1, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(dir_frame, textvariable=self.batch_dir_var).grid(row=1, column=1, padx=6, pady=(8, 0), sticky="ew")
        ttk.Button(dir_frame, text="Procurar...", command=lambda: self.browse_dir(self.batch_dir_var)).grid(row=1, column=2, pady=(8, 0))

        opt_frame = ttk.LabelFrame(frame, text="Otimização (Análise de Imagem)", padding=10)
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
        self.toggle_opt_options()

        param_frame = ttk.LabelFrame(frame, text="Parâmetros de Detecção", padding=10)
        param_frame.grid(row=2, column=0, sticky="ew", pady=(0, 10))

        ttk.Label(param_frame, text="Threshold (fator):").grid(row=0, column=0, sticky="w")
        ttk.Entry(param_frame, textvariable=self.threshold_var, width=10).grid(row=0, column=1, padx=6, sticky="w")
        ttk.Checkbutton(param_frame, text="Copiar arquivos em vez de mover", variable=self.copy_files_var).grid(row=1, column=0, columnspan=2, sticky="w", pady=(8, 0))
        ttk.Checkbutton(param_frame, text="Sobrescrever arquivos existentes no destino", variable=self.batch_overwrite_var).grid(row=2, column=0, columnspan=2, sticky="w", pady=(8, 0))
        ttk.Checkbutton(param_frame, text="Dry-Run (simular sem mover/copiar arquivos)", variable=self.dry_run_var).grid(row=3, column=0, columnspan=2, sticky="w", pady=(8, 0))
        
        action_frame = ttk.Frame(frame)
        action_frame.grid(row=3, column=0, sticky="ew", pady=(0, 10))
        action_frame.columnconfigure(0, weight=1)

        self.btn_run_batch = ttk.Button(action_frame, text="▶ INICIAR ASTROBATCH", command=self.start_batch_processing)
        self.btn_run_batch.grid(row=0, column=0, sticky="ew", ipady=6)
        self.btn_cancel_batch = ttk.Button(action_frame, text="■ CANCELAR", command=self.cancel_processing, state="disabled")
        self.btn_cancel_batch.grid(row=0, column=1, padx=(8, 0), ipady=6)

    def _build_tab_flow(self):
        frame = self.tab_flow
        frame.columnconfigure(0, weight=1)

        dir_frame = ttk.LabelFrame(frame, text="Diretório de Batches", padding=10)
        dir_frame.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        dir_frame.columnconfigure(1, weight=1)
        
        ttk.Label(dir_frame, text="Pasta Base:").grid(row=0, column=0, sticky="w")
        ttk.Entry(dir_frame, textvariable=self.batch_dir_var).grid(row=0, column=1, padx=6, sticky="ew")
        ttk.Button(dir_frame, text="Procurar...", command=lambda: self.browse_dir(self.batch_dir_var)).grid(row=0, column=2)

        param_frame = ttk.LabelFrame(frame, text="Parâmetros", padding=10)
        param_frame.grid(row=1, column=0, sticky="ew", pady=(0, 10))
        
        ttk.Label(param_frame, text="Global Master:").grid(row=0, column=0, sticky="w")
        self.combo_global_master = ttk.Combobox(
            param_frame, 
            textvariable=self.flow_global_master_var, 
            values=["Auto"], 
            width=13, 
            state="readonly"
        )
        self.combo_global_master.grid(row=0, column=1, padx=6, sticky="w")
        
        ttk.Label(param_frame, text="Star Engine:").grid(row=0, column=2, sticky="w")
        ttk.Combobox(param_frame, textvariable=self.flow_engine_var, values= ["DAO","OpenCV"], width=13, state="readonly").grid(row=0, column=3, sticky="w")
        
        ttk.Label(param_frame, text="FWHM Médio (px):").grid(row=1, column=0, sticky="w", pady=(4,0))
        ttk.Entry(param_frame, textvariable=self.flow_fwhm_var, width=15).grid(row=1, column=1, padx=6, sticky="w", pady=(4,0))
        
        ttk.Label(param_frame, text="Sensibilidade (Sigma Threshold):").grid(row=2, column=0, sticky="w", pady=(4,0))
        ttk.Entry(param_frame, textvariable=self.flow_sigma_var, width=15).grid(row=2, column=1, padx=6, sticky="w", pady=(4,0))

        ttk.Label(param_frame, text="Raio de Pareamento (px):").grid(row=3, column=0, sticky="w", pady=(4,0))
        ttk.Entry(param_frame, textvariable=self.flow_matching_radius_var, width=15).grid(row=3, column=1, padx=6, sticky="w", pady=(4,0))

        ttk.Label(param_frame, text="Tolerância de Reprojeção RANSAC:").grid(row=4, column=0, sticky="w", pady=(4,0))
        ttk.Entry(param_frame, textvariable=self.flow_ransac_var, width=15).grid(row=4, column=1, padx=6, sticky="w", pady=(4,0))
        
        ttk.Label(param_frame, text="Mínimo de Estrelas (Min Stars):").grid(row=5, column=0, sticky="w", pady=(4,0))
        ttk.Entry(param_frame, textvariable=self.flow_min_stars_var, width=15).grid(row=5, column=1, padx=6, sticky="w", pady=(4,0))

        ttk.Label(param_frame, text="Mínimo de Inliers RANSAC:").grid(row=6, column=0, sticky="w", pady=(4,0))
        ttk.Entry(param_frame, textvariable=self.flow_min_inliers_var, width=15).grid(row=6, column=1, padx=6, sticky="w", pady=(4,0))

        ttk.Label(param_frame, text="Ratio Mínimo de Inliers (ex: 0.15):").grid(row=7, column=0, sticky="w", pady=(4,0))
        ttk.Entry(param_frame, textvariable=self.flow_min_ratio_var, width=15).grid(row=7, column=1, padx=6, sticky="w", pady=(4,0))

        ttk.Checkbutton(param_frame, text="Gerar Imagens de Diagnóstico (.jpg) nas âncoras", variable=self.flow_debug_var).grid(row=8, column=0, columnspan=2, sticky="w", pady=(8, 0))

        action_frame = ttk.Frame(frame)
        action_frame.grid(row=2, column=0, sticky="ew", pady=(0, 10))
        action_frame.columnconfigure(0, weight=1)
        
        self.btn_preview_flow = ttk.Button(action_frame, text="🔍 PREVIEW DA ÂNCORA", command=self.show_astroflow_preview)
        self.btn_preview_flow.grid(row=0, column=0, sticky="ew", ipady=6)
        
        self.btn_run_flow = ttk.Button(action_frame, text="▶ INICIAR ASTROFLOW", command=self.start_flow_processing)
        self.btn_run_flow.grid(row=0, column=1, sticky="ew", ipady=6)
        
        self.btn_viz_flow = ttk.Button(action_frame, text="📈 VISUALIZAR FLOW", command=self.show_flow_visualization)
        self.btn_viz_flow.grid(row=0, column=2, padx=8, sticky="ew", ipady=6)

        self.btn_cancel_flow = ttk.Button(action_frame, text="■ CANCELAR", command=self.cancel_processing, state="disabled")
        self.btn_cancel_flow.grid(row=0, column=3, padx=(0, 0), ipady=6)

    def _build_tab_align(self):
        """Constrói a interface da aba AstroAlign."""
        frame = self.tab_align
        frame.columnconfigure(0, weight=1)

        dir_frame = ttk.LabelFrame(frame, text="Diretórios de Alinhamento", padding=10)
        dir_frame.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        dir_frame.columnconfigure(1, weight=1)

        ttk.Label(dir_frame, text="Pasta Base (Onde está o global_flow.json):").grid(row=0, column=0, sticky="w")
        ttk.Entry(dir_frame, textvariable=self.batch_dir_var).grid(row=0, column=1, padx=6, sticky="ew")
        ttk.Button(dir_frame, text="Procurar...", command=lambda: self.browse_dir(self.batch_dir_var)).grid(row=0, column=2)

        ttk.Label(dir_frame, text="Pasta Destino (Frames Alinhados):").grid(row=1, column=0, sticky="w", pady=(8, 0))
        ttk.Entry(dir_frame, textvariable=self.align_output_dir_var).grid(row=1, column=1, padx=6, pady=(8, 0), sticky="ew")
        ttk.Button(dir_frame, text="Procurar...", command=lambda: self.browse_dir(self.align_output_dir_var)).grid(row=1, column=2, pady=(8, 0))

        param_frame = ttk.LabelFrame(frame, text="Parâmetros de Warping", padding=10)
        param_frame.grid(row=1, column=0, sticky="ew", pady=(0, 10))
        param_frame.columnconfigure(1, weight=1)

        ttk.Label(param_frame, text="Método de Interpolação:").grid(row=0, column=0, sticky="w")
        interp_combo = ttk.Combobox(
            param_frame, 
            textvariable=self.align_interpolation_var, 
            values=["Nearest", "Bilinear", "Bicubic", "Lanczos"], 
            width=15, 
            state="readonly"
        )
        interp_combo.grid(row=0, column=1, padx=6, sticky="w")

        ttk.Checkbutton(param_frame, text="Preservar Header FITS original", variable=self.align_keep_header_var).grid(row=1, column=0, columnspan=2, sticky="w", pady=(8, 0))
        ttk.Checkbutton(param_frame, text="Sobrescrever arquivos alinhados existentes no destino", variable=self.align_overwrite_var).grid(row=2, column=0, columnspan=2, sticky="w", pady=(8, 0))
        ttk.Checkbutton(param_frame, text="Dry-Run (simular alinhamento sem gravar no disco)", variable=self.align_dry_run_var).grid(row=3, column=0, columnspan=2, sticky="w", pady=(8, 0))

        action_frame = ttk.Frame(frame)
        action_frame.grid(row=2, column=0, sticky="ew", pady=(0, 10))
        action_frame.columnconfigure(0, weight=1)

        self.btn_run_align = ttk.Button(action_frame, text="▶ INICIAR ASTROALIGN", command=self.start_align_processing)
        self.btn_run_align.grid(row=0, column=0, sticky="ew", ipady=6)
        self.btn_cancel_align = ttk.Button(action_frame, text="■ CANCELAR", command=self.cancel_processing, state="disabled")
        self.btn_cancel_align.grid(row=0, column=1, padx=(8, 0), ipady=6)
        
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
        except queue.Empty: pass
        finally: self.after(50, self._drain_log_queue)

    def _lock_ui(self, module_name: str):
        self.save_settings()
        self.console_text.configure(state=tk.NORMAL)
        self.console_text.delete("1.0", tk.END)
        self.console_text.configure(state=tk.DISABLED)
        self.cancel_event.clear()
        
        self.btn_run_batch.configure(state="disabled")
        self.btn_run_flow.configure(state="disabled")
        self.btn_run_align.configure(state="disabled")
        
        if module_name == "Batch":
            self.btn_cancel_batch.configure(state="normal")
            self.status_var.set("Processando AstroBatch...")
        elif module_name == "Flow":
            self.btn_cancel_flow.configure(state="normal")
            self.status_var.set("Processando AstroFlow...")
        elif module_name == "Align":
            self.btn_cancel_align.configure(state="normal") 
            self.status_var.set("Processando AstroAlign...")

    def _unlock_ui(self):
        self.btn_run_batch.configure(state="normal")
        self.btn_run_flow.configure(state="normal")
        self.btn_run_align.configure(state="normal")
        self.btn_cancel_batch.configure(state="disabled")
        self.btn_cancel_flow.configure(state="disabled")
        self.btn_cancel_align.configure(state="disabled")

    def start_batch_processing(self):
        if self.worker and self.worker.is_alive(): return
        try:
            input_dir = Path(self.batch_input_dir_var.get()).expanduser().resolve()
            output_dir = Path(self.batch_dir_var.get()).expanduser().resolve()
            if not input_dir.is_dir(): raise ValueError("A pasta de origem não existe.")
            if input_dir == output_dir: raise ValueError("A pasta de destino deve ser diferente.")
            
            config = ProcessingConfig(
                input_dir=input_dir,
                output_dir=output_dir,
                threshold_factor=float(self.threshold_var.get()),
                crop_size=int(self.crop_size_var.get()),
                dry_run=self.dry_run_var.get(),
                copy_files=self.copy_files_var.get(),
                overwrite=self.batch_overwrite_var.get(),
                opt_method=self.opt_method_var.get(),
                downsample_method=self.downsample_method_var.get(),
                downsample_scale=float(self.downsample_scale_var.get()),
            )
        except Exception as exc:
            messagebox.showerror("Parâmetros inválidos", str(exc))
            return
            
        self._lock_ui("Batch")
        self.worker = threading.Thread(target=self.run_batch_logic, args=(config,), daemon=True)
        self.worker.start()

    def run_batch_logic(self, config: ProcessingConfig):
        from batch_logic import process_fits_logic
        try:
            processed, batches = process_fits_logic(config, self.print_to_console, self.update_progress, self.cancel_event)
            status = f"Cancelado. {processed} processados." if self.cancel_event.is_set() else f"Concluído. {processed} processados, {batches} batches."
            self.after(0, lambda: self.status_var.set(status))
        except Exception as exc:
            self.print_to_console(f"\nERRO FATAL: {exc}\n")
            self.after(0, lambda: self.status_var.set("Erro no AstroBatch."))
        finally:
            self.after(0, self._unlock_ui)

    def start_flow_processing(self):
        if self.worker and self.worker.is_alive(): return
        batch_dir = Path(self.batch_dir_var.get()).expanduser().resolve()
        if not batch_dir.is_dir(): return messagebox.showerror("Erro", "Pasta Base não encontrada.")
        
        # Cria um dicionário seguro mapeando explicitamente as chaves
        flow_reg = self.config_registry["AstroFlow"]
        config = {
            "global_master": flow_reg["global_master"].get(),
            "fwhm": flow_reg["fwhm"].get(),
            "sigma": flow_reg["sigma"].get(),
            "matching_radius": flow_reg["matching_radius"].get(),
            "ransac": flow_reg["ransac"].get(),
            "debug_images": flow_reg["debug_images"].get(),
            "min_stars": flow_reg["min_stars"].get(),
            "min_inliers": flow_reg["min_inliers"].get(),
            "min_ratio": flow_reg["min_ratio"].get(),
            "max_stars": 150
        }
        
        self.save_settings()
        self._lock_ui("Flow")

        self.worker = threading.Thread(target=self.run_flow_logic, args=(batch_dir, config), daemon=True)
        self.worker.start()
    
    def show_astroflow_preview(self):
        """Abre uma janela pop-up com a pré-visualização da detecção de estrelas na âncora."""
        base_dir_str = self.batch_dir_var.get()
        if not base_dir_str:
            return messagebox.showerror("Erro", "Selecione a Pasta Base das Batches primeiro.")
            
        base_dir = Path(base_dir_str).expanduser().resolve()
        batch_folders = sorted([d for d in base_dir.iterdir() if d.is_dir() and "batch" in d.name.lower()])
        
        if not batch_folders:
            return messagebox.showerror("Erro", "Nenhuma pasta de Batch encontrada na pasta base.")
            
        # Pega a primeira batch como referência para o preview
        target_batch = batch_folders[0]
        
        config = {
            "fwhm": self.flow_fwhm_var.get(),
            "sigma": self.flow_sigma_var.get(),
            "max_stars": 250,
            "engine": self.flow_engine_var.get(),
        }
        
        from astroflow_logic import preview_star_detection
        
        try:
            img_preview, count, fwhm_measured = preview_star_detection(target_batch, config)
            if img_preview is None:
                return messagebox.showwarning("Aviso", "Não foi possível carregar a imagem âncora para preview.")
        except Exception as exc:
            return messagebox.showerror("Erro", f"Falha ao gerar preview: {exc}")
            
        # Cria a janela Toplevel para exibir o resultado
        prev_window = tk.Toplevel(self)
        prev_window.title(f"Preview da Âncora ({target_batch.name})")
        prev_window.geometry("650x600")
        
        info_label = ttk.Label(
            prev_window, 
            text=f"Batch: {target_batch.name} | Estrelas Detectadas: {count} | FWHM Base: {fwhm_measured:.1f}px | Engine: {self.flow_engine_var.get()}", 
            font=("Segoe UI", 10, "bold")
        )
        info_label.pack(pady=10)
        
        # Exibe a imagem usando Matplotlib embutido no Tkinter para facilitar o zoom/pan se necessário
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        from matplotlib.figure import Figure
        
        fig = Figure(figsize=(5, 4), dpi=80)
        ax = fig.add_subplot(111)
        
        # OpenCV usa BGR, convertemos para RGB para o Matplotlib exibir corretamente
        ax.imshow(cv2.cvtColor(img_preview, cv2.COLOR_BGR2RGB))
        ax.set_title("Detecção de Estrelas (DAOStarFinder)", color='white', fontsize=10)
        ax.axis("off")
        fig.tight_layout(pad=0.5)
        
        # Fundo escuro nativo para combinar com a UI sem overhead de reestilização complexa
        fig.patch.set_facecolor('#1e1e1e')
        ax.set_facecolor('#1e1e1e')
        
        canvas = FigureCanvasTkAgg(fig, master=prev_window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        hint_label = ttk.Label(
            prev_window, 
            text="Dica: Feche esta janela, ajuste os valores de FWHM ou Sigma na aba e clique em Preview novamente para refinar.", 
            font=("Segoe UI", 8, "italic")
        )
        hint_label.pack(pady=5)
        
    def run_flow_logic(self, batch_dir: Path, config: dict):
        from astroflow_logic import process_all_flows
        try:
            process_all_flows(batch_dir, config, self.print_to_console, self.update_progress, self.cancel_event)
        except Exception as exc:
            self.print_to_console(f"\nERRO FATAL: {exc}\n")
        finally:
            self.after(0, self._unlock_ui)
            self.after(0, lambda: self.status_var.set("Pronto."))

    def cancel_processing(self):
        if self.worker and self.worker.is_alive(): self.cancel_event.set()

    def show_flow_visualization(self):
        """Constrói a visualização 2D multiplicando o Flow Local pelo Global."""
        
        import numpy as np
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        from matplotlib.figure import Figure
        
        base_dir = Path(self.batch_dir_var.get()).expanduser().resolve()
        global_json = base_dir / "global_flow.json"
        
        if not global_json.exists():
            return messagebox.showerror("Erro", "Execute o AstroFlow primeiro. Arquivo global_flow.json ausente.")

        with open(global_json, "r") as f:
            global_data = json.load(f)

        points_x, points_y = [], []
        
        center_pt = np.array([0, 0, 1])

        for batch_name, g_info in global_data["batches"].items():
            g_matrix = np.array(g_info["matrix"])
            local_json = base_dir / batch_name / "flow_local.json"
            
            if not local_json.exists(): continue
            
            with open(local_json, "r") as f:
                local_data = json.load(f)
                
            for frame_name, l_info in local_data["frames"].items():
                l_matrix = np.array(l_info["matrix"])
                
                abs_matrix = np.dot(g_matrix, l_matrix)
                
                transformed_pt = np.dot(abs_matrix, center_pt)
                points_x.append(transformed_pt[0])
                points_y.append(transformed_pt[1])

        viz_window = tk.Toplevel(self)
        viz_window.title("AstroFlow: Trajetória da Abóbada Celeste")
        viz_window.geometry("700x500")
        
        fig = Figure(figsize=(6, 4), dpi=80)
        ax = fig.add_subplot(111)
        ax.plot(points_x, points_y, marker='o', markersize=2, linestyle='-', color='cyan', alpha=0.6, rasterized=True)
        ax.scatter(points_x[0], points_y[0], color='green', s=40, label='Início', zorder=5)
        ax.scatter(points_x[-1], points_y[-1], color='red', s=40, label='Fim', zorder=5)
        
        ax.set_title("Drift Analisado pelo AstroFlow", fontsize=10)
        ax.set_xlabel("Deslocamento X (pixels)", fontsize=9)
        ax.set_ylabel("Deslocamento Y (pixels)", fontsize=9)
        ax.invert_yaxis() 
        ax.grid(True, linestyle='--', alpha=0.2)
        ax.legend(fontsize=8)
        fig.tight_layout(pad=1.0)

        fig.patch.set_facecolor('#1e1e1e')
        ax.set_facecolor('#2d2d2d')
        ax.tick_params(colors='white')
        ax.xaxis.label.set_color('white')
        ax.yaxis.label.set_color('white')
        ax.title.set_color('white')
        
        canvas = FigureCanvasTkAgg(fig, master=viz_window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
    def _update_global_master_options(self, *args):
        """Varre a pasta selecionada e atualiza o Dropdown do Global Master."""
        if not hasattr(self, 'combo_global_master'):
            return # Previne erros se a UI ainda não tiver sido desenhada
            
        base_dir_str = self.batch_dir_var.get()
        valores_dropdown = ["Auto"]
        
        if base_dir_str:
            base_dir = Path(base_dir_str).expanduser().resolve()
            if base_dir.is_dir():
                try:
                    # Busca subpastas que tenham "batch" no nome
                    batches = sorted([d.name for d in base_dir.iterdir() if d.is_dir() and "batch" in d.name.lower()])
                    valores_dropdown.extend(batches)
                except Exception:
                    pass
                    
        self.combo_global_master["values"] = valores_dropdown
        
        # Se a seleção atual não for mais válida (ex: o usuário trocou de pasta), reseta para "Auto"
        if self.flow_global_master_var.get() not in valores_dropdown:
            self.flow_global_master_var.set("Auto")

    def start_align_processing(self):
        if self.worker and self.worker.is_alive(): return
        
        try:
            base_dir = Path(self.batch_dir_var.get()).expanduser().resolve()
            output_dir = Path(self.align_output_dir_var.get()).expanduser().resolve()
            
            if not base_dir.is_dir(): raise ValueError("A pasta base de batches não existe.")
            if base_dir == output_dir: raise ValueError("A pasta de destino deve ser diferente da pasta base.")
            
            overwrite = self.align_overwrite_var.get()
        except Exception as exc:
            messagebox.showerror("Parâmetros inválidos", str(exc))
            return
            
        self._lock_ui("Align")
        self.worker = threading.Thread(target=self.run_align_logic, args=(base_dir, output_dir, overwrite), daemon=True)
        self.worker.start()

    def run_align_logic(self, base_dir: Path, output_dir: Path, overwrite: bool):
        # Importação tardia (Lazy Import) para manter o startup instantâneo
        from astroalign_logic import process_all_alignments
        
        config_dict = {
            "interpolation": self.align_interpolation_var.get(),
            "overwrite": overwrite,
            "dry_run": self.align_dry_run_var.get(),
            "keep_header": self.align_keep_header_var.get()
        }
        
        try:
            process_all_alignments(
                base_dir, 
                output_dir, 
                config_dict, 
                self.print_to_console, 
                self.update_progress, 
                self.cancel_event
            )
            status = "Cancelado." if self.cancel_event.is_set() else "AstroAlign Concluído com Sucesso!"
            self.after(0, lambda: self.status_var.set(status))
        except Exception as exc:
            self.print_to_console(f"\nERRO FATAL NO ASTROALIGN: {exc}\n")
            self.after(0, lambda: self.status_var.set("Erro no AstroAlign."))
        finally:
            self.after(0, self._unlock_ui)
            
    def _on_close(self):
        if self.worker and self.worker.is_alive():
            if not messagebox.askyesno("Processamento ativo", "Processamento ativo. Sair mesmo assim?"):
                return
            self.cancel_event.set()
        self.save_settings()
        self.destroy()

if __name__ == "__main__":
    app = AstroProcessManager()
    app.mainloop()