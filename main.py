# main.py
import json
import math
import queue
import threading
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk

# Importa as lógicas de negócio dos arquivos separados
from batch_logic import ProcessingConfig, process_fits_logic, RESAMPLE_MODES
from astroflow_logic import process_all_batches_flow 

class AstroProcessManager(tk.Tk):
    def __init__(self):
        super().__init__()

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
        """Inicializa as variáveis e cria o registro dinâmico para o JSON."""
        # Variáveis Globais / Compartilhadas
        self.batch_dir_var = tk.StringVar()  # Output do Batch, Input do Flow
        self.status_var = tk.StringVar(value="Pronto.")

        # Variáveis AstroBatch
        self.batch_input_dir_var = tk.StringVar()
        self.opt_method_var = tk.StringVar(value="Crop")
        self.crop_size_var = tk.IntVar(value=1000)
        self.downsample_method_var = tk.StringVar(value="Nearest")
        self.downsample_scale_var = tk.DoubleVar(value=0.25)
        self.threshold_var = tk.DoubleVar(value=3.0)
        self.dry_run_var = tk.BooleanVar(value=True)
        self.copy_files_var = tk.BooleanVar(value=False)

        # Variáveis AstroFlow
        self.flow_fwhm_var = tk.DoubleVar(value=4.0)
        self.flow_max_stars_var = tk.IntVar(value=150)

        # REGISTRO DINÂMICO PARA SAVE/LOAD
        self.config_registry = {
            "Global": {
                "batch_dir": self.batch_dir_var
            },
            "AstroBatch": {
                "input_dir": self.batch_input_dir_var,
                "opt_method": self.opt_method_var,
                "crop_size": self.crop_size_var,
                "downsample_method": self.downsample_method_var,
                "downsample_scale": self.downsample_scale_var,
                "threshold_factor": self.threshold_var,
                "dry_run": self.dry_run_var,
                "copy_files": self.copy_files_var
            },
            "AstroFlow": {
                "fwhm": self.flow_fwhm_var,
                "max_stars": self.flow_max_stars_var
            }
        }

    def load_settings(self):
        """Carrega as configurações iterando pelo registro de módulos."""
        config_file = Path("astro_config.json")
        if not config_file.exists(): return
        
        try:
            with open(config_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                
            for module_name, variables in self.config_registry.items():
                if module_name in data:
                    for key, var in variables.items():
                        if key in data[module_name]:
                            var.set(data[module_name][key])
        except Exception as exc:
            self.print_to_console(f"Aviso: Erro ao carregar configurações: {exc}\n")

    def save_settings(self):
        """Salva as configurações iterando pelo registro de módulos."""
        config_file = Path("astro_config.json")
        data = {}
        
        for module_name, variables in self.config_registry.items():
            data[module_name] = {k: v.get() for k, v in variables.items()}
            
        try:
            with open(config_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4)
        except Exception as exc:
            self.print_to_console(f"Aviso: Erro ao salvar configurações: {exc}\n")

    def create_widgets(self):
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1) # Notebook expande

        # --- NOTEBOOK (SISTEMA DE ABAS) ---
        self.notebook = ttk.Notebook(self)
        self.notebook.grid(row=0, column=0, sticky="nsew", pady=(0, 10))

        # Cria os frames para cada aba
        self.tab_batch = ttk.Frame(self.notebook, padding=10)
        self.tab_flow = ttk.Frame(self.notebook, padding=10)
        
        self.notebook.add(self.tab_batch, text="1. AstroBatch (Separador)")
        self.notebook.add(self.tab_flow, text="2. AstroFlow (Cinemática)")

        # Constroi as UIs dentro de cada aba
        self._build_tab_batch()
        self._build_tab_flow()

        # --- ÁREA COMUM INFERIOR ---
        bottom_frame = ttk.Frame(self)
        bottom_frame.grid(row=1, column=0, sticky="nsew")
        bottom_frame.columnconfigure(0, weight=1)
        bottom_frame.rowconfigure(1, weight=1)

        ttk.Label(bottom_frame, textvariable=self.status_var, font=("Segoe UI", 9, "bold")).grid(row=0, column=0, sticky="sw", pady=(0, 4))

        console_frame = ttk.LabelFrame(bottom_frame, text="Console de Saída Compartilhado", padding=5)
        console_frame.grid(row=1, column=0, sticky="nsew")
        console_frame.columnconfigure(0, weight=1)
        console_frame.rowconfigure(0, weight=1)

        self.console_text = scrolledtext.ScrolledText(console_frame, height=12, state=tk.DISABLED, font=("Consolas", 10), bg="#1e1e1e", fg="#d4d4d4")
        self.console_text.grid(row=0, column=0, sticky="nsew")
        
        # Garante que a raiz redimensione corretamente o console
        self.rowconfigure(1, weight=1)

    def _build_tab_batch(self):
        """Constrói a interface da aba AstroBatch."""
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

        # (Mantendo a Otimização e Parâmetros idênticos ao original...)
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
        ttk.Checkbutton(param_frame, text="Dry-Run (simular sem mover/copiar arquivos)", variable=self.dry_run_var).grid(row=2, column=0, columnspan=2, sticky="w", pady=(8, 0))

        action_frame = ttk.Frame(frame)
        action_frame.grid(row=3, column=0, sticky="ew", pady=(0, 10))
        action_frame.columnconfigure(0, weight=1)

        self.btn_run_batch = ttk.Button(action_frame, text="▶ INICIAR ASTROBATCH", command=self.start_batch_processing)
        self.btn_run_batch.grid(row=0, column=0, sticky="ew", ipady=6)
        self.btn_cancel_batch = ttk.Button(action_frame, text="■ CANCELAR", command=self.cancel_processing, state="disabled")
        self.btn_cancel_batch.grid(row=0, column=1, padx=(8, 0), ipady=6)

    def _build_tab_flow(self):
        """Constrói a interface da aba AstroFlow."""
        frame = self.tab_flow
        frame.columnconfigure(0, weight=1)

        dir_frame = ttk.LabelFrame(frame, text="Diretório Compartilhado", padding=10)
        dir_frame.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        dir_frame.columnconfigure(1, weight=1)
        
        # Note que usamos a mesma variável: self.batch_dir_var
        ttk.Label(dir_frame, text="Pasta Base das Batches:").grid(row=0, column=0, sticky="w")
        ttk.Entry(dir_frame, textvariable=self.batch_dir_var).grid(row=0, column=1, padx=6, sticky="ew")
        ttk.Button(dir_frame, text="Procurar...", command=lambda: self.browse_dir(self.batch_dir_var)).grid(row=0, column=2)

        param_frame = ttk.LabelFrame(frame, text="Parâmetros Cinemáticos", padding=10)
        param_frame.grid(row=1, column=0, sticky="ew", pady=(0, 10))
        
        ttk.Label(param_frame, text="FWHM Médio das Estrelas (px):").grid(row=0, column=0, sticky="w")
        ttk.Entry(param_frame, textvariable=self.flow_fwhm_var, width=10).grid(row=0, column=1, padx=6, sticky="w")
        ttk.Label(param_frame, text="Máximo de estrelas para RANSAC:").grid(row=1, column=0, sticky="w", pady=(8,0))
        ttk.Entry(param_frame, textvariable=self.flow_max_stars_var, width=10).grid(row=1, column=1, padx=6, sticky="w", pady=(8,0))

        action_frame = ttk.Frame(frame)
        action_frame.grid(row=2, column=0, sticky="ew", pady=(0, 10))
        action_frame.columnconfigure(0, weight=1)

        self.btn_run_flow = ttk.Button(action_frame, text="▶ INICIAR ASTROFLOW", command=self.start_flow_processing)
        self.btn_run_flow.grid(row=0, column=0, sticky="ew", ipady=6)
        self.btn_cancel_flow = ttk.Button(action_frame, text="■ CANCELAR", command=self.cancel_processing, state="disabled")
        self.btn_cancel_flow.grid(row=0, column=1, padx=(8, 0), ipady=6)

    # --- FUNÇÕES UTILITÁRIAS ---
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

    def _lock_ui(self, module_name: str):
        """Desabilita botões e limpa o console antes de rodar."""
        self.save_settings()
        self.console_text.configure(state=tk.NORMAL)
        self.console_text.delete("1.0", tk.END)
        self.console_text.configure(state=tk.DISABLED)
        self.cancel_event.clear()
        
        self.btn_run_batch.configure(state="disabled")
        self.btn_run_flow.configure(state="disabled")
        
        if module_name == "Batch":
            self.btn_cancel_batch.configure(state="normal")
            self.status_var.set("Processando AstroBatch...")
        elif module_name == "Flow":
            self.btn_cancel_flow.configure(state="normal")
            self.status_var.set("Processando AstroFlow...")

    def _unlock_ui(self):
        """Habilita botões após o processo finalizar."""
        self.btn_run_batch.configure(state="normal")
        self.btn_run_flow.configure(state="normal")
        self.btn_cancel_batch.configure(state="disabled")
        self.btn_cancel_flow.configure(state="disabled")

    def cancel_processing(self):
        if self.worker and self.worker.is_alive():
            self.cancel_event.set()
            self.status_var.set("Cancelando...")

    # --- INÍCIO: LÓGICA DO ASTROBATCH ---
    def start_batch_processing(self):
        if self.worker and self.worker.is_alive(): return
        
        try:
            input_dir = Path(self.batch_input_dir_var.get()).expanduser().resolve()
            output_dir = Path(self.batch_dir_var.get()).expanduser().resolve()
            if not input_dir.is_dir(): raise ValueError("A pasta de origem não existe.")
            if input_dir == output_dir: raise ValueError("A pasta de destino deve ser diferente.")
            
            config = ProcessingConfig(
                input_dir=input_dir,
                output_dir=output_dir, # Mapeamos batch_dir_var para output_dir internamente
                threshold_factor=float(self.threshold_var.get()),
                crop_size=int(self.crop_size_var.get()),
                dry_run=self.dry_run_var.get(),
                copy_files=self.copy_files_var.get(),
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
        try:
            processed, batches = process_fits_logic(config, self.print_to_console, self.cancel_event)
            status = f"Cancelado. {processed} processados." if self.cancel_event.is_set() else f"Concluído. {processed} processados, {batches} batches."
            self.after(0, lambda: self.status_var.set(status))
        except Exception as exc:
            self.print_to_console(f"\nERRO FATAL: {exc}\n")
            self.after(0, lambda: self.status_var.set("Erro no AstroBatch."))
        finally:
            self.after(0, self._unlock_ui)

    # --- INÍCIO: LÓGICA DO ASTROFLOW ---
    def start_flow_processing(self):
        if self.worker and self.worker.is_alive(): return
        
        try:
            batch_dir = Path(self.batch_dir_var.get()).expanduser().resolve()
            if not batch_dir.is_dir(): raise ValueError("A pasta Base das Batches não existe.")
            
            fwhm = float(self.flow_fwhm_var.get())
            max_stars = int(self.flow_max_stars_var.get())
        except Exception as exc:
            messagebox.showerror("Parâmetros inválidos", str(exc))
            return
            
        self._lock_ui("Flow")
        self.worker = threading.Thread(target=self.run_flow_logic, args=(batch_dir, fwhm, max_stars), daemon=True)
        self.worker.start()

    def run_flow_logic(self, batch_dir: Path, fwhm: float, max_stars: int):
        try:
            # Aqui chamaremos a função wrapper do astroflow_logic.py
            # process_all_batches_flow(batch_dir, fwhm, max_stars, self.print_to_console, self.cancel_event)
            self.print_to_console(f"Iniciando AstroFlow na pasta {batch_dir}...\n(Integração do backend pendente)")
            self.after(0, lambda: self.status_var.set("Concluído."))
        except Exception as exc:
            self.print_to_console(f"\nERRO FATAL: {exc}\n")
            self.after(0, lambda: self.status_var.set("Erro no AstroFlow."))
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