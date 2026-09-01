import json
import queue
import threading
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk

import cv2
import numpy as np


# ============================================================
# Imports da lógica de processamento
# ============================================================
from batch_logic import ProcessingConfig, RESAMPLE_MODES
from astroalign_logic import INTERPOLATION_MODES


class AstroProcessManager(tk.Tk):
    """
    Interface gráfica principal do Astro Process Manager.

    Pipeline:
        1. AstroCalibration
        2. AstroDebayer
        3. AstroBatch
        4. AstroFlow
        5. AstroAlign

    A GUI não executa processamento pesado na thread principal.
    Cada pipeline roda em uma thread de trabalho e comunica:
        - logs -> Queue
        - progresso -> after()
        - cancelamento -> Event
    """

    APP_NAME = "Astro Process Manager"
    CONFIG_FILE = Path("astro_config.json")

    BG = "#ffffff"
    PANEL = "#ffffff"
    BORDER = "#d9dde3"
    TEXT = "#20242a"
    MUTED = "#68717d"
    ACCENT = "#2563eb"
    ACCENT_ACTIVE = "#1d4ed8"
    DANGER = "#dc2626"

    def __init__(self):
        super().__init__()

        self.title(self.APP_NAME)
        self.geometry("1120x900")
        self.minsize(940, 1020)
        self.configure(bg=self.BG)

        self.log_queue = queue.Queue()
        self.worker: threading.Thread | None = None
        self.cancel_event = threading.Event()
        self.custom_anchors = {}

        self._init_variables()
        self._configure_style()
        self.load_settings()
        self._create_widgets()

        self.after(50, self._drain_log_queue)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        # Atualiza opções dependentes da pasta após a UI existir.
        self.after(100, self._update_global_master_options)

    # ========================================================
    # Variáveis
    # ========================================================

    def _init_variables(self):
        # ---------- Global ----------
        self.batch_dir_var = tk.StringVar()
        self.status_var = tk.StringVar(value="Pronto.")
        self.progress_var = tk.DoubleVar(value=0.0)

        # ---------- Calibration ----------
        self.calib_input_var = tk.StringVar()
        self.calib_output_var = tk.StringVar()

        self.apply_dark_var = tk.BooleanVar(value=True)
        self.dark_path_var = tk.StringVar()

        self.apply_flat_var = tk.BooleanVar(value=True)
        self.flat_path_var = tk.StringVar()

        self.calib_create_master_var = tk.BooleanVar(value=True)
        self.calib_overwrite_var = tk.BooleanVar(value=False)

        # ---------- Debayer ----------
        self.debayer_input_var = tk.StringVar()
        self.debayer_output_var = tk.StringVar()
        self.debayer_pattern_var = tk.StringVar(value="Auto")
        self.debayer_method_var = tk.StringVar(value="Bilinear")
        self.debayer_overwrite_var = tk.BooleanVar(value=False)

        # ---------- AstroBatch ----------
        self.batch_input_dir_var = tk.StringVar()
        self.opt_method_var = tk.StringVar(value="Crop")
        self.crop_size_var = tk.IntVar(value=1000)
        self.downsample_method_var = tk.StringVar(value="Nearest")
        self.downsample_scale_var = tk.DoubleVar(value=0.25)
        self.threshold_var = tk.DoubleVar(value=3.0)
        self.dry_run_var = tk.BooleanVar(value=True)
        self.copy_files_var = tk.BooleanVar(value=False)
        self.batch_overwrite_var = tk.BooleanVar(value=False)

        # ---------- AstroFlow ----------
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

        # ---------- AstroAlign ----------
        self.align_output_dir_var = tk.StringVar()
        self.align_interpolation_var = tk.StringVar(value="Lanczos")
        self.align_overwrite_var = tk.BooleanVar(value=False)
        self.align_dry_run_var = tk.BooleanVar(value=False)
        self.align_keep_header_var = tk.BooleanVar(value=True)

        # Registry usado pelo JSON de configurações.
        self.config_registry = {
            "Global": {
                "batch_dir": self.batch_dir_var,
            },
            "AstroCalibration": {
                "input_dir": self.calib_input_var,
                "output_dir": self.calib_output_var,
                "apply_dark": self.apply_dark_var,
                "dark_path": self.dark_path_var,
                "apply_flat": self.apply_flat_var,
                "flat_path": self.flat_path_var,
                "create_master": self.calib_create_master_var,
                "overwrite": self.calib_overwrite_var,
            },
            "AstroDebayer": {
                "input_dir": self.debayer_input_var,
                "output_dir": self.debayer_output_var,
                "pattern": self.debayer_pattern_var,
                "method": self.debayer_method_var,
                "overwrite": self.debayer_overwrite_var,
            },
            "AstroBatch": {
                "input_dir": self.batch_input_dir_var,
                "opt_method": self.opt_method_var,
                "crop_size": self.crop_size_var,
                "downsample_method": self.downsample_method_var,
                "downsample_scale": self.downsample_scale_var,
                "threshold_factor": self.threshold_var,
                "dry_run": self.dry_run_var,
                "copy_files": self.copy_files_var,
                "overwrite": self.batch_overwrite_var,
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
                "engine": self.flow_engine_var,
            },
            "AstroAlign": {
                "output_dir": self.align_output_dir_var,
                "interpolation": self.align_interpolation_var,
                "overwrite": self.align_overwrite_var,
                "dry_run": self.align_dry_run_var,
                "keep_header": self.align_keep_header_var,
            },
        }

    # ========================================================
    # Estilo
    # ========================================================

    def _configure_style(self):
        style = ttk.Style(self)

        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        style.configure(
            ".",
            font=("Segoe UI", 10),
            background=self.BG,
            foreground=self.TEXT,
        )

        style.configure(
            "TFrame",
            background=self.BG,
        )

        style.configure(
            "Panel.TFrame",
            background=self.PANEL,
        )

        style.configure(
            "TLabel",
            background=self.BG,
            foreground=self.TEXT,
        )

        style.configure(
            "Muted.TLabel",
            background=self.BG,
            foreground=self.MUTED,
            font=("Segoe UI", 9),
        )

        style.configure(
            "Title.TLabel",
            background=self.BG,
            foreground=self.TEXT,
            font=("Segoe UI Semibold", 20),
        )

        style.configure(
            "Subtitle.TLabel",
            background=self.BG,
            foreground=self.MUTED,
            font=("Segoe UI", 10),
        )

        style.configure(
            "Section.TLabelframe",
            background=self.PANEL,
            bordercolor=self.BORDER,
            relief="solid",
            borderwidth=1,
        )

        style.configure(
            "Section.TLabelframe.Label",
            background=self.PANEL,
            foreground=self.TEXT,
            font=("Segoe UI Semibold", 10),
        )

        style.configure(
            "TEntry",
            padding=(8, 6),
        )

        style.configure(
            "TCombobox",
            padding=(7, 5),
        )

        style.configure(
            "TButton",
            padding=(12, 7),
            font=("Segoe UI Semibold", 9),
        )

        style.configure(
            "Accent.TButton",
            background=self.ACCENT,
            foreground="white",
            borderwidth=0,
        )
        style.map(
            "Accent.TButton",
            background=[("active", self.ACCENT_ACTIVE), ("pressed", self.ACCENT_ACTIVE)],
            foreground=[("disabled", "#aeb7c4"), ("!disabled", "white")],
        )

        style.configure(
            "Danger.TButton",
            background=self.DANGER,
            foreground="white",
            borderwidth=0,
        )

        style.configure(
            "TNotebook",
            background=self.BG,
            borderwidth=0,
            tabmargins=(0, 0, 0, 0),
        )

        style.configure(
            "TNotebook.Tab",
            background="#e8ebef",
            foreground=self.MUTED,
            padding=(18, 10),
            font=("Segoe UI Semibold", 9),
        )

        style.map(
            "TNotebook.Tab",
            background=[("selected", self.PANEL)],
            foreground=[("selected", self.ACCENT)],
        )

        style.configure(
            "Horizontal.TProgressbar",
            troughcolor="#e2e5e9",
            background=self.ACCENT,
            borderwidth=0,
            thickness=7,
        )

        style.configure(
            "TCheckbutton",
            background=self.PANEL,
        )

        style.configure(
            "TRadiobutton",
            background=self.PANEL,
        )

    # ========================================================
    # Layout principal
    # ========================================================

    def _create_widgets(self):
        self.columnconfigure(0, weight=1)
        self.rowconfigure(2, weight=1)

        # Header
        header = ttk.Frame(self)
        header.grid(row=0, column=0, sticky="ew", padx=22, pady=(18, 10))
        header.columnconfigure(0, weight=1)

        ttk.Label(
            header,
            text="Astro Process Manager",
            style="Title.TLabel",
        ).grid(row=0, column=0, sticky="w")

        ttk.Label(
            header,
            text="Calibração  →  Debayer  →  Batch  →  Flow  →  Align",
            style="Subtitle.TLabel",
        ).grid(row=1, column=0, sticky="w", pady=(2, 0))

        # Pasta global
        global_frame = ttk.LabelFrame(
            self,
            text="Projeto / Diretório de Batches",
            style="Section.TLabelframe",
            padding=12,
        )
        global_frame.grid(row=1, column=0, sticky="ew", padx=22, pady=(0, 10))
        global_frame.columnconfigure(1, weight=1)

        ttk.Label(global_frame, text="Pasta Base:").grid(
            row=0, column=0, sticky="w"
        )

        ttk.Entry(
            global_frame,
            textvariable=self.batch_dir_var,
        ).grid(row=0, column=1, sticky="ew", padx=8)

        ttk.Button(
            global_frame,
            text="Selecionar pasta",
            command=lambda: self.browse_dir(self.batch_dir_var),
        ).grid(row=0, column=2)

        # Notebook
        notebook_container = ttk.Frame(self)
        notebook_container.grid(
            row=2,
            column=0,
            sticky="nsew",
            padx=22,
            pady=(0, 10),
        )
        notebook_container.columnconfigure(0, weight=1)
        notebook_container.rowconfigure(0, weight=1)

        self.notebook = ttk.Notebook(notebook_container)
        self.notebook.grid(row=0, column=0, sticky="nsew")

        self.tab_calib = ttk.Frame(self.notebook, padding=18)
        self.tab_debayer = ttk.Frame(self.notebook, padding=18)
        self.tab_batch = ttk.Frame(self.notebook, padding=18)
        self.tab_flow = ttk.Frame(self.notebook, padding=18)
        self.tab_align = ttk.Frame(self.notebook, padding=18)

        self.notebook.add(self.tab_calib, text="1  Calibration")
        self.notebook.add(self.tab_debayer, text="2  Debayer")
        self.notebook.add(self.tab_batch, text="3  Batch")
        self.notebook.add(self.tab_flow, text="4  Flow")
        self.notebook.add(self.tab_align, text="5  Align")

        self._build_tab_calib()
        self._build_tab_debayer()
        self._build_tab_batch()
        self._build_tab_flow()
        self._build_tab_align()

        # Rodapé
        self._build_footer()

    def _build_footer(self):
        footer = ttk.Frame(self)
        footer.grid(row=3, column=0, sticky="ew", padx=22, pady=(0, 16))
        footer.columnconfigure(0, weight=1)

        status_line = ttk.Frame(footer)
        status_line.grid(row=0, column=0, sticky="ew")
        status_line.columnconfigure(1, weight=1)

        ttk.Label(
            status_line,
            textvariable=self.status_var,
            font=("Segoe UI Semibold", 9),
        ).grid(row=0, column=0, sticky="w")

        self.progress_bar = ttk.Progressbar(
            status_line,
            variable=self.progress_var,
            maximum=100,
            mode="determinate",
        )
        self.progress_bar.grid(
            row=0,
            column=1,
            sticky="ew",
            padx=(14, 0),
        )

        console_frame = ttk.LabelFrame(
            footer,
            text="Console",
            style="Section.TLabelframe",
            padding=7,
        )
        console_frame.grid(row=1, column=0, sticky="ew", pady=(8, 0))
        console_frame.columnconfigure(0, weight=1)

        self.console_text = scrolledtext.ScrolledText(
            console_frame,
            height=7,
            state=tk.DISABLED,
            font=("Cascadia Mono", 9),
            bg="#17191c",
            fg="#d7dce2",
            insertbackground="white",
            relief="flat",
            borderwidth=0,
        )
        self.console_text.grid(row=0, column=0, sticky="ew")

    # ========================================================
    # Helpers visuais
    # ========================================================

    @staticmethod
    def _configure_grid(frame, columns=3):
        for col in range(columns):
            frame.columnconfigure(col, weight=1 if col == 1 else 0)

    def _path_row(self, parent, row, label, variable, browse_command, browse_text="Selecionar"):
        ttk.Label(parent, text=label).grid(
            row=row, column=0, sticky="w", pady=5
        )
        ttk.Entry(parent, textvariable=variable).grid(
            row=row, column=1, sticky="ew", padx=8, pady=5
        )
        ttk.Button(
            parent,
            text=browse_text,
            command=browse_command,
        ).grid(row=row, column=2, pady=5)

    def _action_bar(self, parent, row, run_command, cancel_command, run_text):
        parent.columnconfigure(0, weight=1)

        run_btn = ttk.Button(
            parent,
            text=run_text,
            style="Accent.TButton",
            command=run_command,
        )
        run_btn.grid(row=0, column=0, sticky="ew", ipady=5)

        cancel_btn = ttk.Button(
            parent,
            text="Cancelar",
            style="Danger.TButton",
            command=cancel_command,
            state="disabled",
        )
        cancel_btn.grid(row=0, column=1, padx=(8, 0), ipady=5)

        return run_btn, cancel_btn

    def _description(self, parent, row, text, columnspan=3):
        ttk.Label(
            parent,
            text=text,
            style="Muted.TLabel",
            wraplength=820,
            justify="left",
        ).grid(
            row=row,
            column=0,
            columnspan=columnspan,
            sticky="w",
            pady=(0, 10),
        )

    # ========================================================
    # Calibration
    # ========================================================

    def _build_tab_calib(self):
        frame = self.tab_calib
        frame.columnconfigure(0, weight=1)

        intro = ttk.LabelFrame(
            frame,
            text="Entrada e Saída",
            style="Section.TLabelframe",
            padding=12,
        )
        intro.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        intro.columnconfigure(1, weight=1)

        self._description(
            intro,
            0,
            "Calibre os LIGHTS antes do Debayer. Darks e Flats podem ser "
            "informados como uma pasta de frames ou como um Master já pronto.",
        )

        self._path_row(
            intro,
            1,
            "LIGHTS / RAW:",
            self.calib_input_var,
            lambda: self.browse_dir(self.calib_input_var),
        )

        self._path_row(
            intro,
            2,
            "Saída calibrada:",
            self.calib_output_var,
            lambda: self.browse_dir(self.calib_output_var),
        )

        dark = ttk.LabelFrame(
            frame,
            text="Dark",
            style="Section.TLabelframe",
            padding=12,
        )
        dark.grid(row=1, column=0, sticky="ew", pady=(0, 10))
        dark.columnconfigure(1, weight=1)

        ttk.Checkbutton(
            dark,
            text="Aplicar Dark",
            variable=self.apply_dark_var,
        ).grid(row=0, column=0, sticky="w")

        ttk.Entry(
            dark,
            textvariable=self.dark_path_var,
        ).grid(row=0, column=1, sticky="ew", padx=8)

        ttk.Button(
            dark,
            text="Arquivo...",
            command=lambda: self.browse_file_or_dir(self.dark_path_var),
        ).grid(row=0, column=2)

        ttk.Label(
            dark,
            text="Pasta = gerar Master Dark   •   Arquivo = usar como Master Dark",
            style="Muted.TLabel",
        ).grid(row=1, column=0, columnspan=3, sticky="w", pady=(7, 0))

        flat = ttk.LabelFrame(
            frame,
            text="Flat",
            style="Section.TLabelframe",
            padding=12,
        )
        flat.grid(row=2, column=0, sticky="ew", pady=(0, 10))
        flat.columnconfigure(1, weight=1)

        ttk.Checkbutton(
            flat,
            text="Aplicar Flat",
            variable=self.apply_flat_var,
        ).grid(row=0, column=0, sticky="w")

        ttk.Entry(
            flat,
            textvariable=self.flat_path_var,
        ).grid(row=0, column=1, sticky="ew", padx=8)

        ttk.Button(
            flat,
            text="Arquivo...",
            command=lambda: self.browse_file_or_dir(self.flat_path_var),
        ).grid(row=0, column=2)

        ttk.Label(
            flat,
            text="Pasta = gerar Master Flat   •   Arquivo = usar como Master Flat",
            style="Muted.TLabel",
        ).grid(row=1, column=0, columnspan=3, sticky="w", pady=(7, 0))

        options = ttk.LabelFrame(
            frame,
            text="Opções",
            style="Section.TLabelframe",
            padding=12,
        )
        options.grid(row=3, column=0, sticky="ew", pady=(0, 10))

        ttk.Checkbutton(
            options,
            text="Gerar Masters automaticamente quando receber uma pasta",
            variable=self.calib_create_master_var,
        ).grid(row=0, column=0, sticky="w", pady=2)

        ttk.Checkbutton(
            options,
            text="Sobrescrever arquivos existentes",
            variable=self.calib_overwrite_var,
        ).grid(row=1, column=0, sticky="w", pady=2)

        actions = ttk.Frame(frame)
        actions.grid(row=4, column=0, sticky="ew")

        self.btn_run_calib, self.btn_cancel_calib = self._action_bar(
            actions,
            0,
            self.start_calibration,
            self.cancel_processing,
            "▶  INICIAR CALIBRATION",
        )

    # ========================================================
    # Debayer
    # ========================================================

    def _build_tab_debayer(self):
        frame = self.tab_debayer
        frame.columnconfigure(0, weight=1)

        dirs = ttk.LabelFrame(
            frame,
            text="Entrada e Saída",
            style="Section.TLabelframe",
            padding=12,
        )
        dirs.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        dirs.columnconfigure(1, weight=1)

        self._description(
            dirs,
            0,
            "Converta o mosaico Bayer monocromático calibrado em RGB antes "
            "do AstroFlow/AstroAlign. Isso evita interpolar diretamente a "
            "grade física Bayer durante o warping.",
        )

        self._path_row(
            dirs,
            1,
            "Entrada calibrada:",
            self.debayer_input_var,
            lambda: self.browse_dir(self.debayer_input_var),
        )

        self._path_row(
            dirs,
            2,
            "Saída RGB:",
            self.debayer_output_var,
            lambda: self.browse_dir(self.debayer_output_var),
        )

        params = ttk.LabelFrame(
            frame,
            text="Parâmetros do Debayer",
            style="Section.TLabelframe",
            padding=12,
        )
        params.grid(row=1, column=0, sticky="ew", pady=(0, 10))

        ttk.Label(params, text="Padrão Bayer:").grid(
            row=0, column=0, sticky="w", padx=(0, 8)
        )

        ttk.Combobox(
            params,
            textvariable=self.debayer_pattern_var,
            values=["Auto", "RGGB", "BGGR", "GRBG", "GBRG"],
            state="readonly",
            width=14,
        ).grid(row=0, column=1, sticky="w")

        ttk.Label(params, text="Método:").grid(
            row=0, column=2, sticky="w", padx=(35, 8)
        )

        ttk.Combobox(
            params,
            textvariable=self.debayer_method_var,
            values=["Nearest", "Bilinear", "VNG", "Edge-Aware"],
            state="readonly",
            width=16,
        ).grid(row=0, column=3, sticky="w")

        ttk.Checkbutton(
            params,
            text="Sobrescrever arquivos existentes",
            variable=self.debayer_overwrite_var,
        ).grid(row=1, column=0, columnspan=4, sticky="w", pady=(10, 0))

        actions = ttk.Frame(frame)
        actions.grid(row=2, column=0, sticky="ew")

        self.btn_run_debayer, self.btn_cancel_debayer = self._action_bar(
            actions,
            0,
            self.start_debayer,
            self.cancel_processing,
            "▶  INICIAR DEBAYER",
        )

    # ========================================================
    # Batch
    # ========================================================

    def _build_tab_batch(self):
        frame = self.tab_batch
        frame.columnconfigure(0, weight=1)

        dirs = ttk.LabelFrame(
            frame,
            text="Diretórios",
            style="Section.TLabelframe",
            padding=12,
        )
        dirs.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        dirs.columnconfigure(1, weight=1)

        self._path_row(
            dirs,
            0,
            "Pasta origem:",
            self.batch_input_dir_var,
            lambda: self.browse_dir(self.batch_input_dir_var),
        )

        self._path_row(
            dirs,
            1,
            "Pasta destino:",
            self.batch_dir_var,
            lambda: self.browse_dir(self.batch_dir_var),
        )

        opt = ttk.LabelFrame(
            frame,
            text="Otimização para análise",
            style="Section.TLabelframe",
            padding=12,
        )
        opt.grid(row=1, column=0, sticky="ew", pady=(0, 10))
        opt.columnconfigure(1, weight=1)

        ttk.Radiobutton(
            opt,
            text="Recorte central (Crop)",
            variable=self.opt_method_var,
            value="Crop",
            command=self.toggle_opt_options,
        ).grid(row=0, column=0, sticky="w")

        self.crop_frame = ttk.Frame(opt)
        self.crop_frame.grid(row=0, column=1, sticky="w", padx=20)

        ttk.Label(self.crop_frame, text="Tamanho (px):").pack(side="left")
        self.crop_entry = ttk.Entry(
            self.crop_frame,
            textvariable=self.crop_size_var,
            width=10,
        )
        self.crop_entry.pack(side="left", padx=(7, 0))

        ttk.Radiobutton(
            opt,
            text="Downsampling",
            variable=self.opt_method_var,
            value="Downsampling",
            command=self.toggle_opt_options,
        ).grid(row=1, column=0, sticky="w", pady=(8, 0))

        self.down_frame = ttk.Frame(opt)
        self.down_frame.grid(row=1, column=1, sticky="w", padx=20, pady=(8, 0))

        ttk.Label(self.down_frame, text="Método:").pack(side="left")

        self.down_combo = ttk.Combobox(
            self.down_frame,
            textvariable=self.downsample_method_var,
            values=list(RESAMPLE_MODES),
            state="readonly",
            width=13,
        )
        self.down_combo.pack(side="left", padx=7)

        ttk.Label(self.down_frame, text="Escala:").pack(side="left", padx=(8, 0))

        self.down_scale_entry = ttk.Entry(
            self.down_frame,
            textvariable=self.downsample_scale_var,
            width=8,
        )
        self.down_scale_entry.pack(side="left", padx=7)

        self.toggle_opt_options()

        params = ttk.LabelFrame(
            frame,
            text="Detecção e operação de arquivos",
            style="Section.TLabelframe",
            padding=12,
        )
        params.grid(row=2, column=0, sticky="ew", pady=(0, 10))

        ttk.Label(params, text="Threshold (fator):").grid(
            row=0, column=0, sticky="w"
        )
        ttk.Entry(
            params,
            textvariable=self.threshold_var,
            width=10,
        ).grid(row=0, column=1, sticky="w", padx=8)

        ttk.Checkbutton(
            params,
            text="Copiar em vez de mover",
            variable=self.copy_files_var,
        ).grid(row=1, column=0, columnspan=2, sticky="w", pady=(8, 0))

        ttk.Checkbutton(
            params,
            text="Sobrescrever arquivos existentes",
            variable=self.batch_overwrite_var,
        ).grid(row=2, column=0, columnspan=2, sticky="w", pady=(8, 0))

        ttk.Checkbutton(
            params,
            text="Dry-Run (não alterar arquivos)",
            variable=self.dry_run_var,
        ).grid(row=3, column=0, columnspan=2, sticky="w", pady=(8, 0))

        actions = ttk.Frame(frame)
        actions.grid(row=3, column=0, sticky="ew")

        self.btn_run_batch, self.btn_cancel_batch = self._action_bar(
            actions,
            0,
            self.start_batch_processing,
            self.cancel_processing,
            "▶  INICIAR ASTROBATCH",
        )

    # ========================================================
    # Flow
    # ========================================================

    def _build_tab_flow(self):
        frame = self.tab_flow
        frame.columnconfigure(0, weight=1)

        params = ttk.LabelFrame(
            frame,
            text="Detecção e referência",
            style="Section.TLabelframe",
            padding=12,
        )
        params.grid(row=0, column=0, sticky="ew", pady=(0, 10))

        ttk.Label(params, text="Global Master:").grid(
            row=0, column=0, sticky="w"
        )

        self.combo_global_master = ttk.Combobox(
            params,
            textvariable=self.flow_global_master_var,
            values=["Auto"],
            state="readonly",
            width=16,
        )
        self.combo_global_master.grid(row=0, column=1, sticky="w", padx=8)

        ttk.Label(params, text="Star Engine:").grid(
            row=0, column=2, sticky="w", padx=(30, 8)
        )

        ttk.Combobox(
            params,
            textvariable=self.flow_engine_var,
            values=["DAO", "OpenCV"],
            state="readonly",
            width=14,
        ).grid(row=0, column=3, sticky="w")

        ttk.Label(params, text="FWHM médio (px):").grid(
            row=1, column=0, sticky="w", pady=(10, 0)
        )
        ttk.Entry(
            params,
            textvariable=self.flow_fwhm_var,
            width=14,
        ).grid(row=1, column=1, sticky="w", padx=8, pady=(10, 0))

        ttk.Label(params, text="Sigma threshold:").grid(
            row=1, column=2, sticky="w", padx=(30, 8), pady=(10, 0)
        )
        ttk.Entry(
            params,
            textvariable=self.flow_sigma_var,
            width=14,
        ).grid(row=1, column=3, sticky="w", pady=(10, 0))

        ttk.Label(params, text="Raio de pareamento (px):").grid(
            row=2, column=0, sticky="w", pady=(10, 0)
        )
        ttk.Entry(
            params,
            textvariable=self.flow_matching_radius_var,
            width=14,
        ).grid(row=2, column=1, sticky="w", padx=8, pady=(10, 0))

        ttk.Label(params, text="RANSAC reprojection:").grid(
            row=2, column=2, sticky="w", padx=(30, 8), pady=(10, 0)
        )
        ttk.Entry(
            params,
            textvariable=self.flow_ransac_var,
            width=14,
        ).grid(row=2, column=3, sticky="w", pady=(10, 0))

        ttk.Label(params, text="Mínimo de estrelas:").grid(
            row=3, column=0, sticky="w", pady=(10, 0)
        )
        ttk.Entry(
            params,
            textvariable=self.flow_min_stars_var,
            width=14,
        ).grid(row=3, column=1, sticky="w", padx=8, pady=(10, 0))

        ttk.Label(params, text="Mínimo de inliers:").grid(
            row=3, column=2, sticky="w", padx=(30, 8), pady=(10, 0)
        )
        ttk.Entry(
            params,
            textvariable=self.flow_min_inliers_var,
            width=14,
        ).grid(row=3, column=3, sticky="w", pady=(10, 0))

        ttk.Label(params, text="Ratio mínimo de inliers:").grid(
            row=4, column=0, sticky="w", pady=(10, 0)
        )
        ttk.Entry(
            params,
            textvariable=self.flow_min_ratio_var,
            width=14,
        ).grid(row=4, column=1, sticky="w", padx=8, pady=(10, 0))

        ttk.Checkbutton(
            params,
            text="Gerar imagens de diagnóstico nas âncoras",
            variable=self.flow_debug_var,
        ).grid(row=5, column=0, columnspan=4, sticky="w", pady=(12, 0))

        actions = ttk.Frame(frame)
        actions.grid(row=1, column=0, sticky="ew", pady=(0, 10))

        for col in range(5):
            actions.columnconfigure(col, weight=1)

        self.btn_select_anchor = ttk.Button(
            actions,
            text="🖼  Definir referência",
            command=self.open_anchor_selector,
        )
        self.btn_select_anchor.grid(row=0, column=0, sticky="ew", ipady=5)

        self.btn_preview_flow = ttk.Button(
            actions,
            text="🔍  Preview detecção",
            command=self.show_astroflow_preview,
        )
        self.btn_preview_flow.grid(
            row=0, column=1, sticky="ew", padx=6, ipady=5
        )

        self.btn_run_flow = ttk.Button(
            actions,
            text="▶  Iniciar AstroFlow",
            style="Accent.TButton",
            command=self.start_flow_processing,
        )
        self.btn_run_flow.grid(row=0, column=2, sticky="ew", ipady=5)

        self.btn_viz_flow = ttk.Button(
            actions,
            text="📈  Visualizar",
            command=self.show_flow_visualization,
        )
        self.btn_viz_flow.grid(
            row=0, column=3, sticky="ew", padx=6, ipady=5
        )

        self.btn_cancel_flow = ttk.Button(
            actions,
            text="Cancelar",
            style="Danger.TButton",
            command=self.cancel_processing,
            state="disabled",
        )
        self.btn_cancel_flow.grid(row=0, column=4, sticky="ew", ipady=5)

    # ========================================================
    # Align
    # ========================================================

    def _build_tab_align(self):
        frame = self.tab_align
        frame.columnconfigure(0, weight=1)

        dirs = ttk.LabelFrame(
            frame,
            text="Diretórios de alinhamento",
            style="Section.TLabelframe",
            padding=12,
        )
        dirs.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        dirs.columnconfigure(1, weight=1)

        self._path_row(
            dirs,
            0,
            "Pasta base:",
            self.batch_dir_var,
            lambda: self.browse_dir(self.batch_dir_var),
        )

        self._path_row(
            dirs,
            1,
            "Pasta destino:",
            self.align_output_dir_var,
            lambda: self.browse_dir(self.align_output_dir_var),
        )

        params = ttk.LabelFrame(
            frame,
            text="Warping",
            style="Section.TLabelframe",
            padding=12,
        )
        params.grid(row=1, column=0, sticky="ew", pady=(0, 10))

        ttk.Label(params, text="Interpolação:").grid(
            row=0, column=0, sticky="w"
        )

        interp_values = list(INTERPOLATION_MODES)
        if not interp_values:
            interp_values = ["Nearest", "Bilinear", "Bicubic", "Lanczos"]

        ttk.Combobox(
            params,
            textvariable=self.align_interpolation_var,
            values=interp_values,
            state="readonly",
            width=16,
        ).grid(row=0, column=1, sticky="w", padx=8)

        ttk.Checkbutton(
            params,
            text="Preservar Header FITS original",
            variable=self.align_keep_header_var,
        ).grid(row=1, column=0, columnspan=2, sticky="w", pady=(10, 0))

        ttk.Checkbutton(
            params,
            text="Sobrescrever arquivos existentes",
            variable=self.align_overwrite_var,
        ).grid(row=2, column=0, columnspan=2, sticky="w", pady=(7, 0))

        ttk.Checkbutton(
            params,
            text="Dry-Run (não gravar no disco)",
            variable=self.align_dry_run_var,
        ).grid(row=3, column=0, columnspan=2, sticky="w", pady=(7, 0))

        actions = ttk.Frame(frame)
        actions.grid(row=2, column=0, sticky="ew")

        self.btn_run_align, self.btn_cancel_align = self._action_bar(
            actions,
            0,
            self.start_align_processing,
            self.cancel_processing,
            "▶  INICIAR ASTROALIGN",
        )

    # ========================================================
    # Arquivos / diretórios
    # ========================================================

    def browse_dir(self, var):
        folder = filedialog.askdirectory(parent=self)
        if folder:
            var.set(folder)

    def browse_file(self, var):
        path = filedialog.askopenfilename(
            parent=self,
            filetypes=[
                ("FITS", "*.fits *.fit *.fts"),
                ("Todos os arquivos", "*.*"),
            ],
        )
        if path:
            var.set(path)

    def browse_file_or_dir(self, var):
        """
        Permite escolher tanto uma pasta quanto um arquivo.

        Primeiro tenta pasta; se cancelado, permite selecionar arquivo.
        """
        folder = filedialog.askdirectory(parent=self)
        if folder:
            var.set(folder)
            return

        self.browse_file(var)

    # ========================================================
    # Configuração / estado
    # ========================================================

    def load_settings(self):
        if not self.CONFIG_FILE.exists():
            return

        try:
            with self.CONFIG_FILE.open("r", encoding="utf-8") as f:
                data = json.load(f)

            for module, variables in self.config_registry.items():
                module_data = data.get(module, {})
                for key, variable in variables.items():
                    if key in module_data:
                        try:
                            variable.set(module_data[key])
                        except (tk.TclError, ValueError, TypeError):
                            pass

        except Exception as exc:
            self.print_to_console(
                f"[Config] Não foi possível carregar configurações: {exc}\n"
            )

    def save_settings(self):
        data = {}

        for module, variables in self.config_registry.items():
            data[module] = {}
            for key, variable in variables.items():
                try:
                    data[module][key] = variable.get()
                except Exception:
                    pass

        try:
            with self.CONFIG_FILE.open("w", encoding="utf-8") as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
        except Exception as exc:
            self.print_to_console(
                f"[Config] Não foi possível salvar configurações: {exc}\n"
            )

    def toggle_opt_options(self):
        is_crop = self.opt_method_var.get() == "Crop"

        self.crop_entry.configure(
            state="normal" if is_crop else "disabled"
        )

        self.down_combo.configure(
            state="disabled" if is_crop else "readonly"
        )

        self.down_scale_entry.configure(
            state="disabled" if is_crop else "normal"
        )

    # ========================================================
    # Console / progresso
    # ========================================================

    def print_to_console(self, text: str):
        self.log_queue.put(str(text))

    def _drain_log_queue(self):
        try:
            while True:
                text = self.log_queue.get_nowait()

                self.console_text.configure(state=tk.NORMAL)
                self.console_text.insert(tk.END, text)

                # Evita crescimento exagerado da GUI em processamento longo.
                try:
                    lines = int(self.console_text.index("end-1c").split(".")[0])
                    if lines > 5000:
                        self.console_text.delete("1.0", "1000.0")
                except Exception:
                    pass

                self.console_text.see(tk.END)
                self.console_text.configure(state=tk.DISABLED)

        except queue.Empty:
            pass
        finally:
            self.after(50, self._drain_log_queue)

    def update_progress(self, current: int, total: int, phase_text: str = ""):
        def update():
            if total > 0:
                self.progress_var.set((current / total) * 100.0)
            else:
                self.progress_var.set(0.0)

            if phase_text:
                self.status_var.set(phase_text)

        self.after(0, update)

    # ========================================================
    # Lock / Unlock global
    # ========================================================

    def _lock_ui(self, module_name: str):
        self.save_settings()

        self.console_text.configure(state=tk.NORMAL)
        self.console_text.delete("1.0", tk.END)
        self.console_text.configure(state=tk.DISABLED)

        self.progress_var.set(0)
        self.cancel_event.clear()

        buttons = [
            self.btn_run_calib,
            self.btn_run_debayer,
            self.btn_run_batch,
            self.btn_run_flow,
            self.btn_run_align,
        ]

        for button in buttons:
            button.configure(state="disabled")

        cancel_buttons = [
            self.btn_cancel_calib,
            self.btn_cancel_debayer,
            self.btn_cancel_batch,
            self.btn_cancel_flow,
            self.btn_cancel_align,
        ]

        for button in cancel_buttons:
            button.configure(state="disabled")

        cancel_map = {
            "Calibration": self.btn_cancel_calib,
            "Debayer": self.btn_cancel_debayer,
            "Batch": self.btn_cancel_batch,
            "Flow": self.btn_cancel_flow,
            "Align": self.btn_cancel_align,
        }

        if module_name in cancel_map:
            cancel_map[module_name].configure(state="normal")

        self.status_var.set(f"Processando Astro{module_name}...")

    def _unlock_ui(self):
        buttons = [
            self.btn_run_calib,
            self.btn_run_debayer,
            self.btn_run_batch,
            self.btn_run_flow,
            self.btn_run_align,
        ]

        for button in buttons:
            button.configure(state="normal")

        cancel_buttons = [
            self.btn_cancel_calib,
            self.btn_cancel_debayer,
            self.btn_cancel_batch,
            self.btn_cancel_flow,
            self.btn_cancel_align,
        ]

        for button in cancel_buttons:
            button.configure(state="disabled")

        self.progress_var.set(100)

    # ========================================================
    # Calibration
    # ========================================================

    def start_calibration(self):
        if self.worker and self.worker.is_alive():
            return

        from calibration_logic import run_calibration_pipeline

        try:
            input_dir = Path(self.calib_input_var.get()).expanduser().resolve()
            output_dir = Path(self.calib_output_var.get()).expanduser().resolve()

            if not input_dir.is_dir():
                raise ValueError("A pasta de LIGHTS não existe.")

            if input_dir == output_dir:
                raise ValueError("A pasta de saída deve ser diferente da entrada.")

            if self.apply_dark_var.get() and not self.dark_path_var.get():
                raise ValueError("Dark está habilitado, mas nenhum Dark foi informado.")

            if self.apply_flat_var.get() and not self.flat_path_var.get():
                raise ValueError("Flat está habilitado, mas nenhum Flat foi informado.")

            config = {
                "input_dir": str(input_dir),
                "output_dir": str(output_dir),
                "apply_dark": self.apply_dark_var.get(),
                "dark_path": self.dark_path_var.get(),
                "apply_flat": self.apply_flat_var.get(),
                "flat_path": self.flat_path_var.get(),
                "create_master": self.calib_create_master_var.get(),
                "overwrite": self.calib_overwrite_var.get(),
            }

        except Exception as exc:
            messagebox.showerror("Parâmetros inválidos", str(exc), parent=self)
            return

        self._lock_ui("Calibration")

        self.worker = threading.Thread(
            target=self._run_calibration_worker,
            args=(config,),
            daemon=True,
        )
        self.worker.start()

    def _run_calibration_worker(self, config):
        from calibration_logic import run_calibration_pipeline

        try:
            run_calibration_pipeline(
                config,
                self.print_to_console,
                self.update_progress,
                self.cancel_event,
            )

            status = (
                "Calibration cancelado."
                if self.cancel_event.is_set()
                else "AstroCalibration concluído."
            )
            self.after(0, lambda: self.status_var.set(status))

        except Exception as exc:
            self.print_to_console(
                f"\nERRO FATAL NO ASTROCALIBRATION:\n{exc}\n"
            )
            self.after(
                0,
                lambda: self.status_var.set("Erro no AstroCalibration."),
            )
        finally:
            self.after(0, self._unlock_ui)

    # ========================================================
    # Debayer
    # ========================================================

    def start_debayer(self):
        if self.worker and self.worker.is_alive():
            return

        from debayer_logic import process_debayer

        try:
            input_dir = Path(self.debayer_input_var.get()).expanduser().resolve()
            output_dir = Path(self.debayer_output_var.get()).expanduser().resolve()

            if not input_dir.is_dir():
                raise ValueError("A pasta calibrada não existe.")

            if input_dir == output_dir:
                raise ValueError("A pasta de saída deve ser diferente da entrada.")

            config = {
                "input_dir": str(input_dir),
                "output_dir": str(output_dir),
                "pattern": self.debayer_pattern_var.get(),
                "method": self.debayer_method_var.get(),
                "overwrite": self.debayer_overwrite_var.get(),
            }

        except Exception as exc:
            messagebox.showerror("Parâmetros inválidos", str(exc), parent=self)
            return

        self._lock_ui("Debayer")

        self.worker = threading.Thread(
            target=self._run_debayer_worker,
            args=(config,),
            daemon=True,
        )
        self.worker.start()

    def _run_debayer_worker(self, config):
        from debayer_logic import process_debayer

        try:
            process_debayer(
                config,
                self.print_to_console,
                self.update_progress,
                self.cancel_event,
            )

            status = (
                "Debayer cancelado."
                if self.cancel_event.is_set()
                else "AstroDebayer concluído."
            )
            self.after(0, lambda: self.status_var.set(status))

        except Exception as exc:
            self.print_to_console(
                f"\nERRO FATAL NO ASTRODEBAYER:\n{exc}\n"
            )
            self.after(
                0,
                lambda: self.status_var.set("Erro no AstroDebayer."),
            )
        finally:
            self.after(0, self._unlock_ui)

    # ========================================================
    # Batch
    # ========================================================

    def start_batch_processing(self):
        if self.worker and self.worker.is_alive():
            return

        try:
            input_dir = Path(
                self.batch_input_dir_var.get()
            ).expanduser().resolve()

            output_dir = Path(
                self.batch_dir_var.get()
            ).expanduser().resolve()

            if not input_dir.is_dir():
                raise ValueError("A pasta de origem não existe.")

            if input_dir == output_dir:
                raise ValueError(
                    "A pasta de destino deve ser diferente da origem."
                )

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
            messagebox.showerror(
                "Parâmetros inválidos",
                str(exc),
                parent=self,
            )
            return

        self._lock_ui("Batch")

        self.worker = threading.Thread(
            target=self.run_batch_logic,
            args=(config,),
            daemon=True,
        )
        self.worker.start()

    def run_batch_logic(self, config):
        from batch_logic import process_fits_logic

        try:
            processed, batches = process_fits_logic(
                config,
                self.print_to_console,
                self.update_progress,
                self.cancel_event,
            )

            if self.cancel_event.is_set():
                status = f"Batch cancelado. {processed} processados."
            else:
                status = (
                    f"AstroBatch concluído. "
                    f"{processed} processados, {batches} batches."
                )

            self.after(0, lambda: self.status_var.set(status))

        except Exception as exc:
            self.print_to_console(
                f"\nERRO FATAL NO ASTROBATCH:\n{exc}\n"
            )
            self.after(
                0,
                lambda: self.status_var.set("Erro no AstroBatch."),
            )
        finally:
            self.after(0, self._unlock_ui)

    # ========================================================
    # Flow
    # ========================================================

    def start_flow_processing(self):
        if self.worker and self.worker.is_alive():
            return

        batch_dir = Path(
            self.batch_dir_var.get()
        ).expanduser().resolve()

        if not batch_dir.is_dir():
            messagebox.showerror(
                "Erro",
                "Pasta Base não encontrada.",
                parent=self,
            )
            return

        config = {
            "custom_anchors": dict(self.custom_anchors),
            "global_master": self.flow_global_master_var.get(),
            "fwhm": self.flow_fwhm_var.get(),
            "sigma": self.flow_sigma_var.get(),
            "matching_radius": self.flow_matching_radius_var.get(),
            "ransac": self.flow_ransac_var.get(),
            "debug_images": self.flow_debug_var.get(),
            "min_stars": self.flow_min_stars_var.get(),
            "min_inliers": self.flow_min_inliers_var.get(),
            "min_ratio": self.flow_min_ratio_var.get(),
            "max_stars": 150,
            "engine": self.flow_engine_var.get(),
        }

        self.save_settings()
        self._lock_ui("Flow")

        self.worker = threading.Thread(
            target=self.run_flow_logic,
            args=(batch_dir, config),
            daemon=True,
        )
        self.worker.start()

    def run_flow_logic(self, batch_dir, config):
        from astroflow_logic import process_all_flows

        try:
            process_all_flows(
                batch_dir,
                config,
                self.print_to_console,
                self.update_progress,
                self.cancel_event,
            )

            status = (
                "AstroFlow cancelado."
                if self.cancel_event.is_set()
                else "AstroFlow concluído."
            )
            self.after(0, lambda: self.status_var.set(status))

        except Exception as exc:
            self.print_to_console(
                f"\nERRO FATAL NO ASTROFLOW:\n{exc}\n"
            )
            self.after(
                0,
                lambda: self.status_var.set("Erro no AstroFlow."),
            )
        finally:
            self.after(0, self._unlock_ui)

    # ========================================================
    # Align
    # ========================================================

    def start_align_processing(self):
        if self.worker and self.worker.is_alive():
            return

        try:
            base_dir = Path(
                self.batch_dir_var.get()
            ).expanduser().resolve()

            output_dir = Path(
                self.align_output_dir_var.get()
            ).expanduser().resolve()

            if not base_dir.is_dir():
                raise ValueError("A pasta base de batches não existe.")

            if base_dir == output_dir:
                raise ValueError(
                    "A pasta de destino deve ser diferente da pasta base."
                )

            config = {
                "interpolation": self.align_interpolation_var.get(),
                "overwrite": self.align_overwrite_var.get(),
                "dry_run": self.align_dry_run_var.get(),
                "keep_header": self.align_keep_header_var.get(),
            }

        except Exception as exc:
            messagebox.showerror(
                "Parâmetros inválidos",
                str(exc),
                parent=self,
            )
            return

        self._lock_ui("Align")

        self.worker = threading.Thread(
            target=self.run_align_logic,
            args=(base_dir, output_dir, config),
            daemon=True,
        )
        self.worker.start()

    def run_align_logic(self, base_dir, output_dir, config_dict):
        from astroalign_logic import process_all_alignments

        try:
            process_all_alignments(
                base_dir,
                output_dir,
                config_dict,
                self.print_to_console,
                self.update_progress,
                self.cancel_event,
            )

            status = (
                "AstroAlign cancelado."
                if self.cancel_event.is_set()
                else "AstroAlign concluído."
            )
            self.after(0, lambda: self.status_var.set(status))

        except Exception as exc:
            self.print_to_console(
                f"\nERRO FATAL NO ASTROALIGN:\n{exc}\n"
            )
            self.after(
                0,
                lambda: self.status_var.set("Erro no AstroAlign."),
            )
        finally:
            self.after(0, self._unlock_ui)

    # ========================================================
    # AstroFlow: Preview
    # ========================================================

    def show_astroflow_preview(self):
        base_dir_str = self.batch_dir_var.get()

        if not base_dir_str:
            messagebox.showerror(
                "Erro",
                "Selecione a Pasta Base das Batches primeiro.",
                parent=self,
            )
            return

        base_dir = Path(base_dir_str).expanduser().resolve()

        if not base_dir.is_dir():
            messagebox.showerror(
                "Erro",
                "A Pasta Base não existe.",
                parent=self,
            )
            return

        batch_folders = sorted(
            [
                d
                for d in base_dir.iterdir()
                if d.is_dir() and "batch" in d.name.lower()
            ]
        )

        if not batch_folders:
            messagebox.showerror(
                "Erro",
                "Nenhuma pasta de Batch encontrada.",
                parent=self,
            )
            return

        target_batch = batch_folders[0]

        config = {
            "fwhm": self.flow_fwhm_var.get(),
            "sigma": self.flow_sigma_var.get(),
            "max_stars": 250,
            "engine": self.flow_engine_var.get(),
        }

        from astroflow_logic import preview_star_detection

        try:
            img_preview, count, fwhm_measured = preview_star_detection(
                target_batch,
                config,
            )

            if img_preview is None:
                messagebox.showwarning(
                    "Aviso",
                    "Não foi possível carregar a imagem para preview.",
                    parent=self,
                )
                return

        except Exception as exc:
            messagebox.showerror(
                "Erro",
                f"Falha ao gerar preview:\n{exc}",
                parent=self,
            )
            return

        prev_window = tk.Toplevel(self)
        prev_window.title(f"AstroFlow — Preview — {target_batch.name}")
        prev_window.geometry("850x700")
        prev_window.minsize(650, 500)

        info = ttk.Frame(prev_window, padding=12)
        info.pack(fill=tk.X)

        ttk.Label(
            info,
            text=(
                f"{target_batch.name}   •   "
                f"{count} estrelas   •   "
                f"FWHM {fwhm_measured:.1f}px   •   "
                f"Engine: {self.flow_engine_var.get()}"
            ),
            font=("Segoe UI Semibold", 10),
        ).pack(anchor="w")

        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        from matplotlib.figure import Figure

        fig = Figure(figsize=(8, 5), dpi=90)
        ax = fig.add_subplot(111)

        if img_preview.ndim == 3:
            ax.imshow(cv2.cvtColor(img_preview, cv2.COLOR_BGR2RGB))
        else:
            ax.imshow(img_preview, cmap="gray")

        ax.set_title("Detecção de estrelas")
        ax.axis("off")
        fig.tight_layout(pad=0.5)

        canvas = FigureCanvasTkAgg(fig, master=prev_window)
        canvas.draw()
        canvas.get_tk_widget().pack(
            fill=tk.BOTH,
            expand=True,
            padx=12,
            pady=5,
        )

        ttk.Label(
            prev_window,
            text="Ajuste FWHM/Sigma na aba Flow e execute o preview novamente.",
            style="Muted.TLabel",
        ).pack(pady=(0, 10))

    # ========================================================
    # AstroFlow: visualização
    # ========================================================

    def show_flow_visualization(self):
        base_dir = Path(
            self.batch_dir_var.get()
        ).expanduser().resolve()

        global_json = base_dir / "global_flow.json"

        if not global_json.exists():
            messagebox.showerror(
                "Erro",
                "Execute o AstroFlow primeiro. "
                "global_flow.json não encontrado.",
                parent=self,
            )
            return

        try:
            with global_json.open("r", encoding="utf-8") as f:
                global_data = json.load(f)

            points_x = []
            points_y = []

            center_pt = np.array([0, 0, 1])

            for batch_name, g_info in global_data["batches"].items():
                g_matrix = np.array(g_info["matrix"])
                local_json = base_dir / batch_name / "flow_local.json"

                if not local_json.exists():
                    continue

                with local_json.open("r", encoding="utf-8") as f:
                    local_data = json.load(f)

                for _, l_info in local_data["frames"].items():
                    l_matrix = np.array(l_info["matrix"])
                    abs_matrix = np.dot(g_matrix, l_matrix)

                    transformed_pt = np.dot(abs_matrix, center_pt)

                    points_x.append(transformed_pt[0])
                    points_y.append(transformed_pt[1])

        except Exception as exc:
            messagebox.showerror(
                "Erro",
                f"Falha ao carregar os dados do Flow:\n{exc}",
                parent=self,
            )
            return

        if not points_x:
            messagebox.showwarning(
                "Aviso",
                "Nenhum ponto de trajetória foi encontrado.",
                parent=self,
            )
            return

        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        from matplotlib.figure import Figure

        win = tk.Toplevel(self)
        win.title("AstroFlow — Trajetória")
        win.geometry("850x600")

        fig = Figure(figsize=(8, 5), dpi=90)
        ax = fig.add_subplot(111)

        ax.plot(
            points_x,
            points_y,
            marker="o",
            markersize=2,
            linestyle="-",
            alpha=0.6,
            rasterized=True,
        )

        ax.scatter(
            points_x[0],
            points_y[0],
            s=40,
            label="Início",
            zorder=5,
        )

        ax.scatter(
            points_x[-1],
            points_y[-1],
            s=40,
            label="Fim",
            zorder=5,
        )

        ax.set_title("Drift analisado pelo AstroFlow")
        ax.set_xlabel("Deslocamento X (pixels)")
        ax.set_ylabel("Deslocamento Y (pixels)")
        ax.invert_yaxis()
        ax.grid(True, linestyle="--", alpha=0.2)
        ax.legend(fontsize=8)

        fig.tight_layout()

        canvas = FigureCanvasTkAgg(fig, master=win)
        canvas.draw()
        canvas.get_tk_widget().pack(
            fill=tk.BOTH,
            expand=True,
            padx=10,
            pady=10,
        )

    # ========================================================
    # AstroFlow: seletor de âncora
    # ========================================================

    def open_anchor_selector(self):
        import functools
        import threading

        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        from matplotlib.figure import Figure
        from astroflow_logic import load_fits_data, extract_luminance

        base_dir_str = self.batch_dir_var.get()

        if not base_dir_str:
            messagebox.showerror(
                "Erro",
                "Selecione a Pasta Base primeiro.",
                parent=self,
            )
            return

        base_dir = Path(base_dir_str).expanduser().resolve()

        if not base_dir.is_dir():
            messagebox.showerror(
                "Erro",
                "A Pasta Base não existe.",
                parent=self,
            )
            return

        batch_folders = sorted(
            [
                d
                for d in base_dir.iterdir()
                if d.is_dir() and "batch" in d.name.lower()
            ]
        )

        if not batch_folders:
            messagebox.showerror(
                "Erro",
                "Nenhuma pasta de Batch encontrada.",
                parent=self,
            )
            return

        win = tk.Toplevel(self)
        win.title("AstroFlow — Selecionar referência")
        win.geometry("1000x800")
        win.minsize(750, 600)

        top = ttk.Frame(win, padding=12)
        top.pack(fill=tk.X)

        ttk.Label(top, text="Batch:").pack(side=tk.LEFT)

        batch_combo = ttk.Combobox(
            top,
            values=[b.name for b in batch_folders],
            state="readonly",
            width=18,
        )
        batch_combo.pack(side=tk.LEFT, padx=(7, 15))

        ttk.Label(top, text="Frame:").pack(side=tk.LEFT)

        frame_combo = ttk.Combobox(
            top,
            state="readonly",
            width=34,
        )
        frame_combo.pack(side=tk.LEFT, padx=7)

        image_frame = ttk.Frame(win, padding=(12, 0))
        image_frame.pack(fill=tk.BOTH, expand=True)

        fig = Figure(figsize=(8, 6), dpi=85)
        ax = fig.add_subplot(111)
        ax.axis("off")

        ax_img = ax.imshow(
            np.zeros((10, 10)),
            cmap="gray",
            interpolation="nearest",
            rasterized=True,
            vmin=0,
            vmax=255,
        )

        title_obj = ax.set_title(
            "Stretched Preview",
            fontsize=10,
        )

        canvas = FigureCanvasTkAgg(fig, master=image_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        @functools.lru_cache(maxsize=40)
        def get_preview_data(filepath):
            data, header = load_fits_data(filepath)
            l_data = extract_luminance(data, header)

            h, w = l_data.shape[:2]
            small_w = max(w // 3, 1)
            small_h = max(h // 3, 1)

            return cv2.resize(
                l_data,
                (small_w, small_h),
                interpolation=cv2.INTER_NEAREST,
            )

        generation = {"id": 0}

        def update_frames(event=None):
            b_name = batch_combo.get()
            if not b_name:
                return

            b_path = base_dir / b_name

            fits_files = sorted(
                [
                    f.name
                    for f in b_path.iterdir()
                    if f.is_file()
                    and f.suffix.lower() in {".fit", ".fits", ".fts"}
                ]
            )

            frame_combo.configure(values=fits_files)

            if fits_files:
                selected = self.custom_anchors.get(
                    b_name,
                    fits_files[len(fits_files) // 2],
                )

                if selected not in fits_files:
                    selected = fits_files[0]

                frame_combo.set(selected)
                update_image()

        def update_image(event=None):
            b_name = batch_combo.get()
            f_name = frame_combo.get()

            if not b_name or not f_name:
                return

            generation["id"] += 1
            current_id = generation["id"]

            batch_combo.configure(state="disabled")
            frame_combo.configure(state="disabled")

            def worker():
                try:
                    f_path = base_dir / b_name / f_name
                    small_data = get_preview_data(f_path)

                    median = np.median(small_data)
                    p25, p75 = np.percentile(
                        small_data,
                        [25, 75],
                    )

                    std = max(
                        (p75 - p25) / 1.35,
                        1e-5,
                    )

                    vmin = median - 0.5 * std
                    vmax = median + 6.0 * std

                    denominator = max(vmax - vmin, 1e-5)

                    norm = np.clip(
                        (small_data - vmin) / denominator,
                        0,
                        1,
                    ) * 255

                    img_8u = norm.astype(np.uint8)

                    def update_gui():
                        if current_id != generation["id"]:
                            return

                        ax_img.set_data(img_8u)
                        ax_img.set_extent(
                            (
                                0,
                                img_8u.shape[1],
                                img_8u.shape[0],
                                0,
                            )
                        )

                        title_obj.set_text(
                            f"Stretched Preview — {b_name} / {f_name}"
                        )

                        canvas.draw_idle()

                        batch_combo.configure(state="readonly")
                        frame_combo.configure(state="readonly")

                    self.after(0, update_gui)

                except Exception as exc:
                    def handle_error():
                        batch_combo.configure(state="readonly")
                        frame_combo.configure(state="readonly")
                        self.print_to_console(
                            f"[Preview] Erro: {exc}\n"
                        )

                    self.after(0, handle_error)

            threading.Thread(
                target=worker,
                daemon=True,
            ).start()

        batch_combo.bind(
            "<<ComboboxSelected>>",
            update_frames,
        )

        frame_combo.bind(
            "<<ComboboxSelected>>",
            update_image,
        )

        if batch_folders:
            batch_combo.set(batch_folders[0].name)
            update_frames()

        buttons = ttk.Frame(win, padding=12)
        buttons.pack(fill=tk.X)

        def save_selection():
            b_name = batch_combo.get()
            f_name = frame_combo.get()

            if not b_name or not f_name:
                return

            self.custom_anchors[b_name] = f_name

            self.print_to_console(
                f"[AstroFlow] Referência da {b_name}: {f_name}\n"
            )

            messagebox.showinfo(
                "Referência salva",
                f"Frame definido como referência da {b_name}:\n\n{f_name}",
                parent=win,
            )

        ttk.Button(
            buttons,
            text="✓  Definir como referência",
            style="Accent.TButton",
            command=save_selection,
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5), ipady=5)

        ttk.Button(
            buttons,
            text="Fechar",
            command=win.destroy,
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 0), ipady=5)

    # ========================================================
    # Global Master
    # ========================================================

    def _update_global_master_options(self, *args):
        if not hasattr(self, "combo_global_master"):
            return

        base_dir_str = self.batch_dir_var.get()
        values = ["Auto"]

        if base_dir_str:
            base_dir = Path(
                base_dir_str
            ).expanduser().resolve()

            if base_dir.is_dir():
                try:
                    batches = sorted(
                        [
                            d.name
                            for d in base_dir.iterdir()
                            if d.is_dir()
                            and "batch" in d.name.lower()
                        ]
                    )
                    values.extend(batches)
                except OSError:
                    pass

        self.combo_global_master["values"] = values

        if self.flow_global_master_var.get() not in values:
            self.flow_global_master_var.set("Auto")

    # ========================================================
    # Cancelamento / fechamento
    # ========================================================

    def cancel_processing(self):
        if self.worker and self.worker.is_alive():
            self.print_to_console(
                "\n[GUI] Solicitação de cancelamento enviada...\n"
            )
            self.cancel_event.set()
            self.status_var.set("Cancelamento solicitado...")

    def _on_close(self):
        if self.worker and self.worker.is_alive():
            answer = messagebox.askyesno(
                "Processamento ativo",
                "Existe um processamento em andamento.\n\n"
                "Deseja solicitar o cancelamento e fechar a aplicação?",
                parent=self,
            )

            if not answer:
                return

            self.cancel_event.set()

        self.save_settings()
        self.destroy()


if __name__ == "__main__":
    app = AstroProcessManager()
    app.mainloop()