# ============================================================
# ARQUIVO 2: main.py (Controller Atualizado)
# ============================================================
import json
import queue
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext, ttk

import numpy as np

# ============================================================
# Imports da lógica de processamento
# ============================================================
from batch_logic import ProcessingConfig
from app.application.runner import PipelineRunner
from app.application.log_buffer import ActivityBuffer
from app.application.commands import ResourceSettings
from app.application.pipelines import execute_pipeline
from app.infrastructure.json_store import SettingsRepository
from views.align_view import AlignView
from views.batch_view import BatchView

# Importando as views separadas
from views.calibration_view import CalibrationView
from views.flow_view import FlowView
from views.stacking_view import StackingView
from views.hdr_view import HDRView
from views.hdr_model import HDRViewModel
from views.scrollable_host import ScrollableHost


class AstroProcessManager(tk.Tk):
    APP_NAME = "Astro Process Manager"
    CONFIG_FILE = Path("astro_config.json")
    PIPELINE_BUTTONS = (("Calibration", "calib"), ("Batch", "batch"), ("Flow", "flow"),
                        ("Align", "align"), ("Stack", "stack"), ("HDR", "hdr"))

    BG = "#ffffff"
    PANEL = "#ffffff"
    BORDER = "#d5dde8"
    TEXT = "#172033"
    MUTED = "#526070"
    ACCENT = "#2563eb"
    ACCENT_ACTIVE = "#1d4ed8"
    DANGER = "#dc2626"

    def __init__(self):
        super().__init__()

        self.title(self.APP_NAME)
        self.geometry("1120x900")
        self.minsize(940, 700)
        self.configure(bg=self.BG)

        self.log_queue = ActivityBuffer()
        self.worker = None
        self.runner = PipelineRunner(self.print_to_console)
        self.cancel_event = self.runner.cancel_event
        self._closing = False
        self.custom_anchors = {}

        self._init_variables()
        self._configure_style()
        self.load_settings()
        self._create_widgets()
        self._start_cpu_kernel_warmup()

    def _start_cpu_kernel_warmup(self) -> None:
        """Compile/cache CPU kernels without delaying the UI startup."""

        def warm() -> None:
            try:
                from cpu_kernels import warm_cpu_kernels

                warm_cpu_kernels()
            except Exception as exc:
                self.log_queue.put(f"[CPU] Kernel warm-up unavailable: {exc}\n")

        threading.Thread(target=warm, name="cpu_kernel_warmup", daemon=True).start()

        self.after(50, self._drain_log_queue)
        self.after(50, self._drain_operation_events)
        self.after(500, self._tick_operation_clock)
        self.after(100, self._update_global_master_options)
        self.after(150, self.refresh_flow_reference_preview)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

    def _init_variables(self):
        # ---------- Global ----------
        self.batch_dir_var = tk.StringVar()
        self.status_var = tk.StringVar(value="Pronto.")
        self.progress_var = tk.DoubleVar(value=0.0)
        self.progress_detail_var = tk.StringVar(value="Aguardando uma operacao.")
        self.elapsed_var = tk.StringVar(value="")
        self.console_autoscroll_var = tk.BooleanVar(value=True)
        self._operation_started_at = None
        self._progress_current = 0
        self._progress_total = 0

        # ---------- Calibration ----------
        self.calib_input_var = tk.StringVar()
        self.calib_output_var = tk.StringVar()
        self.apply_dark_var = tk.BooleanVar(value=True)
        self.dark_path_var = tk.StringVar()
        self.apply_flat_var = tk.BooleanVar(value=True)
        self.flat_path_var = tk.StringVar()
        self.calib_create_master_var = tk.BooleanVar(value=True)
        self.calib_overwrite_var = tk.BooleanVar(value=False)
        self.hdr_input_var = tk.StringVar(); self.hdr_output_var = tk.StringVar()
        self.hdr_saturation_var = tk.StringVar(); self.hdr_noise_var = tk.DoubleVar(value=1.0)
        self.hdr_rowband_var = tk.IntVar(value=256); self.hdr_exptime_var = tk.StringVar()

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
        self.flow_profile_var = tk.StringVar(value="Stable")
        self.flow_detector_engine_var = tk.StringVar(value="")
        self.flow_transform_fallback_var = tk.StringVar(value="Disabled")

        # ---------- AstroAlign ----------
        self.align_output_dir_var = tk.StringVar()
        self.align_debayer_pattern_var = tk.StringVar(value="Auto")
        self.align_debayer_method_var = tk.StringVar(value="Bilinear")
        self.align_interpolation_var = tk.StringVar(value="Lanczos")
        self.align_rgb_registration_var = tk.BooleanVar(value=True)
        self.align_overwrite_var = tk.BooleanVar(value=False)
        self.align_dry_run_var = tk.BooleanVar(value=False)
        self.align_keep_header_var = tk.BooleanVar(value=True)
        self.align_delete_intermediates_var = tk.BooleanVar(value=False)
        self.align_compress_output_var = tk.BooleanVar(value=True)
        self.align_profile_var = tk.StringVar(value="Stable")
        self.align_warp_engine_var = tk.StringVar(value="")
        self.resource_memory_var = tk.IntVar(value=512)
        self.resource_workers_var = tk.IntVar(value=2)
        self.align_quality_gate_var = tk.BooleanVar(value=False)
        self.align_quality_shift_var = tk.DoubleVar(value=1.5)

        # ---- AstroStack ----
        # ---- Diretórios ----
        self.stack_output_dir_var = tk.StringVar(value="")
        self.stack_input_dir_var = tk.StringVar(value="")

        # ---- Seleção de Frames ----
        self.stack_selection_mode_var = tk.StringVar(value="BestPercentage")
        self.stack_selection_percentage_var = tk.DoubleVar(value=80.0)
        self.stack_selection_percentage_text_var = tk.StringVar(value="80%")
        self.stack_selection_metric_var = tk.StringVar(value="quality")
        self.stack_trail_filter_var = tk.BooleanVar(value=False)
        self.stack_min_roundness_var = tk.DoubleVar(value=0.65)
        self.stack_min_shape_stars_var = tk.IntVar(value=5)

        # ---- Combinação ----
        self.stack_method_var = tk.StringVar(value="Median")

        # ---- Rejeição de Outliers ----
        self.stack_rejection_method_var = tk.StringVar(value="SigmaClip")
        self.stack_rejection_low_var = tk.DoubleVar(value=3.0)
        self.stack_rejection_high_var = tk.DoubleVar(
            value=3.0
        )  # Corrigido para 3.0 (padrão)

        # ---- Normalização ----
        self.stack_normalize_var = tk.BooleanVar(value=True)
        self.stack_normalize_method_var = tk.StringVar(value="Median")

        # ---- Pós-processamento ----
        self.stack_dither_correction_var = tk.BooleanVar(value=False)

        # ---- Saída ----
        self.stack_output_name_var = tk.StringVar(value="stacked_image.fits")
        self.stack_output_bit_depth_var = tk.StringVar(value="16-bit")
        self.stack_compress_var = tk.BooleanVar(value=True)
        self.stack_profile_var = tk.StringVar(value="Stable")
        self.stack_reducer_engine_var = tk.StringVar(value="")

        # Registry atualizado
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
                "engine_profile": self.flow_profile_var,
                "detector_engine": self.flow_detector_engine_var,
                "transform_fallback": self.flow_transform_fallback_var,
            },
            "AstroAlign": {
                "output_dir": self.align_output_dir_var,
                "debayer_pattern": self.align_debayer_pattern_var,
                "debayer_method": self.align_debayer_method_var,
                "interpolation": self.align_interpolation_var,
                "rgb_registration": self.align_rgb_registration_var,
                "overwrite": self.align_overwrite_var,
                "dry_run": self.align_dry_run_var,
                "keep_header": self.align_keep_header_var,
                "delete_intermediates": self.align_delete_intermediates_var,
                "compress_output": self.align_compress_output_var,
                "engine_profile": self.align_profile_var,
                "warp_engine": self.align_warp_engine_var,
            },
            "AstroStack": {
                # ---- Diretórios ----
                "input_dir": self.stack_input_dir_var,
                "output_dir": self.stack_output_dir_var,
                # ---- Seleção de Frames ----
                "selection_mode": self.stack_selection_mode_var,
                "selection_percentage": self.stack_selection_percentage_var,
                "selection_metric": self.stack_selection_metric_var,
                "trail_filter_enabled": self.stack_trail_filter_var,
                "min_roundness": self.stack_min_roundness_var,
                "min_shape_stars": self.stack_min_shape_stars_var,
                # ---- Combinação ----
                "method": self.stack_method_var,
                # ---- Rejeição de Outliers ----
                "rejection_method": self.stack_rejection_method_var,
                "rejection_low": self.stack_rejection_low_var,
                "rejection_high": self.stack_rejection_high_var,
                # ---- Normalização ----
                "normalize": self.stack_normalize_var,
                "normalize_method": self.stack_normalize_method_var,
                # ---- Pós-processamento ----
                "apply_dither_correction": self.stack_dither_correction_var,
                # ---- Saída ----
                "output_name": self.stack_output_name_var,
                "output_bit_depth": self.stack_output_bit_depth_var,
                "compress_output": self.stack_compress_var,
                "engine_profile": self.stack_profile_var,
                "reducer_engine": self.stack_reducer_engine_var,
            },
            "Resources": {"memory_mb": self.resource_memory_var, "workers": self.resource_workers_var},
            "AlignQuality": {"enabled": self.align_quality_gate_var, "max_shift": self.align_quality_shift_var},
            "AstroHDR": {"input_dir": self.hdr_input_var, "output_path": self.hdr_output_var, "saturation": self.hdr_saturation_var, "noise_floor": self.hdr_noise_var, "row_band": self.hdr_rowband_var, "exptime_override": self.hdr_exptime_var},
        }

    def _configure_style(self):
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        # Tk's native listbox theme does not always inherit ttk combobox
        # colors, so set it explicitly as well.
        self.option_add("*TCombobox*Listbox.background", "#ffffff")
        self.option_add("*TCombobox*Listbox.foreground", self.TEXT)
        self.option_add("*TCombobox*Listbox.selectBackground", "#dbeafe")
        self.option_add("*TCombobox*Listbox.selectForeground", self.TEXT)

        style.configure(
            ".", font=("Segoe UI", 10), background=self.BG, foreground=self.TEXT
        )
        style.configure("TFrame", background=self.BG)
        style.configure("Panel.TFrame", background=self.PANEL)
        style.configure("TLabel", background=self.BG, foreground=self.TEXT)
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
        # Explicit field colors prevent platform themes from rendering typed
        # values with low contrast or inheriting the panel background.
        style.configure(
            "TEntry",
            padding=(8, 6),
            foreground=self.TEXT,
            fieldbackground="#ffffff",
            insertcolor=self.TEXT,
        )
        style.map(
            "TEntry",
            fieldbackground=[("disabled", "#eef2f7"), ("!disabled", "#ffffff")],
            foreground=[("disabled", "#7a8798"), ("!disabled", self.TEXT)],
        )
        style.configure(
            "TCombobox",
            padding=(7, 5),
            foreground=self.TEXT,
            fieldbackground="#ffffff",
            background="#ffffff",
            arrowcolor=self.TEXT,
        )
        style.map(
            "TCombobox",
            fieldbackground=[("readonly", "#ffffff"), ("disabled", "#eef2f7")],
            foreground=[("readonly", self.TEXT), ("disabled", "#7a8798")],
            selectbackground=[("readonly", "#dbeafe")],
            selectforeground=[("readonly", self.TEXT)],
        )
        style.configure("TButton", padding=(12, 7), font=("Segoe UI Semibold", 9))
        style.configure(
            "Accent.TButton", background=self.ACCENT, foreground="white", borderwidth=0
        )
        style.map(
            "Accent.TButton",
            background=[
                ("active", self.ACCENT_ACTIVE),
                ("pressed", self.ACCENT_ACTIVE),
            ],
            foreground=[("disabled", "#aeb7c4"), ("!disabled", "white")],
        )
        style.configure(
            "Danger.TButton", background=self.DANGER, foreground="white", borderwidth=0
        )
        style.configure(
            "TNotebook", background=self.BG, borderwidth=0, tabmargins=(0, 0, 0, 0)
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
        style.configure("TCheckbutton", background=self.PANEL, foreground=self.TEXT)
        style.map(
            "TCheckbutton",
            foreground=[("disabled", "#7a8798"), ("!disabled", self.TEXT)],
        )
        style.configure("TRadiobutton", background=self.PANEL, foreground=self.TEXT)
        style.map(
            "TRadiobutton",
            foreground=[("disabled", "#7a8798"), ("!disabled", self.TEXT)],
        )
        style.configure(
            "TSpinbox",
            foreground=self.TEXT,
            fieldbackground="#ffffff",
            insertcolor=self.TEXT,
        )
        style.configure("TNotebook", tabmargins=(4, 4, 4, 0))

    def _create_widgets(self):
        self.columnconfigure(0, weight=1)
        self.rowconfigure(2, weight=1)

        header = ttk.Frame(self)
        header.grid(row=0, column=0, sticky="ew", padx=22, pady=(18, 10))
        header.columnconfigure(0, weight=1)

        ttk.Label(header, text="Astro Process Manager", style="Title.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        ttk.Label(
            header,
            text="Calibração  →  Batch  →  Flow  →  Align + Debayer  →  Stack / HDR",
            style="Subtitle.TLabel",
        ).grid(row=1, column=0, sticky="w", pady=(2, 0))

        global_frame = ttk.LabelFrame(
            self,
            text="Projeto / Diretório de Batches",
            style="Section.TLabelframe",
            padding=12,
        )
        global_frame.grid(row=1, column=0, sticky="ew", padx=22, pady=(0, 10))
        global_frame.columnconfigure(1, weight=1)

        ttk.Label(global_frame, text="Pasta Base:").grid(row=0, column=0, sticky="w")
        ttk.Entry(global_frame, textvariable=self.batch_dir_var).grid(
            row=0, column=1, sticky="ew", padx=8
        )
        ttk.Button(
            global_frame,
            text="Selecionar pasta",
            command=lambda: self.browse_dir(self.batch_dir_var),
        ).grid(row=0, column=2)

        resources = ttk.Frame(global_frame)
        resources.grid(row=1, column=0, columnspan=3, sticky="w", pady=(8,0))
        ttk.Label(resources, text="Flow / Align: workers máx.").pack(side="left")
        ttk.Entry(resources, textvariable=self.resource_workers_var, width=6).pack(side="left", padx=8)
        ttk.Label(resources, text="RAM estimada (MiB)").pack(side="left")
        ttk.Entry(resources, textvariable=self.resource_memory_var, width=9).pack(side="left", padx=8)

        notebook_container = ttk.Frame(self)
        notebook_container.grid(row=2, column=0, sticky="nsew", padx=22, pady=(0, 10))
        notebook_container.columnconfigure(0, weight=1)
        notebook_container.rowconfigure(0, weight=1)

        self.notebook = ttk.Notebook(notebook_container)
        self.notebook.grid(row=0, column=0, sticky="nsew")

        self._tab_hosts = {}
        for key, label, view_type in (("calib", "1  Calibration", CalibrationView),
                                     ("batch", "2  Batch", BatchView),
                                     ("flow", "3  Flow", FlowView),
                                     ("align", "4  Align", AlignView)):
            host = ScrollableHost(self.notebook)
            view = view_type(host.canvas, app=self)
            host.mount(view)
            setattr(self, f"tab_{key}", view)
            self._tab_hosts[key] = host
            self.notebook.add(host, text=label)
        # Stack already owns a scroll canvas, so it needs no second wrapper.
        self.tab_stack = StackingView(self.notebook, app=self)
        self.notebook.add(self.tab_stack, text="5  Stack")
        hdr_model = HDRViewModel(
            self.hdr_input_var, self.hdr_output_var, self.hdr_saturation_var,
            self.hdr_noise_var, self.hdr_rowband_var, self.hdr_exptime_var,
            lambda: self.browse_dir(self.hdr_input_var),
            lambda: self.browse_save_file(self.hdr_output_var),
            self.use_align_output_for_hdr, self.start_hdr, self.cancel_processing)
        hdr_host = ScrollableHost(self.notebook)
        self.tab_hdr = HDRView(hdr_host.canvas, hdr_model)
        hdr_host.mount(self.tab_hdr)
        self._tab_hosts["hdr"] = hdr_host
        self.btn_run_hdr = self.tab_hdr.run_button
        self.btn_cancel_hdr = self.tab_hdr.cancel_button

        self.notebook.add(hdr_host, text="6  HDR")

        self._build_footer()

    def _build_footer(self):
        footer = ttk.Frame(self)
        footer.grid(row=3, column=0, sticky="ew", padx=22, pady=(0, 16))
        footer.columnconfigure(0, weight=1)

        status_line = ttk.Frame(footer)
        status_line.grid(row=0, column=0, sticky="ew")
        status_line.columnconfigure(0, weight=1)

        ttk.Label(
            status_line, textvariable=self.status_var, font=("Segoe UI Semibold", 9)
        ).grid(row=0, column=0, sticky="w")
        ttk.Label(
            status_line, textvariable=self.elapsed_var, style="Muted.TLabel"
        ).grid(row=0, column=1, sticky="e", padx=(12, 0))
        self.progress_bar = ttk.Progressbar(
            status_line, variable=self.progress_var, maximum=100, mode="determinate"
        )
        self.progress_bar.grid(row=1, column=0, columnspan=2, sticky="ew", pady=(5, 0))
        ttk.Label(
            status_line, textvariable=self.progress_detail_var, style="Muted.TLabel"
        ).grid(row=2, column=0, columnspan=2, sticky="w", pady=(3, 0))

        console_frame = ttk.LabelFrame(
            footer, text="Atividade", style="Section.TLabelframe", padding=7
        )
        console_frame.grid(row=1, column=0, sticky="ew", pady=(8, 0))
        console_frame.columnconfigure(0, weight=1)

        console_toolbar = ttk.Frame(console_frame)
        console_toolbar.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        console_toolbar.columnconfigure(0, weight=1)
        ttk.Label(
            console_toolbar,
            text="Mensagens de processamento, avisos e erros",
            style="Muted.TLabel",
        ).grid(row=0, column=0, sticky="w")
        ttk.Checkbutton(
            console_toolbar, text="Acompanhar", variable=self.console_autoscroll_var
        ).grid(row=0, column=1, padx=(8, 0))
        ttk.Button(console_toolbar, text="Copiar", command=self.copy_console).grid(
            row=0, column=2, padx=(8, 0)
        )
        ttk.Button(console_toolbar, text="Limpar", command=self.clear_console).grid(
            row=0, column=3, padx=(8, 0)
        )

        self.console_text = scrolledtext.ScrolledText(
            console_frame,
            height=9,
            state=tk.DISABLED,
            font=("Cascadia Mono", 9),
            bg="#17191c",
            fg="#d7dce2",
            insertbackground="white",
            relief="flat",
            borderwidth=0,
        )
        self.console_text.grid(row=1, column=0, sticky="ew")
        self.console_text.tag_configure("error", foreground="#ff8a8a")
        self.console_text.tag_configure("warning", foreground="#f7c948")
        self.console_text.tag_configure("success", foreground="#7ee787")
        self.console_text.tag_configure("system", foreground="#8ab4f8")

    def refresh_flow_reference_preview(self):
        if hasattr(self, "tab_flow"):
            self.tab_flow.refresh_reference_preview()

    def browse_dir(self, var):
        folder = filedialog.askdirectory(parent=self)
        if folder:
            var.set(folder)
            self.save_settings()
            if var == self.batch_dir_var:
                self._update_global_master_options()
                self.refresh_flow_reference_preview()

    def browse_file(self, var):
        path = filedialog.askopenfilename(
            parent=self,
            filetypes=[("FITS", "*.fits *.fit *.fts"), ("Todos os arquivos", "*.*")],
        )
        if path:
            self.save_settings()
            var.set(path)

    def browse_file_or_dir(self, var):
        folder = filedialog.askdirectory(parent=self)
        if folder:
            var.set(folder)
            self.save_settings()
            return
        self.browse_file(var)

    def browse_save_file(self, variable):
        path = filedialog.asksaveasfilename(parent=self, defaultextension=".fits",
                                          filetypes=[("FITS", "*.fits *.fit *.fts")])
        if path:
            variable.set(path)
            self.save_settings()

    def use_align_output_for_hdr(self):
        path = self.align_output_dir_var.get().strip()
        if not path:
            self.status_var.set("Defina a saída do Align antes de enviá-la ao HDR.")
            return False
        self.hdr_input_var.set(path)
        self.notebook.select(self._tab_hosts["hdr"])
        self.save_settings()
        self.status_var.set("HDR: pasta do Align selecionada. Escolha o arquivo de saída.")
        return True

    def load_settings(self):
        if not self.CONFIG_FILE.exists():
            return
        try:
            with self.CONFIG_FILE.open("r", encoding="utf-8") as f:
                data = json.load(f)

            for module, variables in self.config_registry.items():
                module_data = data.get(module, {})
                if not isinstance(module_data, dict):
                    continue
                for key, variable in variables.items():
                    if key in module_data:
                        try:
                            value = module_data[key]
                            if module == "AstroStack" and key == "output_bit_depth":
                                value = "16-bit"
                            variable.set(value)
                        except (tk.TclError, ValueError, TypeError):
                            pass

            custom_anchors = data.get("AstroFlow", {}).get("custom_anchors", {})
            if isinstance(custom_anchors, dict):
                self.custom_anchors = {
                    str(batch): str(frame) for batch, frame in custom_anchors.items()
                }
            else:
                self.custom_anchors = {}
        except Exception as exc:
            self.print_to_console(f"[Config] Erro ao carregar configurações: {exc}\n")

    def save_settings(self):
        data = {}
        for module, variables in self.config_registry.items():
            data[module] = {}
            for key, variable in variables.items():
                try:
                    data[module][key] = variable.get()
                except Exception:
                    pass

        data.setdefault("AstroFlow", {})
        data["AstroFlow"]["custom_anchors"] = dict(self.custom_anchors)

        try:
            with self.CONFIG_FILE.open("w", encoding="utf-8") as f:
                json.dump(data, f, indent=4, ensure_ascii=False)
        except Exception as exc:
            self.print_to_console(f"[Config] Erro ao salvar configurações: {exc}\n")

    def toggle_opt_options(self):
        is_crop = self.opt_method_var.get() == "Crop"
        if hasattr(self, "tab_batch"):
            try:
                self.tab_batch.crop_entry.configure(
                    state="normal" if is_crop else "disabled"
                )
                self.tab_batch.down_combo.configure(
                    state="disabled" if is_crop else "readonly"
                )
                self.tab_batch.down_scale_entry.configure(
                    state="disabled" if is_crop else "normal"
                )
            except AttributeError:
                pass

    def print_to_console(self, text: str):
        self.log_queue.put(str(text))

    def clear_console(self):
        self.console_text.configure(state=tk.NORMAL)
        self.console_text.delete("1.0", tk.END)
        self.console_text.configure(state=tk.DISABLED)

    def copy_console(self):
        content = self.console_text.get("1.0", "end-1c")
        self.clipboard_clear()
        self.clipboard_append(content)
        self.status_var.set("Atividade copiada para a area de transferencia.")

    @staticmethod
    def _console_tag(text: str) -> str:
        normalized = text.casefold()
        if "erro" in normalized or "error" in normalized or "falhou" in normalized:
            return "error"
        if "aviso" in normalized or "warning" in normalized or "cancel" in normalized:
            return "warning"
        if "conclu" in normalized or "finaliz" in normalized or "sucesso" in normalized:
            return "success"
        return "system"

    def _drain_log_queue(self):
        # Yield to input/painting even when workers continuously produce logs.
        insert_args = []
        try:
            for _ in range(200):
                text = self.log_queue.get_nowait()
                stamped_text = f"[{time.strftime('%H:%M:%S')}] {text}"
                insert_args.extend((stamped_text, self._console_tag(text)))
        except queue.Empty:
            pass
        try:
            if insert_args:
                self.console_text.configure(state=tk.NORMAL)
                self.console_text.insert(tk.END, *insert_args)
                try:
                    lines = int(self.console_text.index("end-1c").split(".")[0])
                    if lines > 5000:
                        self.console_text.delete("1.0", "1000.0")
                except Exception:
                    pass
                if self.console_autoscroll_var.get():
                    self.console_text.see(tk.END)
        finally:
            if insert_args:
                self.console_text.configure(state=tk.DISABLED)
            self.after(50, self._drain_log_queue)

    @staticmethod
    def _format_duration(seconds: float) -> str:
        seconds = max(0, round(seconds))
        minutes, seconds = divmod(seconds, 60)
        hours, minutes = divmod(minutes, 60)
        if hours:
            return f"{hours:d}:{minutes:02d}:{seconds:02d}"
        return f"{minutes:02d}:{seconds:02d}"

    def _refresh_operation_clock(self):
        if self._operation_started_at is not None:
            elapsed = time.monotonic() - self._operation_started_at
            self.elapsed_var.set(f"Tempo: {self._format_duration(elapsed)}")

    def _tick_operation_clock(self):
        self._refresh_operation_clock()
        self.after(500, self._tick_operation_clock)

    def update_progress(self, current: int, total: int, phase_text: str = ""):
        def update():
            if total > 0:
                safe_current = max(0, min(current, total))
                value = min(99.0, (safe_current / total) * 100.0)
                self.progress_var.set(value)
                self._progress_current = safe_current
                self._progress_total = total
                detail = f"{safe_current}/{total} itens  |  {value:.0f}%"
                if safe_current > 0 and self._operation_started_at is not None:
                    elapsed = time.monotonic() - self._operation_started_at
                    eta = elapsed * (total - safe_current) / safe_current
                    detail += f"  |  previsao: {self._format_duration(eta)}"
                self.progress_detail_var.set(detail)
            else:
                self.progress_var.set(0.0)
                self._progress_current = 0
                self._progress_total = 0
                self.progress_detail_var.set("Preparando operacao...")
            if phase_text:
                self.status_var.set(phase_text)
            self._refresh_operation_clock()

        self.after(0, update)

    def _operation_buttons(self):
        return [(stage, getattr(self, f"btn_run_{suffix}"), getattr(self, f"btn_cancel_{suffix}"))
                for stage, suffix in self.PIPELINE_BUTTONS]

    def _resource_settings(self):
        return ResourceSettings.from_values(self.resource_workers_var.get(), self.resource_memory_var.get())

    def _lock_ui(self, module_name: str):
        self.save_settings()
        self.clear_console()
        self.progress_var.set(0)
        self.progress_detail_var.set("Preparando operacao...")
        self._progress_current = 0
        self._progress_total = 0
        self._operation_started_at = time.monotonic()
        self._refresh_operation_clock()
        self.cancel_event.clear()

        for stage, run_button, cancel_button in self._operation_buttons():
            run_button.configure(state="disabled")
            cancel_button.configure(state="normal" if stage == module_name else "disabled")
        self.status_var.set(f"Processando Astro{module_name}...")

    def _unlock_ui(self):
        for _, run_button, cancel_button in self._operation_buttons():
            run_button.configure(state="normal")
            cancel_button.configure(state="disabled")

        self._operation_started_at = None


    def _start_operation(self, stage, *args):
        if self.runner.busy:
            return
        self._lock_ui(stage)
        try:
            self.worker = self.runner.start(
                stage, lambda log, progress, cancel: execute_pipeline(stage, args, log, progress, cancel))
        except Exception as exc:
            self._finish_operation("failed", f"{stage}: {exc}")

    def _drain_operation_events(self):
        if self._closing:
            return
        progress, result = self.runner.drain()
        if progress is not None:
            self.update_progress(progress.current, progress.total, progress.phase)
        if result is not None:
            # Progress rendering is scheduled before completion on the Tk queue.
            self.after(0, lambda result=result: self._finish_operation(result.outcome, result.message))
        self.after(50, self._drain_operation_events)

    def _run_legacy_stage(self, stage, *args):
        """Synchronous bridge retained for callers migrating to the runner."""
        try:
            result = execute_pipeline(stage, args, self.print_to_console,
                                      getattr(self, "update_progress", lambda *a: None),
                                      getattr(self, "cancel_event", threading.Event()))
            outcome, message = result.outcome, result.message
        except Exception as exc:
            self.print_to_console(f"[{stage}] {exc}\n")
            outcome, message = "failed", f"{stage}: {exc}"
        self.after(0, lambda: self._finish_operation(outcome, message))

    def _finish_operation(self, outcome: str, status: str):
        """Finalize an operation without presenting failures as 100% complete."""
        if outcome == "success":
            self.progress_var.set(100)
            self.progress_detail_var.set("Concluido  |  100%")
        elif outcome == "cancelled":
            self.progress_detail_var.set(f"Cancelado em {self.progress_var.get():.0f}%")
        else:
            self.progress_detail_var.set(
                f"Interrompido em {self.progress_var.get():.0f}% - consulte Atividade"
            )
        self.status_var.set(status)
        self.print_to_console(f"[GUI] {status}\n")
        self._unlock_ui()

    # --------------------------------------------------------
    # AstroCalibration, AstroBatch, AstroFlow Methods [Abreviação mantida idêntica]
    # --------------------------------------------------------
    def start_calibration(self):
        if self.worker and self.worker.is_alive():
            return
        try:
            config = {
                "input_dir": str(
                    Path(self.calib_input_var.get()).expanduser().resolve()
                ),
                "output_dir": str(
                    Path(self.calib_output_var.get()).expanduser().resolve()
                ),
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
        self._start_operation("Calibration", config)

    def _run_calibration_worker(self, config):
        """Compatibility entry point; UI launches use PipelineRunner."""
        AstroProcessManager._run_legacy_stage(self, "Calibration", config)

    def start_batch_processing(self):
        if self.worker and self.worker.is_alive():
            return
        try:
            config = ProcessingConfig(
                input_dir=Path(self.batch_input_dir_var.get()).expanduser().resolve(),
                output_dir=Path(self.batch_dir_var.get()).expanduser().resolve(),
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
            messagebox.showerror("Parâmetros inválidos", str(exc), parent=self)
            return
        self._start_operation("Batch", config)

    def run_batch_logic(self, config):
        """Compatibility entry point; UI launches use PipelineRunner."""
        AstroProcessManager._run_legacy_stage(self, "Batch", config)

    def start_flow_processing(self):
        if self.worker and self.worker.is_alive():
            return
        batch_dir = Path(self.batch_dir_var.get()).expanduser().resolve()
        if not batch_dir.is_dir():
            messagebox.showerror("Erro", "Pasta Base não encontrada.", parent=self)
            return
        try:
            self._resource_settings()
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
                "engine_profile": self.flow_profile_var.get(),
                "detector_engine": self.flow_detector_engine_var.get(),
                "transform_fallback": self.flow_transform_fallback_var.get(),
                "memory_budget_mb": self.resource_memory_var.get(),
                "flow_workers": self.resource_workers_var.get(),
            }
        except (tk.TclError, TypeError, ValueError) as exc:
            messagebox.showerror("Parâmetros inválidos", str(exc), parent=self)
            return
        self.save_settings()
        self._start_operation("Flow", batch_dir, config)

    def run_flow_logic(self, batch_dir, config):
        """Compatibility entry point; UI launches use PipelineRunner."""
        AstroProcessManager._run_legacy_stage(self, "Flow", batch_dir, config)

    def start_align_processing(self):
        if self.worker and self.worker.is_alive():
            return

        try:
            self._resource_settings()
            shift = float(self.align_quality_shift_var.get())
            if not np.isfinite(shift) or shift < 0:
                raise ValueError("Desvio residual deve ser finito e não negativo.")
            base_dir = Path(self.batch_dir_var.get()).expanduser().resolve()
            output_dir = Path(self.align_output_dir_var.get()).expanduser().resolve()

            if not base_dir.is_dir():
                raise ValueError("A pasta base de batches não existe.")

            if base_dir == output_dir:
                raise ValueError("A pasta de destino deve ser diferente da pasta base.")

            config = {
                "debayer_pattern": self.align_debayer_pattern_var.get(),
                "debayer_method": self.align_debayer_method_var.get(),
                "interpolation": self.align_interpolation_var.get(),
                "overwrite": self.align_overwrite_var.get(),
                "dry_run": self.align_dry_run_var.get(),
                "keep_header": self.align_keep_header_var.get(),
                "delete_intermediates": self.align_delete_intermediates_var.get(),
                "compress_output": self.align_compress_output_var.get(),
                "engine_profile": self.align_profile_var.get(),
                "warp_engine": self.align_warp_engine_var.get(),
                "memory_budget_mb": self.resource_memory_var.get(),
                "workers": self.resource_workers_var.get(),
                "quality_gate": self.align_quality_gate_var.get(),
                "quality_max_shift": self.align_quality_shift_var.get(),
            }

        except Exception as exc:
            messagebox.showerror("Parâmetros inválidos", str(exc), parent=self)
            return

        self._start_operation("Align", base_dir, output_dir, config)

    def run_align_logic(self, base_dir, output_dir, config_dict):
        """Compatibility entry point; UI launches use PipelineRunner."""
        AstroProcessManager._run_legacy_stage(self, "Align", base_dir, output_dir, config_dict)

    def start_hdr(self):
        if self.worker and self.worker.is_alive(): return
        try:
            from hdr_logic import build_hdr_config
            folder_text = self.hdr_input_var.get().strip()
            folder = Path(folder_text)
            if not folder_text or not folder.is_dir():
                raise ValueError("Selecione a pasta dos FITS já alinhados.")
            output_text = self.hdr_output_var.get().strip()
            if not output_text:
                raise ValueError("Selecione o arquivo de saída.")
            output = Path(output_text).resolve()
            paths = sorted(p for p in folder.iterdir() if p.suffix.lower() in {".fit", ".fits", ".fts"} and p.resolve() != output)
            if len(paths) < 2:
                raise ValueError("São necessários pelo menos dois FITS alinhados.")
            config = {"input_paths": paths, "output_path": output_text,
                      "noise_floor": self.hdr_noise_var.get(), "row_band": self.hdr_rowband_var.get()}
            if self.hdr_saturation_var.get(): config["saturation"] = float(self.hdr_saturation_var.get())
            if self.hdr_exptime_var.get(): config["exposure_override"] = float(self.hdr_exptime_var.get())
            if not np.isfinite(float(config["noise_floor"])) or float(config["noise_floor"]) <= 0: raise ValueError("Ruído inválido")
            if config["row_band"] <= 0: raise ValueError("A faixa de linhas deve ser positiva.")
            for key in ("saturation", "exposure_override"):
                if key in config and (not np.isfinite(config[key]) or config[key] <= 0):
                    raise ValueError(f"{key}: informe um número positivo e finito.")
            build_hdr_config(config)
        except (tk.TclError, OSError, TypeError, ValueError) as exc:
            messagebox.showerror("HDR", str(exc)); return
        self._start_operation("HDR", config)

    def start_stacking(self):
        if self.worker and self.worker.is_alive():
            return

        # ---- Verifica os critérios de triagem de frames sem guiagem ----
        try:
            trail_filter_enabled = bool(self.stack_trail_filter_var.get())
            min_roundness = float(self.stack_min_roundness_var.get())
            min_shape_stars_value = float(self.stack_min_shape_stars_var.get())
            if not np.isfinite(min_shape_stars_value) or not min_shape_stars_value.is_integer():
                raise ValueError("O mínimo de estrelas medidas deve ser inteiro.")
            min_shape_stars = int(min_shape_stars_value)
        except (tk.TclError, TypeError, ValueError):
            messagebox.showerror(
                "Parâmetros inválidos",
                "Informe uma roundness mínima entre 0 e 1 e pelo menos 1 estrela medida.",
                parent=self,
            )
            return

        if not np.isfinite(min_roundness) or not 0.0 <= min_roundness <= 1.0:
            messagebox.showerror(
                "Parâmetros inválidos",
                "A roundness mínima b/a deve ser um número finito entre 0 e 1.",
                parent=self,
            )
            return
        if not 1 <= min_shape_stars <= 64:
            messagebox.showerror(
                "Parâmetros inválidos",
                "O mínimo de estrelas medidas deve ficar entre 1 e 64.",
                parent=self,
            )
            return

        # ---- Verifica pasta de entrada ----
        input_dir = Path(self.stack_input_dir_var.get()).expanduser().resolve()
        if not input_dir.is_dir():
            messagebox.showerror(
                "Erro",
                "Selecione uma pasta de entrada com os frames alinhados.\n\n"
                "Esta deve ser a pasta de saída do AstroAlign (ex: .../aligned).",
                parent=self,
            )
            return

        # ---- Verifica pasta de saída ----
        output_dir = Path(self.stack_output_dir_var.get()).expanduser().resolve()
        if not str(self.stack_output_dir_var.get()).strip():
            messagebox.showerror(
                "Erro",
                "Selecione uma pasta de saída para a imagem empilhada.",
                parent=self,
            )
            return

        # ---- Verifica se há arquivos de flow (opcional, mas recomendado) ----
        has_flow = False
        for batch_dir in input_dir.iterdir():
            if batch_dir.is_dir() and (batch_dir / "flow_local.json").exists():
                has_flow = True
                break

        if not has_flow:
            # Verifica se há global_flow.json na pasta pai
            if not (input_dir.parent / "global_flow.json").exists():
                # Avisa, mas não impede
                self.print_to_console(
                    "[Stack] Aviso: Nenhum arquivo de flow encontrado.\n"
                    "  O Stacking usará apenas os nomes dos arquivos para organizar os frames.\n"
                )

        # ---- Configuração ----
        config = {
            "base_dir": str(self.batch_dir_var.get()),
            "input_dir": str(input_dir),
            "output_dir": str(output_dir),
            "selection_mode": self.stack_selection_mode_var.get(),
            "selection_percentage": self.stack_selection_percentage_var.get(),
            "selection_metric": self.stack_selection_metric_var.get(),
            "trail_filter_enabled": trail_filter_enabled,
            "min_roundness": min_roundness,
            "min_shape_stars": min_shape_stars,
            "method": self.stack_method_var.get(),
            "rejection_method": self.stack_rejection_method_var.get(),
            "rejection_low": self.stack_rejection_low_var.get(),
            "rejection_high": self.stack_rejection_high_var.get(),
            "normalize": self.stack_normalize_var.get(),
            "normalize_method": self.stack_normalize_method_var.get(),
            "output_name": self.stack_output_name_var.get(),
            "output_bit_depth": self.stack_output_bit_depth_var.get(),
            "compress_output": self.stack_compress_var.get(),
            "apply_dither_correction": self.stack_dither_correction_var.get(),
            "engine_profile": self.stack_profile_var.get(),
            "reducer_engine": self.stack_reducer_engine_var.get(),
        }

        self._start_operation("Stack", input_dir, config)

    def use_align_output_for_stack(self):
        """Use a saída do AstroAlign como entrada apenas após clique explícito."""

        align_output = self.align_output_dir_var.get().strip()
        if not align_output:
            message = (
                "A saída do AstroAlign ainda não foi definida. "
                "Informe essa pasta na aba AstroAlign antes de usá-la aqui."
            )
            self.status_var.set(message)
            self.print_to_console(f"[Stack] {message}\n")
            return False

        self.stack_input_dir_var.set(align_output)
        self.save_settings()
        self.status_var.set("Entrada do AstroStack definida pela saída do AstroAlign.")
        self.print_to_console(
            f"[Stack] Entrada definida pela saída do AstroAlign: {align_output}\n"
        )
        return True

    def apply_unguided_preset(self):
        """Apply the documented preset for screening unguided subframes."""

        self.stack_trail_filter_var.set(True)
        self.stack_min_roundness_var.set(0.65)
        self.stack_min_shape_stars_var.set(5)
        self.stack_selection_mode_var.set("All")
        self.stack_method_var.set("Mean")
        self.stack_rejection_method_var.set("SigmaClip")
        self.stack_profile_var.set("Stable")
        self.save_settings()
        message = (
            "Preset 'Subs sem guiagem' aplicado: filtro ativo, roundness 0.65, "
            "mínimo de 5 estrelas, seleção All, Mean, SigmaClip e Stable."
        )
        self.status_var.set(message)
        self.print_to_console(f"[Stack] {message}\n")

    def run_stacking_logic(self, input_dir, config):
        """Compatibility entry point; UI launches use PipelineRunner."""
        AstroProcessManager._run_legacy_stage(self, "Stack", input_dir, config)

    def show_astroflow_preview(self):
        import cv2
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        from matplotlib.figure import Figure

        from astroflow_logic import preview_star_detection

        base_dir_str = self.batch_dir_var.get()

        if not base_dir_str:
            messagebox.showerror(
                "Erro", "Selecione a Pasta Base das Batches primeiro.", parent=self
            )
            return

        base_dir = Path(base_dir_str).expanduser().resolve()

        if not base_dir.is_dir():
            messagebox.showerror("Erro", "A Pasta Base não existe.", parent=self)
            return

        batch_folders = sorted(
            [d for d in base_dir.iterdir() if d.is_dir() and "batch" in d.name.lower()]
        )

        if not batch_folders:
            messagebox.showerror(
                "Erro", "Nenhuma pasta de Batch encontrada.", parent=self
            )
            return

        target_batch = batch_folders[0]

        config = {
            "fwhm": self.flow_fwhm_var.get(),
            "sigma": self.flow_sigma_var.get(),
            "max_stars": 250,
            "engine": self.flow_engine_var.get(),
            "engine_profile": self.flow_profile_var.get(),
            "detector_engine": self.flow_detector_engine_var.get(),
        }

        try:
            img_preview, count, fwhm_measured = preview_star_detection(
                target_batch, config
            )

            if img_preview is None:
                messagebox.showwarning(
                    "Aviso",
                    "Não foi possível carregar a imagem para preview.",
                    parent=self,
                )
                return

        except Exception as exc:
            messagebox.showerror("Erro", f"Falha ao gerar preview:\n{exc}", parent=self)
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
                f"{target_batch.name}   •   {count} estrelas   •   FWHM {fwhm_measured:.1f}px   •   Engine: {self.flow_engine_var.get()}"
            ),
            font=("Segoe UI Semibold", 10),
        ).pack(anchor="w")

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
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=12, pady=5)

        ttk.Label(
            prev_window,
            text="Ajuste FWHM/Sigma na aba Flow e execute o preview novamente.",
            style="Muted.TLabel",
        ).pack(pady=(0, 10))

    def show_flow_visualization(self):
        import math

        import numpy as np
        from matplotlib.backends.backend_tkagg import (
            FigureCanvasTkAgg,
            NavigationToolbar2Tk,
        )
        from matplotlib.figure import Figure

        base_dir = Path(self.batch_dir_var.get()).expanduser().resolve()
        global_json = base_dir / "global_flow.json"

        if not global_json.exists():
            messagebox.showerror(
                "Erro",
                "Execute o AstroFlow primeiro. global_flow.json não encontrado.",
                parent=self,
            )
            return

        try:
            with global_json.open("r", encoding="utf-8") as f:
                global_data = json.load(f)

            points_x, points_y, rotations, time_seq = [], [], [], []
            center_pt = np.array([0, 0, 1])
            frame_counter = 0

            for batch_name, g_info in global_data.get("batches", {}).items():
                if g_info.get("status") != "accepted" or g_info.get("matrix") is None:
                    continue

                g_matrix = np.array(g_info["matrix"], dtype=np.float64)
                local_json = base_dir / batch_name / "flow_local.json"

                if not local_json.exists():
                    continue

                with local_json.open("r", encoding="utf-8") as f:
                    local_data = json.load(f)

                for _, l_info in local_data.get("frames", {}).items():
                    if (
                        l_info.get("status") != "accepted"
                        or l_info.get("matrix") is None
                    ):
                        continue

                    l_matrix = np.array(l_info["matrix"], dtype=np.float64)
                    abs_matrix = np.dot(g_matrix, l_matrix)

                    transformed_pt = np.dot(abs_matrix, center_pt)

                    # Extrai a rotação da matriz geométrica (em graus)
                    angle_rad = math.atan2(abs_matrix[1, 0], abs_matrix[0, 0])
                    angle_deg = math.degrees(angle_rad)

                    points_x.append(transformed_pt[0])
                    points_y.append(transformed_pt[1])
                    rotations.append(angle_deg)
                    time_seq.append(frame_counter)
                    frame_counter += 1

        except Exception as exc:
            messagebox.showerror(
                "Erro", f"Falha ao carregar os dados do Flow:\n{exc}", parent=self
            )
            return

        if not points_x:
            messagebox.showwarning(
                "Aviso", "Nenhum ponto de trajetória foi encontrado.", parent=self
            )
            return

        win = tk.Toplevel(self)
        win.title("AstroFlow — Trajetória e Rotação 3D")
        win.geometry("950x750")

        # Configura o Figure indicando projeção 3D
        fig = Figure(figsize=(8, 6), dpi=90)
        ax = fig.add_subplot(111, projection="3d")

        # Desenha a linha conectando a trajetória ao longo do tempo
        ax.plot(points_x, points_y, rotations, color="gray", alpha=0.3, linewidth=1)

        # Desenha os pontos coloridos baseados na passagem do tempo (time_seq)
        sc = ax.scatter(
            points_x,
            points_y,
            rotations,
            c=time_seq,
            cmap="plasma",  # Mapas térmicos como plasma ou viridis ficam ótimos em 3D
            marker="o",
            s=25,
            alpha=0.9,
            depthshade=True,
        )

        # Destaca o Início e o Fim
        ax.scatter(
            points_x[0],
            points_y[0],
            rotations[0],
            color="lime",
            s=60,
            label="Primeiro Frame",
            edgecolor="black",
        )
        ax.scatter(
            points_x[-1],
            points_y[-1],
            rotations[-1],
            color="red",
            s=60,
            label="Último Frame",
            edgecolor="black",
        )

        ax.set_title("Drift XY e Rotação de Campo (Field Rotation)", pad=20)
        ax.set_xlabel("Deslocamento X (px)", labelpad=10)
        ax.set_ylabel("Deslocamento Y (px)", labelpad=10)
        ax.set_zlabel("Rotação (graus)", labelpad=10)
        ax.invert_yaxis()  # Inverte o eixo Y para bater com a tela da imagem

        # Barra lateral mostrando o gradiente de tempo
        cbar = fig.colorbar(sc, ax=ax, shrink=0.7, pad=0.1)
        cbar.set_label("Evolução Temporal (Nº do Frame)")

        ax.legend(loc="upper left")
        fig.tight_layout()

        # Renderização do Canvas e inclusão da Barra de Navegação (para rotacionar o 3D)
        canvas = FigureCanvasTkAgg(fig, master=win)
        canvas.draw()

        toolbar_frame = tk.Frame(win)
        toolbar_frame.pack(side=tk.TOP, fill=tk.X)
        toolbar = NavigationToolbar2Tk(canvas, toolbar_frame)
        toolbar.update()

        canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def open_anchor_selector(self, target_batch=None):
        import functools
        import threading

        import cv2
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        from matplotlib.figure import Figure

        from astroflow_logic import extract_luminance, load_fits_data

        base_dir_str = self.batch_dir_var.get()
        if not base_dir_str:
            messagebox.showerror(
                "Erro", "Selecione a Pasta Base primeiro.", parent=self
            )
            return

        base_dir = Path(base_dir_str).expanduser().resolve()
        if not base_dir.is_dir():
            messagebox.showerror("Erro", "A Pasta Base não existe.", parent=self)
            return

        batch_folders = sorted(
            [d for d in base_dir.iterdir() if d.is_dir() and "batch" in d.name.lower()]
        )
        if not batch_folders:
            messagebox.showerror(
                "Erro", "Nenhuma pasta de Batch encontrada.", parent=self
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
            top, values=[b.name for b in batch_folders], state="readonly", width=18
        )
        batch_combo.pack(side=tk.LEFT, padx=(7, 15))

        ttk.Label(top, text="Frame:").pack(side=tk.LEFT)
        frame_combo = ttk.Combobox(top, state="readonly", width=34)
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
        title_obj = ax.set_title("Stretched Preview", fontsize=10)

        canvas = FigureCanvasTkAgg(fig, master=image_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        @functools.lru_cache(maxsize=40)
        def get_preview_data(filepath):
            data, header = load_fits_data(filepath)
            l_data = extract_luminance(data, header)
            h, w = l_data.shape[:2]
            small_w, small_h = max(w // 3, 1), max(h // 3, 1)
            return cv2.resize(
                l_data, (small_w, small_h), interpolation=cv2.INTER_NEAREST
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
                    if f.is_file() and f.suffix.lower() in {".fit", ".fits", ".fts"}
                ]
            )
            frame_combo.configure(values=fits_files)
            if fits_files:
                selected = self.custom_anchors.get(
                    b_name, fits_files[len(fits_files) // 2]
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
                    p25, p75 = np.percentile(small_data, [25, 75])
                    std = max((p75 - p25) / 1.35, 1e-5)
                    vmin, vmax = median - 0.5 * std, median + 6.0 * std
                    norm = (
                        np.clip((small_data - vmin) / max(vmax - vmin, 1e-5), 0, 1)
                        * 255
                    )
                    img_8u = norm.astype(np.uint8)

                    def update_gui():
                        if current_id != generation["id"]:
                            return
                        ax_img.set_data(img_8u)
                        ax_img.set_extent((0, img_8u.shape[1], img_8u.shape[0], 0))
                        title_obj.set_text(f"Stretched Preview — {b_name} / {f_name}")
                        canvas.draw_idle()
                        batch_combo.configure(state="readonly")
                        frame_combo.configure(state="readonly")

                    self.after(0, update_gui)
                except Exception:

                    def handle_error():
                        batch_combo.configure(state="readonly")
                        frame_combo.configure(state="readonly")
                        self.print_to_console(f"[Preview] Erro: {exc}\n")

                    self.after(0, handle_error)

            threading.Thread(target=worker, daemon=True).start()

        batch_combo.bind("<<ComboboxSelected>>", update_frames)
        frame_combo.bind("<<ComboboxSelected>>", update_image)

        if batch_folders:
            if target_batch and target_batch in [b.name for b in batch_folders]:
                batch_combo.set(target_batch)
            else:
                batch_combo.set(batch_folders[0].name)
            update_frames()

        buttons = ttk.Frame(win, padding=12)
        buttons.pack(fill=tk.X)

        def save_selection():
            b_name, f_name = batch_combo.get(), frame_combo.get()
            if not b_name or not f_name:
                return
            self.custom_anchors[b_name] = f_name
            self.save_settings()
            self.print_to_console(f"[AstroFlow] Referência da {b_name}: {f_name}\n")
            self.refresh_flow_reference_preview()
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
        ttk.Button(buttons, text="Fechar", command=win.destroy).pack(
            side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 0), ipady=5
        )

    def _open_anchor_selector_for_batch(self, batch_name):
        self.open_anchor_selector(target_batch=batch_name)

    def _update_global_master_options(self, *args):
        if not hasattr(self, "tab_flow") or not hasattr(
            self.tab_flow, "combo_global_master"
        ):
            return
        base_dir_str = self.batch_dir_var.get()
        values = ["Auto"]
        if base_dir_str:
            base_dir = Path(base_dir_str).expanduser().resolve()
            if base_dir.is_dir():
                try:
                    batches = sorted(
                        [
                            d.name
                            for d in base_dir.iterdir()
                            if d.is_dir() and "batch" in d.name.lower()
                        ]
                    )
                    values.extend(batches)
                except OSError:
                    pass
        self.tab_flow.combo_global_master["values"] = values
        if self.flow_global_master_var.get() not in values:
            self.flow_global_master_var.set("Auto")

    def cancel_processing(self):
        if self.worker and self.worker.is_alive():
            self.print_to_console("\n[GUI] Solicitação de cancelamento enviada...\n")
            self.cancel_event.set()
            self.status_var.set("Cancelamento solicitado...")

    def _on_close(self):
        if self.worker and self.worker.is_alive():
            answer = messagebox.askyesno(
                "Processamento ativo",
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
