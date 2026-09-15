from tkinter import ttk

from batch_logic import RESAMPLE_MODES

from .batch_model import BatchViewModel


class BatchView(ttk.Frame):
    def __init__(self, parent, model: BatchViewModel):
        super().__init__(parent, padding=18)
        self.model = model
        self.columnconfigure(0, weight=1)
        self._build_ui()
        self.after(100, self.model.toggle_options)

    @staticmethod
    def _path_row(parent, row, label, variable, browse_command):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=5)
        ttk.Entry(parent, textvariable=variable).grid(
            row=row, column=1, sticky="ew", padx=8, pady=5
        )
        ttk.Button(parent, text="Selecionar", command=browse_command).grid(
            row=row, column=2, pady=5
        )

    def _build_ui(self):
        dirs = ttk.LabelFrame(self, text="Diretórios", style="Section.TLabelframe", padding=12)
        dirs.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        dirs.columnconfigure(1, weight=1)
        self._path_row(dirs, 0, "Pasta origem:", self.model.input_dir, self.model.browse_input)
        self._path_row(dirs, 1, "Pasta destino:", self.model.output_dir, self.model.browse_output)

        opt = ttk.LabelFrame(self, text="Otimização para análise", style="Section.TLabelframe", padding=12)
        opt.grid(row=1, column=0, sticky="ew", pady=(0, 10))
        opt.columnconfigure(1, weight=1)
        ttk.Radiobutton(
            opt, text="Recorte central (Crop)", variable=self.model.opt_method,
            value="Crop", command=self.model.toggle_options,
        ).grid(row=0, column=0, sticky="w")
        self.crop_frame = ttk.Frame(opt)
        self.crop_frame.grid(row=0, column=1, sticky="w", padx=20)
        ttk.Label(self.crop_frame, text="Tamanho (px):").pack(side="left")
        self.crop_entry = ttk.Entry(self.crop_frame, textvariable=self.model.crop_size, width=10)
        self.crop_entry.pack(side="left", padx=(7, 0))

        ttk.Radiobutton(
            opt, text="Downsampling", variable=self.model.opt_method,
            value="Downsampling", command=self.model.toggle_options,
        ).grid(row=1, column=0, sticky="w", pady=(8, 0))
        self.down_frame = ttk.Frame(opt)
        self.down_frame.grid(row=1, column=1, sticky="w", padx=20, pady=(8, 0))
        ttk.Label(self.down_frame, text="Método:").pack(side="left")
        resample_values = list(RESAMPLE_MODES.keys()) if hasattr(RESAMPLE_MODES, "keys") else list(RESAMPLE_MODES)
        self.down_combo = ttk.Combobox(
            self.down_frame, textvariable=self.model.downsample_method,
            values=resample_values, state="readonly", width=13,
        )
        self.down_combo.pack(side="left", padx=7)
        ttk.Label(self.down_frame, text="Escala:").pack(side="left", padx=(8, 0))
        self.down_scale_entry = ttk.Entry(self.down_frame, textvariable=self.model.downsample_scale, width=8)
        self.down_scale_entry.pack(side="left", padx=7)

        params = ttk.LabelFrame(self, text="Detecção e operação de arquivos", style="Section.TLabelframe", padding=12)
        params.grid(row=2, column=0, sticky="ew", pady=(0, 10))
        ttk.Label(params, text="Threshold (fator):").grid(row=0, column=0, sticky="w")
        ttk.Entry(params, textvariable=self.model.threshold, width=10).grid(row=0, column=1, sticky="w", padx=8)
        ttk.Checkbutton(params, text="Copiar em vez de mover", variable=self.model.copy_files).grid(row=1, column=0, columnspan=2, sticky="w", pady=(8, 0))
        ttk.Checkbutton(params, text="Sobrescrever arquivos existentes", variable=self.model.overwrite).grid(row=2, column=0, columnspan=2, sticky="w", pady=(8, 0))
        ttk.Checkbutton(params, text="Dry-Run (não alterar arquivos)", variable=self.model.dry_run).grid(row=3, column=0, columnspan=2, sticky="w", pady=(8, 0))

        actions = ttk.Frame(self)
        actions.grid(row=3, column=0, sticky="ew")
        actions.columnconfigure(0, weight=1)
        self.run_button = ttk.Button(
            actions, text="▶  INICIAR ASTROBATCH", style="Accent.TButton", command=self.model.start
        )
        self.run_button.grid(row=0, column=0, sticky="ew", ipady=5)
        self.cancel_button = ttk.Button(
            actions, text="Cancelar", style="Danger.TButton", command=self.model.cancel, state="disabled"
        )
        self.cancel_button.grid(row=0, column=1, padx=(8, 0), ipady=5)
