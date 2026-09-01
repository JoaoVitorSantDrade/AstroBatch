from tkinter import ttk

from batch_logic import RESAMPLE_MODES

from .base_view import BaseAstroView


class BatchView(BaseAstroView):
    def __init__(self, parent, app):
        super().__init__(parent, app)
        self.columnconfigure(0, weight=1)
        self._build_ui()

        # Garante que os campos de Crop/Downsample iniciem com o estado correto
        # assim que a UI for desenhada.
        self.after(100, self.app.toggle_opt_options)

    def _build_ui(self):
        dirs = ttk.LabelFrame(
            self, text="Diretórios", style="Section.TLabelframe", padding=12
        )
        dirs.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        dirs.columnconfigure(1, weight=1)

        self._path_row(
            dirs,
            0,
            "Pasta origem:",
            self.app.batch_input_dir_var,
            lambda: self.app.browse_dir(self.app.batch_input_dir_var),
        )

        self._path_row(
            dirs,
            1,
            "Pasta destino:",
            self.app.batch_dir_var,
            lambda: self.app.browse_dir(self.app.batch_dir_var),
        )

        opt = ttk.LabelFrame(
            self,
            text="Otimização para análise",
            style="Section.TLabelframe",
            padding=12,
        )
        opt.grid(row=1, column=0, sticky="ew", pady=(0, 10))
        opt.columnconfigure(1, weight=1)

        ttk.Radiobutton(
            opt,
            text="Recorte central (Crop)",
            variable=self.app.opt_method_var,
            value="Crop",
            command=self.app.toggle_opt_options,
        ).grid(row=0, column=0, sticky="w")

        self.crop_frame = ttk.Frame(opt)
        self.crop_frame.grid(row=0, column=1, sticky="w", padx=20)

        ttk.Label(self.crop_frame, text="Tamanho (px):").pack(side="left")

        # Referência necessária pelo toggle_opt_options no main.py
        self.crop_entry = ttk.Entry(
            self.crop_frame, textvariable=self.app.crop_size_var, width=10
        )
        self.crop_entry.pack(side="left", padx=(7, 0))

        ttk.Radiobutton(
            opt,
            text="Downsampling",
            variable=self.app.opt_method_var,
            value="Downsampling",
            command=self.app.toggle_opt_options,
        ).grid(row=1, column=0, sticky="w", pady=(8, 0))

        self.down_frame = ttk.Frame(opt)
        self.down_frame.grid(row=1, column=1, sticky="w", padx=20, pady=(8, 0))

        ttk.Label(self.down_frame, text="Método:").pack(side="left")

        # Lê os modos disponíveis dinamicamente do batch_logic
        resample_values = (
            list(RESAMPLE_MODES.keys())
            if hasattr(RESAMPLE_MODES, "keys")
            else list(RESAMPLE_MODES)
        )

        self.down_combo = ttk.Combobox(
            self.down_frame,
            textvariable=self.app.downsample_method_var,
            values=resample_values,
            state="readonly",
            width=13,
        )
        self.down_combo.pack(side="left", padx=7)

        ttk.Label(self.down_frame, text="Escala:").pack(side="left", padx=(8, 0))

        self.down_scale_entry = ttk.Entry(
            self.down_frame, textvariable=self.app.downsample_scale_var, width=8
        )
        self.down_scale_entry.pack(side="left", padx=7)

        params = ttk.LabelFrame(
            self,
            text="Detecção e operação de arquivos",
            style="Section.TLabelframe",
            padding=12,
        )
        params.grid(row=2, column=0, sticky="ew", pady=(0, 10))

        ttk.Label(params, text="Threshold (fator):").grid(row=0, column=0, sticky="w")
        ttk.Entry(params, textvariable=self.app.threshold_var, width=10).grid(
            row=0, column=1, sticky="w", padx=8
        )

        ttk.Checkbutton(
            params, text="Copiar em vez de mover", variable=self.app.copy_files_var
        ).grid(row=1, column=0, columnspan=2, sticky="w", pady=(8, 0))

        ttk.Checkbutton(
            params,
            text="Sobrescrever arquivos existentes",
            variable=self.app.batch_overwrite_var,
        ).grid(row=2, column=0, columnspan=2, sticky="w", pady=(8, 0))

        ttk.Checkbutton(
            params, text="Dry-Run (não alterar arquivos)", variable=self.app.dry_run_var
        ).grid(row=3, column=0, columnspan=2, sticky="w", pady=(8, 0))

        actions = ttk.Frame(self)
        actions.grid(row=3, column=0, sticky="ew")

        # Correção Crítica: O retorno do botão agora é ancorado no self.app
        self.app.btn_run_batch, self.app.btn_cancel_batch = self._action_bar(
            actions,
            0,
            self.app.start_batch_processing,
            self.app.cancel_processing,
            "▶  INICIAR ASTROBATCH",
        )
