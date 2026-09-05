from tkinter import ttk

from .base_view import BaseAstroView


class AlignView(BaseAstroView):
    def __init__(self, parent, app):
        super().__init__(parent, app)
        self.columnconfigure(0, weight=1)
        self._build_ui()

    def _build_ui(self):
        # 1. Diretórios
        dirs = ttk.LabelFrame(
            self,
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
            self.app.batch_dir_var,
            lambda: self.app.browse_dir(self.app.batch_dir_var),
        )
        self._path_row(
            dirs,
            1,
            "Pasta destino:",
            self.app.align_output_dir_var,
            lambda: self.app.browse_dir(self.app.align_output_dir_var),
        )

        # 2. Configurações de Debayer
        debayer = ttk.LabelFrame(
            self,
            text="Debayer (Processamento In-Memory)",
            style="Section.TLabelframe",
            padding=12,
        )
        debayer.grid(row=1, column=0, sticky="ew", pady=(0, 10))
        debayer.columnconfigure(2, weight=1)

        ttk.Label(debayer, text="Padrão Bayer:").grid(row=0, column=0, sticky="w")
        ttk.Combobox(
            debayer,
            textvariable=self.app.align_debayer_pattern_var,
            values=["Auto", "RGGB", "BGGR", "GRBG", "GBRG", "Nenhum"],
            state="readonly",
            width=16,
        ).grid(row=0, column=1, sticky="w", padx=8)

        ttk.Label(
            debayer,
            text="Auto lê o padrão do Header FITS.",
            style="Muted.TLabel",
        ).grid(row=0, column=2, sticky="w")

        ttk.Label(debayer, text="Método:").grid(
            row=1, column=0, sticky="w", pady=(10, 0)
        )
        self.debayer_method_combo = ttk.Combobox(
            debayer,
            textvariable=self.app.align_debayer_method_var,
            values=["Bilinear", "VNG", "Menon2007"],
            state="readonly",
            width=16,
        )
        self.debayer_method_combo.grid(
            row=1, column=1, sticky="w", padx=8, pady=(10, 0)
        )

        def _update_debayer_controls(self, *_):
            disabled = self.app.align_debayer_pattern_var.get() == "Nenhum"
            self.debayer_method_combo.configure(
                state="disabled" if disabled else "readonly"
            )

        self.app.align_debayer_pattern_var.trace_add("write", _update_debayer_controls)
        _update_debayer_controls(self)

        ttk.Label(
            debayer,
            text="VNG/Menon preservam detalhes finos e reduzem ruído.",
            style="Muted.TLabel",
        ).grid(row=1, column=2, sticky="w", pady=(10, 0))

        # 3. Warping e Armazenamento
        params = ttk.LabelFrame(
            self,
            text="Warping, Registro e Armazenamento",
            style="Section.TLabelframe",
            padding=12,
        )
        params.grid(row=2, column=0, sticky="ew", pady=(0, 10))

        ttk.Label(params, text="Interpolação geométrica:").grid(
            row=0, column=0, sticky="w"
        )
        ttk.Combobox(
            params,
            textvariable=self.app.align_interpolation_var,
            values=["nearest", "bilinear", "bicubic", "lanczos"],
            state="readonly",
            width=16,
        ).grid(row=0, column=1, sticky="w", padx=8)

        ttk.Label(params, text="Perfil:").grid(row=0, column=2, sticky="w", padx=(18, 0))
        ttk.Combobox(
            params, textvariable=self.app.align_profile_var, values=["Stable", "Fast"],
            state="readonly", width=12,
        ).grid(row=0, column=3, sticky="w", padx=8)
        ttk.Label(params, text="Warp (Avancado):").grid(row=1, column=2, sticky="w", padx=(18, 0))
        ttk.Combobox(
            params, textvariable=self.app.align_warp_engine_var,
            values=["", "opencv-stable", "opencv-fast"], state="readonly", width=18,
        ).grid(row=1, column=3, sticky="w", padx=8)

        # ---> NOVA OPÇÃO: REGISTRO CROMÁTICO <---
        ttk.Checkbutton(
            params,
            text="Registro Cromático Avançado (Realinhar canais R e B usando G como âncora)",
            variable=self.app.align_rgb_registration_var,
        ).grid(row=1, column=0, columnspan=2, sticky="w", pady=(10, 0))

        ttk.Checkbutton(
            params,
            text="Preservar Header FITS original",
            variable=self.app.align_keep_header_var,
        ).grid(row=2, column=0, columnspan=2, sticky="w", pady=(7, 0))

        ttk.Checkbutton(
            params,
            text="Comprimir saída FITS (RICE_1)",
            variable=self.app.align_compress_output_var,
        ).grid(row=3, column=0, columnspan=2, sticky="w", pady=(7, 0))

        ttk.Label(
            params,
            text="Reduz o espaço em disco; desative para leituras mais rápidas no AstroStack.",
            style="Muted.TLabel",
        ).grid(row=4, column=0, columnspan=2, sticky="w", pady=(0, 0))

        ttk.Checkbutton(
            params,
            text="Sobrescrever arquivos existentes no destino",
            variable=self.app.align_overwrite_var,
        ).grid(row=5, column=0, columnspan=2, sticky="w", pady=(7, 0))

        ttk.Checkbutton(
            params,
            text="Apagar batches intermediários após alinhar (Limpeza de Disco)",
            variable=self.app.align_delete_intermediates_var,
        ).grid(row=6, column=0, columnspan=2, sticky="w", pady=(7, 0))

        ttk.Checkbutton(
            params,
            text="Dry-Run (Simular processamento sem gravar no disco)",
            variable=self.app.align_dry_run_var,
        ).grid(row=7, column=0, columnspan=2, sticky="w", pady=(7, 0))

        # 4. Ações
        actions = ttk.Frame(self)
        quality = ttk.LabelFrame(self, text="Verificação após alinhamento", padding=8)
        quality.grid(row=3, column=0, sticky="ew", pady=5)
        ttk.Checkbutton(quality, text="Rejeitar desalinhamento residual", variable=self.app.align_quality_gate_var).grid(row=0, column=0, sticky="w")
        for column, label, variable in (
            (1, "Desvio máx. (px)", self.app.align_quality_shift_var),
        ):
            ttk.Label(quality, text=label).grid(row=0, column=column, padx=6)
            ttk.Entry(quality, textvariable=variable, width=12).grid(row=1, column=column, padx=6)
        actions.grid(row=4, column=0, sticky="ew")

        self.app.btn_run_align, self.app.btn_cancel_align = self._action_bar(
            actions,
            0,
            self.app.start_align_processing,
            self.app.cancel_processing,
            "▶ ALINHAMENTO",
        )
