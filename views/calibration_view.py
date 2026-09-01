from tkinter import ttk

from views.base_view import BaseAstroView


class CalibrationView(BaseAstroView):
    def __init__(self, parent, app):
        super().__init__(parent, app)
        self.columnconfigure(0, weight=1)
        self._build_ui()

    def _build_ui(self):
        intro = ttk.LabelFrame(
            self, text="Entrada e Saída", style="Section.TLabelframe", padding=12
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
            self.app.calib_input_var,
            lambda: self.app.browse_dir(self.app.calib_input_var),
        )
        self._path_row(
            intro,
            2,
            "Saída calibrada:",
            self.app.calib_output_var,
            lambda: self.app.browse_dir(self.app.calib_output_var),
        )

        dark = ttk.LabelFrame(
            self, text="Dark", style="Section.TLabelframe", padding=12
        )
        dark.grid(row=1, column=0, sticky="ew", pady=(0, 10))
        dark.columnconfigure(1, weight=1)

        ttk.Checkbutton(
            dark, text="Aplicar Dark", variable=self.app.apply_dark_var
        ).grid(row=0, column=0, sticky="w")
        ttk.Entry(dark, textvariable=self.app.dark_path_var).grid(
            row=0, column=1, sticky="ew", padx=8
        )
        ttk.Button(
            dark,
            text="Arquivo...",
            command=lambda: self.app.browse_file_or_dir(self.app.dark_path_var),
        ).grid(row=0, column=2)
        ttk.Label(
            dark,
            text="Pasta = gerar Master Dark   •   Arquivo = usar como Master Dark",
            style="Muted.TLabel",
        ).grid(row=1, column=0, columnspan=3, sticky="w", pady=(7, 0))

        flat = ttk.LabelFrame(
            self, text="Flat", style="Section.TLabelframe", padding=12
        )
        flat.grid(row=2, column=0, sticky="ew", pady=(0, 10))
        flat.columnconfigure(1, weight=1)

        ttk.Checkbutton(
            flat, text="Aplicar Flat", variable=self.app.apply_flat_var
        ).grid(row=0, column=0, sticky="w")
        ttk.Entry(flat, textvariable=self.app.flat_path_var).grid(
            row=0, column=1, sticky="ew", padx=8
        )
        ttk.Button(
            flat,
            text="Arquivo...",
            command=lambda: self.app.browse_file_or_dir(self.app.flat_path_var),
        ).grid(row=0, column=2)
        ttk.Label(
            flat,
            text="Pasta = gerar Master Flat   •   Arquivo = usar como Master Flat",
            style="Muted.TLabel",
        ).grid(row=1, column=0, columnspan=3, sticky="w", pady=(7, 0))

        options = ttk.LabelFrame(
            self, text="Opções", style="Section.TLabelframe", padding=12
        )
        options.grid(row=3, column=0, sticky="ew", pady=(0, 10))

        ttk.Checkbutton(
            options,
            text="Gerar Masters automaticamente quando receber uma pasta",
            variable=self.app.calib_create_master_var,
        ).grid(row=0, column=0, sticky="w", pady=2)
        ttk.Checkbutton(
            options,
            text="Sobrescrever arquivos existentes",
            variable=self.app.calib_overwrite_var,
        ).grid(row=1, column=0, sticky="w", pady=2)

        actions = ttk.Frame(self)
        actions.grid(row=4, column=0, sticky="ew")

        # As referências de botão vão para o app para que ele possa gerenciar o bloqueio de UI (lock/unlock)
        self.app.btn_run_calib, self.app.btn_cancel_calib = self._action_bar(
            actions,
            0,
            self.app.start_calibration,
            self.app.cancel_processing,
            "▶  INICIAR CALIBRATION",
        )
