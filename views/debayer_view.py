from tkinter import ttk

from views.base_view import BaseAstroView


class DebayerView(BaseAstroView):
    def __init__(self, parent, app):
        super().__init__(parent, app)
        self.columnconfigure(0, weight=1)
        self._build_ui()

    def _build_ui(self):
        dirs = ttk.LabelFrame(
            self,
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
            self.app.debayer_input_var,
            lambda: self.app.browse_dir(self.app.debayer_input_var),
        )

        self._path_row(
            dirs,
            2,
            "Saída RGB:",
            self.app.debayer_output_var,
            lambda: self.app.browse_dir(self.app.debayer_output_var),
        )

        params = ttk.LabelFrame(
            self,
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
            textvariable=self.app.debayer_pattern_var,
            values=["Auto", "RGGB", "BGGR", "GRBG", "GBRG"],
            state="readonly",
            width=14,
        ).grid(row=0, column=1, sticky="w")

        ttk.Label(params, text="Método:").grid(
            row=0, column=2, sticky="w", padx=(35, 8)
        )

        ttk.Combobox(
            params,
            textvariable=self.app.debayer_method_var,
            values=["Nearest", "Bilinear", "VNG", "Edge-Aware"],
            state="readonly",
            width=16,
        ).grid(row=0, column=3, sticky="w")

        ttk.Checkbutton(
            params,
            text="Sobrescrever arquivos existentes",
            variable=self.app.debayer_overwrite_var,
        ).grid(row=1, column=0, columnspan=4, sticky="w", pady=(10, 0))

        actions = ttk.Frame(self)
        actions.grid(row=2, column=0, sticky="ew")

        self.btn_run_debayer, self.btn_cancel_debayer = self._action_bar(
            actions,
            0,
            self.app.start_debayer,
            self.app.cancel_processing,
            "▶  INICIAR DEBAYER",
        )
