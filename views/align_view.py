from tkinter import ttk

from astroalign_logic import INTERPOLATION_MODES
from views.base_view import BaseAstroView


class AlignView(BaseAstroView):
    def __init__(self, parent, app):
        super().__init__(parent, app)
        self.columnconfigure(0, weight=1)
        self._build_ui()

    def _build_ui(self):
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

        params = ttk.LabelFrame(
            self,
            text="Warping",
            style="Section.TLabelframe",
            padding=12,
        )
        params.grid(row=1, column=0, sticky="ew", pady=(0, 10))

        ttk.Label(params, text="Interpolação:").grid(row=0, column=0, sticky="w")

        interp_values = list(INTERPOLATION_MODES)
        if not interp_values:
            interp_values = ["Nearest", "Bilinear", "Bicubic", "Lanczos"]

        ttk.Combobox(
            params,
            textvariable=self.app.align_interpolation_var,
            values=interp_values,
            state="readonly",
            width=16,
        ).grid(row=0, column=1, sticky="w", padx=8)

        ttk.Checkbutton(
            params,
            text="Preservar Header FITS original",
            variable=self.app.align_keep_header_var,
        ).grid(row=1, column=0, columnspan=2, sticky="w", pady=(10, 0))

        ttk.Checkbutton(
            params,
            text="Sobrescrever arquivos existentes",
            variable=self.app.align_overwrite_var,
        ).grid(row=2, column=0, columnspan=2, sticky="w", pady=(7, 0))

        ttk.Checkbutton(
            params,
            text="Dry-Run (não gravar no disco)",
            variable=self.app.align_dry_run_var,
        ).grid(row=3, column=0, columnspan=2, sticky="w", pady=(7, 0))

        actions = ttk.Frame(self)
        actions.grid(row=2, column=0, sticky="ew")

        self.btn_run_align, self.btn_cancel_align = self._action_bar(
            actions,
            0,
            self.app.start_align_processing,
            self.app.cancel_processing,
            "▶  INICIAR ASTROALIGN",
        )
