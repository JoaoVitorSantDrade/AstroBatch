from tkinter import ttk
from .calibration_model import CalibrationViewModel


class CalibrationView(ttk.Frame):
    def __init__(self, parent, model: CalibrationViewModel):
        super().__init__(parent, padding=18)
        self.model = model
        self.columnconfigure(0, weight=1)
        self._build_ui()

    @staticmethod
    def _path_row(parent, row, label, variable, browse_command):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=5)
        ttk.Entry(parent, textvariable=variable).grid(
            row=row, column=1, sticky="ew", padx=8, pady=5
        )
        ttk.Button(parent, text="Selecionar", command=browse_command).grid(
            row=row, column=2, pady=5
        )

    @staticmethod
    def _description(parent, row, text):
        ttk.Label(
            parent, text=text, style="Muted.TLabel", wraplength=820, justify="left"
        ).grid(row=row, column=0, columnspan=3, sticky="w", pady=(0, 10))

    def _build_ui(self):
        intro = ttk.LabelFrame(
            self, text="Entrada e Saída", style="Section.TLabelframe", padding=12
        )
        intro.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        intro.columnconfigure(1, weight=1)

        self._description(
            intro,
            0,
            "Calibre os LIGHTS antes do Batch. Darks e Flats podem ser "
            "informados como uma pasta de frames ou como um Master já pronto.",
        )

        self._path_row(
            intro,
            1,
            "LIGHTS / RAW:",
            self.model.input_dir,
            self.model.browse_input,
        )

        self._path_row(
            intro,
            2,
            "Saída calibrada:",
            self.model.output_dir,
            self.model.browse_output,
        )

        dark = ttk.LabelFrame(
            self, text="Dark", style="Section.TLabelframe", padding=12
        )
        dark.grid(row=1, column=0, sticky="ew", pady=(0, 10))
        dark.columnconfigure(1, weight=1)

        ttk.Checkbutton(
            dark, text="Aplicar Dark", variable=self.model.apply_dark
        ).grid(row=0, column=0, sticky="w")

        ttk.Entry(dark, textvariable=self.model.dark_path).grid(
            row=0, column=1, sticky="ew", padx=8
        )

        ttk.Button(
            dark,
            text="Arquivo...",
            command=self.model.browse_dark,
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
            flat, text="Aplicar Flat", variable=self.model.apply_flat
        ).grid(row=0, column=0, sticky="w")

        ttk.Entry(flat, textvariable=self.model.flat_path).grid(
            row=0, column=1, sticky="ew", padx=8
        )

        ttk.Button(
            flat,
            text="Arquivo...",
            command=self.model.browse_flat,
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
            variable=self.model.create_master,
        ).grid(row=0, column=0, sticky="w", pady=2)

        ttk.Checkbutton(
            options,
            text="Sobrescrever arquivos existentes",
            variable=self.model.overwrite,
        ).grid(row=1, column=0, sticky="w", pady=2)

        actions = ttk.Frame(self)
        actions.grid(row=4, column=0, sticky="ew")

        actions.columnconfigure(0, weight=1)
        self.run_button = ttk.Button(
            actions, text="▶  INICIAR CALIBRAÇÃO", style="Accent.TButton",
            command=self.model.start,
        )
        self.run_button.grid(row=0, column=0, sticky="ew", ipady=5)
        self.cancel_button = ttk.Button(
            actions, text="Cancelar", style="Danger.TButton",
            command=self.model.cancel, state="disabled",
        )
        self.cancel_button.grid(row=0, column=1, padx=(8, 0), ipady=5)
