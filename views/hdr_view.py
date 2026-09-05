"""Passive HDR form: widget construction receives only bindings and commands."""
from tkinter import ttk
from .hdr_model import HDRViewModel


class HDRView(ttk.Frame):
    def __init__(self, parent, model: HDRViewModel):
        super().__init__(parent, padding=18)
        self.columnconfigure(0, weight=1)
        box = ttk.LabelFrame(self, text="Fusão linear / HDR", padding=12)
        box.grid(row=0, column=0, sticky="ew")
        box.columnconfigure(1, weight=1)
        ttk.Label(box, text="Use FITS calibrados e alinhados. Exposições iguais reduzem ruído; "
                  "detalhes saturados em todos os frames não podem ser recuperados.",
                  wraplength=780, style="Muted.TLabel").grid(row=0, column=0, columnspan=3, sticky="w", pady=(0,12))
        fields = (
            ("Pasta de entrada", model.input_folder, model.browse_input),
            ("Arquivo de saída", model.output_file, model.browse_output),
            ("Saturação no FITS de entrada (opcional)", model.saturation, None),
            ("Ruído por frame (unidades calibradas)", model.noise, None),
            ("Linhas por faixa", model.row_band, None),
            ("Exposição em segundos (opcional)", model.exposure, None),
        )
        self.entries = {}
        for row, (label, variable, browse) in enumerate(fields, 1):
            ttk.Label(box, text=label).grid(row=row, column=0, sticky="w", pady=6)
            entry = ttk.Entry(box, textvariable=variable, width=18)
            entry.grid(row=row, column=1, sticky="ew", padx=10, pady=6)
            self.entries[label] = entry
            if browse:
                ttk.Button(box, text="Selecionar", command=browse).grid(row=row,column=2)
        ttk.Button(box, text="Usar saída do Align", command=model.use_align_output).grid(
            row=7, column=0, columnspan=3, sticky="w", pady=12)
        actions = ttk.Frame(self)
        actions.grid(row=1, column=0, sticky="ew", pady=12)
        actions.columnconfigure(0, weight=1)
        self.run_button = ttk.Button(actions, text="Iniciar fusão", style="Accent.TButton", command=model.start)
        self.run_button.grid(row=0, column=0, sticky="ew", ipady=5)
        self.cancel_button = ttk.Button(actions, text="Cancelar", command=model.cancel, state="disabled")
        self.cancel_button.grid(row=0, column=1, padx=(8,0), ipady=5)
