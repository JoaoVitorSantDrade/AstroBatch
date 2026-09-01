import tkinter as tk
from tkinter import ttk

class BaseAstroView(ttk.Frame):
    """Classe base para todas as abas, contendo os métodos utilitários de UI."""
    def __init__(self, parent, app, padding=18):
        super().__init__(parent, padding=padding)
        self.app = app  # Referência ao AstroProcessManager (Controller)

    def _path_row(self, parent, row, label, variable, browse_command, browse_text="Selecionar"):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=5)
        ttk.Entry(parent, textvariable=variable).grid(row=row, column=1, sticky="ew", padx=8, pady=5)
        ttk.Button(parent, text=browse_text, command=browse_command).grid(row=row, column=2, pady=5)

    def _action_bar(self, parent, row, run_command, cancel_command, run_text):
        parent.columnconfigure(0, weight=1)

        run_btn = ttk.Button(
            parent, text=run_text, style="Accent.TButton", command=run_command
        )
        run_btn.grid(row=0, column=0, sticky="ew", ipady=5)

        cancel_btn = ttk.Button(
            parent, text="Cancelar", style="Danger.TButton", command=cancel_command, state="disabled"
        )
        cancel_btn.grid(row=0, column=1, padx=(8, 0), ipady=5)

        return run_btn, cancel_btn

    def _description(self, parent, row, text, columnspan=3):
        ttk.Label(
            parent, text=text, style="Muted.TLabel", wraplength=820, justify="left"
        ).grid(row=row, column=0, columnspan=columnspan, sticky="w", pady=(0, 10))