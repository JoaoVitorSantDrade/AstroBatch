# stacking_view.py
import tkinter as tk
from tkinter import ttk

from .base_view import BaseAstroView


class StackingView(BaseAstroView):
    """Aba para configuração do AstroStack com scroll e layout completo"""

    def __init__(self, parent, app):
        super().__init__(parent, app)
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)  # Permite scroll expandir
        self._build_ui()

    def _build_ui(self):
        # ============================================================
        # Container com Scroll
        # ============================================================
        canvas_container = ttk.Frame(self)
        canvas_container.grid(row=0, column=0, sticky="nsew")
        canvas_container.columnconfigure(0, weight=1)
        canvas_container.rowconfigure(0, weight=1)

        # Canvas + Scrollbar
        canvas = tk.Canvas(
            canvas_container,
            background=self.app.BG,
            highlightthickness=0,
            borderwidth=0,
        )
        scrollbar = ttk.Scrollbar(
            canvas_container, orient="vertical", command=canvas.yview
        )
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")

        # Frame interno que conterá todo o conteúdo
        inner = ttk.Frame(canvas)
        self.inner_frame = inner

        canvas_window = canvas.create_window(
            (0, 0),
            window=inner,
            anchor="nw",
            width=canvas.winfo_width(),  # Ajusta largura automaticamente
        )

        def update_scroll_region(event=None):
            canvas.configure(scrollregion=canvas.bbox("all"))

        def resize_inner(event):
            # Ajusta a largura do frame interno ao canvas
            canvas.itemconfig(canvas_window, width=event.width)

        inner.bind("<Configure>", update_scroll_region)
        canvas.bind("<Configure>", resize_inner)

        # Mouse wheel
        def on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

        canvas.bind_all("<MouseWheel>", on_mousewheel)

        # ============================================================
        # 1. Diretórios (NOVO!)
        # ============================================================
        dirs_frame = ttk.LabelFrame(
            inner, text="📁 Diretórios", style="Section.TLabelframe", padding=12
        )
        dirs_frame.grid(row=0, column=0, sticky="ew", pady=(0, 10))
        dirs_frame.columnconfigure(1, weight=1)

        # Pasta de entrada (frames alinhados do AstroAlign)
        ttk.Label(dirs_frame, text="Frames alinhados (entrada):").grid(
            row=0, column=0, sticky="w", pady=5
        )
        ttk.Entry(dirs_frame, textvariable=self.app.stack_input_dir_var).grid(
            row=0, column=1, sticky="ew", padx=8, pady=5
        )
        ttk.Button(
            dirs_frame,
            text="Selecionar",
            command=lambda: self.app.browse_dir(self.app.stack_input_dir_var),
        ).grid(row=0, column=2, pady=5)

        ttk.Label(
            dirs_frame,
            text="Use a pasta de saída do AstroAlign (ex: .../aligned) ou a pasta com as batches já alinhadas",
            style="Muted.TLabel",
        ).grid(row=1, column=0, columnspan=3, sticky="w", pady=(0, 8))

        # Pasta de saída (onde salvar a imagem empilhada)
        ttk.Label(dirs_frame, text="Pasta de saída:").grid(
            row=2, column=0, sticky="w", pady=5
        )
        ttk.Entry(dirs_frame, textvariable=self.app.stack_output_dir_var).grid(
            row=2, column=1, sticky="ew", padx=8, pady=5
        )
        ttk.Button(
            dirs_frame,
            text="Selecionar",
            command=lambda: self.app.browse_dir(self.app.stack_output_dir_var),
        ).grid(row=2, column=2, pady=5)

        ttk.Label(
            dirs_frame, text="Onde a imagem empilhada será salva", style="Muted.TLabel"
        ).grid(row=3, column=0, columnspan=3, sticky="w", pady=(0, 0))

        # ============================================================
        # 2. Seleção de Frames
        # ============================================================
        selection_frame = ttk.LabelFrame(
            inner, text="🎯 Seleção de Frames", style="Section.TLabelframe", padding=12
        )
        selection_frame.grid(row=1, column=0, sticky="ew", pady=(0, 10))
        selection_frame.columnconfigure(1, weight=1)

        # Modo de seleção
        ttk.Label(selection_frame, text="Modo:").grid(row=0, column=0, sticky="w")

        self.selection_mode_combo = ttk.Combobox(
            selection_frame,
            textvariable=self.app.stack_selection_mode_var,
            values=["All", "BestPercentage"],
            state="readonly",
            width=16,
        )
        self.selection_mode_combo.grid(row=0, column=1, sticky="w", padx=8)

        ttk.Label(
            selection_frame,
            text="'All' usa todos | 'BestPercentage' usa os melhores X%",
            style="Muted.TLabel",
        ).grid(row=0, column=2, sticky="w", padx=(8, 0))

        # Métrica
        ttk.Label(selection_frame, text="Métrica:").grid(
            row=1, column=0, sticky="w", pady=(8, 0)
        )

        self.selection_metric_combo = ttk.Combobox(
            selection_frame,
            textvariable=self.app.stack_selection_metric_var,
            values=["quality", "fwhm", "star_count", "snr"],
            state="readonly",
            width=16,
        )
        self.selection_metric_combo.grid(
            row=1, column=1, sticky="w", padx=8, pady=(8, 0)
        )

        ttk.Label(
            selection_frame,
            text="'quality' = estrelas / FWHM (recomendado)",
            style="Muted.TLabel",
        ).grid(row=1, column=2, sticky="w", padx=(8, 0), pady=(8, 0))

        # Percentual
        ttk.Label(selection_frame, text="Percentual:").grid(
            row=2, column=0, sticky="w", pady=(8, 0)
        )

        percent_frame = ttk.Frame(selection_frame)
        percent_frame.grid(row=2, column=1, sticky="w", padx=8, pady=(8, 0))

        ttk.Scale(
            percent_frame,
            from_=10,
            to=100,
            variable=self.app.stack_selection_percentage_var,
            orient="horizontal",
            length=150,
            command=self._update_percentage_label,
        ).pack(side="left")

        ttk.Label(
            percent_frame,
            textvariable=self.app.stack_selection_percentage_text_var,
            style="Muted.TLabel",
            width=5,
        ).pack(side="left", padx=(8, 0))

        # ============================================================
        # 3. Combinação
        # ============================================================
        combine_frame = ttk.LabelFrame(
            inner, text="📊 Combinação", style="Section.TLabelframe", padding=12
        )
        combine_frame.grid(row=2, column=0, sticky="ew", pady=(0, 10))
        combine_frame.columnconfigure(1, weight=1)

        ttk.Label(combine_frame, text="Método:").grid(row=0, column=0, sticky="w")
        ttk.Combobox(
            combine_frame,
            textvariable=self.app.stack_method_var,
            values=["Median", "Mean", "Sum", "Maximum", "Minimum"],
            state="readonly",
            width=14,
        ).grid(row=0, column=1, sticky="w", padx=8)

        ttk.Label(
            combine_frame,
            text="Mediana é a mais robusta para astrofotografia",
            style="Muted.TLabel",
        ).grid(row=0, column=2, sticky="w", padx=(8, 0))

        # ============================================================
        # 4. Rejeição de Outliers
        # ============================================================
        rejection_frame = ttk.LabelFrame(
            inner,
            text="🔬 Rejeição de Outliers",
            style="Section.TLabelframe",
            padding=12,
        )
        rejection_frame.grid(row=3, column=0, sticky="ew", pady=(0, 10))
        rejection_frame.columnconfigure(1, weight=1)

        ttk.Label(rejection_frame, text="Método:").grid(row=0, column=0, sticky="w")
        ttk.Combobox(
            rejection_frame,
            textvariable=self.app.stack_rejection_method_var,
            values=["None", "SigmaClip", "Winsorized", "MAD"],
            state="readonly",
            width=14,
        ).grid(row=0, column=1, sticky="w", padx=8)

        ttk.Label(
            rejection_frame,
            text="SigmaClip é o mais comum para astrofotografia",
            style="Muted.TLabel",
        ).grid(row=0, column=2, sticky="w", padx=(8, 0))

        ttk.Label(rejection_frame, text="Sigma baixo:").grid(
            row=1, column=0, sticky="w", pady=(8, 0)
        )
        ttk.Entry(
            rejection_frame, textvariable=self.app.stack_rejection_low_var, width=10
        ).grid(row=1, column=1, sticky="w", padx=8, pady=(8, 0))

        ttk.Label(
            rejection_frame,
            text="Valores abaixo deste sigma são rejeitados",
            style="Muted.TLabel",
        ).grid(row=1, column=2, sticky="w", padx=(8, 0), pady=(8, 0))

        ttk.Label(rejection_frame, text="Sigma alto:").grid(
            row=2, column=0, sticky="w", pady=(8, 0)
        )
        ttk.Entry(
            rejection_frame, textvariable=self.app.stack_rejection_high_var, width=10
        ).grid(row=2, column=1, sticky="w", padx=8, pady=(8, 0))

        ttk.Label(
            rejection_frame,
            text="Valores acima deste sigma são rejeitados",
            style="Muted.TLabel",
        ).grid(row=2, column=2, sticky="w", padx=(8, 0), pady=(8, 0))

        # ============================================================
        # 5. Normalização
        # ============================================================
        norm_frame = ttk.LabelFrame(
            inner, text="⚖️ Normalização", style="Section.TLabelframe", padding=12
        )
        norm_frame.grid(row=4, column=0, sticky="ew", pady=(0, 10))
        norm_frame.columnconfigure(1, weight=1)

        ttk.Checkbutton(
            norm_frame, text="Normalizar frames", variable=self.app.stack_normalize_var
        ).grid(row=0, column=0, columnspan=2, sticky="w")

        ttk.Label(
            norm_frame,
            text="Recomendado para frames com diferentes níveis de fundo",
            style="Muted.TLabel",
        ).grid(row=0, column=2, sticky="w", padx=(8, 0))

        ttk.Label(norm_frame, text="Método:").grid(
            row=1, column=0, sticky="w", pady=(8, 0)
        )
        ttk.Combobox(
            norm_frame,
            textvariable=self.app.stack_normalize_method_var,
            values=["Median", "Mode"],
            state="readonly",
            width=14,
        ).grid(row=1, column=1, sticky="w", padx=8, pady=(8, 0))

        ttk.Label(
            norm_frame, text="Median é o mais robusto", style="Muted.TLabel"
        ).grid(row=1, column=2, sticky="w", padx=(8, 0), pady=(8, 0))

        # ============================================================
        # 6. Pós-processamento
        # ============================================================
        post_frame = ttk.LabelFrame(
            inner, text="🔧 Pós-processamento", style="Section.TLabelframe", padding=12
        )
        post_frame.grid(row=5, column=0, sticky="ew", pady=(0, 10))
        post_frame.columnconfigure(1, weight=1)

        ttk.Checkbutton(
            post_frame,
            text="Corrigir dithering (filtro de mediana)",
            variable=self.app.stack_dither_correction_var,
        ).grid(row=0, column=0, columnspan=2, sticky="w")

        ttk.Label(
            post_frame,
            text="Útil para reduzir padrões de dithering em imagens empilhadas",
            style="Muted.TLabel",
        ).grid(row=0, column=2, sticky="w", padx=(8, 0))

        # ============================================================
        # 7. Saída - Formatos (NOVO!)
        # ============================================================
        output_frame = ttk.LabelFrame(
            inner, text="💾 Saída", style="Section.TLabelframe", padding=12
        )
        output_frame.grid(row=6, column=0, sticky="ew", pady=(0, 10))
        output_frame.columnconfigure(1, weight=1)

        ttk.Label(output_frame, text="Nome do arquivo:").grid(
            row=0, column=0, sticky="w"
        )
        ttk.Entry(
            output_frame, textvariable=self.app.stack_output_name_var, width=30
        ).grid(row=0, column=1, sticky="w", padx=8)

        # Bit depth (NOVO!)
        ttk.Label(output_frame, text="Profundidade de bits:").grid(
            row=1, column=0, sticky="w", pady=(8, 0)
        )
        ttk.Combobox(
            output_frame,
            textvariable=self.app.stack_output_bit_depth_var,
            values=["16-bit", "32-bit"],
            state="readonly",
            width=14,
        ).grid(row=1, column=1, sticky="w", padx=8, pady=(8, 0))

        ttk.Label(
            output_frame,
            text="32-bit preserva mais detalhes, 16-bit reduz tamanho",
            style="Muted.TLabel",
        ).grid(row=1, column=2, sticky="w", padx=(8, 0), pady=(8, 0))

        ttk.Checkbutton(
            output_frame,
            text="Comprimir saída (RICE_1)",
            variable=self.app.stack_compress_var,
        ).grid(row=2, column=0, columnspan=2, sticky="w", pady=(8, 0))

        ttk.Label(
            output_frame,
            text="Recomendado para reduzir o tamanho do arquivo",
            style="Muted.TLabel",
        ).grid(row=2, column=2, sticky="w", padx=(8, 0), pady=(8, 0))

        # ============================================================
        # 8. Ações
        # ============================================================
        actions = ttk.Frame(inner)
        actions.grid(row=7, column=0, sticky="ew", pady=(0, 10))
        actions.columnconfigure(0, weight=1)
        actions.columnconfigure(1, weight=1)

        self.app.btn_run_stack = ttk.Button(
            actions,
            text="▶  INICIAR ASTROSTACK",
            style="Accent.TButton",
            command=self.app.start_stacking,
        )
        self.app.btn_run_stack.grid(row=0, column=0, sticky="ew", ipady=5)

        self.app.btn_cancel_stack = ttk.Button(
            actions,
            text="Cancelar",
            style="Danger.TButton",
            command=self.app.cancel_processing,
            state="disabled",
        )
        self.app.btn_cancel_stack.grid(
            row=0, column=1, sticky="ew", padx=(8, 0), ipady=5
        )

        # Espaço extra no final para garantir que o último elemento seja visível
        ttk.Frame(inner, height=20).grid(row=8, column=0)

    def _update_percentage_label(self, value):
        """Atualiza o label do percentual quando o slider é movido"""
        self.app.stack_selection_percentage_text_var.set(f"{int(float(value))}%")

    def _path_row(
        self, parent, row, label, variable, browse_command, browse_text="Selecionar"
    ):
        """Método auxiliar para criar linha de caminho"""
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=5)
        ttk.Entry(parent, textvariable=variable).grid(
            row=row, column=1, sticky="ew", padx=8, pady=5
        )
        ttk.Button(parent, text=browse_text, command=browse_command).grid(
            row=row, column=2, pady=5
        )
