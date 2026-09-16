# stacking_view.py
import tkinter as tk
from tkinter import ttk

from .base_view import BaseAstroView
from .stack_model import StackViewModel


class StackingView(BaseAstroView):
    """Aba para configuração do AstroStack com scroll e layout completo"""

    def __init__(self, parent, model: StackViewModel):
        super().__init__(parent, model)
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
            background=self.model.BG,
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
            widget = self.winfo_containing(event.x_root, event.y_root)
            while widget is not None:
                if widget is self:
                    canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
                    break
                widget = widget.master

        # Scope the wheel handler to this view.  A global bind_all handler
        # steals scrolling from the other tabs and from the activity panel.
        canvas.bind("<MouseWheel>", on_mousewheel, add="+")

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
        ttk.Entry(dirs_frame, textvariable=self.model.stack_input_dir_var).grid(
            row=0, column=1, sticky="ew", padx=8, pady=5
        )
        ttk.Button(
            dirs_frame,
            text="Selecionar",
            command=lambda: self.model.browse_dir(self.model.stack_input_dir_var),
        ).grid(row=0, column=2, pady=5)
        ttk.Button(
            dirs_frame,
            text="Usar saída do Align",
            command=self.model.use_align_output_for_stack,
        ).grid(row=0, column=3, padx=(8, 0), pady=5)

        ttk.Label(
            dirs_frame,
            text="Use a pasta de saída do AstroAlign (ex: .../aligned) ou a pasta com as batches já alinhadas",
            style="Muted.TLabel",
        ).grid(row=1, column=0, columnspan=4, sticky="w", pady=(0, 8))

        # Pasta de saída (onde salvar a imagem empilhada)
        ttk.Label(dirs_frame, text="Pasta de saída:").grid(
            row=2, column=0, sticky="w", pady=5
        )
        ttk.Entry(dirs_frame, textvariable=self.model.stack_output_dir_var).grid(
            row=2, column=1, sticky="ew", padx=8, pady=5
        )
        ttk.Button(
            dirs_frame,
            text="Selecionar",
            command=lambda: self.model.browse_dir(self.model.stack_output_dir_var),
        ).grid(row=2, column=2, pady=5)

        ttk.Label(
            dirs_frame, text="Onde a imagem empilhada será salva", style="Muted.TLabel"
        ).grid(row=3, column=0, columnspan=4, sticky="w", pady=(0, 0))

        ttk.Label(dirs_frame, text="Perfil de features:").grid(
            row=4, column=0, sticky="w", pady=(8, 0)
        )
        ttk.Combobox(
            dirs_frame,
            textvariable=self.model.stack_feature_profile_var,
            values=["Intelligent", "Legacy"],
            state="readonly",
            width=16,
        ).grid(row=4, column=1, sticky="w", padx=8, pady=(8, 0))
        ttk.Label(
            dirs_frame,
            text="Intelligent usa seleção explicável, trailing conservador e RAM limitada; Legacy preserva o fluxo anterior.",
            style="Muted.TLabel",
            wraplength=520,
            justify="left",
        ).grid(row=4, column=2, columnspan=2, sticky="w", padx=(8, 0), pady=(8, 0))

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
            textvariable=self.model.stack_selection_mode_var,
            values=["All", "BestPercentage", "MultiMetric"],
            state="readonly",
            width=16,
        )
        self.selection_mode_combo.grid(row=0, column=1, sticky="w", padx=8)

        ttk.Label(
            selection_frame,
            text="'All' usa todos | 'BestPercentage' usa métrica única | 'MultiMetric' usa score robusto",
            style="Muted.TLabel",
        ).grid(row=0, column=2, sticky="w", padx=(8, 0))

        # Métrica
        ttk.Label(selection_frame, text="Métrica:").grid(
            row=1, column=0, sticky="w", pady=(8, 0)
        )

        self.selection_metric_combo = ttk.Combobox(
            selection_frame,
            textvariable=self.model.stack_selection_metric_var,
            values=["quality", "fwhm", "star_count", "snr", "roundness"],
            state="readonly",
            width=16,
        )
        self.selection_metric_combo.grid(
            row=1, column=1, sticky="w", padx=8, pady=(8, 0)
        )

        ttk.Label(
            selection_frame,
            text="'quality' = estrelas / FWHM | 'roundness' = b/a",
            style="Muted.TLabel",
        ).grid(row=1, column=2, sticky="w", padx=(8, 0), pady=(8, 0))

        ttk.Label(selection_frame, text="Perfil de seleção:").grid(
            row=7, column=0, sticky="w", pady=(10, 0)
        )
        ttk.Combobox(
            selection_frame,
            textvariable=self.model.stack_selection_profile_var,
            values=["Sharpness", "Balanced", "Signal", "Custom"],
            state="readonly",
            width=16,
        ).grid(row=7, column=1, sticky="w", padx=8, pady=(10, 0))
        ttk.Label(
            selection_frame,
            text="Pesos robustos normalizados por percentis; Custom usa o campo abaixo.",
            style="Muted.TLabel",
        ).grid(row=7, column=2, sticky="w", padx=(8, 0), pady=(10, 0))

        ttk.Label(selection_frame, text="Pesos avançados:").grid(
            row=8, column=0, sticky="w", pady=(8, 0)
        )
        ttk.Entry(
            selection_frame,
            textvariable=self.model.stack_selection_weights_var,
            width=36,
        ).grid(row=8, column=1, sticky="ew", padx=8, pady=(8, 0))
        ttk.Label(
            selection_frame,
            text="Ex.: fwhm=0.4,roundness=0.3,snr=0.3 (aplica-se ao perfil Custom).",
            style="Muted.TLabel",
        ).grid(row=8, column=2, sticky="w", padx=(8, 0), pady=(8, 0))

        ttk.Label(selection_frame, text="Política de trailing:").grid(
            row=9, column=0, sticky="w", pady=(8, 0)
        )
        ttk.Combobox(
            selection_frame,
            textvariable=self.model.stack_trail_policy_var,
            values=["exclude_severe", "weight_only", "report", "off"],
            state="readonly",
            width=16,
        ).grid(row=9, column=1, sticky="w", padx=8, pady=(8, 0))
        ttk.Label(
            selection_frame,
            text="Somente casos graves e confiáveis são excluídos automaticamente; os moderados recebem penalidade.",
            style="Muted.TLabel",
            wraplength=520,
            justify="left",
        ).grid(row=9, column=2, sticky="w", padx=(8, 0), pady=(8, 0))

        # Triagem opcional de frames sem guiagem
        ttk.Checkbutton(
            selection_frame,
            text="Rejeitar subs com rastros",
            variable=self.model.stack_trail_filter_var,
        ).grid(row=2, column=0, columnspan=2, sticky="w", pady=(12, 0))

        ttk.Label(
            selection_frame,
            text="Ativo rejeita medições desconhecidas; em datasets antigos, rode o AstroFlow novamente antes do filtro.",
            style="Muted.TLabel",
            wraplength=420,
            justify="left",
        ).grid(row=2, column=2, sticky="w", padx=(8, 0), pady=(12, 0))

        ttk.Label(selection_frame, text="Roundness mínima (b/a):").grid(
            row=3, column=0, sticky="w", pady=(8, 0)
        )
        ttk.Spinbox(
            selection_frame,
            from_=0.0,
            to=1.0,
            increment=0.01,
            textvariable=self.model.stack_min_roundness_var,
            width=8,
        ).grid(row=3, column=1, sticky="w", padx=8, pady=(8, 0))
        ttk.Label(
            selection_frame,
            text="1 = estrela circular; valores menores aceitam formas mais alongadas.",
            style="Muted.TLabel",
        ).grid(row=3, column=2, sticky="w", padx=(8, 0), pady=(8, 0))

        ttk.Label(selection_frame, text="Mínimo de estrelas medidas:").grid(
            row=4, column=0, sticky="w", pady=(8, 0)
        )
        ttk.Spinbox(
            selection_frame,
            from_=1,
            to=64,
            increment=1,
            textvariable=self.model.stack_min_shape_stars_var,
            width=8,
        ).grid(row=4, column=1, sticky="w", padx=8, pady=(8, 0))
        ttk.Label(
            selection_frame,
            text="Evita decidir a forma com poucas estrelas detectadas.",
            style="Muted.TLabel",
            wraplength=420,
            justify="left",
        ).grid(row=4, column=2, sticky="w", padx=(8, 0), pady=(8, 0))

        ttk.Button(
            selection_frame,
            text="Preset: Subs sem guiagem",
            command=self.model.apply_unguided_preset,
        ).grid(row=5, column=0, columnspan=2, sticky="w", pady=(12, 0))
        ttk.Label(
            selection_frame,
            text="Define filtro 0.65, 5 estrelas, All, Mean, SigmaClip e Stable; All mantém todos os aprovados.",
            style="Muted.TLabel",
            wraplength=420,
            justify="left",
        ).grid(row=5, column=2, sticky="w", padx=(8, 0), pady=(12, 0))

        # Percentual
        ttk.Label(selection_frame, text="Percentual:").grid(
            row=6, column=0, sticky="w", pady=(8, 0)
        )

        percent_frame = ttk.Frame(selection_frame)
        percent_frame.grid(row=6, column=1, sticky="w", padx=8, pady=(8, 0))

        ttk.Scale(
            percent_frame,
            from_=10,
            to=100,
            variable=self.model.stack_selection_percentage_var,
            orient="horizontal",
            length=150,
            command=self._update_percentage_label,
        ).pack(side="left")

        ttk.Label(
            percent_frame,
            textvariable=self.model.stack_selection_percentage_text_var,
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
            textvariable=self.model.stack_method_var,
            values=["Median", "Mean", "QualityWeightedMean", "Sum", "Maximum", "Minimum"],
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
            textvariable=self.model.stack_rejection_method_var,
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
            rejection_frame, textvariable=self.model.stack_rejection_low_var, width=10
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
            rejection_frame, textvariable=self.model.stack_rejection_high_var, width=10
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
            norm_frame, text="Normalizar frames", variable=self.model.stack_normalize_var
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
            textvariable=self.model.stack_normalize_method_var,
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
            variable=self.model.stack_dither_correction_var,
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
            output_frame, textvariable=self.model.stack_output_name_var, width=30
        ).grid(row=0, column=1, sticky="w", padx=8)

        ttk.Label(output_frame, text="Profundidade de bits:").grid(
            row=1, column=0, sticky="w", pady=(8, 0)
        )
        ttk.Label(output_frame, text="16-bit").grid(
            row=1, column=1, sticky="w", padx=8, pady=(8, 0)
        )

        ttk.Label(
            output_frame,
            text="Saída preserva FIT/TIFF da sessão: 16-bit",
            style="Muted.TLabel",
        ).grid(row=1, column=2, sticky="w", padx=(8, 0), pady=(8, 0))

        ttk.Checkbutton(
            output_frame,
            text="Comprimir saída (RICE_1)",
            variable=self.model.stack_compress_var,
        ).grid(row=2, column=0, columnspan=2, sticky="w", pady=(8, 0))

        ttk.Label(
            output_frame,
            text="Recomendado para reduzir o tamanho do arquivo",
            style="Muted.TLabel",
        ).grid(row=2, column=2, sticky="w", padx=(8, 0), pady=(8, 0))

        ttk.Label(output_frame, text="Armazenamento da redução:").grid(
            row=4, column=0, sticky="w", pady=(10, 0)
        )
        ttk.Combobox(
            output_frame,
            textvariable=self.model.stack_reduction_storage_var,
            values=["ram", "disk_legacy", "ram_spill"],
            state="readonly",
            width=16,
        ).grid(row=4, column=1, sticky="w", padx=8, pady=(10, 0))
        ttk.Label(
            output_frame,
            text="RAM é o padrão Intelligent; spill é opt-in e requer pasta/limite explícitos.",
            style="Muted.TLabel",
        ).grid(row=4, column=2, sticky="w", padx=(8, 0), pady=(10, 0))

        ttk.Label(output_frame, text="Pasta de spill (opcional):").grid(
            row=5, column=0, sticky="w", pady=(8, 0)
        )
        ttk.Entry(
            output_frame,
            textvariable=self.model.stack_spill_directory_var,
        ).grid(row=5, column=1, sticky="ew", padx=8, pady=(8, 0))
        ttk.Button(
            output_frame,
            text="Selecionar",
            command=lambda: self.model.browse_dir(self.model.stack_spill_directory_var),
        ).grid(row=5, column=2, sticky="w", pady=(8, 0))

        ttk.Label(output_frame, text="Limite spill (MiB):").grid(
            row=6, column=0, sticky="w", pady=(8, 0)
        )
        ttk.Spinbox(
            output_frame,
            from_=256,
            to=1048576,
            increment=256,
            textvariable=self.model.stack_spill_limit_var,
            width=12,
        ).grid(row=6, column=1, sticky="w", padx=8, pady=(8, 0))
        ttk.Label(
            output_frame,
            text="Nenhum arquivo de spill é criado no modo RAM puro.",
            style="Muted.TLabel",
        ).grid(row=6, column=2, sticky="w", padx=(8, 0), pady=(8, 0))

        # ============================================================
        # 8. Ações
        # ============================================================
        ttk.Label(output_frame, text="Perfil de engine:").grid(
            row=3, column=0, sticky="w", pady=(8, 0)
        )
        ttk.Combobox(
            output_frame, textvariable=self.model.stack_profile_var,
            values=["Stable", "Fast"], state="readonly", width=14,
        ).grid(row=3, column=1, sticky="w", padx=8, pady=(8, 0))
        ttk.Combobox(
            output_frame, textvariable=self.model.stack_reducer_engine_var,
            values=["", "stable-numpy", "fast-numba"], state="readonly", width=16,
        ).grid(row=3, column=2, sticky="w", padx=(8, 0), pady=(8, 0))

        actions = ttk.Frame(inner)
        actions.grid(row=7, column=0, sticky="ew", pady=(0, 10))
        actions.columnconfigure(0, weight=1)
        actions.columnconfigure(1, weight=1)

        self.btn_run_stack = ttk.Button(
            actions,
            text="▶  INICIAR ASTROSTACK",
            style="Accent.TButton",
            command=self.model.start_stacking,
        )
        self.btn_run_stack.grid(row=0, column=0, sticky="ew", ipady=5)

        self.btn_cancel_stack = ttk.Button(
            actions,
            text="Cancelar",
            style="Danger.TButton",
            command=self.model.cancel_processing,
            state="disabled",
        )
        self.btn_cancel_stack.grid(
            row=0, column=1, sticky="ew", padx=(8, 0), ipady=5
        )

        # Espaço extra no final para garantir que o último elemento seja visível
        ttk.Frame(inner, height=20).grid(row=8, column=0)

    def _update_percentage_label(self, value):
        """Atualiza o label do percentual quando o slider é movido"""
        self.model.stack_selection_percentage_text_var.set(f"{int(float(value))}%")

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
