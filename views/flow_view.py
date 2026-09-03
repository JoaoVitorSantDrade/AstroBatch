import threading
import tkinter as tk
from pathlib import Path
from tkinter import ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

from views.base_view import BaseAstroView


class FlowView(BaseAstroView):
    def __init__(self, parent, app):
        super().__init__(parent, app)
        self.columnconfigure(0, weight=1)
        self.rowconfigure(3, weight=1)

        self.reference_preview_frame = None
        self.reference_preview_canvas = None
        self.reference_preview_inner = None
        self.reference_preview_images = []
        self.reference_preview_generation = 0

        self._build_ui()
        self.after(150, self.refresh_reference_preview)

    def _build_ui(self):
        # ============================================================
        # CARD 1: Detecção (Local Flow)
        # ============================================================
        card_local = ttk.LabelFrame(
            self,
            text="1. Extração de Estrelas (Local Flow)",
            style="Section.TLabelframe",
            padding=12,
        )
        card_local.grid(row=0, column=0, sticky="ew", pady=(0, 10))

        for i in range(4):
            card_local.columnconfigure(i, weight=1)

        ttk.Label(card_local, text="Star Engine:").grid(
            row=0, column=0, sticky="w", pady=(0, 5)
        )
        ttk.Combobox(
            card_local,
            textvariable=self.app.flow_engine_var,
            values=["DAO", "OpenCV"],
            state="readonly",
            width=14,
        ).grid(row=0, column=1, sticky="w", padx=8, pady=(0, 5))

        ttk.Label(card_local, text="Mínimo de estrelas:").grid(
            row=0, column=2, sticky="w", pady=(0, 5)
        )
        ttk.Entry(card_local, textvariable=self.app.flow_min_stars_var, width=14).grid(
            row=0, column=3, sticky="w", padx=8, pady=(0, 5)
        )

        ttk.Label(card_local, text="FWHM médio (px):").grid(
            row=1, column=0, sticky="w", pady=(5, 5)
        )
        ttk.Entry(card_local, textvariable=self.app.flow_fwhm_var, width=14).grid(
            row=1, column=1, sticky="w", padx=8, pady=(5, 5)
        )

        ttk.Label(card_local, text="Sigma (Início da Busca):").grid(
            row=1, column=2, sticky="w", pady=(5, 5)
        )
        ttk.Entry(card_local, textvariable=self.app.flow_sigma_var, width=14).grid(
            row=1, column=3, sticky="w", padx=8, pady=(5, 5)
        )

        ttk.Checkbutton(
            card_local,
            text="Gerar imagens de diagnóstico nas âncoras (debug)",
            variable=self.app.flow_debug_var,
        ).grid(row=2, column=0, columnspan=4, sticky="w", pady=(10, 8))

        # Botões de Ação - Local Flow
        advanced = ttk.LabelFrame(
            card_local, text="Engines (Avancado)", style="Section.TLabelframe", padding=8
        )
        advanced.grid(row=3, column=0, columnspan=4, sticky="ew", pady=(0, 8))
        ttk.Label(advanced, text="Perfil:").grid(row=0, column=0, sticky="w")
        ttk.Combobox(
            advanced, textvariable=self.app.flow_profile_var, values=["Stable", "Fast"],
            state="readonly", width=12,
        ).grid(row=0, column=1, sticky="w", padx=8)
        ttk.Label(advanced, text="Detector:").grid(row=0, column=2, sticky="w")
        ttk.Combobox(
            advanced, textvariable=self.app.flow_detector_engine_var,
            values=["", "dao", "opencv-contours", "opencv-components", "sep"],
            state="readonly", width=20,
        ).grid(row=0, column=3, sticky="w", padx=8)
        ttk.Label(advanced, text="Fallback:").grid(row=1, column=0, sticky="w", pady=(6, 0))
        ttk.Combobox(
            advanced, textvariable=self.app.flow_transform_fallback_var,
            values=["Disabled", "astroalign-asterism"], state="readonly", width=20,
        ).grid(row=1, column=1, sticky="w", padx=8, pady=(6, 0))
        ttk.Label(
            advanced, text="Fast escolhe OpenCV components quando Detector estiver vazio.",
            style="Muted.TLabel",
        ).grid(row=1, column=2, columnspan=2, sticky="w", pady=(6, 0))

        actions_local = ttk.Frame(card_local)
        actions_local.grid(row=4, column=0, columnspan=4, sticky="ew")
        actions_local.columnconfigure(2, weight=1)

        self.btn_select_anchor = ttk.Button(
            actions_local,
            text="🖼 Definir referência",
            command=self.app.open_anchor_selector,
        )
        self.btn_select_anchor.grid(row=0, column=0, sticky="w", padx=(0, 6), ipady=3)

        self.btn_preview_flow = ttk.Button(
            actions_local,
            text="🔍 Preview detecção",
            command=self.app.show_astroflow_preview,
        )
        self.btn_preview_flow.grid(row=0, column=1, sticky="w", padx=6, ipady=3)

        self.app.btn_run_flow = ttk.Button(
            actions_local,
            text="▶ Executar AstroFlow (Local + Global)",
            style="Accent.TButton",
            command=self.app.start_flow_processing,
        )
        self.app.btn_run_flow.grid(row=0, column=2, sticky="e", ipady=3)

        # ============================================================
        # CARD 2: Pareamento Global
        # ============================================================
        card_global = ttk.LabelFrame(
            self,
            text="2. Construção da Malha (Global Flow)",
            style="Section.TLabelframe",
            padding=12,
        )
        card_global.grid(row=1, column=0, sticky="ew", pady=(0, 10))

        for i in range(4):
            card_global.columnconfigure(i, weight=1)

        ttk.Label(card_global, text="Global Master:").grid(
            row=0, column=0, sticky="w", pady=(0, 5)
        )
        self.combo_global_master = ttk.Combobox(
            card_global,
            textvariable=self.app.flow_global_master_var,
            values=["Auto"],
            state="readonly",
            width=14,
        )
        self.combo_global_master.grid(row=0, column=1, sticky="w", padx=8, pady=(0, 5))

        ttk.Label(card_global, text="Raio de busca (px):").grid(
            row=0, column=2, sticky="w", pady=(0, 5)
        )
        ttk.Entry(
            card_global, textvariable=self.app.flow_matching_radius_var, width=14
        ).grid(row=0, column=3, sticky="w", padx=8, pady=(0, 5))

        ttk.Label(card_global, text="RANSAC threshold:").grid(
            row=1, column=0, sticky="w", pady=(5, 5)
        )
        ttk.Entry(card_global, textvariable=self.app.flow_ransac_var, width=14).grid(
            row=1, column=1, sticky="w", padx=8, pady=(5, 5)
        )

        ttk.Label(card_global, text="Mínimo inliers / Ratio:").grid(
            row=1, column=2, sticky="w", pady=(5, 5)
        )
        ratio_frame = ttk.Frame(card_global)
        ratio_frame.grid(row=1, column=3, sticky="w", padx=8, pady=(5, 5))
        ttk.Entry(
            ratio_frame, textvariable=self.app.flow_min_inliers_var, width=6
        ).pack(side="left")
        ttk.Label(ratio_frame, text=" / ").pack(side="left")
        ttk.Entry(ratio_frame, textvariable=self.app.flow_min_ratio_var, width=6).pack(
            side="left"
        )

        # Botões de Ação - Global Flow
        actions_global = ttk.Frame(card_global)
        actions_global.grid(row=2, column=0, columnspan=4, sticky="ew", pady=(10, 0))
        actions_global.columnconfigure(0, weight=1)

        ttk.Label(
            actions_global, text="Ajustou o Raio ou o RANSAC?", style="Muted.TLabel"
        ).grid(row=0, column=0, sticky="w")

        self.btn_run_global_only = ttk.Button(
            actions_global,
            text="🌍 Calcular Apenas Malha Global",
            command=self.start_global_flow_only,
        )
        self.btn_run_global_only.grid(row=0, column=1, sticky="e", padx=(0, 6), ipady=3)

        self.btn_viz_flow = ttk.Button(
            actions_global,
            text="📈 Visualizar 3D",
            command=self.app.show_flow_visualization,
        )
        self.btn_viz_flow.grid(row=0, column=2, sticky="e", padx=6, ipady=3)

        self.app.btn_cancel_flow = ttk.Button(
            actions_global,
            text="Cancelar",
            style="Danger.TButton",
            command=self.app.cancel_processing,
            state="disabled",
        )
        self.app.btn_cancel_flow.grid(row=0, column=3, sticky="e", ipady=3)

        # Container de Thumbnails
        self._build_reference_preview(self)

    def _build_reference_preview(self, parent):
        preview_frame = ttk.LabelFrame(
            parent,
            text="Referências selecionadas",
            style="Section.TLabelframe",
            padding=10,
        )
        preview_frame.grid(
            row=3,
            column=0,
            sticky="nsew",
            pady=(0, 10),
        )

        parent.rowconfigure(3, weight=1)
        preview_frame.columnconfigure(0, weight=1)
        preview_frame.rowconfigure(0, weight=1)

        self.reference_preview_frame = preview_frame

        # ----------------------------------------------------
        # Canvas + scrollbar
        # ----------------------------------------------------
        canvas_container = ttk.Frame(preview_frame)
        canvas_container.grid(row=0, column=0, sticky="nsew")

        canvas_container.columnconfigure(0, weight=1)
        canvas_container.rowconfigure(0, weight=1)

        canvas = tk.Canvas(
            canvas_container,
            background=self.app.BG,
            highlightthickness=0,
            borderwidth=0,
        )

        scrollbar = ttk.Scrollbar(
            canvas_container,
            orient="vertical",
            command=canvas.yview,
        )

        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")

        self.reference_preview_canvas = canvas

        # ----------------------------------------------------
        # Frame interno
        # ----------------------------------------------------
        inner = ttk.Frame(canvas)
        self.reference_preview_inner = inner

        window_id = canvas.create_window(
            (0, 0),
            window=inner,
            anchor="nw",
        )

        def update_scroll_region(event=None):
            canvas.configure(scrollregion=canvas.bbox("all"))

        def resize_inner(event):
            canvas.itemconfigure(window_id, width=event.width)

        inner.bind("<Configure>", update_scroll_region)
        canvas.bind("<Configure>", resize_inner)

        # Inicialmente mostra estado vazio.
        self.after(100, self.refresh_reference_preview)

    def start_global_flow_only(self):
        """
        Gatilho para iniciar apenas a lógica de Global Flow
        sem recalcular os Flows Locais.
        Injeta uma flag na configuração para que a lógica principal saiba.
        """
        if self.app.worker and self.app.worker.is_alive():
            return

        base_dir_str = self.app.batch_dir_var.get()
        if not base_dir_str:
            self.app.print_to_console("ERRO: Selecione a Pasta Base primeiro.\n")
            return

        batch_dir = Path(base_dir_str).expanduser().resolve()

        config = {
            "custom_anchors": dict(self.app.custom_anchors),
            "global_master": self.app.flow_global_master_var.get(),
            "fwhm": self.app.flow_fwhm_var.get(),
            "sigma": self.app.flow_sigma_var.get(),
            "matching_radius": self.app.flow_matching_radius_var.get(),
            "ransac": self.app.flow_ransac_var.get(),
            "debug_images": self.app.flow_debug_var.get(),
            "min_stars": self.app.flow_min_stars_var.get(),
            "min_inliers": self.app.flow_min_inliers_var.get(),
            "min_ratio": self.app.flow_min_ratio_var.get(),
            "max_stars": 150,
            "engine": self.app.flow_engine_var.get(),
            "skip_local_flow": True,  # <--- Flag cruciaL adicionada
        }

        self.app.save_settings()
        self.app._lock_ui("Flow")
        self.app.worker = threading.Thread(
            target=self.app.run_flow_logic, args=(batch_dir, config), daemon=True
        )
        self.app.worker.start()

    def _load_reference_thumbnail(self, filepath):
        """
        Carrega uma imagem FITS e cria um thumbnail para o Grid.
        O processamento pesado ocorre fora da thread da GUI.
        """
        from astroflow_logic import extract_luminance, load_fits_data

        data, header = load_fits_data(filepath)
        luminance = extract_luminance(data, header)

        if luminance is None:
            raise ValueError("Não foi possível extrair luminância.")

        luminance = np.asarray(luminance, dtype=np.float32)

        if luminance.ndim != 2:
            raise ValueError(f"Imagem inválida para preview: shape={luminance.shape}")

        h, w = luminance.shape

        # ----------------------------------------------------
        # Resize mantendo proporção
        # ----------------------------------------------------
        max_width = 230
        max_height = 150

        scale = min(max_width / max(w, 1), max_height / max(h, 1), 1.0)
        new_w = max(int(w * scale), 1)
        new_h = max(int(h * scale), 1)

        thumbnail = cv2.resize(
            luminance,
            (new_w, new_h),
            interpolation=cv2.INTER_AREA,
        )

        # ----------------------------------------------------
        # Stretch para visualização
        # ----------------------------------------------------
        median = float(np.median(thumbnail))
        p25, p75 = np.percentile(thumbnail, [25, 75])

        std = max(float((p75 - p25) / 1.35), 1e-6)
        vmin = median - 0.5 * std
        vmax = median + 6.0 * std

        normalized = np.clip(
            (thumbnail - vmin) / max(vmax - vmin, 1e-6),
            0.0,
            1.0,
        )

        return (normalized * 255).astype(np.uint8)

    def refresh_reference_preview(self):
        """
        Reconstrói o Grid de referências selecionadas.
        4 imagens por linha.
        O Grid pode possuir quantas linhas forem necessárias.
        """
        if (
            self.reference_preview_inner is None
            or self.reference_preview_canvas is None
        ):
            return

        base_dir_str = self.app.batch_dir_var.get()
        if not base_dir_str:
            self._show_empty_reference_preview(
                "Selecione uma Pasta Base para visualizar as referências."
            )
            return

        base_dir = Path(base_dir_str).expanduser().resolve()

        if not base_dir.is_dir():
            self._show_empty_reference_preview("A Pasta Base não existe.")
            return

        # ----------------------------------------------------
        # Invalida carregamentos anteriores
        # ----------------------------------------------------
        self.reference_preview_generation += 1
        generation = self.reference_preview_generation
        inner = self.reference_preview_inner

        # ----------------------------------------------------
        # Limpa Grid anterior
        # ----------------------------------------------------
        for widget in inner.winfo_children():
            widget.destroy()

        self.reference_preview_images.clear()

        # ----------------------------------------------------
        # Descobre batches
        # ----------------------------------------------------
        try:
            batch_folders = sorted(
                [
                    d
                    for d in base_dir.iterdir()
                    if d.is_dir() and "batch" in d.name.lower()
                ],
                key=lambda p: p.name.lower(),
            )
        except OSError as exc:
            self._show_empty_reference_preview(
                f"Não foi possível acessar a Pasta Base:\n{exc}"
            )
            return

        # ----------------------------------------------------
        # Somente batches que possuem referência
        # ----------------------------------------------------
        references = []

        for batch_dir in batch_folders:
            frame_name = self.app.custom_anchors.get(batch_dir.name)

            if not frame_name:
                continue

            frame_path = batch_dir / frame_name

            if not frame_path.is_file():
                references.append((batch_dir.name, frame_name, None))
                continue

            references.append((batch_dir.name, frame_name, frame_path))

        if not references:
            self._show_empty_reference_preview(
                "Nenhuma referência personalizada foi definida.\n\n"
                "Use “Definir referência” para selecionar os frames."
            )
            return

        # ----------------------------------------------------
        # Grid 4 colunas
        # ----------------------------------------------------
        for column in range(4):
            inner.columnconfigure(column, weight=1)

        # ----------------------------------------------------
        # Estado de carregamento
        # ----------------------------------------------------
        cards = []

        for index, (batch_name, frame_name, frame_path) in enumerate(references):
            row = index // 4
            column = index % 4

            card = ttk.Frame(
                inner,
                padding=6,
                relief="solid",
            )

            card.grid(row=row, column=column, sticky="nsew", padx=5, pady=5)
            cards.append((card, batch_name, frame_name, frame_path))

        # ----------------------------------------------------
        # Threads individuais para thumbnails
        # ----------------------------------------------------
        for card, batch_name, frame_name, frame_path in cards:
            threading.Thread(
                target=self._load_card,
                args=(card, batch_name, frame_name, frame_path, generation),
                daemon=True,
            ).start()

    # ----------------------------------------------------
    # Carregamento paralelo dos thumbnails e UI
    # ----------------------------------------------------
    def _load_card(self, card, batch_name, frame_name, frame_path, generation):
        if generation != self.reference_preview_generation:
            return

        if frame_path is None:
            self.after(
                0,
                lambda: self._populate_reference_card_error(
                    card, batch_name, frame_name, "Arquivo não encontrado"
                ),
            )
            return

        try:
            image = self._load_reference_thumbnail(frame_path)
            self.after(
                0,
                lambda img=image: self._populate_reference_card(
                    card, batch_name, frame_name, img, generation
                ),
            )
        except Exception:  # Fixed undefined 'exc' variable
            self.after(
                0,
                lambda: self._populate_reference_card_error(
                    card, batch_name, frame_name, str(exc)
                ),
            )
            self.reference_preview_canvas.yview_moveto(0)

    def _populate_reference_card(
        self, card, batch_name, frame_name, img_array, generation
    ):
        if not card.winfo_exists() or generation != self.reference_preview_generation:
            return

        img_pil = Image.fromarray(img_array)
        img_tk = ImageTk.PhotoImage(img_pil)

        # Mantém a referência da imagem para que o garbage collector não a apague
        self.reference_preview_images.append(img_tk)

        img_label = ttk.Label(card, image=img_tk, anchor="center")
        img_label.pack(expand=True, pady=(5, 5))

        ttk.Label(
            card, text=batch_name, font=("Segoe UI Semibold", 9), anchor="center"
        ).pack(fill=tk.X)
        ttk.Label(card, text=frame_name, style="Muted.TLabel", anchor="center").pack(
            fill=tk.X
        )

    def _populate_reference_card_error(self, card, batch_name, frame_name, error):
        if not card.winfo_exists():
            return

        ttk.Label(card, text="⚠", font=("Segoe UI", 22), anchor="center").pack(
            expand=True, pady=(20, 5)
        )

        ttk.Label(
            card, text=batch_name, font=("Segoe UI Semibold", 9), anchor="center"
        ).pack(fill=tk.X)

        ttk.Label(card, text=frame_name, style="Muted.TLabel", anchor="center").pack(
            fill=tk.X
        )

        ttk.Label(
            card, text=error, style="Muted.TLabel", anchor="center", wraplength=210
        ).pack(fill=tk.X, pady=(2, 8))

    def _show_empty_reference_preview(self, message):
        if self.reference_preview_inner is None:
            return

        for widget in self.reference_preview_inner.winfo_children():
            widget.destroy()

        self.reference_preview_images.clear()

        ttk.Label(
            self.reference_preview_inner,
            text=message,
            style="Muted.TLabel",
            justify="center",
            anchor="center",
        ).grid(
            row=0,
            column=0,
            columnspan=4,
            sticky="nsew",
            padx=20,
            pady=40,
        )

        for column in range(4):
            self.reference_preview_inner.columnconfigure(column, weight=1)

    def on_theme_changed(self, is_dark: bool):
        """Atualiza widgets nativos do Tkinter que não herdam o ttk.Style automaticamente."""
        if hasattr(self, "reference_preview_canvas") and self.reference_preview_canvas:
            self.reference_preview_canvas.configure(background=self.app.BG)
