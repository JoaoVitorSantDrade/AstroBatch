import threading
import tkinter as tk
from pathlib import Path
from tkinter import ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

from views.base_view import BaseAstroView
from views.flow_model import FlowViewModel
from views.preview_service import PreviewService


def format_reference_provenance(flow_data, frame_name):
    """Return compact status and graph provenance text for a reference card."""

    if not flow_data:
        return "Flow pendente"
    frame = (flow_data.get("frames") or {}).get(frame_name) or {}
    status = str(frame.get("status", "desconhecido"))
    stale_prefix = "stale • " if flow_data.get("stale") else ""
    if status != "accepted":
        reason = frame.get("reason") or frame.get("rejected_reason") or frame.get("error")
        return f"{stale_prefix}{status} • {reason or 'sem motivo registrado'}"
    method = frame.get("recovery_method") or "direct"
    hops = int(frame.get("hop_count", 0) or 0)
    parent = frame.get("relative_to") or "reference"
    confidence = frame.get("confidence")
    details = [method, f"hops={hops}", f"parent={parent}"]
    if confidence:
        details.append(f"confidence={confidence}")
    return stale_prefix + "accepted • " + " • ".join(details)


class FlowView(BaseAstroView):
    def __init__(self, parent, model: FlowViewModel):
        super().__init__(parent, model)
        self.columnconfigure(0, weight=1)
        self.rowconfigure(3, weight=1)

        self.reference_preview_frame = None
        self.reference_preview_canvas = None
        self.reference_preview_inner = None
        self.reference_preview_images = []
        self.reference_preview_generation = 0
        self._preview_cards = {}
        self._preview_service = PreviewService(self._load_reference_thumbnail, max_workers=2, max_pending=32)
        self._preview_after_id = self.after(150, self.refresh_reference_preview)
        self.bind("<Destroy>", self._on_destroy, add="+")

        self._build_ui()

    def _on_destroy(self, _event=None):
        self.reference_preview_generation += 1
        self._preview_service.close()
        after_id = self._preview_after_id
        if after_id:
            try:
                self.after_cancel(after_id)
            except tk.TclError:
                pass
            self._preview_after_id = None

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
            textvariable=self.model.flow_engine_var,
            values=["DAO", "OpenCV"],
            state="readonly",
            width=14,
        ).grid(row=0, column=1, sticky="w", padx=8, pady=(0, 5))

        ttk.Label(card_local, text="Mínimo de estrelas:").grid(
            row=0, column=2, sticky="w", pady=(0, 5)
        )
        ttk.Entry(card_local, textvariable=self.model.flow_min_stars_var, width=14).grid(
            row=0, column=3, sticky="w", padx=8, pady=(0, 5)
        )

        ttk.Label(card_local, text="FWHM médio (px):").grid(
            row=1, column=0, sticky="w", pady=(5, 5)
        )
        ttk.Entry(card_local, textvariable=self.model.flow_fwhm_var, width=14).grid(
            row=1, column=1, sticky="w", padx=8, pady=(5, 5)
        )

        ttk.Label(card_local, text="Sigma (Início da Busca):").grid(
            row=1, column=2, sticky="w", pady=(5, 5)
        )
        ttk.Entry(card_local, textvariable=self.model.flow_sigma_var, width=14).grid(
            row=1, column=3, sticky="w", padx=8, pady=(5, 5)
        )

        ttk.Checkbutton(
            card_local,
            text="Gerar imagens de diagnóstico nas âncoras (debug)",
            variable=self.model.flow_debug_var,
        ).grid(row=2, column=0, columnspan=4, sticky="w", pady=(10, 8))

        # Botões de Ação - Local Flow
        advanced = ttk.LabelFrame(
            card_local, text="Engines (Avancado)", style="Section.TLabelframe", padding=8
        )
        advanced.grid(row=3, column=0, columnspan=4, sticky="ew", pady=(0, 8))
        ttk.Label(advanced, text="Perfil:").grid(row=0, column=0, sticky="w")
        ttk.Combobox(
            advanced, textvariable=self.model.flow_profile_var, values=["Stable", "Fast"],
            state="readonly", width=12,
        ).grid(row=0, column=1, sticky="w", padx=8)
        ttk.Label(advanced, text="Detector:").grid(row=0, column=2, sticky="w")
        ttk.Combobox(
            advanced, textvariable=self.model.flow_detector_engine_var,
            values=["", "dao", "opencv-contours", "opencv-components", "sep"],
            state="readonly", width=20,
        ).grid(row=0, column=3, sticky="w", padx=8)
        ttk.Label(advanced, text="Fallback:").grid(row=1, column=0, sticky="w", pady=(6, 0))
        ttk.Combobox(
            advanced, textvariable=self.model.flow_transform_fallback_var,
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
            command=self.model.open_anchor_selector,
        )
        self.btn_select_anchor.grid(row=0, column=0, sticky="w", padx=(0, 6), ipady=3)

        self.btn_preview_flow = ttk.Button(
            actions_local,
            text="🔍 Preview detecção",
            command=self.model.show_astroflow_preview,
        )
        self.btn_preview_flow.grid(row=0, column=1, sticky="w", padx=6, ipady=3)

        self.btn_run_flow = ttk.Button(
            actions_local,
            text="▶ Executar AstroFlow (Local + Global)",
            style="Accent.TButton",
            command=self.model.start_flow_processing,
        )
        self.btn_run_flow.grid(row=0, column=2, sticky="e", ipady=3)

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
            textvariable=self.model.flow_global_master_var,
            values=["Auto"],
            state="readonly",
            width=14,
        )
        self.combo_global_master.grid(row=0, column=1, sticky="w", padx=8, pady=(0, 5))

        ttk.Label(card_global, text="Raio de busca (px):").grid(
            row=0, column=2, sticky="w", pady=(0, 5)
        )
        ttk.Entry(
            card_global, textvariable=self.model.flow_matching_radius_var, width=14
        ).grid(row=0, column=3, sticky="w", padx=8, pady=(0, 5))

        ttk.Label(card_global, text="RANSAC threshold:").grid(
            row=1, column=0, sticky="w", pady=(5, 5)
        )
        ttk.Entry(card_global, textvariable=self.model.flow_ransac_var, width=14).grid(
            row=1, column=1, sticky="w", padx=8, pady=(5, 5)
        )

        ttk.Label(card_global, text="Mínimo inliers / Ratio:").grid(
            row=1, column=2, sticky="w", pady=(5, 5)
        )
        ratio_frame = ttk.Frame(card_global)
        ratio_frame.grid(row=1, column=3, sticky="w", padx=8, pady=(5, 5))
        ttk.Entry(
            ratio_frame, textvariable=self.model.flow_min_inliers_var, width=6
        ).pack(side="left")
        ttk.Label(ratio_frame, text=" / ").pack(side="left")
        ttk.Entry(ratio_frame, textvariable=self.model.flow_min_ratio_var, width=6).pack(
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
            command=self.model.show_flow_visualization,
        )
        self.btn_viz_flow.grid(row=0, column=2, sticky="e", padx=6, ipady=3)

        self.btn_cancel_flow = ttk.Button(
            actions_global,
            text="Cancelar",
            style="Danger.TButton",
            command=self.model.cancel_processing,
            state="disabled",
        )
        self.btn_cancel_flow.grid(row=0, column=3, sticky="e", ipady=3)

        # ============================================================
        # CARD 3: análise temporal (metadata-only)
        # ============================================================
        temporal = ttk.LabelFrame(
            self,
            text="3. Sessão temporal e seeing (somente revisão)",
            style="Section.TLabelframe",
            padding=10,
        )
        temporal.grid(row=2, column=0, sticky="ew", pady=(0, 10))
        temporal.columnconfigure(4, weight=1)
        ttk.Checkbutton(
            temporal,
            text="Gerar agrupamento por DATE-OBS",
            variable=self.model.flow_temporal_enabled_var,
        ).grid(row=0, column=0, columnspan=2, sticky="w")
        ttk.Label(temporal, text="Pausa (min):").grid(row=0, column=2, sticky="e", padx=(12, 4))
        ttk.Entry(temporal, textvariable=self.model.flow_temporal_gap_var, width=8).grid(row=0, column=3, sticky="w")
        ttk.Label(temporal, text="Sigma robusto:").grid(row=0, column=4, sticky="e", padx=(12, 4))
        ttk.Entry(temporal, textvariable=self.model.flow_temporal_seeing_sigma_var, width=8).grid(row=0, column=5, sticky="w")
        ttk.Label(
            temporal,
            text="Frames sem horário ficam marcados como desconhecidos; nenhuma exclusão é automática.",
            style="Muted.TLabel",
        ).grid(row=1, column=0, columnspan=5, sticky="w", pady=(6, 0))
        ttk.Button(
            temporal,
            text="📊 Ver análise da sessão",
            command=self.model.show_temporal_analysis,
        ).grid(row=1, column=5, sticky="e", pady=(6, 0))

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
            background=self.model.BG,
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
        if self.model.is_busy and self.model.is_busy():
            return

        base_dir_str = self.model.batch_dir_var.get()
        if not base_dir_str:
            self.model.print_to_console("ERRO: Selecione a Pasta Base primeiro.\n")
            return

        batch_dir = Path(base_dir_str).expanduser().resolve()

        from app.application.commands import FlowCommand

        try:
            config = FlowCommand.from_values(
                batch_dir,
                custom_anchors=dict(self.model.custom_anchors),
                global_master=self.model.flow_global_master_var.get(),
                fwhm=self.model.flow_fwhm_var.get(),
                sigma=self.model.flow_sigma_var.get(),
                matching_radius=self.model.flow_matching_radius_var.get(),
                ransac=self.model.flow_ransac_var.get(),
                debug_images=self.model.flow_debug_var.get(),
                min_stars=self.model.flow_min_stars_var.get(),
                min_inliers=self.model.flow_min_inliers_var.get(),
                min_ratio=self.model.flow_min_ratio_var.get(),
                max_stars=150,
                engine=self.model.flow_engine_var.get(),
                engine_profile=self.model.flow_profile_var.get(),
                detector_engine=self.model.flow_detector_engine_var.get(),
                transform_fallback=self.model.flow_transform_fallback_var.get(),
                temporal_analysis_enabled=self.model.flow_temporal_enabled_var.get(),
                temporal_gap_minutes=self.model.flow_temporal_gap_var.get(),
                temporal_seeing_sigma=self.model.flow_temporal_seeing_sigma_var.get(),
                memory_budget_mb=self.model.resource_memory_var.get(),
                flow_workers=self.model.resource_workers_var.get(),
                skip_local_flow=True,
            )
        except (TypeError, ValueError) as exc:
            self.model.print_to_console(f"[Flow] Parâmetros inválidos: {exc}\n")
            return

        self.model.save_settings()
        self.model._start_operation("Flow", config)

    def _load_reference_thumbnail(self, filepath):
        """
            Carrega uma imagem FIT/TIFF e cria um thumbnail para o Grid.
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

        base_dir_str = self.model.batch_dir_var.get()
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
            frame_name = self.model.custom_anchors.get(batch_dir.name)

            if not frame_name:
                continue

            frame_path = batch_dir / frame_name
            try:
                from astroalign_logic import load_local_flow

                local_flow = load_local_flow(batch_dir)
            except Exception as exc:
                local_flow = None
                self.model.print_to_console(
                    f"[Flow] Não foi possível carregar a proveniência de {batch_dir.name}: {exc}\n"
                )
            provenance = format_reference_provenance(local_flow, frame_name)

            if not frame_path.is_file():
                references.append((batch_dir.name, frame_name, None, provenance))
                continue

            references.append((batch_dir.name, frame_name, frame_path, provenance))

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

        for index, (batch_name, frame_name, frame_path, provenance) in enumerate(references):
            row = index // 4
            column = index % 4

            card = ttk.Frame(
                inner,
                padding=6,
                relief="solid",
            )

            card.grid(row=row, column=column, sticky="nsew", padx=5, pady=5)
            cards.append((card, batch_name, frame_name, frame_path, provenance))

        self._preview_cards.clear()
        requests = []
        for index, (card, batch_name, frame_name, frame_path, provenance) in enumerate(cards):
            key = (generation, index)
            self._preview_cards[key] = (card, batch_name, frame_name, provenance)
            if frame_path is None:
                self._populate_reference_card_error(
                    card, batch_name, frame_name,
                    f"{provenance}\nArquivo não encontrado",
                )
            else:
                requests.append((key, frame_path))
        self._preview_service.replace(generation, requests)
        self._schedule_preview_poll(generation)

    def _schedule_preview_poll(self, generation):
        if generation != self.reference_preview_generation:
            return
        if self._preview_after_id:
            try:
                self.after_cancel(self._preview_after_id)
            except tk.TclError:
                pass
        self._preview_after_id = self.after(40, lambda: self._poll_preview_results(generation))

    def _poll_preview_results(self, generation):
        self._preview_after_id = None
        if generation != self.reference_preview_generation:
            return
        for result in self._preview_service.drain():
            if result.generation != generation:
                continue
            card_info = self._preview_cards.get(result.key)
            if not card_info:
                continue
            card, batch_name, frame_name, provenance = card_info
            if result.error:
                self._populate_reference_card_error(card, batch_name, frame_name, result.error)
            else:
                self._populate_reference_card(
                    card, batch_name, frame_name, result.image, generation, provenance
                )
        self._schedule_preview_poll(generation)

    def _populate_reference_card(
        self, card, batch_name, frame_name, img_array, generation, provenance
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
        ttk.Label(
            card, text=provenance, style="Muted.TLabel", anchor="center",
            wraplength=210, justify="center",
        ).pack(fill=tk.X, pady=(2, 2))

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
            self.reference_preview_canvas.configure(background=self.model.BG)
