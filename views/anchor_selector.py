"""Presentation controller for Flow reference selection.

The controller owns dialog state, preview generations and pending/applied
reference transitions.  It communicates with the application through small
callbacks, so the dialog never reaches into the root controller or performs a
processing operation itself.
"""

from functools import lru_cache
from pathlib import Path
from typing import Callable, MutableMapping

import tkinter as tk
from tkinter import messagebox, ttk

from views.preview_service import PreviewService


class AnchorSelectionController:
    """Own the reference selector window and its asynchronous preview state."""

    FITS_SUFFIXES = {".fit", ".fits", ".fts"}

    def __init__(
        self,
        parent,
        batch_dir_var,
        custom_anchors: MutableMapping[str, str],
        on_apply_reference: Callable[[Path, str, str], bool],
        on_commit_reference: Callable[[str, str], None],
        on_log: Callable[[str], None] | None = None,
        preview_service_factory=PreviewService,
    ):
        self.parent = parent
        self.batch_dir_var = batch_dir_var
        self.custom_anchors = custom_anchors
        self.on_apply_reference = on_apply_reference
        self.on_commit_reference = on_commit_reference
        self.on_log = on_log or (lambda _message: None)
        self.preview_service_factory = preview_service_factory
        self.pending_reference_changes: dict[str, str] = {}
        self._active_reference: tuple[str, str] | None = None
        self._window = None
        self._close_window = None

    @property
    def active_reference(self) -> tuple[str, str] | None:
        return self._active_reference

    def set_pending_reference(self, batch_name: str, frame_name: str) -> None:
        batch = str(batch_name).strip()
        frame = str(frame_name).strip()
        if batch and frame:
            self.pending_reference_changes[batch] = frame

    def request_apply(self, base_dir: Path, batch_name: str, frame_name: str) -> bool:
        """Request an asynchronous rebase through the application callback.

        ``False`` means that no operation was started.  This is used for a
        batch without an existing Flow revision, where the selection is simply
        saved for the next Flow run, and for rejected/invalid requests.
        """

        batch = str(batch_name).strip()
        frame = str(frame_name).strip()
        if not batch or not frame or self._active_reference is not None:
            return False
        self.set_pending_reference(batch, frame)
        self._active_reference = (batch, frame)
        try:
            started = bool(self.on_apply_reference(Path(base_dir), batch, frame))
        except Exception as exc:
            self.on_log(f"[AstroFlow] Rebase da referência falhou: {exc}\n")
            started = False
        if not started:
            self.pending_reference_changes.pop(batch, None)
            self._active_reference = None
        return started

    def finish_reference_change(self, outcome: str) -> None:
        """Commit or discard the active selection after runner completion."""

        active = self._active_reference
        if active is None:
            return
        batch, frame = active
        if outcome == "success":
            self.on_commit_reference(batch, frame)
        # A user may have selected a different frame while the rebase was
        # running.  Do not erase that newer pending choice when the older
        # operation completes.
        if self.pending_reference_changes.get(batch) == frame:
            self.pending_reference_changes.pop(batch, None)
        self._active_reference = None

    def close(self) -> None:
        """Close the active dialog and invalidate its preview callbacks."""

        if self._close_window is not None:
            self._close_window()
        self._window = None
        self._close_window = None

    def open(self, target_batch: str | None = None):
        """Open the selector, returning its Toplevel or ``None`` on failure."""

        import cv2
        import numpy as np
        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        from matplotlib.figure import Figure

        from astroflow_logic import extract_luminance, load_fits_data

        base_dir_str = str(self.batch_dir_var.get() or "").strip()
        if not base_dir_str:
            messagebox.showerror(
                "Erro", "Selecione a Pasta Base primeiro.", parent=self.parent
            )
            return None
        base_dir = Path(base_dir_str).expanduser().resolve()
        if not base_dir.is_dir():
            messagebox.showerror("Erro", "A Pasta Base não existe.", parent=self.parent)
            return None

        try:
            batch_folders = sorted(
                [
                    d for d in base_dir.iterdir()
                    if d.is_dir() and "batch" in d.name.lower()
                ],
                key=lambda path: path.name.lower(),
            )
        except OSError as exc:
            messagebox.showerror(
                "Erro", f"Não foi possível acessar as batches: {exc}", parent=self.parent
            )
            return None
        if not batch_folders:
            messagebox.showerror(
                "Erro", "Nenhuma pasta de Batch encontrada.", parent=self.parent
            )
            return None

        self.close()
        win = tk.Toplevel(self.parent)
        self._window = win
        win.title("AstroFlow — Selecionar referência")
        win.geometry("1000x800")
        win.minsize(750, 600)

        closed = {"value": False}
        poll_id = {"value": None}
        generation = {"id": 0}
        selection = {"batch": "", "frame": ""}

        top = ttk.Frame(win, padding=12)
        top.pack(fill=tk.X)
        ttk.Label(top, text="Batch:").pack(side=tk.LEFT)
        batch_combo = ttk.Combobox(
            top, values=[batch.name for batch in batch_folders],
            state="readonly", width=18,
        )
        batch_combo.pack(side=tk.LEFT, padx=(7, 15))
        ttk.Label(top, text="Frame:").pack(side=tk.LEFT)
        frame_combo = ttk.Combobox(top, state="readonly", width=34)
        frame_combo.pack(side=tk.LEFT, padx=7)

        image_frame = ttk.Frame(win, padding=(12, 0))
        image_frame.pack(fill=tk.BOTH, expand=True)
        fig = Figure(figsize=(8, 6), dpi=85)
        ax = fig.add_subplot(111)
        ax.axis("off")
        ax_img = ax.imshow(
            np.zeros((10, 10)), cmap="gray", interpolation="nearest",
            rasterized=True, vmin=0, vmax=255,
        )
        title_obj = ax.set_title("Stretched Preview", fontsize=10)
        canvas = FigureCanvasTkAgg(fig, master=image_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        @lru_cache(maxsize=40)
        def get_preview_data(filepath):
            data, header = load_fits_data(filepath)
            luminance = extract_luminance(data, header)
            height, width = luminance.shape[:2]
            small_w, small_h = max(width // 3, 1), max(height // 3, 1)
            return cv2.resize(
                luminance, (small_w, small_h), interpolation=cv2.INTER_NEAREST
            )

        def load_dialog_preview(filepath):
            small_data = get_preview_data(filepath)
            median = np.median(small_data)
            p25, p75 = np.percentile(small_data, [25, 75])
            spread = max((p75 - p25) / 1.35, 1e-5)
            vmin, vmax = median - 0.5 * spread, median + 6.0 * spread
            return (
                np.clip(
                    (small_data - vmin) / max(vmax - vmin, 1e-5), 0, 1
                ) * 255
            ).astype(np.uint8)

        preview_service = self.preview_service_factory(
            load_dialog_preview, max_workers=2, max_pending=4
        )

        def close_selector():
            if closed["value"]:
                return
            closed["value"] = True
            preview_service.close()
            if poll_id["value"] is not None:
                try:
                    win.after_cancel(poll_id["value"])
                except tk.TclError:
                    pass
                poll_id["value"] = None
            if win.winfo_exists():
                win.destroy()
            if self._window is win:
                self._window = None
                self._close_window = None

        self._close_window = close_selector
        win.protocol("WM_DELETE_WINDOW", close_selector)

        def poll_preview(current_id):
            poll_id["value"] = None
            if closed["value"] or current_id != generation["id"]:
                return
            for result in preview_service.drain():
                if result.generation != current_id:
                    continue
                if result.error:
                    self.on_log(f"[Preview] Erro: {result.error}\n")
                else:
                    image = result.image
                    ax_img.set_data(image)
                    ax_img.set_extent((0, image.shape[1], image.shape[0], 0))
                    title_obj.set_text(
                        f"Stretched Preview — {selection['batch']} / {selection['frame']}"
                    )
                    canvas.draw_idle()
                batch_combo.configure(state="readonly")
                frame_combo.configure(state="readonly")
            if not closed["value"]:
                poll_id["value"] = win.after(40, lambda: poll_preview(current_id))

        def update_image(_event=None):
            batch_name, frame_name = batch_combo.get(), frame_combo.get()
            if not batch_name or not frame_name:
                return
            generation["id"] += 1
            current_id = generation["id"]
            selection["batch"], selection["frame"] = batch_name, frame_name
            batch_combo.configure(state="disabled")
            frame_combo.configure(state="disabled")
            frame_path = base_dir / batch_name / frame_name
            preview_service.replace(current_id, [((batch_name, frame_name), frame_path)])
            poll_preview(current_id)

        def update_frames(_event=None):
            batch_name = batch_combo.get()
            if not batch_name:
                return
            batch_path = base_dir / batch_name
            try:
                fits_files = sorted(
                    [
                        file.name for file in batch_path.iterdir()
                        if file.is_file() and file.suffix.lower() in self.FITS_SUFFIXES
                    ]
                )
            except OSError as exc:
                frame_combo.configure(values=[])
                self.on_log(f"[Preview] Não foi possível listar {batch_name}: {exc}\n")
                return
            frame_combo.configure(values=fits_files)
            if not fits_files:
                frame_combo.set("")
                return
            selected = self.pending_reference_changes.get(
                batch_name, self.custom_anchors.get(batch_name, fits_files[len(fits_files) // 2])
            )
            if selected not in fits_files:
                selected = fits_files[0]
            frame_combo.set(selected)
            update_image()

        def save_selection():
            batch_name, frame_name = batch_combo.get(), frame_combo.get()
            if not batch_name or not frame_name:
                return
            self.set_pending_reference(batch_name, frame_name)
            self.on_log(
                f"[AstroFlow] Referência pendente para {batch_name}: {frame_name}. "
                "Clique em Aplicar referência para confirmar.\n"
            )
            messagebox.showinfo(
                "Referência selecionada",
                f"Frame definido como referência da {batch_name}:\n\n{frame_name}\n\n"
                "A alteração ainda não foi aplicada.",
                parent=win,
            )

        def apply_selection():
            batch_name, frame_name = batch_combo.get(), frame_combo.get()
            if not batch_name or not frame_name:
                return
            self.request_apply(base_dir, batch_name, frame_name)

        batch_combo.bind("<<ComboboxSelected>>", update_frames)
        frame_combo.bind("<<ComboboxSelected>>", update_image)

        if target_batch and target_batch in [batch.name for batch in batch_folders]:
            batch_combo.set(target_batch)
        else:
            batch_combo.set(batch_folders[0].name)
        update_frames()

        buttons = ttk.Frame(win, padding=12)
        buttons.pack(fill=tk.X)
        ttk.Button(
            buttons, text="✓  Definir como referência",
            style="Accent.TButton", command=save_selection,
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5), ipady=5)
        ttk.Button(
            buttons, text="Aplicar referência", command=apply_selection,
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5, ipady=5)
        ttk.Button(
            buttons, text="Fechar", command=close_selector,
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(5, 0), ipady=5)
        return win
