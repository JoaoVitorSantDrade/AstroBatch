from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QLabel, QPlainTextEdit, QSplitter, QVBoxLayout, QWidget
from astropy.io import fits


class FitsViewer(QWidget):
    """Read-only FITS inspection with percentile auto-stretch and scalable preview."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.image = QLabel("Select a FITS artifact to preview.")
        self.image.setAlignment(Qt.AlignCenter)
        self.image.setMinimumSize(360, 260)
        self.image.setStyleSheet("background: #101827; color: #dbeafe; border-radius: 6px;")
        self.metadata = QPlainTextEdit()
        self.metadata.setReadOnly(True)
        splitter = QSplitter(Qt.Vertical)
        splitter.addWidget(self.image)
        splitter.addWidget(self.metadata)
        splitter.setSizes([420, 180])
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(splitter)

    def load_path(self, path: str | Path) -> None:
        image_path = Path(path)
        try:
            with fits.open(image_path, memmap=False, ignore_missing_end=True) as hdul:
                hdu = next(hdu for hdu in hdul if hdu.is_image and hdu.data is not None)
                data = np.asarray(hdu.data, dtype=np.float32)
                header = hdu.header
            if data.ndim == 3:
                data = data[0] if data.shape[0] in (3, 4) else data[..., 0]
            finite = data[np.isfinite(data)]
            if finite.size == 0:
                raise ValueError("Image has no finite pixels")
            low, high = np.percentile(finite, (1, 99.5))
            scaled = np.clip((data - low) / max(high - low, 1e-12) * 255, 0, 255).astype(np.uint8)
            qimage = QImage(scaled.data, scaled.shape[1], scaled.shape[0], scaled.strides[0], QImage.Format_Grayscale8).copy()
            pixmap = QPixmap.fromImage(qimage).scaled(self.image.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.image.setPixmap(pixmap)
            fields = [f"{image_path.name}", f"Shape: {tuple(data.shape)}", f"Auto-stretch: p1={low:.5g}, p99.5={high:.5g}", "", "FITS metadata:"]
            fields.extend(f"{card.keyword} = {card.value}" for card in header.cards[:80])
            self.metadata.setPlainText("\n".join(fields))
        except Exception as exc:
            self.image.setText(f"Could not preview FITS:\n{exc}")
            self.metadata.clear()
