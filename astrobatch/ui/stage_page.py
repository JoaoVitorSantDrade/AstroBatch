from __future__ import annotations

import json
from pathlib import Path

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (QFormLayout, QFrame, QHBoxLayout, QLabel, QLineEdit,
                               QPlainTextEdit, QPushButton, QToolButton, QVBoxLayout, QWidget)

from astrobatch.core.models import Artifact, Stage
from astrobatch.services.pipeline import STAGE_LABELS


class StagePage(QWidget):
    run_requested = Signal(object)
    browse_requested = Signal(object, str)
    preview_requested = Signal(str)

    def __init__(self, stage: Stage, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.stage = stage
        self._artifacts: list[Artifact] = []
        self.title = QLabel(STAGE_LABELS[stage])
        self.title.setObjectName("stageTitle")
        self.status = QLabel("Ready")
        self.status.setObjectName("statusText")
        self.input_edit = QLineEdit()
        self.output_edit = QLineEdit()
        self.advanced = QPlainTextEdit("{}")
        self.advanced.setPlaceholderText('Advanced settings as JSON, for example: {"overwrite": true}')
        self.advanced.setMaximumBlockCount(100)
        self.advanced.setFixedHeight(115)
        self.advanced_toggle = QToolButton(text="Advanced settings")
        self.advanced_toggle.setCheckable(True)
        self.advanced_toggle.setArrowType(self.advanced_toggle.arrowType())
        self.advanced.setVisible(False)
        self.advanced_toggle.toggled.connect(self.advanced.setVisible)
        self.run_button = QPushButton(f"Run {STAGE_LABELS[stage]}")
        self.run_button.clicked.connect(lambda: self.run_requested.emit(self.stage))
        self.artifact_area = QVBoxLayout()
        self.artifact_area.setContentsMargins(0, 0, 0, 0)
        form = QFormLayout()
        form.addRow("Input", self._path_row(self.input_edit, "input"))
        form.addRow("Output", self._path_row(self.output_edit, "output"))
        intro = QLabel(self._description())
        intro.setWordWrap(True)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(14)
        layout.addWidget(self.title)
        layout.addWidget(intro)
        layout.addWidget(self.status)
        layout.addLayout(form)
        layout.addWidget(self.advanced_toggle)
        layout.addWidget(self.advanced)
        layout.addWidget(self.run_button)
        layout.addWidget(QLabel("Published artifacts"))
        layout.addLayout(self.artifact_area)
        layout.addStretch()

    def _path_row(self, editor: QLineEdit, field: str) -> QWidget:
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(editor)
        browse = QPushButton("Browse…")
        browse.clicked.connect(lambda: self.browse_requested.emit(self.stage, field))
        layout.addWidget(browse)
        return container

    def _description(self) -> str:
        return {Stage.IMPORT: "Select the folder containing the raw FITS source frames.", Stage.CALIBRATE: "Apply dark and flat calibration and publish normalized 16-bit FITS frames.", Stage.BATCH: "Group calibrated frames into analysis batches.", Stage.FLOW: "Detect anchors and calculate local/global frame flow.", Stage.ALIGN: "Apply the calculated transforms and publish aligned FITS frames and masks.", Stage.STACK: "Preflight memory, select frames, reject outliers, and produce a 16-bit FITS stack.", Stage.REVIEW: "Inspect the final published stack and its FITS metadata."}[self.stage]

    def set_project_data(self, source_dir: str, settings: dict, artifact: Artifact | None) -> None:
        self.input_edit.setText(str(settings.get("input_dir", source_dir)))
        self.output_edit.setText(str(settings.get("output_dir", "")))
        advanced = {key: value for key, value in settings.items() if key not in {"input_dir", "output_dir"}}
        self.advanced.setPlainText(json.dumps(advanced, indent=2, ensure_ascii=False))
        self.set_artifacts([artifact] if artifact else [])

    def collect_settings(self) -> dict:
        try:
            advanced = json.loads(self.advanced.toPlainText() or "{}")
        except json.JSONDecodeError as exc:
            raise ValueError(f"Advanced settings must be valid JSON: {exc.msg}") from exc
        if not isinstance(advanced, dict):
            raise ValueError("Advanced settings must be a JSON object.")
        if self.input_edit.text().strip():
            advanced["input_dir"] = self.input_edit.text().strip()
        if self.output_edit.text().strip():
            advanced["output_dir"] = self.output_edit.text().strip()
        return advanced

    def set_running(self, running: bool) -> None:
        self.run_button.setDisabled(running)

    def set_status(self, text: str) -> None:
        self.status.setText(text)

    def set_artifacts(self, artifacts: list[Artifact]) -> None:
        while self.artifact_area.count():
            item = self.artifact_area.takeAt(0)
            if item.widget(): item.widget().deleteLater()
        for artifact in artifacts:
            card = QFrame()
            card.setFrameShape(QFrame.StyledPanel)
            row = QHBoxLayout(card)
            row.addWidget(QLabel(f"{artifact.name}: {artifact.path}"))
            if artifact.kind == "fits":
                preview = QPushButton("Preview")
                preview.clicked.connect(lambda _, path=artifact.path: self.preview_requested.emit(path))
                row.addWidget(preview)
            self.artifact_area.addWidget(card)
