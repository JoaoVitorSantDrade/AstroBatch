from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QCheckBox, QComboBox, QDoubleSpinBox, QFormLayout, QFrame, QGridLayout, QHBoxLayout, QLabel, QLineEdit, QPlainTextEdit, QPushButton, QScrollArea, QSpinBox, QToolButton, QVBoxLayout, QWidget

from astrobatch.core.models import Artifact, Stage
from astrobatch.services.pipeline import STAGE_LABELS


@dataclass(frozen=True)
class Field:
    section: str
    key: str
    label: str
    kind: str
    default: Any
    options: tuple[str, ...] = ()
    help_text: str = ""


INPUT = Field("Data locations", "input_dir", "Input folder", "path", "", help_text="Leave blank to use the preceding stage output.")
OUTPUT = Field("Data locations", "output_dir", "Output folder", "path", "", help_text="Leave blank to use this project's managed output folder.")
SCHEMAS: dict[Stage, tuple[Field, ...]] = {
    Stage.IMPORT: (Field("Source frames", "input_dir", "Source FITS folder", "path", "", help_text="The original LIGHT frames. Import validates this folder only."),),
    Stage.CALIBRATE: (INPUT, OUTPUT, Field("Calibration masters", "apply_dark", "Apply Dark", "bool", False), Field("Calibration masters", "dark_path", "Dark frames or Master FITS", "path", ""), Field("Calibration masters", "apply_flat", "Apply Flat", "bool", False), Field("Calibration masters", "flat_path", "Flat frames or Master FITS", "path", ""), Field("Output behavior", "create_master", "Create masters from folders", "bool", True), Field("Output behavior", "overwrite", "Overwrite existing output", "bool", False)),
    Stage.BATCH: (INPUT, OUTPUT, Field("Analysis", "opt_method", "Optimization method", "choice", "Crop", ("Crop", "Downsampling")), Field("Analysis", "crop_size", "Central crop size (px)", "int", 1000), Field("Analysis", "downsample_method", "Downsampling method", "choice", "Nearest", ("Nearest", "Bilinear", "Lanczos")), Field("Analysis", "downsample_scale", "Downsampling scale", "float", 0.25), Field("Analysis", "threshold_factor", "Batch threshold factor", "float", 3.0), Field("File handling", "copy_files", "Copy instead of move", "bool", True), Field("File handling", "overwrite", "Overwrite existing batches", "bool", False), Field("File handling", "dry_run", "Dry run (do not write files)", "bool", False)),
    Stage.FLOW: (INPUT, Field("Star detection", "engine", "Detection engine", "choice", "DAO", ("DAO", "OpenCV")), Field("Star detection", "fwhm", "Expected FWHM", "float", 4.0), Field("Star detection", "sigma", "Detection sigma", "float", 5.0), Field("Matching", "global_master", "Global master", "choice", "Auto", ("Auto",)), Field("Matching", "matching_radius", "Matching radius (px)", "int", 15), Field("Matching", "ransac", "RANSAC tolerance", "float", 3.0), Field("Matching", "min_stars", "Minimum stars", "int", 4), Field("Matching", "min_inliers", "Minimum inliers", "int", 4), Field("Matching", "min_ratio", "Minimum inlier ratio", "float", 0.15), Field("Diagnostics", "debug_images", "Write debug images", "bool", False)),
    Stage.ALIGN: (INPUT, OUTPUT, Field("Debayer", "debayer_pattern", "Bayer pattern", "choice", "Auto", ("Auto", "RGGB", "BGGR", "GRBG", "GBRG", "Nenhum")), Field("Debayer", "debayer_method", "Debayer method", "choice", "Bilinear", ("Bilinear", "VNG", "Menon2007")), Field("Registration", "interpolation", "Geometric interpolation", "choice", "lanczos", ("nearest", "bilinear", "bicubic", "lanczos")), Field("Registration", "rgb_registration", "Advanced RGB registration", "bool", True), Field("Output behavior", "keep_header", "Preserve FITS headers", "bool", True), Field("Output behavior", "compress_output", "Compress FITS output (RICE_1)", "bool", True), Field("Output behavior", "overwrite", "Overwrite existing output", "bool", False), Field("Output behavior", "delete_intermediates", "Delete intermediate batches", "bool", False), Field("Output behavior", "dry_run", "Dry run (do not write files)", "bool", False)),
    Stage.STACK: (INPUT, OUTPUT, Field("Frame selection", "selection_mode", "Selection mode", "choice", "BestPercentage", ("All", "BestPercentage")), Field("Frame selection", "selection_percentage", "Best frames (%)", "float", 80.0), Field("Frame selection", "selection_metric", "Ranking metric", "choice", "quality", ("quality", "fwhm", "star_count", "snr")), Field("Combination", "method", "Combine method", "choice", "Median", ("Median", "Mean", "Sum", "Maximum", "Minimum")), Field("Rejection", "rejection_method", "Outlier rejection", "choice", "SigmaClip", ("None", "SigmaClip", "Winsorized", "MAD")), Field("Rejection", "rejection_low", "Low threshold", "float", 3.0), Field("Rejection", "rejection_high", "High threshold", "float", 3.0), Field("Normalization", "normalize", "Normalize frames", "bool", True), Field("Normalization", "normalize_method", "Normalization method", "choice", "Median", ("Median", "Mode")), Field("Normalization", "apply_dither_correction", "Apply dither correction", "bool", False), Field("Output", "output_name", "Output filename", "text", "stacked_image.fits"), Field("Output", "output_bit_depth", "Output bit depth", "choice", "16-bit", ("16-bit",)), Field("Output", "compress_output", "Compress FITS output", "bool", True), Field("Performance", "memory_budget_mb", "Memory budget (MiB)", "int", 4096), Field("Performance", "workers", "Worker threads (0 = automatic)", "int", 0)),
    Stage.REVIEW: (),
}
STAGE_COPY = {Stage.IMPORT: ("Bring in your FITS session", "Validate your original lights and establish the project source."), Stage.CALIBRATE: ("Correct the original lights", "Choose calibration inputs and output behavior without leaving the workflow."), Stage.BATCH: ("Build analysis batches", "Tune grouping behavior, frame optimization, and file handling."), Stage.FLOW: ("Map stars and transforms", "Control star detection and matching quality before alignment."), Stage.ALIGN: ("Publish aligned frames", "Control debayering, interpolation, chromatic registration, and FITS output."), Stage.STACK: ("Build the final image", "Control selection, combine/rejection strategy, output, and resources."), Stage.REVIEW: ("Inspect the final result", "Open the final FITS stack to inspect its auto-stretched data and metadata.")}


class StagePage(QWidget):
    run_requested = Signal(object)
    browse_requested = Signal(object, str)
    preview_requested = Signal(str)

    def __init__(self, stage: Stage, parent=None):
        super().__init__(parent)
        self.stage, self.controls = stage, {}
        self.status = QLabel("Choose or create a project first")
        self.status.setObjectName("stageStatus")
        self.advanced = QPlainTextEdit("{}")
        self.advanced.setPlaceholderText('Optional extra settings, e.g. {"custom_anchors": {...}}')
        self.advanced.setFixedHeight(120)
        self.advanced.setVisible(False)
        self.artifacts = QVBoxLayout()
        self._build()

    def _card(self, heading, copy=""):
        card = QFrame(); card.setObjectName("card")
        layout = QVBoxLayout(card); layout.setContentsMargins(22, 20, 22, 20); layout.setSpacing(9)
        title = QLabel(heading); title.setObjectName("cardTitle"); layout.addWidget(title)
        if copy:
            label = QLabel(copy); label.setObjectName("muted"); label.setWordWrap(True); layout.addWidget(label)
        return card, layout

    def _field_control(self, field):
        if field.kind == "bool": control = QCheckBox()
        elif field.kind == "choice": control = QComboBox(); control.addItems(field.options)
        elif field.kind == "int": control = QSpinBox(); control.setRange(0, 1_000_000)
        elif field.kind == "float": control = QDoubleSpinBox(); control.setRange(0.0, 1_000_000.0); control.setDecimals(3); control.setSingleStep(0.1)
        else: control = QLineEdit()
        self.controls[field.key] = control
        if field.kind != "path": return control
        wrapper = QWidget(); row = QHBoxLayout(wrapper); row.setContentsMargins(0, 0, 0, 0)
        control.setPlaceholderText(field.help_text or "Leave blank for the default")
        browse = QPushButton("Browse"); browse.clicked.connect(lambda: self.browse_requested.emit(self.stage, field.key))
        row.addWidget(control, 1); row.addWidget(browse)
        return wrapper

    def _settings(self, layout):
        groups = {}
        for field in SCHEMAS[self.stage]: groups.setdefault(field.section, []).append(field)
        grid = QGridLayout()
        grid.setHorizontalSpacing(16)
        grid.setVerticalSpacing(16)
        grid.setColumnStretch(0, 1)
        grid.setColumnStretch(1, 1)
        for index, (section, fields) in enumerate(groups.items()):
            card, card_layout = self._card(section)
            form = QFormLayout(); form.setLabelAlignment(Qt.AlignLeft); form.setVerticalSpacing(11)
            for field in fields: form.addRow(QLabel(field.label), self._field_control(field))
            card_layout.addLayout(form)
            grid.addWidget(card, index // 2, index % 2)
        layout.addLayout(grid)

    def _build(self):
        title, copy = STAGE_COPY[self.stage]
        scroll = QScrollArea(); scroll.setWidgetResizable(True); scroll.setFrameShape(QFrame.NoFrame)
        content = QWidget(); layout = QVBoxLayout(content); layout.setContentsMargins(42, 36, 42, 42); layout.setSpacing(16)
        eyebrow = QLabel(f"PIPELINE  /  {STAGE_LABELS[self.stage].upper()}"); eyebrow.setObjectName("eyebrow")
        headline = QLabel(title); headline.setObjectName("pageTitle")
        intro = QLabel(copy); intro.setObjectName("pageIntro"); intro.setWordWrap(True)
        layout.addWidget(eyebrow); layout.addWidget(headline); layout.addWidget(intro)
        status_card, status_layout = self._card("Stage status"); status_layout.addWidget(self.status); layout.addWidget(status_card)
        run_card, run_layout = self._card("Ready when you are", "AstroBatch runs one heavy task at a time and keeps the interface responsive.")
        self.run_button = QPushButton(f"Run {STAGE_LABELS[self.stage]}"); self.run_button.setObjectName("primaryButton")
        self.run_button.clicked.connect(lambda: self.run_requested.emit(self.stage)); run_layout.addWidget(self.run_button); layout.addWidget(run_card)
        self._settings(layout)
        advanced_card, advanced_layout = self._card("Extra settings", "Only niche values not represented above belong here.")
        toggle = QToolButton(text="Show extra settings"); toggle.setCheckable(True); toggle.toggled.connect(self.advanced.setVisible); toggle.toggled.connect(lambda opened: toggle.setText("Hide extra settings" if opened else "Show extra settings"))
        advanced_layout.addWidget(toggle); advanced_layout.addWidget(self.advanced); layout.addWidget(advanced_card)
        artifact_card, artifact_layout = self._card("Published artifacts"); artifact_layout.addLayout(self.artifacts); layout.addWidget(artifact_card); layout.addStretch()
        scroll.setWidget(content); root = QVBoxLayout(self); root.setContentsMargins(0, 0, 0, 0); root.addWidget(scroll)

    def _read(self, field):
        control = self.controls[field.key]
        if field.kind == "bool": return control.isChecked()
        if field.kind == "choice": return control.currentText()
        if field.kind == "int": return control.value()
        if field.kind == "float": return control.value()
        return control.text().strip()

    def _write(self, field, value):
        control = self.controls[field.key]
        if field.kind == "bool": control.setChecked(bool(value))
        elif field.kind == "choice": control.setCurrentText(str(value))
        elif field.kind == "int": control.setValue(int(value or 0))
        elif field.kind == "float": control.setValue(float(value or 0.0))
        else: control.setText(str(value or ""))

    def set_project_data(self, source_dir, settings, artifact):
        for field in SCHEMAS[self.stage]:
            default = source_dir if self.stage is Stage.IMPORT and field.key == "input_dir" else field.default
            self._write(field, settings.get(field.key, default))
        known = {field.key for field in SCHEMAS[self.stage]}
        self.advanced.setPlainText(json.dumps({key: value for key, value in settings.items() if key not in known}, indent=2, ensure_ascii=False))
        self.set_status("Ready to run" if source_dir or artifact else "Choose a source folder to continue"); self.set_artifacts([artifact] if artifact else [])

    def collect_settings(self):
        try: settings = json.loads(self.advanced.toPlainText() or "{}")
        except json.JSONDecodeError as exc: raise ValueError(f"Extra settings must be valid JSON: {exc.msg}") from exc
        if not isinstance(settings, dict): raise ValueError("Extra settings must be a JSON object.")
        for field in SCHEMAS[self.stage]:
            value = self._read(field)
            settings[field.key] = None if field.kind == "int" and field.key == "workers" and value == 0 else value
        return settings

    def set_running(self, running): self.run_button.setDisabled(running)
    def set_status(self, text): self.status.setText(text)

    def set_artifacts(self, artifacts):
        while self.artifacts.count():
            item = self.artifacts.takeAt(0)
            if item.widget(): item.widget().deleteLater()
        if not artifacts:
            empty = QLabel("Nothing published yet."); empty.setObjectName("muted"); self.artifacts.addWidget(empty); return
        for artifact in artifacts:
            card = QFrame(); card.setObjectName("artifactCard"); row = QHBoxLayout(card)
            label = QLabel(f"{artifact.name}\n{artifact.path}"); label.setWordWrap(True); row.addWidget(label, 1)
            if artifact.kind == "fits":
                preview = QPushButton("Preview FITS"); preview.clicked.connect(lambda _, path=artifact.path: self.preview_requested.emit(path)); row.addWidget(preview)
            self.artifacts.addWidget(card)
