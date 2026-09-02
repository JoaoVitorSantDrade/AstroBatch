from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QApplication, QDockWidget, QFileDialog, QInputDialog, QLabel,
                               QMainWindow, QMessageBox, QProgressBar, QStackedWidget,
                               QTextEdit, QToolBar, QWidget)

from astrobatch.core.jobs import JobManager
from astrobatch.core.models import Artifact, JobEvent, JobEventKind, Stage
from astrobatch.project.store import ProjectFormatError, ProjectStore
from astrobatch.project.workspace import Project
from astrobatch.services.pipeline import PipelineService, STAGE_LABELS
from .fits_viewer import FitsViewer
from .jobs import QtJobBridge
from .stage_page import StagePage


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("AstroBatch V2")
        self.resize(1280, 840)
        self.store, self.project = ProjectStore(), None
        self.manager, self.bridge = JobManager(), QtJobBridge()
        self.manager.add_listener(self.bridge.emit_event)
        self.bridge.event_received.connect(self._on_event)
        self.pages = {stage: StagePage(stage) for stage in Stage}
        self.stack = QStackedWidget()
        for page in self.pages.values():
            self.stack.addWidget(page)
            page.run_requested.connect(self.run_stage)
            page.browse_requested.connect(self.browse_path)
            page.preview_requested.connect(self.preview_fits)
        self.setCentralWidget(self.stack)
        self._build_chrome()
        self._apply_style()
        self.show_stage(Stage.IMPORT)

    def _build_chrome(self) -> None:
        toolbar = QToolBar("Project")
        toolbar.setMovable(False)
        toolbar.addAction("New project", self.new_project)
        toolbar.addAction("Open project", self.open_project)
        toolbar.addSeparator()
        for stage in Stage:
            toolbar.addAction(STAGE_LABELS[stage], lambda checked=False, item=stage: self.show_stage(item))
        toolbar.addSeparator()
        self.cancel_action = toolbar.addAction("Cancel job", self.manager.cancel)
        self.cancel_action.setEnabled(False)
        self.progress = QProgressBar()
        self.progress.setFixedWidth(180)
        toolbar.addWidget(self.progress)
        self.addToolBar(toolbar)
        self.log = QTextEdit(readOnly=True)
        dock = QDockWidget("Processing log", self)
        dock.setWidget(self.log)
        self.addDockWidget(Qt.BottomDockWidgetArea, dock)
        self.viewer = FitsViewer()
        viewer_dock = QDockWidget("FITS viewer", self)
        viewer_dock.setWidget(self.viewer)
        self.addDockWidget(Qt.RightDockWidgetArea, viewer_dock)

    def _apply_style(self) -> None:
        self.setStyleSheet("QMainWindow { background: #f8fafc; } #stageTitle { font-size: 26px; font-weight: 700; color: #172554; } #statusText { color: #1d4ed8; font-weight: 600; } QPushButton { padding: 7px 12px; } QLineEdit, QPlainTextEdit { padding: 6px; }")

    def new_project(self) -> None:
        root = QFileDialog.getExistingDirectory(self, "Choose a new project workspace")
        if not root: return
        name, accepted = QInputDialog.getText(self, "Project name", "Name", text=Path(root).name)
        if not accepted: return
        source = QFileDialog.getExistingDirectory(self, "Choose source FITS folder (optional)")
        self.project = self.store.create(Path(root), name, Path(source) if source else None)
        self.log.append(f"Created project: {self.project.workspace.root}")
        self.refresh_pages()

    def open_project(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open AstroBatch V2 project", filter="AstroBatch project (astrobatch.project.json)")
        if not path: return
        try:
            self.project = self.store.load(Path(path))
            self.log.append(f"Opened project: {self.project.name}")
            self.refresh_pages()
        except ProjectFormatError as exc:
            QMessageBox.critical(self, "Cannot open project", str(exc))

    def show_stage(self, stage: Stage) -> None:
        self.stack.setCurrentWidget(self.pages[stage])

    def refresh_pages(self) -> None:
        if not self.project: return
        self.setWindowTitle(f"AstroBatch V2 — {self.project.name}")
        for stage, page in self.pages.items():
            page.set_project_data(self.project.source_dir, self.project.settings.get(stage.value, {}), self.project.artifact_for(stage))

    def browse_path(self, stage: Stage, field: str) -> None:
        directory = QFileDialog.getExistingDirectory(self, f"Select {field} directory")
        if directory:
            (self.pages[stage].input_edit if field == "input" else self.pages[stage].output_edit).setText(directory)

    def run_stage(self, stage: Stage) -> None:
        if not self.project:
            QMessageBox.information(self, "Create a project", "Create or open a V2 project before running a stage.")
            return
        try:
            self.project.settings[stage.value] = self.pages[stage].collect_settings()
            self.store.save(self.project)
            service = PipelineService(self.project)
            self.manager.start(stage, lambda context: service.run(stage, context))
            self.cancel_action.setEnabled(True)
            for page in self.pages.values(): page.set_running(True)
        except (ValueError, RuntimeError) as exc:
            QMessageBox.warning(self, "Cannot start stage", str(exc))

    def preview_fits(self, path: str) -> None:
        self.viewer.load_path(path)

    def _on_event(self, event: JobEvent) -> None:
        if event.kind in {JobEventKind.LOG, JobEventKind.WARNING, JobEventKind.FAILED}:
            self.log.append(f"[{event.stage.value}] {event.message}")
        if event.kind is JobEventKind.PROGRESS and event.progress is not None:
            self.progress.setValue(round(event.progress))
            self.pages[event.stage].set_status(event.message or f"{event.progress:.0f}%")
        if event.kind is JobEventKind.ARTIFACT and event.artifact and self.project:
            self.project.artifacts[event.stage.value] = event.artifact
            self.pages[event.stage].set_artifacts([event.artifact])
        if event.kind in {JobEventKind.COMPLETED, JobEventKind.CANCELLED, JobEventKind.FAILED}:
            self.cancel_action.setEnabled(False)
            for page in self.pages.values(): page.set_running(False)
            self.pages[event.stage].set_status(event.message)
            if self.project and event.kind is JobEventKind.COMPLETED:
                artifacts = [self.project.artifact_for(event.stage)] if self.project.artifact_for(event.stage) else []
                self.project.record_stage(event.stage, event.message, [item for item in artifacts if item], event.details)
                self.store.save(self.project)
            if event.kind is JobEventKind.FAILED:
                QMessageBox.critical(self, f"{STAGE_LABELS[event.stage]} failed", event.message)
