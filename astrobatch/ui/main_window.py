from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDockWidget, QFileDialog, QFrame, QHBoxLayout, QInputDialog, QLabel, QMainWindow, QMessageBox, QProgressBar, QPushButton, QStackedWidget, QTextEdit, QVBoxLayout, QWidget

from astrobatch.core.jobs import JobManager
from astrobatch.core.models import JobEvent, JobEventKind, Stage
from astrobatch.project.store import ProjectFormatError, ProjectStore
from astrobatch.services.pipeline import PipelineService, STAGE_LABELS
from .fits_viewer import FitsViewer
from .jobs import QtJobBridge
from .stage_page import StagePage
from .theme import apply_enterprise_theme
from .welcome_page import WelcomePage


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("AstroBatch V2")
        self.resize(1380, 900)
        self.setMinimumSize(1080, 720)
        self.store, self.project = ProjectStore(), None
        self.manager, self.bridge = JobManager(), QtJobBridge()
        self.manager.add_listener(self.bridge.emit_event)
        self.bridge.event_received.connect(self._on_event)
        self.pages = {stage: StagePage(stage) for stage in Stage}
        self._build_shell()
        apply_enterprise_theme(self)

    def _build_shell(self):
        shell = QWidget()
        row = QHBoxLayout(shell)
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(0)
        sidebar = self._build_sidebar()
        row.addWidget(sidebar)
        content = QWidget()
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(0)
        content_layout.addWidget(self._build_header())
        self.stack = QStackedWidget()
        self.welcome = WelcomePage()
        self.welcome.new_project_requested.connect(self.new_project)
        self.welcome.open_project_requested.connect(self.open_project)
        self.stack.addWidget(self.welcome)
        for page in self.pages.values():
            self.stack.addWidget(page)
            page.run_requested.connect(self.run_stage)
            page.browse_requested.connect(self.browse_path)
            page.preview_requested.connect(self.preview_fits)
        content_layout.addWidget(self.stack, 1)
        row.addWidget(content, 1)
        self.setCentralWidget(shell)
        self._build_docks()
        self.show_welcome()

    def _build_header(self):
        header = QFrame()
        header.setObjectName("topHeader")
        header.setFixedHeight(76)
        layout = QHBoxLayout(header)
        layout.setContentsMargins(30, 14, 30, 14)
        layout.setSpacing(10)
        title_block = QVBoxLayout()
        self.header_title = QLabel("Project workspace")
        self.header_title.setObjectName("headerTitle")
        self.header_meta = QLabel("Create a project to configure a full FITS pipeline")
        self.header_meta.setObjectName("headerMeta")
        title_block.addWidget(self.header_title)
        title_block.addWidget(self.header_meta)
        layout.addLayout(title_block)
        layout.addStretch()
        self.header_badge = QLabel("No active job")
        self.header_badge.setObjectName("headerBadge")
        layout.addWidget(self.header_badge)
        return header

    def _build_sidebar(self):
        sidebar = QFrame()
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(258)
        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(20, 24, 20, 20)
        layout.setSpacing(8)
        brand = QLabel("ASTROBATCH")
        brand.setObjectName("brand")
        version = QLabel("V2  •  FITS PIPELINE")
        version.setObjectName("brandMeta")
        layout.addWidget(brand)
        layout.addWidget(version)
        layout.addSpacing(25)
        self.project_label = QLabel("No project open\nCreate one to begin")
        self.project_label.setObjectName("projectLabel")
        layout.addWidget(self.project_label)
        create = QPushButton("+  New project")
        create.setObjectName("sidebarPrimary")
        create.clicked.connect(self.new_project)
        open_project = QPushButton("Open project")
        open_project.setObjectName("sidebarButton")
        open_project.clicked.connect(self.open_project)
        layout.addWidget(create)
        layout.addWidget(open_project)
        layout.addSpacing(18)
        nav_label = QLabel("PIPELINE")
        nav_label.setObjectName("navLabel")
        layout.addWidget(nav_label)
        self.nav_buttons = {}
        for index, stage in enumerate(Stage, start=1):
            button = QPushButton(f"{index:02d}   {STAGE_LABELS[stage]}")
            button.setObjectName("navButton")
            button.setCheckable(True)
            button.clicked.connect(lambda _, selected=stage: self.show_stage(selected))
            self.nav_buttons[stage] = button
            layout.addWidget(button)
        layout.addStretch()
        self.cancel_button = QPushButton("Cancel running task")
        self.cancel_button.setObjectName("cancelButton")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self.manager.cancel)
        layout.addWidget(self.cancel_button)
        return sidebar

    def _build_docks(self):
        self.log = QTextEdit(readOnly=True)
        self.log.setPlaceholderText("Processing messages appear here.")
        log_dock = QDockWidget("Run log", self)
        log_dock.setObjectName("logDock")
        log_dock.setWidget(self.log)
        self.addDockWidget(Qt.BottomDockWidgetArea, log_dock)
        log_dock.hide()
        self.viewer = FitsViewer()
        viewer_dock = QDockWidget("FITS viewer", self)
        viewer_dock.setWidget(self.viewer)
        self.addDockWidget(Qt.RightDockWidgetArea, viewer_dock)
        viewer_dock.hide()
        self.progress = QProgressBar()
        self.statusBar().addPermanentWidget(self.progress)
        self.statusBar().showMessage("Create a project to begin")

    def show_welcome(self):
        self.stack.setCurrentWidget(self.welcome)
        self.header_title.setText("Project workspace")
        self.header_meta.setText("Create a project to configure a full FITS pipeline")
        for button in self.nav_buttons.values(): button.setChecked(False)

    def show_stage(self, stage: Stage):
        self.stack.setCurrentWidget(self.pages[stage])
        self.header_title.setText(STAGE_LABELS[stage])
        self.header_meta.setText("Configure settings, validate inputs, then run this stage")
        for item, button in self.nav_buttons.items(): button.setChecked(item == stage)

    def new_project(self):
        root = QFileDialog.getExistingDirectory(self, "Choose the project workspace")
        if not root: return
        name, accepted = QInputDialog.getText(self, "Project name", "Name", text=Path(root).name)
        if not accepted: return
        source = QFileDialog.getExistingDirectory(self, "Choose the source FITS folder (optional)")
        self.project = self.store.create(Path(root), name, Path(source) if source else None)
        self.log.append(f"Created project: {self.project.workspace.root}")
        self.refresh_project()
        self.show_stage(Stage.IMPORT)

    def open_project(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open AstroBatch project", filter="AstroBatch project (astrobatch.project.json)")
        if not path: return
        try:
            self.project = self.store.load(Path(path))
            self.log.append(f"Opened project: {self.project.name}")
            self.refresh_project()
            self.show_stage(Stage.IMPORT)
        except ProjectFormatError as exc:
            QMessageBox.critical(self, "Cannot open project", str(exc))

    def refresh_project(self):
        if not self.project: return
        self.project_label.setText(f"{self.project.name}\n{self.project.workspace.root}")
        self.setWindowTitle(f"AstroBatch V2 — {self.project.name}")
        self.header_title.setText(self.project.name)
        self.header_meta.setText(self.project.workspace.root.as_posix())
        for stage, page in self.pages.items(): page.set_project_data(self.project.source_dir, self.project.settings.get(stage.value, {}), self.project.artifact_for(stage))
        self.statusBar().showMessage(f"Project open: {self.project.name}")

    def browse_path(self, stage, field):
        if field in {"dark_path", "flat_path"}:
            picker = QMessageBox(self)
            picker.setWindowTitle("Calibration source")
            picker.setText("Use an existing Master FITS file or a folder of calibration frames?")
            file_button = picker.addButton("Master FITS file", QMessageBox.ActionRole)
            folder_button = picker.addButton("Frames folder", QMessageBox.ActionRole)
            picker.addButton(QMessageBox.Cancel)
            picker.exec()
            if picker.clickedButton() is file_button:
                selected, _ = QFileDialog.getOpenFileName(self, "Choose a master FITS file", filter="FITS files (*.fit *.fits *.fts);;All files (*)")
            elif picker.clickedButton() is folder_button:
                selected = QFileDialog.getExistingDirectory(self, "Choose calibration frames folder")
            else:
                selected = ""
        else:
            selected = QFileDialog.getExistingDirectory(self, f"Choose {field.replace('_', ' ')} folder")
        if selected:
            self.pages[stage].controls[field].setText(selected)

    def run_stage(self, stage):
        if not self.project:
            QMessageBox.information(self, "Start with a project", "Create or open a project before running a pipeline stage.")
            return
        try:
            self.project.settings[stage.value] = self.pages[stage].collect_settings()
            self.store.save(self.project)
            service = PipelineService(self.project)
            self.manager.start(stage, lambda context: service.run(stage, context))
            self.cancel_button.setEnabled(True)
            self.header_badge.setText(f"Running {STAGE_LABELS[stage]}")
            for page in self.pages.values(): page.set_running(True)
        except (ValueError, RuntimeError) as exc:
            QMessageBox.warning(self, "Cannot start stage", str(exc))

    def preview_fits(self, path): self.viewer.load_path(path)

    def _on_event(self, event: JobEvent):
        if event.kind in {JobEventKind.LOG, JobEventKind.WARNING, JobEventKind.FAILED}: self.log.append(f"[{event.stage.value}] {event.message}")
        if event.kind is JobEventKind.PROGRESS and event.progress is not None:
            self.progress.setValue(round(event.progress))
            self.pages[event.stage].set_status(event.message or f"{event.progress:.0f}%")
            self.statusBar().showMessage(event.message or f"Running {STAGE_LABELS[event.stage]}")
        if event.kind is JobEventKind.ARTIFACT and event.artifact and self.project:
            self.project.artifacts[event.stage.value] = event.artifact
            self.pages[event.stage].set_artifacts([event.artifact])
        if event.kind in {JobEventKind.COMPLETED, JobEventKind.CANCELLED, JobEventKind.FAILED}:
            self.cancel_button.setEnabled(False)
            self.header_badge.setText("No active job")
            for page in self.pages.values(): page.set_running(False)
            self.pages[event.stage].set_status(event.message)
            if self.project and event.kind is JobEventKind.COMPLETED:
                artifact = self.project.artifact_for(event.stage)
                self.project.record_stage(event.stage, event.message, [artifact] if artifact else [], event.details)
                self.store.save(self.project)
            if event.kind is JobEventKind.FAILED: QMessageBox.critical(self, f"{STAGE_LABELS[event.stage]} failed", event.message)
