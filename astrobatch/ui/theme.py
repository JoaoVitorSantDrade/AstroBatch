"""Central visual system for the AstroBatch desktop application."""

from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication, QWidget


def apply_enterprise_theme(window: QWidget) -> None:
    """Apply the shared high-contrast blue/slate desktop theme."""
    app = QApplication.instance()
    if app is not None:
        palette = app.palette()
        palette.setColor(QPalette.Window, QColor("#F8FAFC"))
        palette.setColor(QPalette.WindowText, QColor("#0F172A"))
        palette.setColor(QPalette.Base, QColor("#FFFFFF"))
        palette.setColor(QPalette.AlternateBase, QColor("#F1F5F9"))
        palette.setColor(QPalette.Text, QColor("#111827"))
        palette.setColor(QPalette.Button, QColor("#FFFFFF"))
        palette.setColor(QPalette.ButtonText, QColor("#1E293B"))
        palette.setColor(QPalette.Highlight, QColor("#2563EB"))
        palette.setColor(QPalette.HighlightedText, QColor("#FFFFFF"))
        app.setPalette(palette)

    window.setStyleSheet("""
        QMainWindow { background: #F8FAFC; color: #0F172A; }
        #sidebar { background: #0F172A; }
        #brand { color: #F8FAFC; font-size: 20px; font-weight: 800; letter-spacing: 1.6px; }
        #brandMeta, #navLabel { color: #94A3B8; font-size: 10px; font-weight: 700; letter-spacing: 1.1px; }
        #projectLabel { color: #DCE6F4; background: #18243A; border: 1px solid #263650; border-radius: 8px; padding: 12px; line-height: 1.3; }
        #sidebarPrimary, #sidebarButton, #navButton, #cancelButton { text-align: left; border: 0; border-radius: 6px; padding: 10px 12px; }
        #sidebarPrimary { color: #FFFFFF; background: #2563EB; font-weight: 700; }
        #sidebarPrimary:hover { background: #1D4ED8; }
        #sidebarButton, #navButton { background: transparent; color: #C5D0E0; }
        #sidebarButton:hover, #navButton:hover { background: #1E293B; color: #FFFFFF; }
        #navButton:checked { background: #1D4ED8; color: #FFFFFF; font-weight: 700; }
        #cancelButton { background: #3B1F2A; color: #FBC1CC; }
        #topHeader { background: #FFFFFF; border-bottom: 1px solid #D9E1EC; }
        #headerTitle { color: #0F172A; font-size: 17px; font-weight: 750; }
        #headerMeta { color: #64748B; font-size: 12px; }
        #headerBadge { color: #1D4ED8; background: #EFF6FF; border: 1px solid #BFDBFE; border-radius: 12px; padding: 5px 9px; font-weight: 700; }
        #eyebrow { color: #64748B; font-size: 10px; font-weight: 800; letter-spacing: 1.2px; }
        #welcomeTitle { color: #0F172A; font-size: 40px; font-weight: 800; line-height: 1.12; }
        #pageTitle { color: #0F172A; font-size: 30px; font-weight: 800; }
        #pageIntro { color: #526070; font-size: 14px; max-width: 760px; line-height: 1.45; }
        #card { background: #FFFFFF; border: 1px solid #D9E1EC; border-radius: 8px; }
        #cardTitle { color: #172033; font-size: 15px; font-weight: 750; }
        #muted { color: #64748B; line-height: 1.35; }
        #stageStatus { color: #1D4ED8; background: #EFF6FF; border: 1px solid #BFDBFE; border-radius: 6px; padding: 8px 10px; font-weight: 700; }
        #workflowStep { background: #F8FAFC; border: 1px solid #D9E1EC; border-radius: 7px; }
        #stepNumber { color: #2563EB; font-size: 11px; font-weight: 800; }
        #stepTitle { color: #172033; font-size: 16px; font-weight: 750; }
        #primaryButton { background: #2563EB; color: #FFFFFF; border: 1px solid #2563EB; border-radius: 6px; padding: 10px 18px; font-weight: 700; }
        #primaryButton:hover { background: #1D4ED8; border-color: #1D4ED8; }
        #artifactCard { background: #F8FAFC; border: 1px solid #D9E1EC; border-radius: 6px; padding: 3px; }
        QLabel { color: #1E293B; }
        QLineEdit, QPlainTextEdit, QTextEdit, QComboBox, QSpinBox, QDoubleSpinBox {
            color: #111827; background: #FFFFFF; border: 1px solid #AAB7C8;
            border-radius: 6px; padding: 8px; selection-background-color: #2563EB; selection-color: #FFFFFF;
        }
        QLineEdit:focus, QPlainTextEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus { border: 2px solid #2563EB; }
        QCheckBox { color: #1E293B; spacing: 8px; }
        QCheckBox::indicator { width: 17px; height: 17px; border: 1px solid #7E8EA3; border-radius: 3px; background: #FFFFFF; }
        QCheckBox::indicator:checked { background: #2563EB; border-color: #2563EB; }
        QPushButton { border: 1px solid #AAB7C8; border-radius: 6px; padding: 8px 12px; background: #FFFFFF; color: #1E293B; }
        QPushButton:hover { background: #F1F5F9; border-color: #7E8EA3; }
        QToolButton { color: #1D4ED8; font-weight: 700; border: 0; padding: 4px 0; }
        QStatusBar { background: #FFFFFF; color: #526070; border-top: 1px solid #D9E1EC; }
        QProgressBar { border: 1px solid #CBD5E1; border-radius: 5px; background: #F1F5F9; text-align: center; min-height: 16px; }
        QProgressBar::chunk { background: #2563EB; border-radius: 4px; }
        QDockWidget::title { background: #FFFFFF; color: #172033; padding: 9px; border-bottom: 1px solid #D9E1EC; font-weight: 700; }
    """)
