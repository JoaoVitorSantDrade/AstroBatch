from __future__ import annotations

import sys


def main() -> int:
    from PySide6.QtGui import QFont
    from PySide6.QtWidgets import QApplication
    from astrobatch.ui.main_window import MainWindow

    app = QApplication(sys.argv)
    # Prefer the standard Windows UI font; Qt falls back cleanly on other OSes.
    app.setFont(QFont("Segoe UI", 10))
    app.setApplicationName("AstroBatch V2")
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
