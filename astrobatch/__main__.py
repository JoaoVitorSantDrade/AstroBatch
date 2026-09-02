from __future__ import annotations

import sys


def main() -> int:
    from PySide6.QtWidgets import QApplication
    from astrobatch.ui.main_window import MainWindow

    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
