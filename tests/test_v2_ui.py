from __future__ import annotations

import os
import unittest

try:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    from astrobatch.ui.main_window import MainWindow
except ImportError:
    QApplication = None


@unittest.skipIf(QApplication is None, "PySide6 is not installed in this environment")
class V2UiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.application = QApplication.instance() or QApplication([])

    def test_shell_exposes_all_guided_pipeline_stages(self) -> None:
        window = MainWindow()
        try:
            self.assertEqual(len(window.pages), 7)
            self.assertEqual(window.windowTitle(), "AstroBatch V2")
            self.assertFalse(window.cancel_action.isEnabled())
        finally:
            window.close()
