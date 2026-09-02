from __future__ import annotations

import os
import unittest

try:
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QPalette
    from PySide6.QtTest import QTest
    from astrobatch.ui.main_window import MainWindow
    from astrobatch.ui.stage_page import SCHEMAS
    from astrobatch.core.models import Stage
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
            self.assertFalse(window.cancel_button.isEnabled())
            self.assertIs(window.stack.currentWidget(), window.welcome)
        finally:
            window.close()

    def test_stage_schemas_keep_the_v1_processing_controls_visible(self) -> None:
        self.assertTrue({"apply_dark", "dark_path", "apply_flat", "flat_path"}.issubset({field.key for field in SCHEMAS[Stage.CALIBRATE]}))
        self.assertTrue({"opt_method", "crop_size", "downsample_scale", "threshold_factor"}.issubset({field.key for field in SCHEMAS[Stage.BATCH]}))
        self.assertTrue({"engine", "fwhm", "sigma", "matching_radius", "ransac"}.issubset({field.key for field in SCHEMAS[Stage.FLOW]}))
        self.assertTrue({"debayer_pattern", "interpolation", "rgb_registration", "compress_output"}.issubset({field.key for field in SCHEMAS[Stage.ALIGN]}))
        self.assertTrue({"method", "rejection_method", "normalize", "output_name", "memory_budget_mb", "workers"}.issubset({field.key for field in SCHEMAS[Stage.STACK]}))

    def test_inputs_are_interactive_and_have_high_text_contrast(self) -> None:
        window = MainWindow()
        try:
            window.show_stage(Stage.STACK)
            output_name = window.pages[Stage.STACK].controls["output_name"]
            window.show()
            self.application.processEvents()
            output_name.clear()
            QTest.mouseClick(output_name, Qt.LeftButton)
            QTest.keyClicks(output_name, "final_stack.fits")
            self.assertEqual(output_name.text(), "final_stack.fits")
            base = output_name.palette().color(QPalette.Base)
            text = output_name.palette().color(QPalette.Text)
            self.assertGreaterEqual(self._contrast_ratio(base, text), 7.0)
        finally:
            window.close()

    def test_sidebar_navigation_and_native_stack_controls_work(self) -> None:
        window = MainWindow()
        try:
            window.show()
            self.application.processEvents()
            QTest.mouseClick(window.nav_buttons[Stage.STACK], Qt.LeftButton)
            self.assertIs(window.stack.currentWidget(), window.pages[Stage.STACK])
            self.assertEqual(window.header_title.text(), "Stack")
            self.assertTrue(window.nav_buttons[Stage.STACK].isChecked())
            page = window.pages[Stage.STACK]
            page.controls["method"].setCurrentText("Mean")
            page.controls["normalize"].setChecked(False)
            page.controls["memory_budget_mb"].setValue(2048)
            settings = page.collect_settings()
            self.assertEqual(settings["method"], "Mean")
            self.assertFalse(settings["normalize"])
            self.assertEqual(settings["memory_budget_mb"], 2048)
        finally:
            window.close()

    @staticmethod
    def _contrast_ratio(first, second) -> float:
        def luminance(color) -> float:
            channels = [color.redF(), color.greenF(), color.blueF()]
            linear = [value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4 for value in channels]
            return 0.2126 * linear[0] + 0.7152 * linear[1] + 0.0722 * linear[2]
        first_luminance, second_luminance = luminance(first), luminance(second)
        return (max(first_luminance, second_luminance) + 0.05) / (min(first_luminance, second_luminance) + 0.05)
