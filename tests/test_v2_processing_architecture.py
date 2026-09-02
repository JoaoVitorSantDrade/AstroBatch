from __future__ import annotations

import inspect
import unittest
from pathlib import Path

from astrobatch.processing import align, batch, calibration, cpu_kernels, flow, stacking
from astrobatch.services.pipeline import PipelineService


class V2ProcessingArchitectureTests(unittest.TestCase):
    def test_scientific_engines_are_owned_by_the_v2_package(self) -> None:
        package_root = Path(__file__).parents[1] / "astrobatch" / "processing"
        for module in (align, batch, calibration, cpu_kernels, flow, stacking):
            self.assertTrue(Path(inspect.getfile(module)).is_relative_to(package_root))

    def test_pipeline_service_uses_only_v2_processing_modules(self) -> None:
        source = inspect.getsource(PipelineService)
        self.assertIn("astrobatch.processing.calibration", source)
        self.assertIn("astrobatch.processing.batch", source)
        self.assertIn("astrobatch.processing.flow", source)
        self.assertIn("astrobatch.processing.align", source)
        self.assertIn("astrobatch.processing.stacking", source)
        self.assertNotIn("from calibration_logic", source)

    def test_processing_layer_does_not_depend_on_a_ui_framework(self) -> None:
        package_root = Path(__file__).parents[1] / "astrobatch" / "processing"
        forbidden = ("tkinter", "customtkinter", "PySide6", "PyQt")
        for path in package_root.glob("*.py"):
            content = path.read_text(encoding="utf-8")
            self.assertFalse(any(value in content for value in forbidden), path.name)
