from __future__ import annotations

import unittest
from pathlib import Path


ROOT = Path(__file__).parents[1]


class V1UiContractTests(unittest.TestCase):
    def test_all_v1_stage_options_remain_bound_to_native_controls(self) -> None:
        views = "\n".join(path.read_text(encoding="utf-8") for path in (ROOT / "views").glob("*_view.py"))
        required = {
            "apply_dark_var", "apply_flat_var", "calib_overwrite_var",
            "opt_method_var", "crop_size_var", "downsample_method_var", "threshold_var",
            "flow_engine_var", "flow_fwhm_var", "flow_sigma_var", "flow_matching_radius_var",
            "align_debayer_pattern_var", "align_interpolation_var", "align_rgb_registration_var",
            "stack_selection_mode_var", "stack_method_var", "stack_rejection_method_var",
            "stack_normalize_var", "stack_output_name_var", "stack_compress_var",
        }
        calibration_model_names = {
            "apply_dark_var": "apply_dark",
            "apply_flat_var": "apply_flat",
            "calib_overwrite_var": "overwrite",
            "batch_input_dir_var": "input_dir",
            "batch_dir_var": "output_dir",
            "opt_method_var": "opt_method",
            "crop_size_var": "crop_size",
            "downsample_method_var": "downsample_method",
            "downsample_scale_var": "downsample_scale",
            "threshold_var": "threshold",
            "copy_files_var": "copy_files",
            "batch_overwrite_var": "overwrite",
            "dry_run_var": "dry_run",
        }
        missing = sorted(
            name for name in required
            if f"self.app.{name}" not in views
            and f"self.model.{calibration_model_names.get(name, name)}" not in views
        )
        self.assertEqual(missing, [])

    def test_calibration_view_uses_passive_model_bindings(self) -> None:
        source = (ROOT / "views" / "calibration_view.py").read_text(encoding="utf-8")
        self.assertIn("CalibrationViewModel", source)
        self.assertNotIn("self.app.", source)

    def test_batch_view_uses_passive_model_bindings(self) -> None:
        source = (ROOT / "views" / "batch_view.py").read_text(encoding="utf-8")
        self.assertIn("BatchViewModel", source)
        self.assertNotIn("self.app.", source)

    def test_theme_declares_high_contrast_editable_field_states(self) -> None:
        source = (ROOT / "main.py").read_text(encoding="utf-8")
        for token in ("fieldbackground=\"#ffffff\"", "insertcolor=self.TEXT", "selectforeground", "TCombobox*Listbox.foreground"):
            self.assertIn(token, source)

    def test_stack_mousewheel_binding_is_scoped_to_its_canvas(self) -> None:
        source = (ROOT / "views" / "stacking_view.py").read_text(encoding="utf-8")
        self.assertIn('canvas.bind("<MouseWheel>"', source)
        self.assertNotIn("bind_all(\"<MouseWheel>\"", source)
