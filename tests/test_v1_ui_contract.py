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
        missing = sorted(name for name in required if f"self.app.{name}" not in views)
        self.assertEqual(missing, [])

    def test_theme_declares_high_contrast_editable_field_states(self) -> None:
        source = (ROOT / "main.py").read_text(encoding="utf-8")
        for token in ("fieldbackground=\"#ffffff\"", "insertcolor=self.TEXT", "selectforeground", "TCombobox*Listbox.foreground"):
            self.assertIn(token, source)
