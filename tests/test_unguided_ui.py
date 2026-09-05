from __future__ import annotations

import tkinter as tk
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import main


class FakeVar:
    def __init__(self, value):
        self.value = value

    def get(self):
        return self.value

    def set(self, value):
        self.value = value


class UnguidedUiBehaviorTests(unittest.TestCase):
    def _preset_app(self):
        return SimpleNamespace(
            stack_trail_filter_var=FakeVar(False),
            stack_min_roundness_var=FakeVar(0.2),
            stack_min_shape_stars_var=FakeVar(2),
            stack_selection_mode_var=FakeVar("BestPercentage"),
            stack_method_var=FakeVar("Median"),
            stack_rejection_method_var=FakeVar("MAD"),
            stack_profile_var=FakeVar("Fast"),
            status_var=FakeVar("Pronto."),
            saved=0,
            messages=[],
        )

    def test_unguided_preset_updates_all_requested_controls(self):
        app = self._preset_app()
        app.save_settings = lambda: setattr(app, "saved", app.saved + 1)
        app.print_to_console = app.messages.append

        main.AstroProcessManager.apply_unguided_preset(app)

        self.assertTrue(app.stack_trail_filter_var.get())
        self.assertEqual(app.stack_min_roundness_var.get(), 0.65)
        self.assertEqual(app.stack_min_shape_stars_var.get(), 5)
        self.assertEqual(app.stack_selection_mode_var.get(), "All")
        self.assertEqual(app.stack_method_var.get(), "Mean")
        self.assertEqual(app.stack_rejection_method_var.get(), "SigmaClip")
        self.assertEqual(app.stack_profile_var.get(), "Stable")
        self.assertEqual(app.saved, 1)
        self.assertIn("Subs sem guiagem", app.status_var.get())

    def test_blank_align_output_keeps_existing_stack_input(self):
        app = SimpleNamespace(
            align_output_dir_var=FakeVar("   "),
            stack_input_dir_var=FakeVar("C:/already/aligned"),
            status_var=FakeVar("Pronto."),
            messages=[],
        )
        app.save_settings = lambda: self.fail("blank Align output must not save")
        app.print_to_console = app.messages.append

        result = main.AstroProcessManager.use_align_output_for_stack(app)

        self.assertFalse(result)
        self.assertEqual(app.stack_input_dir_var.get(), "C:/already/aligned")
        self.assertIn("saída do AstroAlign", app.status_var.get())
        self.assertTrue(app.messages)

    def test_align_output_is_copied_only_when_explicit_handoff_runs(self):
        app = SimpleNamespace(
            align_output_dir_var=FakeVar("C:/aligned"),
            stack_input_dir_var=FakeVar("C:/old"),
            status_var=FakeVar("Pronto."),
            messages=[],
        )
        app.saved = 0
        app.save_settings = lambda: setattr(app, "saved", app.saved + 1)
        app.print_to_console = app.messages.append

        result = main.AstroProcessManager.use_align_output_for_stack(app)

        self.assertTrue(result)
        self.assertEqual(app.stack_input_dir_var.get(), "C:/aligned")
        self.assertEqual(app.saved, 1)

    def test_invalid_roundness_shows_error_without_starting_worker(self):
        app = SimpleNamespace(
            worker=None,
            stack_trail_filter_var=FakeVar(False),
            stack_min_roundness_var=FakeVar("nan"),
            stack_min_shape_stars_var=FakeVar(5),
        )

        with patch.object(main.messagebox, "showerror") as showerror:
            main.AstroProcessManager.start_stacking(app)

        showerror.assert_called_once()
        self.assertIsNone(app.worker)

    def test_invalid_star_count_shows_error_without_starting_worker(self):
        app = SimpleNamespace(
            worker=None,
            stack_trail_filter_var=FakeVar(False),
            stack_min_roundness_var=FakeVar(0.65),
            stack_min_shape_stars_var=FakeVar("1.5"),
        )

        with patch.object(main.messagebox, "showerror") as showerror:
            main.AstroProcessManager.start_stacking(app)

        showerror.assert_called_once()
        self.assertIsNone(app.worker)

    def test_live_tk_variables_and_controls_are_bound_when_tk_is_available(self):
        try:
            with patch.object(main.AstroProcessManager, "load_settings"), patch.object(
                main.AstroProcessManager, "save_settings"
            ), patch.object(
                main.AstroProcessManager, "_start_cpu_kernel_warmup"
            ):
                app = main.AstroProcessManager()
        except (tk.TclError, RuntimeError) as exc:
            self.skipTest(f"Tk indisponível neste ambiente: {exc}")

        try:
            app.withdraw()
            self.assertIs(
                app.config_registry["AstroStack"]["trail_filter_enabled"],
                app.stack_trail_filter_var,
            )
            self.assertIs(
                app.config_registry["AstroStack"]["min_roundness"],
                app.stack_min_roundness_var,
            )
            self.assertIs(
                app.config_registry["AstroStack"]["min_shape_stars"],
                app.stack_min_shape_stars_var,
            )

            check = next(
                widget
                for widget in app.tab_stack.inner_frame.winfo_children()[1].winfo_children()
                if isinstance(widget, tk.Widget)
                and widget.winfo_class() == "TCheckbutton"
                and widget.cget("text") == "Rejeitar subs com rastros"
            )
            self.assertEqual(str(app.stack_trail_filter_var), check.cget("variable"))
        finally:
            app.destroy()


if __name__ == "__main__":
    unittest.main()
