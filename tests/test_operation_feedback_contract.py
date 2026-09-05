from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
import threading
from main import AstroProcessManager


ROOT = Path(__file__).parents[1]


class OperationFeedbackContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.source = (ROOT / "main.py").read_text(encoding="utf-8")

    def test_progress_exposes_count_percent_and_time_estimate(self) -> None:
        for token in (
            "progress_detail_var",
            "elapsed_var",
            "previsao:",
            "safe_current = max(0, min(current, total))",
        ):
            self.assertIn(token, self.source)

    def test_terminal_outcome_does_not_turn_errors_or_cancellations_into_success(self) -> None:
        self.assertIn('if outcome == "success":', self.source)
        self.assertIn('elif outcome == "cancelled":', self.source)
        self.assertNotIn("self.progress_var.set(100)\n\n    #", self.source)

    def test_stack_buttons_follow_the_same_locking_contract_as_other_stages(self) -> None:
        app = SimpleNamespace(PIPELINE_BUTTONS=AstroProcessManager.PIPELINE_BUTTONS,
                              save_settings=Mock(), clear_console=Mock(), progress_var=Mock(),
                              progress_detail_var=Mock(), _refresh_operation_clock=Mock(),
                              cancel_event=threading.Event(), status_var=Mock())
        for _, suffix in app.PIPELINE_BUTTONS:
            setattr(app, f"btn_run_{suffix}", Mock())
            setattr(app, f"btn_cancel_{suffix}", Mock())
        app._operation_buttons = lambda: AstroProcessManager._operation_buttons(app)
        AstroProcessManager._lock_ui(app, "Stack")
        for stage, run, cancel in app._operation_buttons():
            run.configure.assert_called_with(state="disabled")
            cancel.configure.assert_called_with(state="normal" if stage == "Stack" else "disabled")
        AstroProcessManager._unlock_ui(app)
        for _, run, cancel in app._operation_buttons():
            run.configure.assert_called_with(state="normal")
            cancel.configure.assert_called_with(state="disabled")

    def test_activity_log_has_readable_controls_and_severity_tags(self) -> None:
        for token in (
            'text="Copiar"',
            'text="Limpar"',
            'text="Acompanhar"',
            'tag_configure("error"',
            'tag_configure("warning"',
            'tag_configure("success"',
        ):
            self.assertIn(token, self.source)


if __name__ == "__main__":
    unittest.main()
