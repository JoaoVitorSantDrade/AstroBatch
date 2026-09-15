import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from views.anchor_selector import AnchorSelectionController


class _Value:
    def __init__(self, value=""):
        self.value = value

    def get(self):
        return self.value


class AnchorSelectionControllerTests(unittest.TestCase):
    def make_controller(self, apply_callback):
        committed = []
        controller = AnchorSelectionController(
            Mock(),
            _Value(),
            {},
            on_apply_reference=apply_callback,
            on_commit_reference=lambda batch, frame: committed.append((batch, frame)),
        )
        return controller, committed

    def test_pending_reference_commits_only_after_success(self):
        controller, committed = self.make_controller(lambda *_args: True)
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            self.assertTrue(controller.request_apply(base, "batch_001", "02.fits"))
            self.assertEqual(controller.pending_reference_changes, {"batch_001": "02.fits"})
            self.assertEqual(committed, [])

            controller.finish_reference_change("failed")
            self.assertEqual(committed, [])
            self.assertIsNone(controller.active_reference)
            self.assertEqual(controller.pending_reference_changes, {})

            self.assertTrue(controller.request_apply(base, "batch_001", "03.fits"))
            controller.finish_reference_change("success")
            self.assertEqual(committed, [("batch_001", "03.fits")])
            self.assertEqual(controller.pending_reference_changes, {})

    def test_reference_apply_is_not_reentered(self):
        controller, committed = self.make_controller(lambda *_args: True)
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            self.assertTrue(controller.request_apply(base, "batch_001", "02.fits"))
            self.assertFalse(controller.request_apply(base, "batch_001", "03.fits"))
            controller.set_pending_reference("batch_001", "04.fits")
            controller.finish_reference_change("success")
        self.assertEqual(committed, [("batch_001", "02.fits")])
        self.assertEqual(controller.pending_reference_changes, {"batch_001": "04.fits"})

    def test_no_revision_apply_clears_pending_without_async_commit(self):
        controller, committed = self.make_controller(lambda *_args: False)
        with tempfile.TemporaryDirectory() as directory:
            self.assertFalse(controller.request_apply(Path(directory), "batch_001", "02.fits"))
        self.assertIsNone(controller.active_reference)
        self.assertEqual(controller.pending_reference_changes, {})
        self.assertEqual(committed, [])


if __name__ == "__main__":
    unittest.main()
