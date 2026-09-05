import builtins
import queue
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

import batch_logic as batch
from main import AstroProcessManager


class BatchReliabilityTests(unittest.TestCase):
    def run_batch(self, root, cancel, progress):
        source = root / "input"
        source.mkdir()
        (source / "frame_001.fits").write_bytes(b"fixture")
        config = batch.ProcessingConfig(source, root / "out", 3, 16, False,
                                        True, False, "Crop", "Nearest", 0.5)
        with patch.object(batch, "prepare_fits_file", return_value=(
            source / "frame_001.fits", np.ones((4, 4), np.float32), None
        )):
            return batch.process_fits_logic(config, Mock(), progress, cancel)

    def test_disk_failure_reaches_caller(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(batch.shutil, "copy2", side_effect=OSError("disk full")):
                with self.assertRaisesRegex(OSError, "Falha ao salvar 1"):
                    self.run_batch(Path(directory), threading.Event(), Mock())

    def test_cancel_does_not_emit_completion(self):
        cancel = threading.Event()
        cancel.set()
        progress = Mock()
        with tempfile.TemporaryDirectory() as directory:
            self.run_batch(Path(directory), cancel, progress)
        self.assertFalse(any(call.args[2] == "Concluído." for call in progress.call_args_list))

    def test_unexpected_processing_failure_stops_disk_worker(self):
        threads = []
        real_thread = threading.Thread
        def track_thread(*args, **kwargs):
            thread = real_thread(*args, **kwargs)
            threads.append(thread)
            return thread
        progress = Mock(side_effect=[None, RuntimeError("callback failed")])
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(batch.threading, "Thread", side_effect=track_thread):
                with self.assertRaisesRegex(RuntimeError, "callback failed"):
                    self.run_batch(Path(directory), threading.Event(), progress)
        self.assertTrue(threads)
        self.assertTrue(all(not thread.is_alive() for thread in threads))


class UIResponsivenessTests(unittest.TestCase):
    def test_log_burst_yields_and_preserves_order_and_tags(self):
        app = SimpleNamespace(log_queue=queue.Queue(), console_text=Mock(),
                              console_autoscroll_var=Mock(), after=Mock(),
                              _console_tag=AstroProcessManager._console_tag,
                              _drain_log_queue=Mock())
        app.console_text.index.return_value = "201.0"
        for index in range(1000):
            app.log_queue.put(f"error {index}\n")
        AstroProcessManager._drain_log_queue(app)
        self.assertEqual(app.log_queue.qsize(), 800)
        app.console_text.insert.assert_called_once()
        arguments = app.console_text.insert.call_args.args
        self.assertIn("error 0\n", arguments[1])
        self.assertIn("error 199\n", arguments[-2])
        self.assertEqual(arguments[-1], "error")
        app.after.assert_called_once()

    def test_worker_import_failure_always_finishes(self):
        real_import = builtins.__import__
        app = SimpleNamespace(print_to_console=Mock(), _finish_operation=Mock(),
                              status_var=Mock())
        app.after = lambda delay, callback: callback()
        def missing_import(name, *args, **kwargs):
            if name in {"calibration_logic", "batch_logic", "astroflow_logic",
                        "astroalign_logic", "stacking_logic"}:
                raise ImportError("missing optional dependency")
            return real_import(name, *args, **kwargs)
        for method, args in (
            ("_run_calibration_worker", ({},)), ("run_batch_logic", ({},)),
            ("run_flow_logic", (Path('.'), {})),
            ("run_align_logic", (Path('.'), Path('.'), {})),
            ("run_stacking_logic", (Path('.'), {})),
        ):
            with self.subTest(method=method), patch("builtins.__import__", missing_import):
                getattr(AstroProcessManager, method)(app, *args)
                self.assertEqual(app._finish_operation.call_args.args[0], "failed")
                app._finish_operation.reset_mock()


if __name__ == "__main__":
    unittest.main()
