from __future__ import annotations

import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import astroalign_logic as align


class AlignSchedulerTests(unittest.TestCase):
    def test_pipeline_submits_frames_from_all_batches_to_one_scheduler(self) -> None:
        first, second = Path("batch_001"), Path("batch_002")
        local = {
            "frames": {
                "frame.fit": {"status": "accepted", "matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]}
            }
        }
        global_flow = {
            "batches": {
                first.name: {"status": "accepted", "matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]},
                second.name: {"status": "accepted", "matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]]},
            }
        }
        progress: list[tuple[int, int, str]] = []
        with tempfile.TemporaryDirectory() as temp, patch.object(align, "load_global_flow", return_value=global_flow), patch.object(
            align, "find_batch_folders", return_value=[first, second]
        ), patch.object(align, "load_local_flow", return_value=local), patch.object(
            align, "get_optimal_worker_count", return_value=2
        ), patch.object(align, "_process_single_alignment", side_effect=lambda name, *_: (name, None)) as worker:
            result = align.process_all_alignments(
                Path(temp), Path(temp) / "out", {"dry_run": True}, lambda _: None,
                lambda current, total, message: progress.append((current, total, message)), threading.Event(),
            )
        self.assertEqual(result, (2, 0))
        self.assertEqual(worker.call_count, 2)
        self.assertEqual(progress[-1][:2], (2, 2))


if __name__ == "__main__":
    unittest.main()
