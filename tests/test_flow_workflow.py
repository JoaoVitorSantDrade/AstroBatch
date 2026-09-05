from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
import unittest
from unittest.mock import patch

import numpy as np
from astropy.io import fits

import astroflow_logic as flow


class FlowWorkflowTests(unittest.TestCase):
    def test_confidence_uses_spatial_and_phase_evidence(self):
        limits = {"min_ratio": 0.15, "max_rms": 4.0}
        self.assertEqual(flow._classify_flow_confidence({"phase_response": .3, "inlier_ratio": .8, "rms": 1., "spatial_inlier_coverage": .04}, limits), "accepted")
        self.assertEqual(flow._classify_flow_confidence({"phase_response": .01, "inlier_ratio": .8, "rms": 1., "spatial_inlier_coverage": .04}, limits), "low_confidence")

    def test_header_budget_clamps_workers_without_loading_pixels(self):
        with TemporaryDirectory() as td:
            path = Path(td) / "frame.fits"
            fits.PrimaryHDU(np.zeros((20, 20), dtype=np.uint16)).writeto(path)
            stars = np.array([[2, 2], [4, 2], [6, 2], [2, 6], [6, 6]], dtype=np.float32)
            result_frame = lambda p, *args: (p.name, {"path": p, "data": None, "phase_data": np.zeros((8, 8), np.float32), "stars": stars, "fwhm": 2., "metrics": {}, "status": "prepared"})
            with patch.object(flow, "_process_single_frame", side_effect=result_frame):
                result = flow.process_local_flow(Path(td), {"flow_workers": 8, "memory_budget_mb": 64, "min_stars": 4}, lambda _: None, Event())
            self.assertEqual(result["total_frames"], 1)
            self.assertLessEqual(result["valid_frames"], 1)

    def test_phase_correlation_is_reused_for_relaxed_attempt(self):
        with patch("astroflow_logic.cv2.phaseCorrelate", return_value=((0., 0.), 1.0)) as phase:
            flow._match_incremental_stars(np.zeros((0, 2)), np.zeros((0, 2)), (0., 0.), 1.)
        self.assertEqual(phase.call_count, 0)

    def test_manual_anchor_contract_preserves_all_names(self):
        with TemporaryDirectory() as td:
            root = Path(td)
            for name in ("01.fits", "02.fits", "03.fits"):
                fits.PrimaryHDU(np.zeros((8, 8), dtype=np.uint16)).writeto(root / name)
            stars = np.array([[2, 2], [4, 2], [6, 2], [2, 6], [6, 6]], dtype=np.float32)
            def prepared(p, *args):
                return p.name, {"path": p, "data": None, "phase_data": np.zeros((8, 8), np.float32), "stars": stars, "fwhm": 2., "metrics": {}, "status": "prepared"}
            with patch.object(flow, "_process_single_frame", side_effect=prepared):
                result = flow.process_local_flow(root, {"custom_anchors": {root.name: "02.fits"}, "min_stars": 4}, lambda _: None, Event())
            with (root / "flow_local.json").open(encoding="utf-8") as handle:
                persisted = __import__("json").load(handle)
            self.assertEqual(persisted["batch_anchor"], "02.fits")
            self.assertEqual(set(persisted["frames"]), {"01.fits", "02.fits", "03.fits"})


if __name__ == "__main__":
    unittest.main()
