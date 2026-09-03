from __future__ import annotations

import unittest

import numpy as np

import astroflow_logic as flow


class FlowEngineTests(unittest.TestCase):
    def test_fast_profile_chooses_components_when_legacy_detector_is_unchanged(self) -> None:
        self.assertEqual(
            flow._flow_detector_choice({"engine": "DAO", "engine_profile": "Fast"}),
            "opencv-components",
        )

    def test_connected_components_detector_returns_coordinate_contract(self) -> None:
        image = np.zeros((96, 96), dtype=np.float32)
        for x, y in ((20, 20), (45, 35), (70, 70)):
            image[y - 2 : y + 3, x - 2 : x + 3] = 100.0
        coords, fwhm, metrics = flow.detect_stars(
            image, 4.0, 1.0, 20, "opencv-components", "Fast"
        )
        self.assertEqual(coords.ndim, 2)
        self.assertEqual(coords.shape[1], 2)
        self.assertGreaterEqual(len(coords), 3)
        self.assertGreater(fwhm, 0.0)
        self.assertEqual(metrics["star_count"], len(coords))


if __name__ == "__main__":
    unittest.main()
