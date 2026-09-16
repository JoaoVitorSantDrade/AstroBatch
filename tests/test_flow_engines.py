from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import astroflow_logic as flow


class FlowEngineTests(unittest.TestCase):
    def test_fast_profile_chooses_components_when_legacy_detector_is_unchanged(self) -> None:
        self.assertEqual(
            flow._flow_detector_choice({"engine": "DAO", "engine_profile": "Fast"}),
            "opencv-components",
        )

    def test_connected_components_detector_returns_coordinate_contract(self) -> None:
        image = np.random.default_rng(42).normal(0.0, 1.0, (96, 96)).astype(np.float32)
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

    def test_dao_stats_cache_reuses_image_statistics_without_changing_catalogue(self) -> None:
        image = np.random.default_rng(42).normal(0.0, 1.0, (96, 96)).astype(np.float32)
        for x, y in ((20, 20), (45, 35), (70, 70)):
            image[y - 2 : y + 3, x - 2 : x + 3] = 100.0
        cache: dict = {}
        first = flow.detect_stars_dao(image, 4.0, 5.0, 20, stats_cache=cache)
        self.assertIn("dao_stats", cache)
        self.assertIn("dao_convolved", cache)
        reference_second = flow.detect_stars_dao(image, 4.0, 4.0, 20, stats_cache=None)
        expected_second = flow.detect_stars_dao(image, 4.0, 4.0, 20, stats_cache={})
        with patch.object(flow, "sigma_clipped_stats", side_effect=AssertionError("stats recomputed")):
            second = flow.detect_stars_dao(image, 4.0, 4.0, 20, stats_cache=cache)
        self.assertTrue(cache.get("dao_low_xypos_computed"))
        self.assertTrue(cache.get("dao_catalog_ready"))
        np.testing.assert_array_equal(second[0], expected_second[0])
        np.testing.assert_array_equal(second[0], reference_second[0])
        self.assertEqual(second[1], expected_second[1])
        self.assertEqual(second[1], reference_second[1])
        self.assertEqual(second[2], expected_second[2])
        self.assertEqual(second[2], reference_second[2])
        self.assertEqual(first[2]["mean"], second[2]["mean"])
        self.assertEqual(first[2]["median"], second[2]["median"])
        self.assertEqual(first[2]["std"], second[2]["std"])

    def test_dao_private_module_absence_keeps_public_fallback(self) -> None:
        image = np.random.default_rng(7).normal(0.0, 1.0, (96, 96)).astype(np.float32)
        for x, y in ((20, 20), (45, 35), (70, 70)):
            image[y - 2 : y + 3, x - 2 : x + 3] = 100.0
        with patch.object(flow, "_DAOFINDER_MODULE", None):
            fallback = flow.detect_stars_dao(image, 4.0, 5.0, 20, stats_cache={})
        direct = flow.detect_stars_dao(image, 4.0, 5.0, 20, stats_cache=None)
        np.testing.assert_array_equal(fallback[0], direct[0])
        self.assertEqual(fallback[1:], direct[1:])

    def test_dao_peak_helper_absence_keeps_cached_finder_fallback(self) -> None:
        image = np.random.default_rng(8).normal(0.0, 1.0, (96, 96)).astype(np.float32)
        for x, y in ((20, 20), (45, 35), (70, 70)):
            image[y - 2 : y + 3, x - 2 : x + 3] = 100.0
        with patch.object(flow, "_DAO_FAST_CIRCULAR_PEAKS", None):
            fallback = flow.detect_stars_dao(image, 4.0, 5.0, 20, stats_cache={})
        direct = flow.detect_stars_dao(image, 4.0, 5.0, 20, stats_cache=None)
        np.testing.assert_array_equal(fallback[0], direct[0])
        self.assertEqual(fallback[1:], direct[1:])

    def test_failed_frame_still_exposes_explicit_unknown_timestamp(self) -> None:
        with patch.object(flow, "load_fits_data", side_effect=OSError("broken FITS")):
            name, frame = flow._process_single_frame(
                Path("broken.fits"),
                4.0,
                5.0,
                150,
                4,
                "DAO",
                "Stable",
            )
        self.assertEqual(name, "broken.fits")
        self.assertEqual(frame["status"], "error")
        self.assertEqual(frame["timestamp_state"], "unknown")
        self.assertIsNone(frame["timestamp_normalized"])

    def test_global_anchor_reuse_requires_untruncated_local_catalogue(self) -> None:
        stars = np.arange(72, dtype=np.float32).reshape(36, 2)
        info = {
            "anchor_stars": stars,
            "anchor_shape": (96, 96),
            "fwhm": 4.0,
            "anchor_detection": {
                "fwhm": 4.0,
                "sigma": 5.0,
                "sigma_used": 5.0,
                "max_stars": 150,
                "catalog_truncated": False,
                "engine": "DAO",
                "engine_profile": "Stable",
            },
        }
        reused = flow._cached_anchor_for_global(info, 4.0, 5.0, "DAO", "Stable")
        self.assertIsNotNone(reused)
        info["anchor_detection"]["catalog_truncated"] = True
        self.assertIsNone(flow._cached_anchor_for_global(info, 4.0, 5.0, "DAO", "Stable"))


if __name__ == "__main__":
    unittest.main()
