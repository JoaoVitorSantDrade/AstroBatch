import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from frame_quality import measure_star_shapes


def _add_gaussian(image: np.ndarray, x: float, y: float, sx: float, sy: float) -> None:
    yy, xx = np.indices(image.shape, dtype=np.float64)
    image += 100.0 * np.exp(
        -(((xx - x) ** 2) / (2.0 * sx**2) + ((yy - y) ** 2) / (2.0 * sy**2))
    )


class FrameQualityTests(unittest.TestCase):
    def test_circular_stars_are_round_and_trailed_stars_are_elongated(self) -> None:
        coords = np.asarray(
            [[20, 20], [40, 20], [60, 20], [20, 40], [40, 40], [60, 40], [20, 60], [40, 60], [60, 60]],
            dtype=np.float32,
        )
        circular = np.full((80, 80), 10.0, dtype=np.float32)
        trailed = np.full((80, 80), 10.0, dtype=np.float32)
        for x, y in coords:
            _add_gaussian(circular, x, y, 2.0, 2.0)
            _add_gaussian(trailed, x, y, 1.0, 4.0)

        circular_result = measure_star_shapes(circular, coords)
        trailed_result = measure_star_shapes(trailed, coords)

        self.assertEqual(circular_result["shape_star_count"], 9)
        self.assertGreater(circular_result["roundness"], 0.9)
        self.assertLess(trailed_result["roundness"], 0.6)
        self.assertGreater(trailed_result["elongation"], 1.5)

    def test_hot_pixel_and_edge_candidates_are_excluded_without_mutating_input(self) -> None:
        image = np.full((48, 48), 5.0, dtype=np.float32)
        _add_gaussian(image, 24, 24, 2.0, 2.0)
        _add_gaussian(image, 10, 10, 2.0, 2.0)
        image[9, 9] = 1.0e6
        image_before = image.copy()

        result = measure_star_shapes(
            image,
            np.asarray([[24, 24], [10, 10], [2, 2], [np.nan, 10]], dtype=np.float32),
            radius=100,
        )

        self.assertEqual(result["shape_star_count"], 1)
        self.assertIsNotNone(result["roundness"])
        np.testing.assert_array_equal(image, image_before)

    def test_blank_nan_and_malformed_inputs_return_json_safe_empty_metrics(self) -> None:
        image = np.zeros((32, 32), dtype=np.float32)
        image[10:20, 10:20] = np.nan
        result = measure_star_shapes(image, np.asarray([[16, 16], [1, 1]]))

        self.assertEqual(result, {"roundness": None, "shape_star_count": 0, "shape_fwhm": None, "elongation": None})
        self.assertEqual(measure_star_shapes(image, np.empty((0, 2))), {"roundness": None, "shape_star_count": 0, "shape_fwhm": None, "elongation": None})
        with self.assertRaises(ValueError):
            measure_star_shapes(np.zeros((2, 2, 2)), np.empty((0, 2)))

    def test_sample_count_is_bounded(self) -> None:
        image = np.full((160, 160), 3.0, dtype=np.float32)
        coords = []
        for y in range(12, 148, 10):
            for x in range(12, 148, 10):
                _add_gaussian(image, x, y, 1.5, 1.5)
                coords.append((x, y))

        result = measure_star_shapes(image, np.asarray(coords, dtype=np.float32), radius=3)
        self.assertLessEqual(result["shape_star_count"], 64)

    def test_flow_preparation_does_not_retain_raw_full_resolution_array(self) -> None:
        # A 24 MP float32 frame would occupy 96 MB (about 91.6 MiB).  Prepared flow
        # frames only need phase_data, stars, and metadata after this point.
        import astroflow_logic as flow

        image = np.zeros((32, 32), dtype=np.float32)
        stars = np.asarray([[8, 8], [16, 8], [24, 8], [8, 16], [16, 16]], dtype=np.float32)
        detector_metrics = {"star_count": len(stars), "fwhm": 2.0, "valid": True}
        with patch.object(flow, "load_fits_data", return_value=(image, {})), patch.object(
            flow, "detect_stars", return_value=(stars, 2.0, detector_metrics)
        ):
            _, prepared = flow._process_single_frame(
                Path("synthetic.fits"), 2.0, 5.0, 20, 1, "DAO"
            )

        self.assertEqual(prepared["status"], "prepared")
        self.assertIsNone(prepared["data"])
        self.assertIsNotNone(prepared["phase_data"])


if __name__ == "__main__":
    unittest.main()
