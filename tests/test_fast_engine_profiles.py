from __future__ import annotations

import unittest

import numpy as np

import astroalign_logic as align
import stacking_logic as stacking


class FastEngineProfileTests(unittest.TestCase):
    def test_fast_lanczos_warp_keeps_shape_dtype_and_border_contract(self) -> None:
        image = np.arange(64, dtype=np.float32).reshape(8, 8)
        matrix = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        output = align.warp_frame(image, matrix, "lanczos", engine_profile="Fast")
        self.assertEqual(output.shape, image.shape)
        self.assertEqual(output.dtype, np.float32)
        self.assertTrue(np.allclose(output[:, 0], 0.0))

    def test_fast_mean_reducer_matches_stable_within_float32_tolerance(self) -> None:
        rng = np.random.default_rng(7)
        values = rng.normal(size=(7, 32, 32)).astype(np.float32)
        masks = rng.random(values.shape) > 0.2
        stable = stacking.reject_and_combine_block(
            values, masks, "Mean", "None", 3.0, 3.0, engine_profile="Stable"
        )
        fast = stacking.reject_and_combine_block(
            values, masks, "Mean", "None", 3.0, 3.0, engine_profile="Fast", kernel_parallel=True
        )
        np.testing.assert_allclose(fast, stable, rtol=2e-6, atol=2e-6, equal_nan=True)


if __name__ == "__main__":
    unittest.main()
