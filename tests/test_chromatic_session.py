from __future__ import annotations

import unittest

import cv2
import numpy as np

import astroalign_logic as align


class ChromaticSessionModelTests(unittest.TestCase):
    def _sample(self, seed: int):
        rng = np.random.default_rng(seed)
        yy, xx = np.indices((160, 160))
        base = np.zeros((160, 160), dtype=np.float32)
        for x, y in ((18, 22), (62, 34), (110, 28), (137, 74), (32, 106), (84, 120), (124, 140), (145, 115)):
            base += np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * 2.2**2)).astype(np.float32) * 1000
        red = cv2.warpAffine(base, np.asarray([[1, 0, -0.35], [0, 1, 0.45]], np.float32), (160, 160))
        blue = cv2.warpAffine(base, np.asarray([[1, 0, 0.55], [0, 1, -0.25]], np.float32), (160, 160))
        noise = lambda: rng.normal(0, 0.5, base.shape).astype(np.float32)
        return base + noise(), red + noise(), blue + noise()

    def test_global_model_aggregates_channels_and_is_applied(self):
        model = align.fit_chromatic_session_model((self._sample(seed) for seed in range(4)), min_samples=3)
        self.assertIsNotNone(model)
        self.assertTrue(model["accepted"])
        self.assertEqual(set(model["channels"]), {"0", "2"})
        sample = self._sample(99)
        rgb = np.dstack((sample[1], sample[0], sample[2]))
        diagnostics = {}
        corrected = align.warp_frame(
            rgb,
            np.eye(3),
            "bilinear",
            rgb_registration=True,
            engine_profile="Fast",
            diagnostics=diagnostics,
            rgb_registration_mode="session-auto",
            rgb_session_model=model,
        )
        self.assertEqual(corrected.shape, rgb.shape)
        self.assertEqual(diagnostics["rgb_models"][0]["source"], "session")

    def test_invalid_model_falls_back_to_hybrid_without_raising(self):
        sample = self._sample(1)
        rgb = np.dstack((sample[1], sample[0], sample[2]))
        corrected = align.warp_frame(
            rgb,
            np.eye(3),
            "bilinear",
            rgb_registration=True,
            engine_profile="Fast",
            rgb_registration_mode="session-auto",
            rgb_session_model={"accepted": False},
        )
        self.assertEqual(corrected.shape, rgb.shape)


if __name__ == "__main__":
    unittest.main()
