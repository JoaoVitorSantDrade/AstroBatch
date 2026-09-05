from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
from astropy.io import fits

import astroalign_logic as align


class AlignQualityDeliveryTests(unittest.TestCase):
    def test_worker_quality_gate_accepts_correct_translation_and_rejects_identity(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); batch = root / "batch_001"; out = root / "out"
            batch.mkdir()
            anchor = np.zeros((700, 900), np.float32)
            anchor[180:195, 300:325] = 1000
            anchor[500:510, 700:715] = 500
            shifted = cv2.warpAffine(anchor, np.float32([[1, 0, 5], [0, 1, 0]]), (900, 700))
            fits.PrimaryHDU(anchor).writeto(batch / "anchor.fits")
            fits.PrimaryHDU(shifted).writeto(batch / "shifted.fits")
            ref = align.prepare_reference_preview(batch / "anchor.fits", np.eye(3), "bilinear")
            cfg = align._build_align_config(root, out, {"quality_gate": True, "overwrite": True})
            good = align._process_single_alignment("shifted.fits", {"matrix": [[1, 0, -5], [0, 1, 0], [0, 0, 1]]},
                batch, out, np.eye(3).tolist(), "bilinear", cfg, reference_preview=ref)
            self.assertIsNone(good[1])
            bad = align._process_single_alignment("shifted.fits", {"matrix": np.eye(3).tolist()},
                batch, out / "bad", np.eye(3).tolist(), "bilinear", cfg, reference_preview=ref)
            self.assertIsNotNone(bad[1])
    def test_correct_translation_has_small_residual_against_reference(self):
        ref = np.zeros((700, 900), np.float32)
        ref[180:190, 300:320] = 100
        moved = cv2.warpAffine(ref, np.float32([[1, 0, 0], [0, 1, 0]]), (900, 700))
        q = align.estimate_alignment_quality(ref, moved)
        self.assertLess(q["rms"], 0.2)

    def test_blank_frames_do_not_claim_confidence(self):
        blank = np.zeros((64, 64), np.float32)
        q = align.estimate_alignment_quality(blank, blank)
        self.assertLess(q["confidence"], 0.05)

    def test_reference_preview_is_bounded(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "large.fits"
            fits.PrimaryHDU(np.zeros((1200, 1400), dtype=np.float32)).writeto(path)
            preview, mask = align.prepare_reference_preview(path, np.eye(3), "bilinear")
            self.assertLessEqual(max(preview.shape[:2]), 512)
            self.assertEqual(preview.shape[:2], mask.shape)

    def test_zero_and_negative_pixels_remain_valid(self):
        data = np.array([[0.0, -2.0], [3.0, 4.0]], np.float32)
        mask = align.generate_valid_mask(data.shape, np.eye(3))
        self.assertEqual(int(mask.sum()), 4)

    def test_rgb_correction_sign(self):
        ref = np.zeros((96, 96), np.float32)
        ref[40:45, 50:55] = 1
        channel = np.roll(np.roll(ref, -2, axis=0), 3, axis=1)
        dx, dy, confidence = align.rgb_registration_shift(ref, channel)
        self.assertGreater(confidence, 0.05)
        self.assertAlmostEqual(dx, -3, delta=0.5)
        self.assertAlmostEqual(dy, 2, delta=0.5)


if __name__ == "__main__":
    unittest.main()
