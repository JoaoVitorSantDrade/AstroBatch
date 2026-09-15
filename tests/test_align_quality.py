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

    def test_quality_gate_reports_insufficient_overlap_from_graph_cache(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); batch = root / "batch_001"; out = root / "out"
            batch.mkdir()
            data = np.zeros((128, 128), np.float32)
            data[40:48, 50:58] = 1000
            fits.PrimaryHDU(data).writeto(batch / "frame.fits")
            cfg = align._build_align_config(root, out, {"quality_gate": True, "overwrite": True})
            empty_preview = (np.zeros((64, 64), np.float32), np.zeros((64, 64), np.uint8))
            result = align._process_single_alignment(
                "frame.fits",
                {"matrix": np.eye(3).tolist(), "_quality_reference_labels": ["parent"]},
                batch,
                out,
                np.eye(3).tolist(),
                "bilinear",
                cfg,
                reference_previews={"parent": empty_preview},
            )
            self.assertIn("insufficient_overlap", result[1])

    def test_quality_gate_never_accepts_a_frame_against_its_own_preview(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); batch = root / "batch_001"; out = root / "out"
            batch.mkdir()
            data = np.zeros((96, 96), np.float32)
            data[35:45, 42:52] = 1000
            fits.PrimaryHDU(data).writeto(batch / "frame.fits")
            preview = align.prepare_reference_preview(
                batch / "frame.fits", np.eye(3), "bilinear"
            )
            cfg = align._build_align_config(root, out, {"quality_gate": True, "overwrite": True})
            result = align._process_single_alignment(
                "frame.fits",
                {
                    "matrix": np.eye(3).tolist(),
                    "_quality_reference_labels": ["self"],
                    "_quality_frame_identity": ("batch_001", "frame.fits"),
                },
                batch, out, np.eye(3).tolist(), "bilinear", cfg,
                reference_previews={"self": preview},
                reference_preview_sources={"self": ("batch_001", "frame.fits")},
            )
            self.assertIn("unverified", result[1])

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

    def test_similarity_rgb_registration_corrects_scale_and_translation(self):
        reference = np.zeros((256, 256), np.float32)
        for x, y in ((40, 50), (100, 70), (170, 40), (220, 150),
                     (60, 200), (150, 210), (210, 220), (30, 150),
                     (120, 130), (190, 100)):
            cv2.circle(reference, (x, y), 3, 1000, -1)
        target_matrix = np.asarray(
            [[1.002, 0.0005, -1.2], [-0.0005, 1.002, 0.8]],
            dtype=np.float32,
        )
        channel = cv2.warpAffine(
            reference,
            cv2.invertAffineTransform(target_matrix),
            (256, 256),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
        )
        rgb = np.dstack((channel, reference, reference))
        diagnostics = {}
        corrected = align.warp_frame(
            rgb,
            np.eye(3),
            "bilinear",
            rgb_registration=True,
            engine_profile="Fast",
            diagnostics=diagnostics,
            rgb_registration_mode="similarity",
        )
        self.assertIn(0, diagnostics["rgb_models"])
        self.assertLess(
            float(np.mean((corrected[:, :, 0] - reference) ** 2)),
            float(np.mean((channel - reference) ** 2)) * 0.2,
        )


if __name__ == "__main__":
    unittest.main()
