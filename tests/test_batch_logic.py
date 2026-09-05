from __future__ import annotations

import math
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

import batch_logic as batch


def _reference_score(current: np.ndarray, previous: np.ndarray) -> float:
    """The pre-optimization scorer, retained as a focused test oracle."""
    diff = cv2.subtract(current, previous)
    if np.isnan(diff).any() or np.isinf(diff).any():
        valid = np.isfinite(diff)
        if not valid.any():
            return math.nan
        diff = diff[valid]
    diff -= np.mean(diff)
    return float(np.linalg.norm(diff) / math.sqrt(diff.size))


class ComparisonScoreTests(unittest.TestCase):
    def test_finite_score_matches_reference_without_mutating_inputs(self) -> None:
        rng = np.random.default_rng(20260904)
        current = rng.normal(1000, 20, size=(64, 71)).astype(np.float32)
        previous = rng.normal(1000, 20, size=current.shape).astype(np.float32)
        current_before = current.copy()
        previous_before = previous.copy()

        actual = batch.comparison_score(current, previous)

        self.assertAlmostEqual(actual, _reference_score(current, previous), places=6)
        np.testing.assert_array_equal(current, current_before)
        np.testing.assert_array_equal(previous, previous_before)

    def test_mixed_invalid_score_matches_reference(self) -> None:
        rng = np.random.default_rng(20260904)
        current = rng.normal(size=(512, 512)).astype(np.float32)
        previous = rng.normal(size=current.shape).astype(np.float32)
        current[::37, ::43] = np.nan
        previous[::53, ::47] = np.inf

        actual = batch.comparison_score(current, previous)
        expected = _reference_score(current, previous)

        self.assertTrue(np.isclose(actual, expected, rtol=2e-6, atol=2e-6))

    def test_all_invalid_score_is_nan(self) -> None:
        current = np.full((8, 9), np.nan, dtype=np.float32)
        previous = np.full((8, 9), np.inf, dtype=np.float32)

        self.assertTrue(math.isnan(batch.comparison_score(current, previous)))

    def test_empty_score_is_nan(self) -> None:
        current = np.empty((0, 4), dtype=np.float32)
        previous = np.empty((0, 4), dtype=np.float32)

        self.assertTrue(math.isnan(batch.comparison_score(current, previous)))

    def test_shape_mismatch_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            batch.comparison_score(
                np.zeros((4, 5), dtype=np.float32),
                np.zeros((4, 6), dtype=np.float32),
            )

    def test_subtraction_overflow_becomes_all_invalid(self) -> None:
        maximum = np.finfo(np.float32).max
        current = np.full((4, 4), maximum, dtype=np.float32)
        previous = np.full((4, 4), -maximum, dtype=np.float32)

        with np.errstate(over="ignore", invalid="ignore"):
            actual = batch.comparison_score(current, previous)

        self.assertTrue(math.isnan(actual))

    def test_large_dc_offset_uses_stable_variance(self):
        current = np.full((1024, 1024), 1e9, dtype=np.float32)
        current[::2] += 64
        previous = np.zeros_like(current)
        expected = float(np.std(current, dtype=np.float64))
        self.assertAlmostEqual(batch.comparison_score(current, previous), expected, places=6)

    def test_large_scores_agree_with_high_precision_reference(self):
        rng = np.random.default_rng(42)
        current = rng.normal(1000, 20, (1024, 1024)).astype(np.float32)
        previous = rng.normal(1000, 20, current.shape).astype(np.float32)
        expected = float(np.std(current - previous, dtype=np.float64))
        self.assertAlmostEqual(batch.comparison_score(current, previous), expected, places=6)


class BatchTransferTests(unittest.TestCase):
    def test_failed_overwrite_preserves_both_files_and_cleans_temporary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, destination = root / "source.fits", root / "destination.fits"
            source.write_bytes(b"new")
            destination.write_bytes(b"original")
            def partial_copy(src, dst):
                Path(dst).write_bytes(b"partial")
                raise OSError("disk full")
            with patch.object(batch.shutil, "copy2", side_effect=partial_copy):
                with self.assertRaises(OSError):
                    batch.transfer_batch_file(source, destination, "move", True)
            self.assertEqual(source.read_bytes(), b"new")
            self.assertEqual(destination.read_bytes(), b"original")
            self.assertEqual(len(list(root.iterdir())), 2)

    def test_successful_overwrite_copy_and_move(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for action in ("copy", "move"):
                source, destination = root / "source.fits", root / "destination.fits"
                source.write_bytes(b"new")
                destination.write_bytes(b"old")
                batch.transfer_batch_file(source, destination, action, True)
                self.assertEqual(destination.read_bytes(), b"new")
                self.assertEqual(source.exists(), action == "copy")

    def test_same_file_and_disabled_overwrite_preserve_data(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, destination = root / "source.fits", root / "destination.fits"
            source.write_bytes(b"new")
            destination.write_bytes(b"old")
            with self.assertRaises(ValueError):
                batch.transfer_batch_file(source, source, "move", True)
            with self.assertRaises(FileExistsError):
                batch.transfer_batch_file(source, destination, "copy", False)
            self.assertEqual(destination.read_bytes(), b"old")


if __name__ == "__main__":
    unittest.main()
