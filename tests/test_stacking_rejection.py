from __future__ import annotations

import time
import unittest
import warnings
from unittest import mock

import numpy as np

import astrobatch.processing.stacking as stacking


def _reference_reject(
    values: np.ndarray,
    masks: np.ndarray,
    method: str,
    low: float,
    high: float,
) -> np.ndarray:
    masked = values.astype(np.float32, copy=True)
    masked[~masks] = np.nan
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        if method == "SigmaClip":
            center = np.nanmedian(masked, axis=0)
            std = np.nanstd(masked, axis=0)
            valid = (masked >= center - low * std) & (masked <= center + high * std)
            valid |= ~(np.isfinite(std) & (std > 1e-10))
            masked[~valid] = np.nan
        elif method == "MAD":
            center = np.nanmedian(masked, axis=0)
            mad = np.nanmedian(np.abs(masked - center), axis=0)
            sigma = np.float32(1.4826) * mad
            valid = (masked >= center - low * sigma) & (masked <= center + high * sigma)
            valid |= ~(np.isfinite(mad) & (mad > 1e-10))
            masked[~valid] = np.nan
        elif method == "Winsorized":
            p_low = np.nanpercentile(masked, min(49.0, max(0.0, low * 10.0)), axis=0)
            p_high = np.nanpercentile(
                masked,
                max(51.0, min(100.0, 100.0 - high * 10.0)),
                axis=0,
            )
            valid_range = np.isfinite(p_low) & np.isfinite(p_high) & (p_high >= p_low)
            masked = np.where(valid_range, np.clip(masked, p_low, p_high), masked)
        return np.asarray(np.nanmedian(masked, axis=0), dtype=np.float32)


class RejectionPerformanceTests(unittest.TestCase):
    def setUp(self) -> None:
        rng = np.random.default_rng(42)
        self.values = rng.normal(1000, 15, size=(15, 96, 96)).astype(np.float32)
        self.values[0, 10:20, 10:20] += 800
        self.valid = np.ones_like(self.values, dtype=bool)

    def test_all_valid_fast_path_matches_nan_reference(self) -> None:
        for method in ("SigmaClip", "MAD", "Winsorized"):
            with self.subTest(method=method):
                expected = _reference_reject(self.values, self.valid, method, 3.0, 3.0)
                actual = stacking._reject_cpu(self.values, self.valid, method, 3.0, 3.0)
                np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-5)

    def test_partial_mask_uses_existing_nan_semantics(self) -> None:
        masks = self.valid.copy()
        masks[:, :8, :8] = False
        expected = _reference_reject(self.values, masks, "SigmaClip", 3.0, 3.0)
        actual = stacking._reject_cpu(self.values, masks, "SigmaClip", 3.0, 3.0)
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-5)

    def test_winsorized_partial_mask_matches_nanpercentile(self) -> None:
        masks = self.valid.copy()
        masks[:, ::3, ::4] = False
        expected = _reference_reject(self.values, masks, "Winsorized", 3.0, 3.0)
        actual = stacking._reject_cpu(self.values, masks, "Winsorized", 3.0, 3.0)
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-4)

    def test_all_invalid_pixels_do_not_emit_warning(self) -> None:
        masks = np.zeros_like(self.valid)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = stacking._reject_cpu(self.values, masks, "SigmaClip", 3.0, 3.0)
        self.assertFalse(caught)
        self.assertTrue(np.isnan(result).all())

    def test_no_outlier_fast_path_avoids_nanmedian(self) -> None:
        values = np.arange(9 * 4 * 4, dtype=np.float32).reshape(9, 4, 4)
        masks = np.ones_like(values, dtype=bool)
        with mock.patch.object(stacking.np, "nanmedian", side_effect=AssertionError):
            result = stacking._reject_cpu(values, masks, "SigmaClip", 1_000.0, 1_000.0)
        np.testing.assert_array_equal(result, np.median(values, axis=0))

    def test_fast_path_is_no_slower_than_nan_reference(self) -> None:
        def elapsed(callback):
            started = time.perf_counter()
            callback()
            return time.perf_counter() - started

        stacking._reject_cpu(self.values, self.valid, "SigmaClip", 3.0, 3.0)
        fast = min(
            elapsed(lambda: stacking._reject_cpu(self.values, self.valid, "SigmaClip", 3.0, 3.0))
            for _ in range(3)
        )
        reference = min(
            elapsed(lambda: _reference_reject(self.values, self.valid, "SigmaClip", 3.0, 3.0))
            for _ in range(3)
        )
        self.assertLessEqual(fast, reference * 1.05)


if __name__ == "__main__":
    unittest.main()
