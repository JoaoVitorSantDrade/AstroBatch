from __future__ import annotations

import tempfile
import threading
import unittest
from pathlib import Path

import numpy as np
from astropy.io import fits

import astrobatch.processing.calibration as calibration
from astrobatch.processing.cpu_kernels import calibrate_inplace, masked_extrema, warm_cpu_kernels
import astrobatch.processing.stacking as stacking


class CpuKernelExactnessTests(unittest.TestCase):
    def test_calibration_kernel_is_bitwise_identical(self) -> None:
        rng = np.random.default_rng(20260902)
        data = rng.normal(100, 10, size=(11, 13)).astype(np.float32)
        dark = rng.normal(2, 1, size=data.shape).astype(np.float32)
        flat = rng.uniform(0.02, 2, size=data.shape).astype(np.float32)
        flat[0, 0] = 0.0
        expected = data.copy()
        expected -= dark
        np.divide(expected, flat, out=expected, where=np.isfinite(flat) & (flat > 0.01))
        actual = calibrate_inplace(data.copy(), dark, flat)
        np.testing.assert_array_equal(actual, expected)

    def test_masked_extrema_is_bitwise_identical(self) -> None:
        values = np.asarray(
            [[[1.0, np.nan], [np.inf, -2.0]], [[2.0, 3.0], [4.0, -3.0]]],
            dtype=np.float32,
        )
        masks = np.asarray([[[1, 1], [1, 0]], [[1, 1], [0, 1]]], dtype=np.uint8)
        valid = masks.astype(bool) & ~np.isnan(values)
        np.testing.assert_array_equal(
            masked_extrema(values, masks, True), np.max(np.where(valid, values, -np.inf), axis=0)
        )
        np.testing.assert_array_equal(
            masked_extrema(values, masks, False), np.min(np.where(valid, values, np.inf), axis=0)
        )

    def test_warmup_compiles_kernels(self) -> None:
        warm_cpu_kernels()


class CalibrationBandTests(unittest.TestCase):
    def test_master_bands_match_full_stack_median(self) -> None:
        rng = np.random.default_rng(42)
        frames = [rng.normal(size=(9, 7)).astype(np.float32) for _ in range(5)]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for index, frame in enumerate(frames):
                fits.PrimaryHDU(frame).writeto(root / f"frame_{index}.fits")
            output = root / "master.fits"
            actual = calibration.make_master(root, output, lambda _: None, threading.Event())
            expected = np.median(np.stack(frames, axis=0), axis=0).astype(np.float32)
            np.testing.assert_array_equal(actual, expected)
            with fits.open(output) as hdul:
                self.assertEqual(hdul[0].data.dtype.kind, "u")
                self.assertEqual(hdul[0].data.dtype.itemsize, 2)
                self.assertTrue(hdul[0].header["CALNORM"])


class GpuRemovalTests(unittest.TestCase):
    def test_legacy_gpu_config_is_ignored(self) -> None:
        config = stacking._build_config_from_dict(Path("input"), {"use_gpu": True})
        self.assertFalse(hasattr(config, "use_gpu"))
