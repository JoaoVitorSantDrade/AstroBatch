from __future__ import annotations

import tempfile
import threading
import unittest
from pathlib import Path

import numpy as np
from astropy.io import fits

import astrobatch.processing.calibration as calibration
import astrobatch.processing.stacking as stacking


class BitDepthPolicyTests(unittest.TestCase):
    def test_stack_defaults_to_16_bit_output(self) -> None:
        self.assertEqual(stacking.StackingConfig().output_bit_depth, "16-bit")
        config = stacking._build_config_from_dict(Path("input"), {})
        self.assertEqual(config.output_bit_depth, "16-bit")
        legacy = stacking._build_config_from_dict(
            Path("input"), {"output_bit_depth": "32-bit"}
        )
        self.assertEqual(legacy.output_bit_depth, "16-bit")

    def test_stack_16_bit_conversion_is_explicit(self) -> None:
        config = stacking.StackingConfig(output_bit_depth="16-bit")
        converted = stacking.convert_output_dtype(
            np.asarray([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32), config
        )
        self.assertEqual(converted.dtype, np.uint16)
        self.assertEqual(int(converted.min()), 0)
        self.assertEqual(int(converted.max()), 65535)

    def test_calibration_output_is_normalized_uint16(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "calibrated.fits"
            data = np.asarray([[-0.5, 1.25]], dtype=np.float32)
            calibration.save_uint16_fits(data, None, path, -0.5, 1.25)
            with fits.open(path) as hdul:
                self.assertEqual(hdul[0].data.dtype.kind, "u")
                self.assertEqual(hdul[0].data.dtype.itemsize, 2)
                np.testing.assert_array_equal(
                    hdul[0].data, np.asarray([[0, 65535]], dtype=np.uint16)
                )
                self.assertTrue(hdul[0].header["CALNORM"])
            np.testing.assert_allclose(
                calibration.load_calibration_master(path), data, rtol=0.0, atol=1e-7
            )

    def test_calibration_uses_one_range_for_all_lights(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            output = root / "output"
            source.mkdir()
            fits.PrimaryHDU(np.asarray([[10, 20]], dtype=np.uint16)).writeto(
                source / "light_1.fits"
            )
            fits.PrimaryHDU(np.asarray([[30, 40]], dtype=np.uint16)).writeto(
                source / "light_2.fits"
            )
            calibration.run_calibration_pipeline(
                {
                    "input_dir": str(source),
                    "output_dir": str(output),
                    "apply_dark": False,
                    "apply_flat": False,
                    "overwrite": True,
                },
                lambda _: None,
                lambda *_: None,
                threading.Event(),
            )
            with fits.open(output / "light_1.fits") as hdul:
                np.testing.assert_array_equal(
                    hdul[0].data, np.asarray([[0, 21845]], dtype=np.uint16)
                )
                self.assertEqual(hdul[0].header["CALMIN"], 10.0)
                self.assertEqual(hdul[0].header["CALMAX"], 40.0)
            with fits.open(output / "light_2.fits") as hdul:
                np.testing.assert_array_equal(
                    hdul[0].data, np.asarray([[43690, 65535]], dtype=np.uint16)
                )
