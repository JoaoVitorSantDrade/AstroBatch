from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from astropy.io import fits

import astroflow_logic as flow


class FlowFitsIoTests(unittest.TestCase):
    def test_raw_mmap_scaling_matches_astropy_physical_values(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            path = Path(temporary_dir) / "scaled.fits"
            image = fits.PrimaryHDU(
                np.arange(24, dtype=np.int16).reshape(4, 6)
            )
            image.header["BSCALE"] = 2.0
            image.header["BZERO"] = 100.0
            fits.HDUList([image]).writeto(path)

            actual, header = flow.load_fits_data(path)
            with fits.open(path, memmap=False, do_not_scale_image_data=False) as hdul:
                expected = np.asarray(hdul[0].data, dtype=np.float32).copy()

            self.assertEqual(actual.dtype, np.float32)
            self.assertEqual(header["BZERO"], 100.0)
            self.assertTrue(np.array_equal(actual, expected, equal_nan=True))

    def test_unscaled_float32_is_detached_from_source_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            path = Path(temporary_dir) / "float.fits"
            source = np.arange(16, dtype=np.float32).reshape(4, 4)
            fits.PrimaryHDU(source).writeto(path)

            actual, _ = flow.load_fits_data(path)
            self.assertEqual(actual.dtype, np.float32)
            self.assertTrue(np.array_equal(actual, source))
            self.assertTrue(actual.flags.owndata)


if __name__ == "__main__":
    unittest.main()
