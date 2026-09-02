from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from astropy.io import fits

import astroalign_logic as align


class AstroAlignOutputTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_compressed_output_is_the_default_and_preserves_rgb_layout(self) -> None:
        data = np.arange(18, dtype=np.float32).reshape(2, 3, 3)
        mask = np.ones((2, 3), dtype=np.uint8)
        header = fits.Header({"OBJECT": "M42"})
        output = self.root / "compressed.fits"

        align.save_aligned_fits(data, mask, header, output)

        with fits.open(output) as hdul:
            self.assertIsInstance(hdul[1], fits.CompImageHDU)
            self.assertIsInstance(hdul[2], fits.CompImageHDU)
            self.assertEqual(hdul[1].compression_type, "RICE_1")
            self.assertEqual(hdul[2].compression_type, "PLIO_1")
            self.assertEqual(hdul[1].header["OBJECT"], "M42")
            np.testing.assert_array_equal(hdul[1].data, np.moveaxis(data, -1, 0))
            np.testing.assert_array_equal(hdul[2].data, mask)

    def test_uncompressed_output_uses_image_hdus(self) -> None:
        data = np.arange(12, dtype=np.float32).reshape(3, 4)
        mask = np.ones((3, 4), dtype=np.uint8)
        header = fits.Header({"OBJECT": "M31"})
        output = self.root / "uncompressed.fits"

        align.save_aligned_fits(data, mask, header, output, compress_output=False)

        with fits.open(output) as hdul:
            self.assertIsInstance(hdul[1], fits.ImageHDU)
            self.assertNotIsInstance(hdul[1], fits.CompImageHDU)
            self.assertIsInstance(hdul[2], fits.ImageHDU)
            self.assertNotIn("ZCMPTYPE", hdul[1].header)
            self.assertEqual(hdul[1].header["OBJECT"], "M31")
            np.testing.assert_array_equal(hdul[1].data, data.astype(np.uint16))
            np.testing.assert_array_equal(hdul[2].data, mask)

    def test_compress_output_configuration_defaults_to_true(self) -> None:
        config = align._build_align_config(self.root, self.root / "out", {})
        uncompressed = align._build_align_config(
            self.root,
            self.root / "out",
            {"compress_output": False},
        )

        self.assertTrue(config.compress_output)
        self.assertFalse(uncompressed.compress_output)


if __name__ == "__main__":
    unittest.main()
