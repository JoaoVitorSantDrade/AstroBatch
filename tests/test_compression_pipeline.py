from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from astropy.io import fits

import astroalign_logic as align
import stacking_logic as stacking


class CompressionPipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _write_aligned_inputs(self, input_dir: Path, compress_output: bool) -> None:
        input_dir.mkdir()
        for index in range(3):
            data = np.full((4, 4), 1000 + index, dtype=np.float32)
            align.save_aligned_fits(
                data,
                np.ones((4, 4), dtype=np.uint8),
                fits.Header({"OBJECT": "M42"}),
                input_dir / f"frame_{index}.fits",
                compress_output=compress_output,
            )

    def _stack(self, input_dir: Path, output_dir: Path) -> np.ndarray:
        result = stacking.process_stacking(
            stacking.StackingConfig(
                input_dir=input_dir,
                output_dir=output_dir,
                selection_mode="All",
                method="Mean",
                rejection_method="None",
                normalize=False,
                compress_output=False,
                workers=1,
            )
        )
        self.assertEqual(result["status"], "success")
        with fits.open(result["output_path"]) as hdul:
            return np.asarray(hdul[0].data, dtype=np.float32)

    def test_compressed_and_uncompressed_aligned_inputs_stack_equivalently(self) -> None:
        compressed_input = self.root / "compressed"
        uncompressed_input = self.root / "uncompressed"
        self._write_aligned_inputs(compressed_input, compress_output=True)
        self._write_aligned_inputs(uncompressed_input, compress_output=False)

        compressed_result = self._stack(compressed_input, self.root / "compressed_out")
        uncompressed_result = self._stack(
            uncompressed_input,
            self.root / "uncompressed_out",
        )

        np.testing.assert_allclose(compressed_result, uncompressed_result)
        self.assertTrue((compressed_input / stacking.FITS_CACHE_DIR_NAME).is_dir())
        self.assertFalse((uncompressed_input / stacking.FITS_CACHE_DIR_NAME).exists())

    def test_rgb_median_stack_reduces_substacks_channel_by_channel(self) -> None:
        input_dir = self.root / "rgb"
        input_dir.mkdir()
        for index in range(3):
            data = np.empty((4, 4, 3), dtype=np.float32)
            data[..., 0] = 100 + index
            data[..., 1] = 200 + index
            data[..., 2] = 300 + index
            align.save_aligned_fits(
                data,
                np.ones((4, 4), dtype=np.uint8),
                fits.Header({"OBJECT": "M42"}),
                input_dir / f"rgb_{index}.fits",
                compress_output=False,
            )

        result = stacking.process_stacking(
            stacking.StackingConfig(
                input_dir=input_dir,
                output_dir=self.root / "rgb_out",
                selection_mode="All",
                method="Median",
                rejection_method="SigmaClip",
                normalize=False,
                workers=1,
            )
        )

        self.assertEqual(result["status"], "success")
        self.assertEqual(result["shape"], (3, 4, 4))


if __name__ == "__main__":
    unittest.main()
