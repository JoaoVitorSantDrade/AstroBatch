from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
from astropy.io import fits

import stacking_logic as stacking


class RamStackTests(unittest.TestCase):
    def _write_inputs(self, root: Path, count: int = 12) -> Path:
        input_dir = root / "input"
        input_dir.mkdir()
        yy, xx = np.indices((40, 48))
        base = (xx * 17 + yy * 11).astype(np.float32)
        for index in range(count):
            data = base + np.float32(index)
            fits.PrimaryHDU(data).writeto(input_dir / f"frame_{index:03d}.fits")
        return input_dir

    def _config(self, input_dir: Path, output_dir: Path, **kwargs):
        values = dict(
            base_dir=input_dir,
            input_dir=input_dir,
            output_dir=output_dir,
            selection_mode="All",
            rejection_method="None",
            method="Median",
            workers=1,
            memory_budget_mb=1024,
            cache_decompressed_fits=True,
            engine_profile="Stable",
        )
        values.update(kwargs)
        return stacking.StackingConfig(**values)

    def test_ram_mode_matches_legacy_stable_product_and_does_not_cache_inputs(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            input_dir = self._write_inputs(root)
            disk = stacking.process_stacking(self._config(input_dir, root / "disk"))
            ram = stacking.process_stacking(
                self._config(input_dir, root / "ram", reduction_storage="ram")
            )
            self.assertEqual(disk["status"], "success")
            self.assertEqual(ram["status"], "success")
            with fits.open(disk["output_path"]) as expected, fits.open(ram["output_path"]) as actual:
                np.testing.assert_array_equal(expected[0].data, actual[0].data)
                np.testing.assert_array_equal(expected["VALID_MASK"].data, actual["VALID_MASK"].data)
            self.assertFalse((input_dir / ".astrostack_fits_cache").exists())
            self.assertEqual(ram["reduction_storage"], "ram")

    def test_ram_mode_matches_legacy_rgb_mean_rounding(self):
        """The RAM tree must retain Stable's coverage-weighted RGB mean."""

        with TemporaryDirectory() as directory:
            root = Path(directory)
            input_dir = root / "rgb_input"
            input_dir.mkdir()
            yy, xx = np.indices((40, 48))
            base = (xx * 2 + yy).astype(np.float32)
            for index in range(9):
                data = np.stack(
                    (base + index, base + 2 * index, base + 3 * index)
                ).astype(np.float32)
                fits.PrimaryHDU(data).writeto(input_dir / f"frame_{index:03d}.fits")

            values = self._config(
                input_dir,
                root / "disk_rgb",
                method="Mean",
                rejection_method="None",
            )
            disk = stacking.process_stacking(values)
            values.output_dir = root / "ram_rgb"
            values.reduction_storage = "ram"
            ram = stacking.process_stacking(values)
            self.assertEqual(disk["status"], "success")
            self.assertEqual(ram["status"], "success")
            with fits.open(disk["output_path"]) as expected, fits.open(ram["output_path"]) as actual:
                np.testing.assert_array_equal(expected[0].data, actual[0].data)
                np.testing.assert_array_equal(expected["VALID_MASK"].data, actual["VALID_MASK"].data)

    def test_ram_intelligent_stable_preserves_legacy_rejection_order(self):
        """Changing storage must not change Stable rejection topology."""

        with TemporaryDirectory() as directory:
            root = Path(directory)
            input_dir = root / "intelligent_input"
            input_dir.mkdir()
            rng = np.random.default_rng(20260915)
            height, width = 24, 28
            for index in range(17):
                data = rng.normal(100.0 + index * 0.5, 8.0, (height, width)).astype(np.float32)
                if index % 4 == 0:
                    data[5:9, 7:11] += np.float32(1500.0)
                mask = np.ones((height, width), dtype=np.uint8)
                mask[:, (index * 3) % width : ((index * 3) % width) + 2] = 0
                fits.HDUList([
                    fits.PrimaryHDU(data),
                    fits.ImageHDU(mask, name="VALID_MASK"),
                ]).writeto(input_dir / f"frame_{index:03d}.fits")

            common = self._config(
                input_dir,
                root / "disk_intelligent",
                selection_mode="All",
                method="Median",
                rejection_method="SigmaClip",
                normalize=False,
                feature_profile="Intelligent",
                trail_policy="report",
            )
            disk = stacking.process_stacking(common)
            common.output_dir = root / "ram_intelligent"
            common.reduction_storage = "ram"
            ram = stacking.process_stacking(common)
            self.assertEqual(disk["status"], "success")
            self.assertEqual(ram["status"], "success")
            with fits.open(disk["output_path"], memmap=False) as expected, fits.open(ram["output_path"], memmap=False) as actual:
                np.testing.assert_array_equal(expected[0].data, actual[0].data)
                np.testing.assert_array_equal(expected["VALID_MASK"].data, actual["VALID_MASK"].data)

    def test_ram_stable_three_leaf_remainder_keeps_left_to_right_tree(self):
        """The non-power-of-two carry remainder must stay on the right."""

        with TemporaryDirectory() as directory:
            root = Path(directory)
            input_dir = root / "three_leaf_input"
            input_dir.mkdir()
            rng = np.random.default_rng(20260916)
            for index in range(3):
                data = rng.normal(100.0, 12.0, (32, 36)).astype(np.float32)
                if index == 2:
                    data[8:14, 9:15] += np.float32(500.0)
                fits.PrimaryHDU(data).writeto(input_dir / f"frame_{index:03d}.fits")

            common = self._config(
                input_dir,
                root / "disk_three_leaf",
                method="Median",
                rejection_method="SigmaClip",
                normalize=False,
                feature_profile="Intelligent",
                trail_policy="report",
            )
            disk = stacking.process_stacking(common)
            common.output_dir = root / "ram_three_leaf"
            common.reduction_storage = "ram"
            ram = stacking.process_stacking(common)
            self.assertEqual(disk["status"], "success")
            self.assertEqual(ram["status"], "success")
            with fits.open(disk["output_path"], memmap=False) as expected, fits.open(ram["output_path"], memmap=False) as actual:
                np.testing.assert_array_equal(expected[0].data, actual[0].data)
                np.testing.assert_array_equal(expected["VALID_MASK"].data, actual["VALID_MASK"].data)

    def test_explicit_spill_is_bounded_and_cleaned(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            input_dir = self._write_inputs(root, count=8)
            spill = root / "spill"
            result = stacking.process_stacking(
                self._config(
                    input_dir,
                    root / "out",
                    reduction_storage="ram_spill",
                    spill_directory=spill,
                    spill_limit_mb=256,
                )
            )
            self.assertEqual(result["status"], "success")
            self.assertGreater(result["spill_bytes_written"], 0)
            self.assertFalse(list(spill.glob(".astrostack-spill-*")))

    def test_quality_weighted_mean_retains_uint16_and_weight_telemetry(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            input_dir = self._write_inputs(root, count=6)
            result = stacking.process_stacking(
                self._config(
                    input_dir,
                    root / "out",
                    reduction_storage="ram",
                    method="QualityWeightedMean",
                    feature_profile="Intelligent",
                    selection_mode="MultiMetric",
                )
            )
            self.assertEqual(result["status"], "success")
            self.assertTrue(result["quality_weighted"])
            with fits.open(result["output_path"]) as hdul:
                self.assertEqual(hdul[0].data.dtype, np.dtype("uint16"))


if __name__ == "__main__":
    unittest.main()
