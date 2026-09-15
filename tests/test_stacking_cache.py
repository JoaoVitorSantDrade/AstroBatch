from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
from astropy.io import fits

import stacking_logic as stacking


class FitsCacheTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.cache_dir = self.root / stacking.FITS_CACHE_DIR_NAME
        self.messages: list[str] = []

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _write_scaled_frame(self, path: Path, with_mask: bool = False) -> None:
        image = fits.PrimaryHDU(np.arange(16, dtype=np.int16).reshape(4, 4))
        image.header["BSCALE"] = 2.0
        image.header["BZERO"] = 100.0
        hdus: list[fits.hdu.base.ExtensionHDU | fits.PrimaryHDU] = [image]
        if with_mask:
            hdus.append(
                fits.ImageHDU(
                    np.ones((4, 4), dtype=np.int8), name="VALID_MASK"
                )
            )
        fits.HDUList(hdus).writeto(path)

    def _write_compressed_frame(self, path: Path, with_mask: bool = False) -> None:
        image = fits.CompImageHDU(
            np.arange(16, dtype=np.uint16).reshape(4, 4),
            compression_type="RICE_1",
        )
        hdus: list[fits.hdu.base.ExtensionHDU | fits.PrimaryHDU] = [
            fits.PrimaryHDU(),
            image,
        ]
        if with_mask:
            hdus.append(
                fits.CompImageHDU(
                    np.ones((4, 4), dtype=np.uint8),
                    name="VALID_MASK",
                    compression_type="PLIO_1",
                )
            )
        fits.HDUList(hdus).writeto(path)

    def _write_unscaled_frame(self, path: Path) -> None:
        fits.PrimaryHDU(np.arange(16, dtype=np.int16).reshape(4, 4)).writeto(path)

    def _frame_and_geometry(
        self, path: Path
    ) -> tuple[stacking.FrameInfo, stacking.FrameGeometry]:
        geometry = stacking.inspect_fits(path)
        return (
            stacking.FrameInfo(
                path=path,
                name=path.name,
                batch="root",
                metrics={},
                quality=1.0,
                star_count=1.0,
                fwhm=1.0,
                snr=1.0,
                rms=1.0,
                has_valid_mask=geometry.mask_hdu_index is not None,
                shape=(geometry.height, geometry.width),
                image_kind=geometry.image_kind,
                channels=geometry.channels,
            ),
            geometry,
        )

    def _cache_once(
        self, path: Path
    ) -> tuple[stacking.FrameInfo, stacking.FitsCacheStats]:
        frame, geometry = self._frame_and_geometry(path)
        _, stats = stacking.cache_decompressed_frames(
            [frame],
            {path: geometry},
            self.cache_dir,
            self.messages.append,
            None,
        )
        return frame, stats

    def test_second_identical_run_is_a_hit_without_rewrite(self) -> None:
        source = self.root / "compressed.fits"
        self._write_compressed_frame(source, with_mask=True)

        first_frame, first_stats = self._cache_once(source)
        cached_path = first_frame.path
        cached_mtime = cached_path.stat().st_mtime_ns

        second_frame, second_stats = self._cache_once(source)

        self.assertEqual(first_stats.rebuilt, 1)
        self.assertEqual(second_stats.hits, 1)
        self.assertEqual(second_stats.rebuilt, 0)
        self.assertEqual(second_frame.path, cached_path)
        self.assertEqual(cached_path.stat().st_mtime_ns, cached_mtime)
        self.assertTrue(second_frame.has_valid_mask)

    def test_changed_source_creates_a_new_cache_entry(self) -> None:
        source = self.root / "compressed.fits"
        self._write_compressed_frame(source)
        _, first_stats = self._cache_once(source)

        with fits.open(source, mode="update") as hdul:
            hdul[0].header["HISTORY"] = "cache identity changed"
        os.utime(source, None)

        _, second_stats = self._cache_once(source)

        self.assertEqual(first_stats.rebuilt, 1)
        self.assertEqual(second_stats.rebuilt, 1)
        self.assertEqual(len(list(self.cache_dir.glob("*.fits"))), 2)

    def test_corrupt_cache_is_rebuilt(self) -> None:
        source = self.root / "compressed.fits"
        self._write_compressed_frame(source)
        first_frame, _ = self._cache_once(source)
        first_frame.path.write_bytes(b"not a FITS file")

        second_frame, stats = self._cache_once(source)

        self.assertEqual(stats.rebuilt, 1)
        self.assertEqual(stats.hits, 0)
        self.assertEqual(stacking.inspect_fits(second_frame.path).height, 4)

    def test_scaled_uncompressed_frame_uses_raw_storage_cache(self) -> None:
        source = self.root / "scaled.fits"
        self._write_scaled_frame(source)

        frame, stats = self._cache_once(source)

        self.assertEqual(stats.scaled, 1)
        self.assertEqual(stats.rebuilt, 1)
        self.assertNotEqual(frame.path, source)
        self.assertTrue(stacking.inspect_fits(source).requires_scaling)
        cached_geometry = stacking.inspect_fits(frame.path)
        self.assertTrue(cached_geometry.cache_raw_storage)
        self.assertFalse(cached_geometry.requires_scaling)

        with fits.open(
            source,
            memmap=False,
            do_not_scale_image_data=False,
        ) as original_hdul:
            original = np.array(original_hdul[0].data, dtype=np.float32, copy=True)
        with fits.open(
            frame.path,
            memmap=True,
            do_not_scale_image_data=True,
        ) as cached_hdul:
            raw = np.array(cached_hdul[0].data, copy=True)
        restored = stacking._restore_cached_physical_values(raw, cached_geometry)
        self.assertTrue(np.array_equal(original, restored))

    def test_streaming_raw_scaling_matches_astropy_without_cache(self) -> None:
        source = self.root / "scaled-stream.fits"
        self._write_scaled_frame(source)
        geometry = stacking.inspect_fits(source)

        with stacking._open_streaming_fits(source, geometry) as hdul:
            raw = stacking._read_hdu_section(
                hdul[geometry.hdu_index],
                (slice(1, 3), slice(1, 4)),
            )
            restored = stacking._restore_streaming_physical_values(raw, geometry)
        with fits.open(source, memmap=False, do_not_scale_image_data=False) as hdul:
            expected = np.asarray(hdul[geometry.hdu_index].data[1:3, 1:4], dtype=np.float32)

        self.assertTrue(np.array_equal(restored, expected, equal_nan=True))

    def test_mixed_sources_cache_scaled_and_compressed_science_frames(self) -> None:
        compressed = self.root / "compressed.fits"
        uncompressed = self.root / "scaled.fits"
        self._write_compressed_frame(compressed)
        self._write_scaled_frame(uncompressed)
        compressed_frame, compressed_geometry = self._frame_and_geometry(compressed)
        uncompressed_frame, uncompressed_geometry = self._frame_and_geometry(uncompressed)

        _, stats = stacking.cache_decompressed_frames(
            [compressed_frame, uncompressed_frame],
            {compressed: compressed_geometry, uncompressed: uncompressed_geometry},
            self.cache_dir,
            self.messages.append,
            None,
        )

        self.assertEqual(stats.rebuilt, 2)
        self.assertEqual(stats.scaled, 1)
        self.assertNotEqual(compressed_frame.path, compressed)
        self.assertNotEqual(uncompressed_frame.path, uncompressed)

    def test_discovery_ignores_input_sidecar_cache(self) -> None:
        source = self.root / "frame.fits"
        self._write_unscaled_frame(source)
        self.cache_dir.mkdir()
        self._write_unscaled_frame(self.cache_dir / "cached.fits")
        (self.cache_dir / ".partial.fits.tmp").write_bytes(b"temporary")

        frames, metadata = stacking.discover_aligned_frames(self.root)

        self.assertEqual(frames, [source])
        self.assertEqual(len(metadata), 1)
        self.assertEqual(next(iter(metadata.values()))["frame_count"], 1)


if __name__ == "__main__":
    unittest.main()
