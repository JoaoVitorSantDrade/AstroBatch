from __future__ import annotations

import queue
import tempfile
import unittest
from pathlib import Path

import numpy as np
from astropy.io import fits

import astrobatch.processing.stacking as stacking


class SubstackProgressTests(unittest.TestCase):
    def test_substack_reports_open_read_combine_and_completed_band(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            paths: list[Path] = []
            for index in range(3):
                path = root / f"frame_{index}.fits"
                fits.PrimaryHDU(np.full((8, 8), index, dtype=np.float32)).writeto(path)
                paths.append(path)

            geometries = {path: stacking.inspect_fits(path) for path in paths}
            frames = [
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
                    has_valid_mask=False,
                    shape=(8, 8),
                    image_kind="Mono",
                    channels=1,
                )
                for path in paths
            ]
            events: queue.SimpleQueue[tuple[int, int, str]] = queue.SimpleQueue()

            stacking._process_substack(
                frames,
                geometries,
                [1.0] * len(frames),
                [None] * len(frames),
                stacking.StackingConfig(
                    method="Mean",
                    rejection_method="None",
                    workers=1,
                ),
                root / "leaf.fits",
                8,
                8,
                1,
                None,
                leaf_index=0,
                progress_queue=events,
            )

            messages: list[str] = []
            completed = 0
            while True:
                try:
                    _, increment, message = events.get_nowait()
                except queue.Empty:
                    break
                completed += increment
                messages.append(message)

            self.assertTrue(any("opening" in message for message in messages))
            self.assertTrue(any("reading band" in message for message in messages))
            self.assertTrue(any("combining band" in message for message in messages))
            self.assertTrue(any("completed band" in message for message in messages))
            self.assertEqual(completed, 1)


if __name__ == "__main__":
    unittest.main()
