from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

import numpy as np
from astropy.io import fits

from astrobatch.core.resources import estimate_resources
from astrobatch.core.fits import inspect_fits


class V2ResourceTests(unittest.TestCase):
    def test_estimate_always_chooses_a_bounded_row_count(self) -> None:
        estimate = estimate_resources(200, 6000, 4000, budget_mb=1024)
        self.assertGreaterEqual(estimate.safe_chunk_rows, 32)
        self.assertLessEqual(estimate.safe_chunk_rows, 4000)
        self.assertLessEqual(estimate.budget_mb, estimate.available_mb * 3 // 4)

    def test_import_fits_inspection_uses_portable_non_memmapped_read(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_dir:
            path = Path(temporary_dir) / "source.fits"
            fits.PrimaryHDU(np.ones((4, 5), dtype=np.uint16)).writeto(path)
            shape, _ = inspect_fits(path)
            self.assertEqual(shape, (4, 5))
