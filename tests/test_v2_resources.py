from __future__ import annotations

import unittest

from astrobatch.core.resources import estimate_resources


class V2ResourceTests(unittest.TestCase):
    def test_estimate_always_chooses_a_bounded_row_count(self) -> None:
        estimate = estimate_resources(200, 6000, 4000, budget_mb=1024)
        self.assertGreaterEqual(estimate.safe_chunk_rows, 32)
        self.assertLessEqual(estimate.safe_chunk_rows, 4000)
        self.assertLessEqual(estimate.budget_mb, estimate.available_mb * 3 // 4)
