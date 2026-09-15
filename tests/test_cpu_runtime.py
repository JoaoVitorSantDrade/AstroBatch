from __future__ import annotations

import unittest
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from unittest.mock import patch

import numpy as np

from app.engines.execution import ExecutionBudget
from cpu_runtime import (
    avx2_available,
    configure_opencv_threads,
    configure_worker_runtime,
    numpy_cpu_features,
    recommended_numba_threads,
    simd_level,
)


class CPURuntimeTests(unittest.TestCase):
    def test_feature_dispatch_is_explicit_and_safe_for_fallback(self):
        self.assertTrue(avx2_available({"AVX2": True}))
        self.assertFalse(avx2_available({"AVX2": False}))
        self.assertEqual(simd_level({"AVX2": True}), "AVX2")
        self.assertEqual(simd_level({}), "baseline")

    def test_thread_ceiling_uses_eight_physical_cores_on_5800x_shape(self):
        with patch("cpu_runtime._cpu_counts", return_value=(16, 8)):
            self.assertEqual(recommended_numba_threads(), 8)
            self.assertEqual(recommended_numba_threads(3), 3)
            self.assertEqual(ExecutionBudget.for_pipeline(1).kernel_threads, 8)
            self.assertEqual(ExecutionBudget.for_pipeline(4).kernel_threads, 1)

    def test_runtime_feature_map_is_a_boolean_mapping(self):
        features = numpy_cpu_features()
        self.assertIsInstance(features, dict)
        self.assertTrue(all(isinstance(key, str) and isinstance(value, bool) for key, value in features.items()))

    def test_budget_diagnostics_make_oversubscription_policy_visible(self):
        budget = ExecutionBudget.for_pipeline(1)
        self.assertEqual(
            set(budget.as_dict()),
            {"worker_count", "kernel_parallel", "kernel_threads", "max_in_flight"},
        )
        self.assertEqual(budget.as_dict()["worker_count"], 1)
        self.assertTrue(budget.as_dict()["kernel_parallel"])

    def test_parallel_extrema_and_sum_keep_serial_pixel_results(self):
        from cpu_kernels import _masked_extrema_parallel, masked_extrema, masked_sum_count

        rng = np.random.default_rng(17)
        values = rng.normal(size=(4, 512, 512)).astype(np.float32)
        values[0, 0, 0] = np.nan
        masks = (rng.random(values.shape) > 0.1).astype(np.uint8)
        serial_max = masked_extrema(values, masks, True)
        parallel_max = _masked_extrema_parallel(
            np.ascontiguousarray(values), np.ascontiguousarray(masks), True
        )
        self.assertEqual(serial_max.tobytes(), parallel_max.tobytes())
        serial_sum, serial_count = masked_sum_count(values, masks, parallel=False)
        parallel_sum, parallel_count = masked_sum_count(values, masks, parallel=True)
        self.assertEqual(serial_sum.tobytes(), parallel_sum.tobytes())
        self.assertEqual(serial_count.tobytes(), parallel_count.tobytes())

    def test_small_parallel_requests_use_serial_path_without_changing_results(self):
        from cpu_kernels import masked_sum_count, weighted_merge

        rng = np.random.default_rng(91)
        values = rng.normal(size=(3, 16, 16)).astype(np.float32)
        masks = (rng.random(values.shape) > 0.2).astype(np.uint8)
        serial_sum, serial_count = masked_sum_count(values, masks, parallel=False)
        requested_sum, requested_count = masked_sum_count(values, masks, parallel=True)
        self.assertEqual(serial_sum.tobytes(), requested_sum.tobytes())
        self.assertEqual(serial_count.tobytes(), requested_count.tobytes())

        leaves = rng.normal(size=(2, 16, 16)).astype(np.float32)
        counts = rng.integers(0, 4, size=leaves.shape, dtype=np.uint32)
        first = weighted_merge(leaves, counts)
        second = weighted_merge(leaves, counts)
        self.assertEqual(first.tobytes(), second.tobytes())

    def test_executor_initializer_applies_numba_mask_inside_worker(self):
        import numba

        with ThreadPoolExecutor(
            max_workers=2,
            initializer=partial(configure_worker_runtime, 1),
        ) as executor:
            observed = list(executor.map(lambda _: numba.get_num_threads(), range(4)))
        self.assertEqual(observed, [1, 1, 1, 1])

    def test_opencv_pool_can_be_bounded_and_restored(self):
        import cv2

        previous = int(cv2.getNumThreads())
        try:
            self.assertEqual(configure_opencv_threads(1), 1)
        finally:
            cv2.setNumThreads(previous)


if __name__ == "__main__":
    unittest.main()
