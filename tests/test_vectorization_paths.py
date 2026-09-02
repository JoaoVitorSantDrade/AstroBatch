from __future__ import annotations

import unittest
import warnings

import numpy as np

import astrobatch.processing.flow as flow
import astrobatch.processing.stacking as stacking

try:
    import astrobatch.processing.align as align
except ModuleNotFoundError:
    align = None


def _reference_matches(previous, current, shift, radius):
    """Pre-vectorization greedy matching rule."""
    from scipy.spatial import KDTree

    distances, indices = KDTree(previous).query(
        current + np.asarray(shift, dtype=np.float32), distance_upper_bound=radius
    )
    candidates = sorted(
        (float(distance), int(previous_index), current_index)
        for current_index, (distance, previous_index) in enumerate(zip(distances, indices))
        if np.isfinite(distance) and previous_index < len(previous)
    )
    used, reference, target = set(), [], []
    for _, previous_index, current_index in candidates:
        if previous_index not in used:
            used.add(previous_index)
            reference.append(previous[previous_index])
            target.append(current[current_index])
    return np.asarray(reference, dtype=np.float32), np.asarray(target, dtype=np.float32)


class AstroFlowVectorizationTests(unittest.TestCase):
    def test_incremental_matching_preserves_greedy_order(self) -> None:
        previous = np.asarray([[0, 0], [10, 0], [20, 0], [30, 0]], dtype=np.float32)
        current = np.asarray([[0.1, 0], [0.2, 0], [10.1, 0], [20.1, 0]], dtype=np.float32)
        expected = _reference_matches(previous, current, (0, 0), 2.0)
        actual = flow._match_incremental_stars(previous, current, (0, 0), 2.0)
        np.testing.assert_array_equal(actual[0], expected[0])
        np.testing.assert_array_equal(actual[1], expected[1])

    def test_quad_hash_is_rotation_and_scale_invariant(self) -> None:
        points = np.asarray([[0, 0], [3, 1], [1, 5], [5, 4]], dtype=np.float32)
        rotated_scaled = points @ np.asarray([[0, -2], [2, 0]], dtype=np.float32) + 100
        first = flow._build_quad_hash(points)
        second = flow._build_quad_hash(rotated_scaled)
        self.assertIsNotNone(first)
        self.assertIsNotNone(second)
        np.testing.assert_allclose(first[0], second[0], rtol=1e-6, atol=1e-6)


class StackingVectorizationTests(unittest.TestCase):
    def test_masked_combine_matches_nan_reducers(self) -> None:
        rng = np.random.default_rng(20260902)
        values = rng.normal(size=(7, 9, 11)).astype(np.float32)
        values[0, 0, 0] = np.nan
        masks = rng.random(values.shape) > 0.2
        masked = np.where(masks, values, np.nan)
        references = {
            "Median": np.nanmedian(masked, axis=0),
            "Mean": np.nanmean(masked, axis=0),
            "Sum": np.nansum(masked, axis=0),
            "Maximum": np.nanmax(masked, axis=0),
            "Minimum": np.nanmin(masked, axis=0),
        }
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            for method, expected in references.items():
                with self.subTest(method=method):
                    actual = stacking._combine_masked_cpu(values, masks, method)
                    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


@unittest.skipIf(align is None, "alignment optional dependencies are not installed")
class AstroAlignVectorizationTests(unittest.TestCase):
    def test_identity_warp_preserves_mono_and_rgb(self) -> None:
        matrix = np.asarray([[1, 0, 0], [0, 1, 0]], dtype=np.float32)
        mono = np.arange(48, dtype=np.float32).reshape(6, 8)
        rgb = np.dstack((mono, mono + 100, mono + 200))
        np.testing.assert_array_equal(align.warp_frame(mono, matrix, "nearest"), mono)
        np.testing.assert_array_equal(align.warp_frame(rgb, matrix, "nearest"), rgb)
        np.testing.assert_array_equal(align.generate_valid_mask(mono.shape, matrix), 1)

    def test_integer_translation_has_expected_border_mask(self) -> None:
        matrix = np.asarray([[1, 0, 2], [0, 1, -1]], dtype=np.float32)
        mask = align.generate_valid_mask((5, 6), matrix)
        expected = np.zeros((5, 6), dtype=np.uint8)
        expected[:4, 2:] = 1
        np.testing.assert_array_equal(mask, expected)

    def test_legacy_cubic_modes_remain_pixel_equivalent(self) -> None:
        from skimage.transform import AffineTransform, warp

        image = np.arange(7 * 9 * 3, dtype=np.float32).reshape(7, 9, 3)
        matrix = np.asarray([[1, 0.03, 1.2], [-0.02, 1, -0.7]], dtype=np.float32)
        homogeneous = np.eye(3, dtype=np.float64)
        homogeneous[:2, :] = matrix
        expected = np.stack(
            [
                warp(
                    image[:, :, channel],
                    AffineTransform(matrix=homogeneous).inverse,
                    order=3,
                    mode="constant",
                    cval=0.0,
                    preserve_range=True,
                )
                for channel in range(3)
            ],
            axis=-1,
        ).astype(np.float32)
        for mode in ("bicubic", "lanczos"):
            np.testing.assert_array_equal(
                align.warp_frame(image, matrix, mode), expected
            )
