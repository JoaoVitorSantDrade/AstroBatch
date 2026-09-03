from __future__ import annotations

import sys
import types
import unittest
from unittest.mock import patch

import numpy as np

from app.engines.astroalign_fallback import estimate_asterism_transform


class AstroalignFallbackTests(unittest.TestCase):
    def test_uses_existing_coordinates_and_returns_homogeneous_matrix(self) -> None:
        class FakeTransform:
            params = np.array([[1.0, 0.0, 3.0], [0.0, 1.0, -2.0], [0.0, 0.0, 1.0]])

            def __call__(self, points):
                return np.asarray(points) + np.array([3.0, -2.0])

        source = np.array([[0.0, 0.0], [2.0, 1.0], [4.0, 3.0]])
        target = source + np.array([3.0, -2.0])
        fake = types.SimpleNamespace(find_transform=lambda *args, **kwargs: (FakeTransform(), (source, target)))
        with patch.dict(sys.modules, {"astroalign": fake}):
            matrix, metrics = estimate_asterism_transform(source, target)
        np.testing.assert_allclose(matrix, FakeTransform.params)
        self.assertEqual(metrics["matches"], 3)
        self.assertEqual(metrics["inliers"], 3)
        self.assertEqual(metrics["recovery_method"], "astroalign_asterism")

    def test_rejects_insufficient_coordinates_before_importing_dependency(self) -> None:
        with self.assertRaises(ValueError):
            estimate_asterism_transform(np.zeros((2, 2)), np.zeros((2, 2)))


if __name__ == "__main__":
    unittest.main()
