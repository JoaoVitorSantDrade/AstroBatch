from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import unittest

from stacking_features import (
    assess_trailing,
    build_quality_scores,
    parse_selection_weights,
    score_to_weight,
)


class StackingFeatureTests(unittest.TestCase):
    def _frame(self, name: str, **metrics):
        return SimpleNamespace(path=Path(name), name=name, metrics=metrics)

    def test_custom_weights_accept_json_and_compact_forms(self):
        self.assertEqual(
            parse_selection_weights('{"fwhm": 0.4, "snr": 0.6}'),
            {"fwhm": 0.4, "snr": 0.6},
        )
        self.assertEqual(
            parse_selection_weights("fwhm=0.4;roundness=0.3,snr=0.3"),
            {"fwhm": 0.4, "roundness": 0.3, "snr": 0.3},
        )

    def test_percentile_score_is_explainable_and_deterministic(self):
        frames = [
            self._frame("b.fits", fwhm=4, roundness=0.8, shape_star_count=10),
            self._frame("a.fits", fwhm=2, roundness=0.95, shape_star_count=20),
            self._frame("c.fits", fwhm=6, roundness=0.7, shape_star_count=5),
        ]
        scores = build_quality_scores(frames, "Sharpness")
        self.assertTrue(scores[str(Path("a.fits"))].eligible)
        self.assertGreater(scores[str(Path("a.fits"))].score, scores[str(Path("c.fits"))].score)
        self.assertIn("fwhm", scores[str(Path("a.fits"))].components)
        self.assertEqual(
            build_quality_scores(list(reversed(frames)), "Sharpness")[str(Path("a.fits"))].score,
            scores[str(Path("a.fits"))].score,
        )

    def test_trailing_policy_is_conservative_and_penalizes_moderate(self):
        severe = assess_trailing(
            {
                "shape_star_count": 24,
                "roundness": 0.45,
                "elongation": 2.0,
                "trail_excess_px": 2.5,
                "trail_coherence": 0.9,
            },
            median_roundness=0.90,
            robust_sigma_roundness=0.04,
            median_excess=0.15,
            robust_sigma_excess=0.15,
        )
        self.assertEqual(severe.classification, "severe")
        self.assertEqual(score_to_weight(None, severe.classification), 0.0)
        moderate = assess_trailing(
            {
                "shape_star_count": 24,
                "roundness": 0.70,
                "elongation": 1.35,
                "trail_excess_px": 0.8,
                "trail_coherence": 0.55,
            },
            median_roundness=0.90,
            robust_sigma_roundness=0.04,
            median_excess=0.15,
            robust_sigma_excess=0.15,
        )
        self.assertEqual(moderate.classification, "moderate")
        self.assertAlmostEqual(score_to_weight(None, moderate.classification), 0.65)


if __name__ == "__main__":
    unittest.main()
