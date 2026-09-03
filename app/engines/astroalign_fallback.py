"""Optional Astroalign asterism matcher, isolated from Flow orchestration."""

from __future__ import annotations

from typing import Any

import numpy as np


def estimate_asterism_transform(
    source: np.ndarray, target: np.ndarray, max_control_points: int = 50
) -> tuple[np.ndarray, dict[str, Any]]:
    """Find a similarity transform from already-detected star coordinates.

    Astroalign accepts point coordinates directly, avoiding its image detector
    and allowing the caller to retain one source-of-truth for star metrics.
    """
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if len(source) < 3 or len(target) < 3:
        raise ValueError("Astroalign fallback needs at least three stars per frame.")

    try:
        import astroalign
    except ImportError as exc:  # pragma: no cover - availability is registry tested
        raise RuntimeError("Astroalign is not installed.") from exc

    transform, (source_points, target_points) = astroalign.find_transform(
        source[:max_control_points], target[:max_control_points], max_control_points=max_control_points
    )
    matrix = np.asarray(transform.params, dtype=np.float64)
    transformed = transform(source_points)
    errors = np.linalg.norm(transformed - target_points, axis=1)
    rms = float(np.sqrt(np.mean(errors**2))) if len(errors) else 999.0
    return matrix, {
        "matches": int(len(source_points)),
        "inliers": int(len(source_points)),
        "inlier_ratio": 1.0,
        "rms": rms,
        "recovery_method": "astroalign_asterism",
    }
