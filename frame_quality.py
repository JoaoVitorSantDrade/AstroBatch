"""Small, dependency-free measurements used to qualify detected stars.

The flow detector already gives us coordinates for likely stars.  This module
only inspects a bounded cutout around those coordinates; it deliberately does
not run another source finder or alter the input image.
"""

from __future__ import annotations

import numpy as np


_MAX_SHAPE_STARS = 64
_MAX_RADIUS = 8


def _empty_measurement() -> dict[str, float | int | None]:
    return {
        "roundness": None,
        "shape_star_count": 0,
        "shape_fwhm": None,
        "elongation": None,
    }


def _usable_star_shape(cutout: np.ndarray, radius: int) -> tuple[float, float, float] | None:
    """Return roundness, geometric FWHM and elongation for one cutout.

    The detector coordinates are only approximate, so moments are taken from
    the complete local window and measured relative to the weighted centroid.
    Border pixels provide a local background estimate.  A single-pixel peak
    is rejected as a hot/blank-patch artefact instead of being allowed to
    manufacture an apparently precise shape.
    """

    finite = np.isfinite(cutout)
    expected_pixels = cutout.size
    if int(finite.sum()) < max(9, expected_pixels // 2):
        return None

    # The outer ring is generally outside the PSF and is more reliable than a
    # global background for short exposures with gradients or vignetting.
    border = np.zeros(cutout.shape, dtype=bool)
    border[0, :] = border[-1, :] = True
    border[:, 0] = border[:, -1] = True
    border_values = cutout[border & finite]
    if border_values.size < 4:
        return None
    background = float(np.median(border_values))
    if not np.isfinite(background):
        return None

    residual = cutout.astype(np.float64, copy=False) - background
    # Use a robust noise floor when one is available.  With an exactly flat
    # synthetic background MAD is zero, correctly retaining positive signal.
    border_residual = residual[border & finite]
    mad = float(np.median(np.abs(border_residual - np.median(border_residual))))
    noise = 1.4826 * mad
    threshold = 3.0 * noise if np.isfinite(noise) and noise > 0 else 0.0
    weights = np.where(finite & (residual > threshold), residual, 0.0)
    positive = weights[weights > 0]
    if positive.size < 4:
        return None

    total = float(np.sum(positive, dtype=np.float64))
    if not np.isfinite(total) or total <= 0:
        return None

    # A hot pixel or a mostly blank cutout has a dominant isolated sample.
    # True stellar PSFs have several neighbouring positive samples, including
    # for the smallest practical detector FWHM.
    ordered = np.sort(positive)
    peak = float(ordered[-1])
    second = float(ordered[-2]) if ordered.size > 1 else 0.0
    if peak / total > 0.72 or (second > 0 and peak / second > 8.0):
        return None

    yy, xx = np.indices(cutout.shape, dtype=np.float64)
    sum_x = float(np.sum(weights * xx, dtype=np.float64) / total)
    sum_y = float(np.sum(weights * yy, dtype=np.float64) / total)
    if not np.isfinite(sum_x) or not np.isfinite(sum_y):
        return None
    # A centroid at the edge of the window means the candidate is usually an
    # edge source or a contaminated patch, for which moments are unreliable.
    centre = float(radius)
    if max(abs(sum_x - centre), abs(sum_y - centre)) > radius * 0.65:
        return None

    dx = xx - sum_x
    dy = yy - sum_y
    var_x = float(np.sum(weights * dx * dx, dtype=np.float64) / total)
    var_y = float(np.sum(weights * dy * dy, dtype=np.float64) / total)
    cov_xy = float(np.sum(weights * dx * dy, dtype=np.float64) / total)
    covariance = np.asarray([[var_x, cov_xy], [cov_xy, var_y]], dtype=np.float64)
    if not np.isfinite(covariance).all():
        return None
    eigenvalues = np.linalg.eigvalsh(covariance)
    minor_var, major_var = float(eigenvalues[0]), float(eigenvalues[1])
    if minor_var <= 1.0e-6 or major_var <= minor_var:
        # Equal variances are fine; only a degenerate (single-line) source is
        # unusable.  The small floor also keeps ratios JSON-safe and bounded.
        if major_var <= 1.0e-6:
            return None
        minor_var = max(minor_var, 0.0)

    major_sigma = float(np.sqrt(max(major_var, 0.0)))
    minor_sigma = float(np.sqrt(max(minor_var, 0.0)))
    if major_sigma <= 1.0e-3 or not np.isfinite(major_sigma + minor_sigma):
        return None
    roundness = float(np.clip(minor_sigma / major_sigma, 0.0, 1.0))
    elongation = float(max(1.0, major_sigma / max(minor_sigma, 1.0e-6)))
    # FWHM from the geometric mean of the two second-moment sigmas.  It is
    # stable for mildly trailed stars and retains the existing detector FWHM
    # semantics as an independent quality metric.
    shape_fwhm = float(2.354820045 * np.sqrt(max(major_sigma * minor_sigma, 0.0)))
    values = (roundness, shape_fwhm, elongation)
    if not all(np.isfinite(value) for value in values):
        return None
    return values


def measure_star_shapes(
    data: np.ndarray, stars: np.ndarray, radius: int = _MAX_RADIUS
) -> dict[str, float | int | None]:
    """Measure robust shape statistics around detected ``(x, y)`` stars.

    At most 64 coordinates are sampled and the radius is capped at eight
    pixels, keeping this suitable for every frame in an unguided sequence.
    Non-edge, finite, background-subtracted cutouts contribute to medians.
    The returned values contain only Python scalars and ``None`` so they can
    be written directly to the existing flow JSON.
    """

    if np.asarray(data).ndim != 2:
        raise ValueError("data must be a 2D image")
    result = _empty_measurement()
    image = np.asarray(data)
    try:
        coordinates = np.asarray(stars, dtype=np.float64)
    except (TypeError, ValueError):
        return result
    if coordinates.size == 0:
        return result
    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        return result

    try:
        bounded_radius = min(_MAX_RADIUS, max(1, int(radius)))
    except (TypeError, ValueError, OverflowError):
        bounded_radius = _MAX_RADIUS
    height, width = image.shape
    usable: list[tuple[float, float, float]] = []
    # Coordinates from the detector are priority ordered.  Taking the first
    # bounded sample preserves that priority and makes the operation stable.
    for x, y in coordinates[:_MAX_SHAPE_STARS]:
        if not np.isfinite(x) or not np.isfinite(y):
            continue
        cx, cy = int(np.rint(x)), int(np.rint(y))
        if (
            cx - bounded_radius < 0
            or cy - bounded_radius < 0
            or cx + bounded_radius >= width
            or cy + bounded_radius >= height
        ):
            continue
        cutout = image[
            cy - bounded_radius : cy + bounded_radius + 1,
            cx - bounded_radius : cx + bounded_radius + 1,
        ]
        shape = _usable_star_shape(cutout, bounded_radius)
        if shape is not None:
            usable.append(shape)

    if not usable:
        return result
    values = np.asarray(usable, dtype=np.float64)
    result["roundness"] = float(np.clip(np.median(values[:, 0]), 0.0, 1.0))
    result["shape_star_count"] = int(len(usable))
    result["shape_fwhm"] = float(np.median(values[:, 1]))
    result["elongation"] = float(max(1.0, np.median(values[:, 2])))
    return result

