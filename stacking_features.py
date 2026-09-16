"""Deterministic quality features shared by Flow and Stack.

The functions in this module deliberately operate on small Python mappings or
FrameInfo-like objects.  They do not read FITS files and never change pixel
values.  Keeping the policy separate from the reducers makes it possible to
test selection, trailing classification and weighting without allocating a
full camera frame.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from collections.abc import Mapping, Sequence
from typing import Any
import json

import numpy as np


SELECTION_PROFILES: dict[str, dict[str, float]] = {
    "Sharpness": {
        "fwhm": 0.40,
        "roundness": 0.30,
        "alignment_rms": 0.20,
        "star_count": 0.10,
    },
    "Balanced": {
        "fwhm": 0.25,
        "roundness": 0.20,
        "star_count": 0.20,
        "snr": 0.15,
        "alignment_rms": 0.10,
        "coverage": 0.10,
    },
    "Signal": {
        "star_count": 0.30,
        "snr": 0.25,
        "coverage": 0.20,
        "fwhm": 0.15,
        "roundness": 0.10,
    },
}


def parse_selection_weights(value: Any) -> dict[str, float]:
    """Parse the advanced-weight field used by the native Stack form.

    Both a JSON object and a compact ``metric=value,metric=value`` form are
    accepted. Invalid/negative values are ignored here and rejected later by
    the command validator when no usable weight remains.
    """

    if isinstance(value, Mapping):
        raw = value
    elif value is None:
        return {}
    else:
        text = str(value).strip()
        if not text:
            return {}
        try:
            parsed = json.loads(text)
        except (TypeError, ValueError, json.JSONDecodeError):
            parsed = {}
            for token in text.replace(";", ",").split(","):
                if "=" not in token:
                    continue
                key, number = token.split("=", 1)
                parsed[key.strip()] = number.strip()
        raw = parsed if isinstance(parsed, Mapping) else {}
    result: dict[str, float] = {}
    for key, number in raw.items():
        name = str(key).strip()
        parsed = _finite(number)
        if name in _METRIC_ALIASES and parsed is not None and parsed > 0:
            result[name] = parsed
    return result

_LOWER_IS_BETTER = {"fwhm", "alignment_rms"}
_METRIC_ALIASES: dict[str, tuple[str, ...]] = {
    "fwhm": ("shape_fwhm", "fwhm"),
    "roundness": ("roundness",),
    "star_count": ("shape_star_count", "star_count"),
    "snr": ("snr", "signal_to_noise"),
    "alignment_rms": ("alignment_rms", "rms"),
    "coverage": ("coverage", "valid_fraction"),
}


def _finite(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _nested_metric(metrics: Mapping[str, Any], name: str) -> float | None:
    for alias in _METRIC_ALIASES.get(name, (name,)):
        value = metrics.get(alias)
        number = _finite(value)
        if number is not None:
            return number
    alignment = metrics.get("alignment_quality")
    if isinstance(alignment, Mapping):
        if name == "alignment_rms":
            return _finite(alignment.get("rms"))
        if name == "coverage":
            return _finite(alignment.get("coverage"))
    return None


def frame_metric(frame: Any, name: str) -> float | None:
    """Read a metric from a FrameInfo-like object without imposing a class."""

    metrics = getattr(frame, "metrics", None)
    if not isinstance(metrics, Mapping):
        metrics = {}
    value = _nested_metric(metrics, name)
    if value is not None:
        return value
    # FrameInfo stores the common Flow values as attributes as well.  The
    # mapping remains authoritative when both forms exist.
    return _finite(getattr(frame, name, None))


def _rank(values: Sequence[float | None], *, higher_is_better: bool) -> list[float | None]:
    """Return deterministic percentile ranks in [0, 1], preserving ties."""

    finite = [(index, value) for index, value in enumerate(values) if value is not None]
    if not finite:
        return [None] * len(values)
    ordered = sorted(finite, key=lambda item: (item[1], item[0]))
    ranks: list[float | None] = [None] * len(values)
    n = len(ordered)
    cursor = 0
    while cursor < n:
        end = cursor + 1
        while end < n and ordered[end][1] == ordered[cursor][1]:
            end += 1
        # Average rank is deterministic for ties and avoids depending on the
        # order in which inspection futures completed.
        percentile = ((cursor + end - 1) * 0.5) / max(1, n - 1)
        for position in range(cursor, end):
            ranks[ordered[position][0]] = percentile
        cursor = end
    if not higher_is_better:
        ranks = [None if value is None else 1.0 - value for value in ranks]
    return ranks


@dataclass(frozen=True, slots=True)
class FrameScore:
    """A reproducible multi-metric score and its explainable components."""

    score: float | None
    components: dict[str, float]
    missing: tuple[str, ...]
    available_weight: float
    eligible: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "score": self.score,
            "components": dict(self.components),
            "missing": list(self.missing),
            "available_weight": self.available_weight,
            "eligible": self.eligible,
        }


def frame_key(frame: Any) -> str:
    """Return a stable key for reports and lookups across pipeline stages."""

    path = getattr(frame, "path", None)
    if path is not None:
        return str(path)
    name = getattr(frame, "name", None)
    if name is not None:
        return str(name)
    return f"object:{id(frame)}"


def build_quality_scores(
    frames: Sequence[Any],
    profile: str = "Balanced",
    custom_weights: Mapping[str, Any] | None = None,
) -> dict[str, FrameScore]:
    """Build robust percentile scores for a collection of frames.

    A frame with less
    than 60% of the requested weight represented by finite metrics is marked
    ineligible instead of receiving a misleading zero score.
    """

    profile_name = str(profile or "Balanced")
    if profile_name == "Custom":
        raw = parse_selection_weights(custom_weights)
        weights = {
            key: max(0.0, float(value))
            for key, value in raw.items()
            if key in _METRIC_ALIASES and _finite(value) is not None
        }
    else:
        weights = dict(SELECTION_PROFILES.get(profile_name, SELECTION_PROFILES["Balanced"]))
    total_weight = sum(weights.values())
    if total_weight <= 0:
        weights = dict(SELECTION_PROFILES["Balanced"])
        total_weight = sum(weights.values())

    raw_values = {
        metric: [frame_metric(frame, metric) for frame in frames]
        for metric in weights
    }
    ranks = {
        metric: _rank(values, higher_is_better=metric not in _LOWER_IS_BETTER)
        for metric, values in raw_values.items()
    }
    result: dict[str, FrameScore] = {}
    for index, frame in enumerate(frames):
        components: dict[str, float] = {}
        missing: list[str] = []
        available = 0.0
        weighted = 0.0
        for metric, weight in weights.items():
            value = ranks[metric][index]
            if value is None:
                missing.append(metric)
                continue
            components[metric] = float(np.clip(value, 0.0, 1.0))
            available += weight
            weighted += weight * components[metric]
        ratio = available / total_weight if total_weight else 0.0
        eligible = bool(available > 0 and ratio >= 0.60)
        score = float(weighted / available) if eligible else None
        result[frame_key(frame)] = FrameScore(
            score=score,
            components=components,
            missing=tuple(missing),
            available_weight=float(available),
            eligible=eligible,
        )
    return result


def score_to_weight(score: FrameScore | None, trail_class: str | None = None) -> float:
    """Convert an explainable score into a bounded scalar stack weight."""

    base = 1.0 if score is None or score.score is None else 0.25 + 0.75 * float(np.clip(score.score, 0.0, 1.0))
    trail_factor = {
        "moderate": 0.65,
        "severe": 0.0,
        "unreliable": 1.0,
        "none": 1.0,
    }.get(str(trail_class or "unreliable"), 1.0)
    return float(np.clip(base * trail_factor, 0.0, 2.0))


@dataclass(frozen=True, slots=True)
class TrailAssessment:
    classification: str
    cause: str
    confidence: float
    penalty: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "classification": self.classification,
            "cause": self.cause,
            "confidence": self.confidence,
            "penalty": self.penalty,
        }


def _robust_location(values: Sequence[float | None]) -> tuple[float | None, float | None]:
    finite = np.asarray([value for value in values if value is not None and math.isfinite(value)], dtype=np.float64)
    if finite.size == 0:
        return None, None
    median = float(np.median(finite))
    mad = float(np.median(np.abs(finite - median)))
    return median, 1.4826 * mad


def assess_trailing(
    metrics: Mapping[str, Any],
    *,
    median_roundness: float | None = None,
    robust_sigma_roundness: float | None = None,
    median_excess: float | None = None,
    robust_sigma_excess: float | None = None,
) -> TrailAssessment:
    """Classify a frame using shape metrics and session-relative thresholds."""

    count = _finite(metrics.get("shape_star_count"))
    roundness = _finite(metrics.get("roundness"))
    elongation = _finite(metrics.get("elongation"))
    excess = _finite(metrics.get("trail_excess_px"))
    direction = _finite(metrics.get("trail_coherence"))
    tangential = _finite(metrics.get("tangential_coherence"))
    if count is None or roundness is None or elongation is None or excess is None:
        return TrailAssessment("unreliable", "unclassified", 0.0, 1.0)
    if median_roundness is None:
        median_roundness = roundness
    if robust_sigma_roundness is None or robust_sigma_roundness <= 0:
        robust_sigma_roundness = 0.05
    if median_excess is None:
        median_excess = excess
    if robust_sigma_excess is None or robust_sigma_excess <= 0:
        robust_sigma_excess = 0.25

    coherent = max(direction or 0.0, tangential or 0.0)
    cause = "unclassified"
    if tangential is not None and tangential >= max(direction or 0.0, 0.45):
        cause = "field_rotation"
    elif direction is not None and direction >= 0.45:
        cause = "tracking"
    elif elongation >= 1.25:
        cause = "optical_or_mixed"

    severe_roundness = min(0.65, median_roundness - 3.0 * robust_sigma_roundness)
    moderate_roundness = min(0.75, median_roundness - 2.0 * robust_sigma_roundness)
    severe_excess = max(1.0, median_excess + 3.0 * robust_sigma_excess)
    moderate_excess = max(0.5, median_excess + 2.0 * robust_sigma_excess)
    severe = (
        count >= 8
        and roundness <= severe_roundness
        and elongation >= 1.5
        and excess >= severe_excess
        and coherent >= 0.65
    )
    moderate = (
        count >= 8
        and roundness <= moderate_roundness
        and elongation >= 1.25
        and excess >= moderate_excess
        and coherent >= 0.40
    )
    if severe:
        confidence = float(np.clip(0.55 + 0.35 * coherent, 0.0, 1.0))
        return TrailAssessment("severe", cause, confidence, 0.0)
    if moderate:
        confidence = float(np.clip(0.35 + 0.45 * coherent, 0.0, 1.0))
        return TrailAssessment("moderate", cause, confidence, 0.65)
    if count < 8:
        return TrailAssessment("unreliable", "unclassified", 0.0, 1.0)
    return TrailAssessment("none", cause if cause != "unclassified" else "unclassified", float(np.clip(0.5 + 0.3 * coherent, 0.0, 1.0)), 1.0)


def annotate_trailing(frames: Sequence[Any]) -> dict[Any, TrailAssessment]:
    """Classify all frames against a robust session baseline and annotate them."""

    mappings: list[Mapping[str, Any]] = []
    for frame in frames:
        metrics = getattr(frame, "metrics", None)
        mappings.append(metrics if isinstance(metrics, Mapping) else {})
    roundness_median, roundness_sigma = _robust_location([_finite(item.get("roundness")) for item in mappings])
    excess_median, excess_sigma = _robust_location([_finite(item.get("trail_excess_px")) for item in mappings])
    result: dict[str, TrailAssessment] = {}
    for frame, metrics in zip(frames, mappings, strict=True):
        assessment = assess_trailing(
            metrics,
            median_roundness=roundness_median,
            robust_sigma_roundness=roundness_sigma,
            median_excess=excess_median,
            robust_sigma_excess=excess_sigma,
        )
        result[frame_key(frame)] = assessment
        if isinstance(getattr(frame, "metrics", None), dict):
            frame.metrics["trail_class"] = assessment.classification
            frame.metrics["trail_cause"] = assessment.cause
            frame.metrics["trail_confidence"] = assessment.confidence
            frame.metrics["trail_penalty"] = assessment.penalty
    return result


__all__ = [
    "FrameScore",
    "SELECTION_PROFILES",
    "parse_selection_weights",
    "TrailAssessment",
    "annotate_trailing",
    "assess_trailing",
    "build_quality_scores",
    "frame_key",
    "frame_metric",
    "score_to_weight",
]
