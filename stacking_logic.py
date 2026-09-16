"""
stacking_logic.py

AstroStack - empilhamento otimizado para o AstroProcessManager.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import queue
import tempfile
import threading
import time
import uuid
import warnings
from collections.abc import Callable
from concurrent.futures import (
    FIRST_COMPLETED,
    ThreadPoolExecutor,
    as_completed,
    wait,
)
from dataclasses import dataclass, field, replace
from functools import partial
from pathlib import Path
from typing import Any, Literal

import numpy as np
from astropy.io import fits
from astropy.stats import SigmaClip
from astropy.utils.exceptions import AstropyWarning
from photutils.background import Background2D, MedianBackground

from app.engines import EngineDescriptor, EngineProfile, ExecutionBudget, registry
from cpu_kernels import apply_scale_and_mask_inplace, masked_extrema, masked_sum_count, weighted_merge
from cpu_runtime import configure_opencv_threads, configure_worker_runtime, physical_core_count
from stacking_features import (
    FrameScore,
    SELECTION_PROFILES,
    annotate_trailing,
    build_quality_scores,
    frame_key,
    parse_selection_weights,
    score_to_weight,
)
from image_io import (
    IMAGE_SUFFIXES,
    TIFF_SUFFIXES,
    read_tiff,
    read_tiff_sidecar_arrays,
    tiff_shape,
    write_npz_sidecar,
    write_sidecar_json,
    write_tiff,
    validate_single_format,
)

try:
    from pyinstrument import Profiler

    HAS_PYINSTRUMENT = True
except ImportError:
    HAS_PYINSTRUMENT = False

warnings.simplefilter("ignore", category=AstropyWarning)

FITS_SUFFIXES = IMAGE_SUFFIXES
FITS_CACHE_DIR_NAME = ".astrostack_fits_cache"
FITS_CACHE_FORMAT_VERSION = 1
DEFAULT_CHUNK_SIZE = 2048
DEFAULT_MEMORY_BUDGET_MB = 4096
DEFAULT_NORMALIZATION_MAX_SAMPLES = 1_000_000
MAX_WORKERS = 16
# Empirical upper bound for the float/mask/rejection working set per input
# pixel in a band.  The previous 24-byte factor split 4 GiB across four
# workers into ~60-row bands for the 3-channel Lagoon capture; that doubled
# FITS passes without improving stability.  Twelve bytes keeps the measured
# peak below the configured budget while materially reducing band overhead.
STACK_BAND_BYTES_PER_PIXEL = 12
# Keep the leaf partition and binary reduction topology independent of the
# requested executor width.  Concurrency may change completion order, but it
# must not change which frames are accumulated together in Stable mode.
DETERMINISTIC_LEAF_COUNT = 4


class StackingCancelled(RuntimeError):
    """Cancelamento cooperativo solicitado pelo usuário."""


@dataclass
class StackingConfig:
    base_dir: Path = field(default_factory=Path)
    input_dir: Path = field(default_factory=Path)
    output_dir: Path = field(default_factory=Path)

    # Keep the historical positional fields first.  New feature controls are
    # appended below so third-party callers constructing this dataclass
    # positionally retain the original argument mapping.
    selection_mode: Literal["All", "BestPercentage", "MultiMetric"] = "BestPercentage"
    selection_percentage: float = 80.0
    selection_metric: Literal["quality", "fwhm", "star_count", "snr", "roundness"] = "quality"
    trail_filter_enabled: bool = False
    min_roundness: float = 0.65
    min_shape_stars: int = 5

    method: Literal["Median", "Mean", "QualityWeightedMean", "Sum", "Maximum", "Minimum"] = "Median"

    rejection_method: Literal["None", "SigmaClip", "Winsorized", "MAD"] = "SigmaClip"
    rejection_low: float = 3.0
    rejection_high: float = 3.0

    normalize: bool = True
    normalize_method: Literal["Median", "Mode"] = "Median"
    apply_dither_correction: bool = False
    remove_background: bool = False

    output_name: str = "stacked_image.fits"
    output_format: Literal["auto", "fits", "tiff"] = "auto"
    # Final presentation/output defaults to the acquisition-compatible 16-bit
    # FITS format. Internal calibration and stack math remain float32.
    output_bit_depth: Literal["16-bit"] = "16-bit"
    compress_output: bool = True

    workers: int | None = None
    chunk_size: int = DEFAULT_CHUNK_SIZE
    memory_budget_mb: int = DEFAULT_MEMORY_BUDGET_MB
    normalization_max_samples: int = DEFAULT_NORMALIZATION_MAX_SAMPLES
    io_queue_factor: int = 2
    cache_decompressed_fits: bool = True
    engine_profile: str = "Stable"
    reducer_engine: str | None = None

    # New feature controls are intentionally appended after the legacy
    # positional fields; keyword callers can opt into them without affecting
    # existing integrations.
    feature_profile: Literal["Legacy", "Intelligent"] = "Legacy"
    selection_profile: Literal["Sharpness", "Balanced", "Signal", "Custom"] = "Balanced"
    selection_weights: dict[str, float] = field(default_factory=dict)
    trail_policy: Literal["off", "exclude_severe", "weight_only", "report"] = "off"
    reduction_storage: Literal["disk_legacy", "ram", "ram_spill"] = "disk_legacy"
    spill_directory: Path | None = None
    spill_limit_mb: int = 0

    @property
    def worker_count(self) -> int:
        try:
            cpu_count = getattr(os, "process_cpu_count", os.cpu_count)() or 1
        except Exception:
            cpu_count = os.cpu_count() or 1

        if self.workers is not None:
            try:
                return max(1, min(MAX_WORKERS, 8, physical_core_count(), cpu_count, int(self.workers)))
            except (TypeError, ValueError):
                pass

        return max(1, min(MAX_WORKERS, 8, physical_core_count(), cpu_count))

    @property
    def effective_chunk_size(self) -> int:
        return max(32, min(2048, int(self.chunk_size)))

    @property
    def effective_io_queue(self) -> int:
        return max(1, min(8, int(self.io_queue_factor)))


@dataclass(slots=True)
class FrameInfo:
    path: Path
    name: str
    batch: str
    metrics: dict[str, Any]
    quality: float
    star_count: float
    fwhm: float
    snr: float
    rms: float
    has_valid_mask: bool
    shape: tuple[int, ...]
    image_kind: str
    channels: int
    # Loaded once per stacking run.  Reading a compressed VALID_MASK section
    # for every block/channel was a major part of the profile.
    valid_mask: np.ndarray | None = None


@dataclass(slots=True)
class FrameGeometry:
    height: int
    width: int
    image_kind: str
    channels: int
    hdu_index: int
    mask_hdu_index: int | None
    science_compressed: bool = False
    # True when the source science HDU carries FITS scaling/BLANK cards and
    # therefore cannot be safely memory-mapped by Astropy.  The flag is kept
    # separate from ``science_compressed`` because dedicated-camera captures
    # commonly store uint16 pixels as signed int16 + BZERO without tile
    # compression.
    requires_scaling: bool = False
    source_bscale: float = 1.0
    source_bzero: float = 0.0
    source_blank: int | None = None
    cache_raw_storage: bool = False
    cache_bscale: float = 1.0
    cache_bzero: float = 0.0
    cache_blank: int | None = None


@dataclass(slots=True)
class BlockRead:
    data: np.ndarray
    mask: np.ndarray


@dataclass(slots=True)
class SubstackInfo:
    path: Path
    frame_count: int
    # Logical position in the deterministic reduction tree.  Completion order
    # of worker futures must never decide scientific accumulation order.
    order: int = 0


@dataclass(slots=True)
class FitsCacheStats:
    hits: int = 0
    rebuilt: int = 0
    direct: int = 0
    skipped: int = 0
    # Number of uncompressed-but-scaled science HDUs that were eligible for
    # the raw-storage cache.  This is diagnostic only and does not alter the
    # legacy counters consumed by callers.
    scaled: int = 0


def get_optimal_worker_count() -> int:
    try:
        cpu_count = getattr(os, "process_cpu_count", os.cpu_count)() or 1
    except Exception:
        cpu_count = os.cpu_count() or 1
    return max(1, min(MAX_WORKERS, 8, physical_core_count(), cpu_count))


def check_cancel(cancel_event: threading.Event | None) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise StackingCancelled("Operação cancelada pelo usuário.")


def discover_aligned_frames(
    input_dir: Path,
) -> tuple[list[Path], dict[str, dict[str, Any]]]:
    if not input_dir.exists():
        return [], {}

    found: list[Path] = []
    batch_metadata: dict[str, dict[str, Any]] = {}

    for root, dirnames, filenames in os.walk(input_dir):
        # The persistent decompression cache lives next to the aligned frames,
        # but it must never become another input batch on the next run.
        dirnames[:] = [
            dirname
            for dirname in dirnames
            if dirname.casefold() != FITS_CACHE_DIR_NAME.casefold()
        ]
        root_path = Path(root)

        for filename in filenames:
            path = root_path / filename

            if path.suffix.casefold() not in FITS_SUFFIXES:
                continue

            found.append(path)
            batch_name = root_path.name or "root"

            meta = batch_metadata.setdefault(
                batch_name,
                {"path": root_path, "frame_count": 0, "frames": []},
            )
            meta["frame_count"] += 1
            meta["frames"].append(filename)

    found.sort(key=lambda p: (str(p.parent).casefold(), p.name.casefold()))
    return found, batch_metadata


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        value = json.load(file)
    return value if isinstance(value, dict) else {}


def load_flow_cache(
    base_dir: Path,
    batch_metadata: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    cache: dict[str, dict[str, Any]] = {}

    if not base_dir or not str(base_dir).strip() or str(base_dir) == ".":
        return cache

    try:
        from astroalign_logic import load_local_flow
    except Exception:
        load_local_flow = None

    for batch_name in batch_metadata:
        try:
            if load_local_flow is not None:
                value = load_local_flow(base_dir / batch_name)
                if isinstance(value, dict):
                    cache[batch_name] = value
                continue
            flow_path = base_dir / batch_name / "flow_local.json"
            if flow_path.is_file():
                cache[batch_name] = load_json(flow_path)
        except Exception:
            cache[batch_name] = {}

    root_flow = base_dir / "flow_local.json"
    if root_flow.is_file() and "root" not in cache:
        try:
            cache["root"] = load_json(root_flow)
        except Exception:
            cache["root"] = {}

    return cache


def _find_primary_image_hdu(
    hdul: fits.HDUList,
) -> tuple[int, fits.hdu.base.NonstandardHDU]:
    for index, hdu in enumerate(hdul):
        if hdu.name == "VALID_MASK" or not hdu.is_image:
            continue

        shape = getattr(hdu, "shape", None)
        if shape is None:
            continue

        ndim = len(shape)
        if ndim in (2, 3):
            return index, hdu

    raise ValueError("Nenhuma imagem FITS 2D/3D encontrada.")


def _classify_shape(
    shape: tuple[int, ...],
    header: fits.Header,
) -> tuple[str, int]:
    del header

    if len(shape) == 2:
        return "Mono", 1

    if len(shape) != 3:
        raise ValueError(f"Dimensão FITS não suportada: {shape}")

    if shape[0] in (3, 4):
        return "RGB", int(shape[0])

    if shape[-1] in (3, 4):
        return "RGB", int(shape[-1])

    raise ValueError(f"Formato RGB não reconhecido: {shape}")


def inspect_fits(filepath: Path) -> FrameGeometry:
    if filepath.suffix.casefold() in TIFF_SUFFIXES:
        try:
            shape = tiff_shape(filepath)
        except Exception:
            data, _header = read_tiff(filepath)
            shape = tuple(np.asarray(data).shape)
        if len(shape) == 3 and shape[-1] in (3, 4):
            image_kind, channels = "RGB", int(shape[-1])
        else:
            image_kind, channels = _classify_shape(shape, fits.Header())
        if image_kind == "Mono":
            height, width = shape
        elif shape[0] in (3, 4) and shape[-1] not in (3, 4):
            _, height, width = shape
        else:
            height, width, _ = shape
        sidecars = read_tiff_sidecar_arrays(filepath)
        mask_index = -1 if "valid_mask" in sidecars or "sat_mask" in sidecars else None
        return FrameGeometry(
            height=int(height), width=int(width), image_kind=image_kind,
            channels=channels, hdu_index=0, mask_hdu_index=mask_index,
            science_compressed=False, requires_scaling=False,
            source_bscale=1.0, source_bzero=0.0, source_blank=None,
            cache_raw_storage=False, cache_bscale=1.0, cache_bzero=0.0,
            cache_blank=None,
        )
    with (
        fits.open(
            filepath,
            memmap=False,
            lazy_load_hdus=True,
            do_not_scale_image_data=True,  # Evita cálculos caros de BSCALE/BZERO durante inspeção
        ) as hdul
    ):
        hdu_index, image_hdu = _find_primary_image_hdu(hdul)
        shape = tuple(image_hdu.shape)
        image_kind, channels = _classify_shape(shape, image_hdu.header)

        mask_hdu_index = None
        for index, hdu in enumerate(hdul):
            if hdu.name == "VALID_MASK" and hdu.is_image:
                mask_hdu_index = index
                break

        if image_kind == "Mono":
            height, width = shape
        elif shape[0] in (3, 4):
            _, height, width = shape
        else:
            height, width, _ = shape

        return FrameGeometry(
            height=int(height),
            width=int(width),
            image_kind=image_kind,
            channels=channels,
            hdu_index=hdu_index,
            mask_hdu_index=mask_hdu_index,
            science_compressed=_is_compressed_hdu(image_hdu),
            requires_scaling=_has_active_scaling(image_hdu),
            source_bscale=float(image_hdu.header.get("BSCALE", 1.0)),
            source_bzero=float(image_hdu.header.get("BZERO", 0.0)),
            source_blank=(
                int(image_hdu.header["BLANK"])
                if "BLANK" in image_hdu.header
                else None
            ),
            cache_raw_storage=bool(image_hdu.header.get("ASTRAW", False)),
            cache_bscale=float(image_hdu.header.get("ASTBSCL", 1.0)),
            cache_bzero=float(image_hdu.header.get("ASTBZRO", 0.0)),
            cache_blank=(
                int(image_hdu.header["ASTBLNK"])
                if "ASTBLNK" in image_hdu.header
                else None
            ),
        )


def load_frame_metrics(filepath: Path) -> dict[str, Any]:
    metrics_path = filepath.parent / f"{filepath.stem}_metrics.json"

    metrics = {}
    try:
        if metrics_path.exists():
            metrics = load_json(metrics_path)
    except Exception:
        pass
    # Keep post-warp quality namespaced: it must not overwrite Flow's star RMS
    # or silently change frame selection in existing configurations.
    try:
        alignment_path = filepath.with_suffix(filepath.suffix + ".align.json")
        if alignment_path.exists():
            report = load_json(alignment_path)
            metrics["alignment_quality"] = report.get("quality", {})
            metrics["alignment_decision"] = report.get("decision", "unverified")
    except (OSError, ValueError, TypeError):
        pass
    return metrics


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
        return number if math.isfinite(number) else default
    except (TypeError, ValueError):
        return default


def _inspect_single_frame(
    filepath: Path, batch_name: str, flow_cache: dict[str, dict[str, Any]]
) -> FrameInfo | None:
    metrics = load_frame_metrics(filepath)

    flow = flow_cache.get(batch_name, {})
    frame_flow = (
        flow.get("frames", {}).get(filepath.name, {}) if isinstance(flow, dict) else {}
    )

    if isinstance(frame_flow, dict):
        metrics = {**metrics, **frame_flow}

    geometry = inspect_fits(filepath)

    star_count = _safe_float(metrics.get("star_count"), 0.0)
    fwhm = _safe_float(metrics.get("fwhm"), 0.0)
    snr = _safe_float(metrics.get("snr"), 0.0)
    rms = _safe_float(metrics.get("rms"), 999.0)

    quality = star_count / max(fwhm, 0.1) if fwhm > 0 else star_count

    if 0 < rms < 999:
        quality /= 1.0 + rms / 10.0

    has_valid_mask = geometry.mask_hdu_index is not None
    if has_valid_mask:
        quality *= 1.1

    metrics["quality"] = quality
    metrics["has_valid_mask"] = has_valid_mask

    shape = (
        (geometry.height, geometry.width)
        if geometry.image_kind == "Mono"
        else (geometry.height, geometry.width, geometry.channels)
    )

    return FrameInfo(
        path=filepath,
        name=filepath.name,
        batch=batch_name,
        metrics=metrics,
        quality=float(quality),
        star_count=star_count,
        fwhm=fwhm,
        snr=snr,
        rms=rms,
        has_valid_mask=has_valid_mask,
        shape=tuple(shape),
        image_kind=geometry.image_kind,
        channels=geometry.channels,
    )


def build_frame_infos(
    fits_files: list[Path],
    batch_metadata: dict[str, dict[str, Any]],
    flow_cache: dict[str, dict[str, Any]],
    app_print: Callable[[str], None],
    cancel_event: threading.Event | None = None,
) -> list[FrameInfo]:
    path_to_batch = {Path(meta["path"]): name for name, meta in batch_metadata.items()}

    result: list[FrameInfo] = []

    # Inspeção em paralelo para aniquilar o gargalo do header mapping do Astropy
    inspection_workers = get_optimal_worker_count()
    configure_opencv_threads(1 if inspection_workers > 1 else physical_core_count())
    with ThreadPoolExecutor(
        max_workers=inspection_workers, thread_name_prefix="AstroInspect",
        initializer=partial(configure_worker_runtime, 1),
    ) as executor:
        futures = {}
        for filepath in fits_files:
            batch_path = filepath.parent
            batch_name = path_to_batch.get(batch_path, batch_path.name or "root")
            future = executor.submit(
                _inspect_single_frame, filepath, batch_name, flow_cache
            )
            futures[future] = filepath

        report_step = max(1, len(fits_files) // 10)

        for index, future in enumerate(as_completed(futures), start=1):
            check_cancel(cancel_event)
            filepath = futures[future]

            try:
                info = future.result()
                if info:
                    result.append(info)
            except Exception as exc:
                app_print(f"⚠️ Ignorando {filepath.name}: {exc}")

            if index % report_step == 0 or index == len(fits_files):
                app_print(f"🔎 Metadados: {index}/{len(fits_files)}")

    return result


def trail_exclusion_reason(frame: FrameInfo, config: StackingConfig) -> str | None:
    """Unknown shape is never silently accepted by the opt-in trail filter."""
    if not config.trail_filter_enabled:
        return None
    roundness = _safe_float(frame.metrics.get("roundness"), -1.0)
    count = _safe_float(frame.metrics.get("shape_star_count"), 0.0)
    if not 0 < roundness <= 1 or count < config.min_shape_stars:
        return "missing_shape_measurement"
    if roundness < config.min_roundness:
        return "elongated_stars"
    return None


def _feature_trail_exclusion_reason(frame: FrameInfo, config: StackingConfig) -> str | None:
    """Apply the Intelligent policy without changing the Legacy filter."""

    if config.feature_profile != "Intelligent":
        return trail_exclusion_reason(frame, config)
    if config.trail_policy == "exclude_severe":
        classification = str(frame.metrics.get("trail_class", "unreliable"))
        confidence = _safe_float(frame.metrics.get("trail_confidence"), 0.0)
        if classification == "severe" and confidence >= 0.80:
            return "severe_trailing"
    if config.trail_filter_enabled:
        return trail_exclusion_reason(frame, config)
    return None


def _annotate_scores(
    frames: list[FrameInfo],
    config: StackingConfig,
) -> dict[str, FrameScore]:
    if config.feature_profile == "Intelligent":
        annotate_trailing(frames)
    if config.selection_mode != "MultiMetric" and config.method != "QualityWeightedMean":
        return {}
    scores = build_quality_scores(
        frames,
        config.selection_profile,
        config.selection_weights,
    )
    for frame in frames:
        score = scores.get(frame_key(frame))
        if score is None:
            continue
        frame.metrics["quality_score"] = score.score
        frame.metrics["quality_score_components"] = dict(score.components)
        frame.metrics["quality_score_missing"] = list(score.missing)
        frame.metrics["quality_score_available_weight"] = score.available_weight
        frame.metrics["quality_score_eligible"] = score.eligible
    return scores


def compute_quality_weights(
    all_frames: list[FrameInfo],
    selected_frames: list[FrameInfo],
    config: StackingConfig,
    scores: dict[str, FrameScore] | None = None,
) -> list[float]:
    """Return bounded, explainable scalar weights in selected-frame order."""

    if config.method != "QualityWeightedMean" and config.feature_profile != "Intelligent":
        return [1.0] * len(selected_frames)
    # Selection and weighting are consecutive consumers of the same feature
    # snapshot.  Accepting it from the caller avoids rerunning the bounded
    # trailing classifier and percentile ranks (which used to happen up to
    # three times for an Intelligent/QWM stack).  Direct callers retain the
    # old behavior by omitting the optional argument.
    if scores is None:
        scores = _annotate_scores(all_frames, config)
    raw: list[float] = []
    for frame in selected_frames:
        assessment = str(frame.metrics.get("trail_class", "unreliable"))
        value = score_to_weight(scores.get(frame_key(frame)), assessment)
        if config.trail_policy == "weight_only" and assessment == "severe":
            value = max(0.05, value)
        elif config.trail_policy in {"off", "report"} and assessment == "severe":
            value = max(0.25, value)
        raw.append(float(value))
    finite = [value for value in raw if math.isfinite(value) and value > 0]
    median = float(np.median(np.asarray(finite, dtype=np.float64))) if finite else 1.0
    if not math.isfinite(median) or median <= 0:
        median = 1.0
    normalized: list[float] = []
    for frame, value in zip(selected_frames, raw, strict=True):
        weight = float(np.clip(value / median, 0.25, 2.0))
        frame.metrics["stack_weight"] = weight
        normalized.append(weight)
    return normalized


def select_frames(
    all_frames: list[FrameInfo],
    config: StackingConfig,
    scores: dict[str, FrameScore] | None = None,
) -> list[FrameInfo]:
    if not all_frames:
        return []

    if scores is None:
        scores = _annotate_scores(all_frames, config)
    all_frames = [
        frame for frame in all_frames
        if _feature_trail_exclusion_reason(frame, config) is None
    ]
    # Metadata inspection completes out of order; ties must be reproducible.
    all_frames = sorted(all_frames, key=lambda frame: (str(frame.path).casefold(), str(frame.path)))
    if config.selection_mode == "All":
        return list(all_frames)

    if config.selection_mode == "MultiMetric":
        if scores is None:
            scores = _annotate_scores(all_frames, config)
        ranked = [
            frame for frame in all_frames
            if scores.get(frame_key(frame), FrameScore(None, {}, (), 0.0, False)).eligible
        ]
        ranked.sort(
            key=lambda frame: (
                -(scores[frame_key(frame)].score or 0.0),
                str(frame.path).casefold(),
                str(frame.path),
            )
        )
        if not ranked:
            # An old Flow report may not have any of the Intelligent metrics.
            # Keep the session usable and make the fallback explicit in its
            # selection report instead of silently rejecting every frame.
            for frame in all_frames:
                frame.metrics["quality_score_fallback"] = "legacy_quality"
            ranked = list(all_frames)
        percentage = max(1.0, min(100.0, float(config.selection_percentage)))
        n_select = max(min(3, len(ranked)), int(len(ranked) * percentage / 100.0))
        return ranked[: min(n_select, len(ranked))]

    metric_name = config.selection_metric

    if metric_name == "quality":
        key = lambda frame: frame.quality
        reverse = True
    elif metric_name == "fwhm":
        key = lambda frame: frame.fwhm if frame.fwhm > 0 else float("inf")
        reverse = False
    elif metric_name == "star_count":
        key = lambda frame: frame.star_count
        reverse = True
    elif metric_name == "roundness":
        key = lambda frame: _safe_float(frame.metrics.get("roundness"), 0.0)
        reverse = True
    else:
        key = lambda frame: frame.snr
        reverse = True

    sorted_frames = sorted(all_frames, key=key, reverse=reverse)
    valid: list[FrameInfo] = []

    for frame in sorted_frames:
        value = key(frame)

        if not isinstance(value, (int, float)) or not math.isfinite(value):
            continue

        if metric_name == "fwhm":
            if value <= 0:
                continue
        elif metric_name == "roundness":
            if not 0 < value <= 1:
                continue
        elif value <= 0:
            continue

        valid.append(frame)

    if not valid:
        valid = [] if metric_name == "roundness" else list(all_frames)

    percentage = max(
        1.0,
        min(100.0, float(config.selection_percentage)),
    )

    minimum = min(3, len(valid))
    n_select = max(minimum, int(len(valid) * percentage / 100.0))
    n_select = min(n_select, len(valid))

    return valid[:n_select]


def write_selection_report(
    all_frames: list[FrameInfo], selected_frames: list[FrameInfo], config: StackingConfig,
) -> Path:
    """Persist selection decisions before cache conversion or image reduction."""
    selected = {frame.path for frame in selected_frames}
    entries = []
    for frame in sorted(all_frames, key=lambda frame: str(frame.path)):
        reason = _feature_trail_exclusion_reason(frame, config)
        if frame.path in selected:
            reason = "selected"
        elif reason is None:
            reason = "not_selected_by_metric"
        entries.append({
            "path": str(frame.path), "batch": frame.batch,
            "selected": frame.path in selected, "reason": reason,
            "roundness": _safe_float(frame.metrics.get("roundness"), None),
            "shape_star_count": _safe_float(frame.metrics.get("shape_star_count"), 0),
            "fwhm": _safe_float(frame.fwhm, None),
            "quality": _safe_float(frame.quality, None),
            "trail_class": frame.metrics.get("trail_class"),
            "trail_cause": frame.metrics.get("trail_cause"),
            "trail_confidence": _safe_float(frame.metrics.get("trail_confidence"), None),
            "quality_score": frame.metrics.get("quality_score"),
            "quality_score_components": frame.metrics.get("quality_score_components", {}),
            "quality_score_missing": frame.metrics.get("quality_score_missing", []),
            "quality_score_available_weight": frame.metrics.get("quality_score_available_weight"),
            "stack_weight": frame.metrics.get("stack_weight", 1.0),
        })
    report = {
        "schema_version": 1, "stage": "selection_before_stacking",
        "total_frames": len(all_frames), "selected_frames": len(selected_frames),
        "settings": {"trail_filter_enabled": config.trail_filter_enabled,
                     "feature_profile": config.feature_profile,
                     "min_roundness": config.min_roundness,
                     "min_shape_stars": config.min_shape_stars,
                     "selection_mode": config.selection_mode,
                     "selection_metric": config.selection_metric,
                     "selection_profile": config.selection_profile,
                     "selection_weights": parse_selection_weights(config.selection_weights),
                     "trail_policy": config.trail_policy,
                     "method": config.method,
                     "reduction_storage": config.reduction_storage,
                     "selection_percentage": config.selection_percentage},
        "frames": entries,
    }
    config.output_dir.mkdir(parents=True, exist_ok=True)
    path = config.output_dir / f"{Path(config.output_name).stem}_selection.json"
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def _read_hdu_section(
    hdu: Any,
    slices: tuple[slice, ...],
) -> np.ndarray:
    section = getattr(hdu, "section", None)

    if section is not None:
        try:
            value = section[slices]
            return np.asarray(value, dtype=np.float32)
        except Exception:
            pass

    return np.asarray(hdu.data[slices], dtype=np.float32)


def _read_mask_section(
    hdu: Any,
    y1: int,
    y2: int,
    x1: int,
    x2: int,
) -> np.ndarray:
    try:
        section = getattr(hdu, "section", None)

        if section is not None:
            value = section[y1:y2, x1:x2]
        else:
            value = hdu.data[y1:y2, x1:x2]

        return np.asarray(value) > 0
    except Exception:
        return np.ones((y2 - y1, x2 - x1), dtype=bool)


def _load_full_valid_mask(
    frame: FrameInfo,
    geometry: FrameGeometry,
) -> np.ndarray | None:
    """Load a frame mask once so block reads do not decompress it repeatedly."""
    if frame.path.suffix.casefold() in TIFF_SUFFIXES:
        arrays = read_tiff_sidecar_arrays(frame.path)
        values = arrays.get("valid_mask")
        if values is None:
            return None
        mask = np.asarray(values, dtype=bool)
        expected_shape = (geometry.height, geometry.width)
        return mask if mask.shape == expected_shape else None
    if geometry.mask_hdu_index is None:
        return None

    with fits.open(
        frame.path,
        memmap=False,
        lazy_load_hdus=True,
        do_not_scale_image_data=False,
    ) as hdul:
        mask_hdu = hdul[geometry.mask_hdu_index]
        try:
            section = getattr(mask_hdu, "section", None)
            values = section[...] if section is not None else mask_hdu.data
            mask = np.asarray(values) > 0
        except Exception:
            return None

    expected_shape = (geometry.height, geometry.width)
    if mask.shape != expected_shape:
        return None
    return np.asarray(mask, dtype=bool)


def _fits_cache_key(path: Path) -> str:
    stat = path.stat()
    identity = f"{path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}"
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


def _is_compressed_hdu(hdu: Any) -> bool:
    return isinstance(hdu, fits.CompImageHDU)


def _has_active_scaling(hdu: Any) -> bool:
    header = hdu.header
    return (
        _is_compressed_hdu(hdu)
        or "BLANK" in header
        or float(header.get("BSCALE", 1.0)) != 1.0
        or float(header.get("BZERO", 0.0)) != 0.0
    )


def _cache_file_is_raw(path: Path, expected_key: str) -> bool:
    try:
        with fits.open(
            path,
            memmap=False,
            lazy_load_hdus=True,
            do_not_scale_image_data=True,
        ) as hdul:
            _, image_hdu = _find_primary_image_hdu(hdul)
            header = image_hdu.header
            if (
                int(header.get("ASTCVER", 0)) != FITS_CACHE_FORMAT_VERSION
                or str(header.get("ASTCKEY", "")) != expected_key
                or not bool(header.get("ASTRAW", False))
            ):
                return False
            if _has_active_scaling(image_hdu):
                return False
        return True
    except Exception:
        return False


def _raw_image_header(header: fits.Header) -> fits.Header:
    """Keep science metadata but remove compression/scaling implementation cards."""
    raw_header = header.copy()
    structural = {
        "XTENSION",
        "BITPIX",
        "NAXIS",
        "PCOUNT",
        "GCOUNT",
        "THEAP",
        "BSCALE",
        "BZERO",
        "BLANK",
    }
    for keyword in list(raw_header):
        if keyword in structural or keyword.startswith("Z"):
            del raw_header[keyword]
    return raw_header


def _cache_header_for_raw_storage(
    header: fits.Header,
    source_key: str,
) -> fits.Header:
    raw_header = _raw_image_header(header)
    raw_header["ASTRAW"] = (True, "AstroStack raw-storage cache")
    raw_header["ASTCVER"] = (FITS_CACHE_FORMAT_VERSION, "AstroStack cache format")
    raw_header["ASTCKEY"] = source_key
    raw_header["ASTBSCL"] = (float(header.get("BSCALE", 1.0)), "Original BSCALE")
    raw_header["ASTBZRO"] = (float(header.get("BZERO", 0.0)), "Original BZERO")
    if "BLANK" in header:
        raw_header["ASTBLNK"] = (int(header["BLANK"]), "Original BLANK")
    return raw_header


def _restore_cached_physical_values(
    data: np.ndarray,
    geometry: FrameGeometry,
) -> np.ndarray:
    if not geometry.cache_raw_storage:
        return np.asarray(data, dtype=np.float32)

    raw = np.asarray(data, dtype=np.float32)
    values = raw * np.float32(geometry.cache_bscale) + np.float32(geometry.cache_bzero)
    if geometry.cache_blank is not None:
        values = np.where(raw == geometry.cache_blank, np.nan, values)
    return np.asarray(values, dtype=np.float32)


def _restore_streaming_physical_values(
    data: np.ndarray,
    geometry: FrameGeometry,
) -> np.ndarray:
    """Restore physical values from a raw streaming FITS section.

    The streaming opener deliberately requests ``do_not_scale_image_data`` so
    uncompressed camera files with BZERO/BSCALE can stay memory-mapped.  Raw
    cache entries carry the original cards as ``AST*`` metadata; source files
    use their normal FITS cards.  Both paths use float32 operations in the
    same order as Astropy's scaler and preserve BLANK as NaN.
    """
    if geometry.cache_raw_storage:
        return _restore_cached_physical_values(data, geometry)
    if not geometry.requires_scaling:
        return np.asarray(data, dtype=np.float32)

    raw = np.asarray(data, dtype=np.float32)
    values = raw * np.float32(geometry.source_bscale) + np.float32(
        geometry.source_bzero
    )
    if geometry.source_blank is not None:
        values = np.where(raw == geometry.source_blank, np.nan, values)
    return np.asarray(values, dtype=np.float32)


def _valid_cached_geometry(
    cached_path: Path,
    source_key: str,
    source_geometry: FrameGeometry,
) -> FrameGeometry | None:
    """Return a reusable cache geometry without reading the source FITS."""
    if not cached_path.is_file() or not _cache_file_is_raw(cached_path, source_key):
        return None

    try:
        candidate = inspect_fits(cached_path)
    except Exception:
        return None

    if (
        candidate.height != source_geometry.height
        or candidate.width != source_geometry.width
        or candidate.image_kind != source_geometry.image_kind
        or candidate.channels != source_geometry.channels
        or (candidate.mask_hdu_index is not None)
        != (source_geometry.mask_hdu_index is not None)
        or not candidate.cache_raw_storage
    ):
        return None
    return candidate


def cache_decompressed_frames(
    frames: list[FrameInfo],
    geometries: dict[Path, FrameGeometry],
    cache_dir: Path,
    app_print: Callable[[str], None],
    cancel_event: threading.Event | None,
) -> tuple[dict[Path, FrameGeometry], FitsCacheStats]:
    """Persist scaled/compressed FITS pixels as raw FITS without RAM caching.

    The cache key includes source path, size and mtime.  A changed source gets a
    different entry automatically; existing entries are safe to reuse between
    runs.  Only one frame is decoded at a time.  In addition to tile-compressed
    HDUs, this covers the uncompressed ``BZERO/BSCALE`` representation emitted
    by many dedicated cameras.  Astropy disables mmap for those files, so
    caching their raw storage is what turns the hot band reads into contiguous
    mmap slices while restoring the original physical float32 values exactly.
    """
    updated: dict[Path, FrameGeometry] = {}
    stats = FitsCacheStats()

    for index, frame in enumerate(frames, start=1):
        check_cancel(cancel_event)
        source_path = frame.path
        geometry = geometries[source_path]
        key = _fits_cache_key(source_path)
        cached_path = cache_dir / f"{key}.fits"

        try:
            cached_geometry = _valid_cached_geometry(cached_path, key, geometry)

            if cached_geometry is not None:
                frame.path = cached_path
                frame.has_valid_mask = cached_geometry.mask_hdu_index is not None
                frame.valid_mask = None
                updated[cached_path] = cached_geometry
                stats.hits += 1
                if index % max(1, len(frames) // 20) == 0 or index == len(frames):
                    app_print(f"[Stack] FITS cache: {index}/{len(frames)}")
                continue

            with fits.open(
                source_path,
                memmap=False,
                lazy_load_hdus=True,
                do_not_scale_image_data=True,
            ) as hdul:
                image_hdu = hdul[geometry.hdu_index]
                mask_hdu = (
                    hdul[geometry.mask_hdu_index]
                    if geometry.mask_hdu_index is not None
                    else None
                )
                cache_candidate = _is_compressed_hdu(image_hdu) or _has_active_scaling(
                    image_hdu
                )
                if not cache_candidate:
                    updated[source_path] = geometry
                    stats.direct += 1
                    if index % max(1, len(frames) // 20) == 0 or index == len(frames):
                        app_print(f"[Stack] FITS cache: {index}/{len(frames)}")
                    continue

                if not _is_compressed_hdu(image_hdu):
                    # Keep this distinction visible in diagnostics: the
                    # source is a normal image HDU, but its BZERO/BSCALE/BLANK
                    # cards prevented mmap and caused the real-capture I/O
                    # hotspot.
                    stats.scaled += 1

                # Preserve FITS's raw signed storage. Original scaling is
                # stored as AST* metadata and applied only to read bands.
                image_data = np.asarray(image_hdu.data)
                hdus: list[Any] = [
                    fits.PrimaryHDU(
                        image_data,
                        header=_cache_header_for_raw_storage(image_hdu.header, key),
                    )
                ]
                if mask_hdu is not None:
                    hdus.append(
                        fits.ImageHDU(
                            np.asarray(mask_hdu.data, dtype=np.int8),
                            name="VALID_MASK",
                        )
                    )

            cache_dir.mkdir(parents=True, exist_ok=True)
            temporary_path = cache_dir / f".{key}.{uuid.uuid4().hex}.tmp"
            try:
                fits.HDUList(hdus).writeto(temporary_path, overwrite=True)
                temporary_path.replace(cached_path)
            finally:
                temporary_path.unlink(missing_ok=True)
            cached_geometry = inspect_fits(cached_path)

            frame.path = cached_path
            frame.has_valid_mask = cached_geometry.mask_hdu_index is not None
            frame.valid_mask = None
            updated[cached_path] = cached_geometry
            stats.rebuilt += 1
        except Exception as exc:
            # A cache failure must never prevent stacking the original frame.
            updated[source_path] = geometry
            stats.skipped += 1
            app_print(f"[Stack] Cache skipped for {frame.name}: {exc}")

        if index % max(1, len(frames) // 20) == 0 or index == len(frames):
            app_print(f"[Stack] FITS cache: {index}/{len(frames)}")

    return updated, stats


def _apply_translation(
    data: np.ndarray,
    dx: float,
    dy: float,
) -> np.ndarray:
    ix = int(round(dx))
    iy = int(round(dy))

    if ix == 0 and iy == 0:
        return data

    shifted = np.zeros_like(data)

    src_y1 = max(0, -iy)
    src_y2 = min(data.shape[0], data.shape[0] - iy)
    src_x1 = max(0, -ix)
    src_x2 = min(data.shape[1], data.shape[1] - ix)

    dst_y1 = max(0, iy)
    dst_y2 = dst_y1 + max(0, src_y2 - src_y1)
    dst_x1 = max(0, ix)
    dst_x2 = dst_x1 + max(0, src_x2 - src_x1)

    if src_y2 <= src_y1 or src_x2 <= src_x1:
        return shifted

    shifted[dst_y1:dst_y2, dst_x1:dst_x2] = data[
        src_y1:src_y2,
        src_x1:src_x2,
    ]
    return shifted


def read_frame_block(
    frame: FrameInfo,
    geometry: FrameGeometry,
    y1: int,
    y2: int,
    x1: int,
    x2: int,
    channel: int | None,
    normalization_factor: float,
    dither_shift: tuple[float, float] | None = None,
) -> BlockRead:
    ix = int(round(dither_shift[0])) if dither_shift else 0
    iy = int(round(dither_shift[1])) if dither_shift else 0

    out_h = y2 - y1
    out_w = x2 - x1

    src_y1 = y1 - iy
    src_y2 = y2 - iy
    src_x1 = x1 - ix
    src_x2 = x2 - ix

    img_h = geometry.height
    img_w = geometry.width

    valid_src_y1 = max(0, src_y1)
    valid_src_y2 = min(img_h, src_y2)
    valid_src_x1 = max(0, src_x1)
    valid_src_x2 = min(img_w, src_x2)

    data_out = np.zeros((out_h, out_w), dtype=np.float32)
    mask_out = np.zeros((out_h, out_w), dtype=bool)

    if valid_src_y2 <= valid_src_y1 or valid_src_x2 <= valid_src_x1:
        return BlockRead(data=data_out, mask=mask_out)

    dst_y1 = valid_src_y1 - src_y1
    dst_y2 = dst_y1 + (valid_src_y2 - valid_src_y1)
    dst_x1 = valid_src_x1 - src_x1
    dst_x2 = dst_x1 + (valid_src_x2 - valid_src_x1)

    # TIFF inputs are accepted as a compatibility path.  Prefer tifffile's
    # memory map for uncompressed captures so a full frame is never copied;
    # compressed TIFFs fall back to the bounded reader in image_io.
    if frame.path.suffix.casefold() in TIFF_SUFFIXES:
        try:
            import tifffile
            source = tifffile.memmap(frame.path)
        except Exception:
            source, _ = read_tiff(frame.path)
        source = np.asarray(source)
        if geometry.image_kind == "Mono":
            raw_data = np.asarray(
                source[valid_src_y1:valid_src_y2, valid_src_x1:valid_src_x2],
                dtype=np.float32,
            )
        elif source.ndim == 3 and source.shape[0] == geometry.channels and source.shape[-1] != geometry.channels:
            if channel is None:
                raise ValueError("Canal não informado para imagem RGB.")
            raw_data = np.asarray(
                source[channel, valid_src_y1:valid_src_y2, valid_src_x1:valid_src_x2],
                dtype=np.float32,
            )
        else:
            if channel is None:
                raise ValueError("Canal não informado para imagem RGB.")
            raw_data = np.asarray(
                source[valid_src_y1:valid_src_y2, valid_src_x1:valid_src_x2, channel],
                dtype=np.float32,
            )
        if normalization_factor != 1.0:
            raw_data *= np.float32(normalization_factor)
        finite = np.isfinite(raw_data)
        raw_mask = finite
        if frame.valid_mask is not None:
            raw_mask &= frame.valid_mask[valid_src_y1:valid_src_y2, valid_src_x1:valid_src_x2]
        else:
            arrays = read_tiff_sidecar_arrays(frame.path)
            valid_sidecar = arrays.get("valid_mask")
            sat_sidecar = arrays.get("sat_mask")
            if valid_sidecar is not None:
                raw_mask &= np.asarray(valid_sidecar[valid_src_y1:valid_src_y2, valid_src_x1:valid_src_x2], dtype=bool)
            if sat_sidecar is not None:
                raw_mask &= ~np.asarray(sat_sidecar[valid_src_y1:valid_src_y2, valid_src_x1:valid_src_x2], dtype=bool)
        if not finite.all():
            raw_data = np.where(finite, raw_data, 0.0).astype(np.float32, copy=False)
        data_out[dst_y1:dst_y2, dst_x1:dst_x2] = raw_data
        mask_out[dst_y1:dst_y2, dst_x1:dst_x2] = raw_mask
        return BlockRead(data=data_out, mask=np.asarray(mask_out, dtype=bool))

    with fits.open(
        frame.path,
        memmap=False,
        lazy_load_hdus=True,
        do_not_scale_image_data=False,
    ) as hdul_local:
        image_hdu = hdul_local[geometry.hdu_index]

        if geometry.image_kind == "Mono":
            raw_data = _read_hdu_section(
                image_hdu,
                (slice(valid_src_y1, valid_src_y2), slice(valid_src_x1, valid_src_x2)),
            )
        elif image_hdu.shape[0] == geometry.channels:
            if channel is None:
                raise ValueError("Canal não informado para imagem RGB.")

            raw_data = _read_hdu_section(
                image_hdu,
                (
                    slice(channel, channel + 1),
                    slice(valid_src_y1, valid_src_y2),
                    slice(valid_src_x1, valid_src_x2),
                ),
            )[0]
        else:
            if channel is None:
                raise ValueError("Canal não informado para imagem RGB.")

            raw_data = _read_hdu_section(
                image_hdu,
                (
                    slice(valid_src_y1, valid_src_y2),
                    slice(valid_src_x1, valid_src_x2),
                    slice(channel, channel + 1),
                ),
            )[:, :, 0]

        raw_data = _restore_cached_physical_values(raw_data, geometry)

        if normalization_factor != 1.0:
            raw_data *= np.float32(normalization_factor)

        finite = np.isfinite(raw_data)

        if frame.valid_mask is not None:
            external_mask = frame.valid_mask[
                valid_src_y1:valid_src_y2,
                valid_src_x1:valid_src_x2,
            ]
            raw_mask = finite & external_mask
        elif geometry.mask_hdu_index is not None:
            mask_hdu = hdul_local[geometry.mask_hdu_index]
            external_mask = _read_mask_section(
                mask_hdu,
                valid_src_y1,
                valid_src_y2,
                valid_src_x1,
                valid_src_x2,
            )
            raw_mask = finite & external_mask
        else:
            raw_mask = finite

        if not finite.all():
            raw_data = np.where(finite, raw_data, 0.0).astype(
                np.float32,
                copy=False,
            )

    data_out[dst_y1:dst_y2, dst_x1:dst_x2] = raw_data
    mask_out[dst_y1:dst_y2, dst_x1:dst_x2] = raw_mask

    return BlockRead(
        data=data_out,
        mask=np.asarray(mask_out, dtype=bool),
    )


def load_normalization_sample(
    frame: FrameInfo,
    geometry: FrameGeometry,
    max_samples: int,
) -> tuple[np.ndarray, np.ndarray | None]:
    if frame.path.suffix.casefold() in TIFF_SUFFIXES:
        try:
            import tifffile
            source = np.asarray(tifffile.memmap(frame.path))
        except Exception:
            source, _ = read_tiff(frame.path)
            source = np.asarray(source)
        height, width = geometry.height, geometry.width
        pixels = max(1, height * width)
        spatial_step = max(1, int(math.sqrt(pixels / max(1, max_samples))))
        if geometry.image_kind == "Mono":
            sample = source[0:height:spatial_step, 0:width:spatial_step]
        elif source.ndim == 3 and source.shape[-1] in (3, 4):
            sample = np.moveaxis(source[0:height:spatial_step, 0:width:spatial_step, :], -1, 0)
        else:
            sample = source[:, 0:height:spatial_step, 0:width:spatial_step]
        arrays = read_tiff_sidecar_arrays(frame.path)
        mask = arrays.get("valid_mask")
        sat = arrays.get("sat_mask")
        if mask is not None:
            mask = np.asarray(mask[0:height:spatial_step, 0:width:spatial_step], dtype=bool)
        if sat is not None:
            sat_mask = np.asarray(sat[0:height:spatial_step, 0:width:spatial_step], dtype=bool)
            mask = ~sat_mask if mask is None else mask & ~sat_mask
        return np.asarray(sample, dtype=np.float32), mask
    with fits.open(
        frame.path,
        memmap=False,
        lazy_load_hdus=True,
        do_not_scale_image_data=False,
    ) as hdul:
        image_hdu = hdul[geometry.hdu_index]
        shape = tuple(image_hdu.shape)

        height = geometry.height
        width = geometry.width
        pixels = max(1, height * width)

        spatial_step = max(
            1,
            int(math.sqrt(pixels / max(1, max_samples))),
        )

        y_slice = slice(0, height, spatial_step)
        x_slice = slice(0, width, spatial_step)

        if geometry.image_kind == "Mono":
            sample = _read_hdu_section(
                image_hdu,
                (y_slice, x_slice),
            )
        elif shape[0] == geometry.channels:
            sample = _read_hdu_section(
                image_hdu,
                (slice(None), y_slice, x_slice),
            )
        else:
            sample = _read_hdu_section(
                image_hdu,
                (y_slice, x_slice, slice(None)),
            )

        mask = None

        if frame.valid_mask is not None:
            mask = frame.valid_mask[::spatial_step, ::spatial_step]
        elif geometry.mask_hdu_index is not None:
            mask_hdu = hdul[geometry.mask_hdu_index]
            mask = _read_mask_section(
                mask_hdu,
                0,
                height,
                0,
                width,
            )
            mask = mask[::spatial_step, ::spatial_step]

        return _restore_cached_physical_values(sample, geometry), mask


def _sample_valid_values(
    values: np.ndarray,
    mask: np.ndarray | None,
    max_samples: int,
) -> np.ndarray:
    if values.ndim == 2:
        flat = values.ravel()

        if mask is not None:
            flat = flat[mask.ravel()]
    elif mask is not None:
        if values.shape[0] == 3 or values.shape[0] == 4:
            flat = values[:, mask].ravel()
        else:
            flat = values[mask[..., None]].ravel()
    else:
        flat = values.ravel()

    flat = flat[np.isfinite(flat)]

    if flat.size <= max_samples:
        return flat

    step = max(1, flat.size // max_samples)
    return flat[::step][:max_samples]


def estimate_normalization_value(
    frame: FrameInfo,
    geometry: FrameGeometry,
    method: str,
    max_samples: int,
) -> float:
    data, mask = load_normalization_sample(
        frame,
        geometry,
        max_samples,
    )

    sample = _sample_valid_values(
        data,
        mask,
        max_samples,
    )

    if sample.size == 0:
        return 0.0

    if method == "Mode":
        p05, p95 = np.percentile(sample, [5.0, 95.0])

        if p95 > p05:
            hist, edges = np.histogram(
                sample,
                bins=64,
                range=(p05, p95),
            )

            if hist.size and np.any(hist):
                index = int(np.argmax(hist))
                return float((edges[index] + edges[index + 1]) * 0.5)

    return float(np.median(sample))


def calculate_normalization_factors(
    selected_frames: list[FrameInfo],
    geometries: dict[Path, FrameGeometry],
    config: StackingConfig,
    app_print: Callable[[str], None],
    cancel_event: threading.Event | None,
) -> list[float] | None:
    if not config.normalize:
        return [1.0] * len(selected_frames)

    if not selected_frames:
        return []

    app_print(f"\n⚖️ Calculando normalização ({config.normalize_method})...")

    check_cancel(cancel_event)

    reference_value = estimate_normalization_value(
        selected_frames[0],
        geometries[selected_frames[0].path],
        config.normalize_method,
        config.normalization_max_samples,
    )

    if not math.isfinite(reference_value) or reference_value <= 0:
        reference_value = 1.0

    factors: list[float] = []
    report_step = max(1, len(selected_frames) // 20)

    for index, frame in enumerate(selected_frames):
        check_cancel(cancel_event)

        current_value = estimate_normalization_value(
            frame,
            geometries[frame.path],
            config.normalize_method,
            config.normalization_max_samples,
        )

        if not math.isfinite(current_value) or current_value <= 0:
            factor = 1.0
        else:
            factor = reference_value / current_value

        factors.append(float(factor))

        if index % report_step == 0 or index == len(selected_frames) - 1:
            app_print(f"   Normalização: {index + 1}/{len(selected_frames)}")

    return factors


def _nanpercentile_axis0(
    values: np.ndarray,
    percentile: float,
) -> np.ndarray:
    """Compute a NaN-aware percentile without NumPy's per-column Python loop."""
    sorted_values = np.sort(values, axis=0)
    valid_count = np.sum(~np.isnan(values), axis=0, dtype=np.intp)
    rank = (valid_count - 1) * (percentile / 100.0)
    lower_index = np.floor(np.maximum(rank, 0)).astype(np.intp)
    upper_index = np.ceil(np.maximum(rank, 0)).astype(np.intp)
    lower = np.take_along_axis(sorted_values, lower_index[None, ...], axis=0)[0]
    upper = np.take_along_axis(sorted_values, upper_index[None, ...], axis=0)[0]
    fraction = rank - lower_index
    result = lower + (upper - lower) * fraction
    return np.where(valid_count > 0, result, np.nan).astype(np.float32, copy=False)


def _nanmedian_axis0_no_warning(values: np.ndarray) -> np.ndarray:
    """NaN median that skips all-NaN columns without emitting a warning."""
    flat = np.asarray(values, dtype=np.float32).reshape(values.shape[0], -1)
    valid_columns = np.any(~np.isnan(flat), axis=0)
    result = np.full(flat.shape[1], np.nan, dtype=np.float32)
    if np.any(valid_columns):
        result[valid_columns] = np.nanmedian(flat[:, valid_columns], axis=0)
    return result.reshape(values.shape[1:])


def _combine_masked_impl(
    values: np.ndarray,
    masks: np.ndarray,
    combine_method: str,
    profile: EngineProfile = EngineProfile.STABLE,
    kernel_parallel: bool = False,
) -> np.ndarray:
    """Combine a block without materialising a NaN-filled float copy."""
    values = np.asarray(values, dtype=np.float32)
    # Exclude NaN, but deliberately retain infinity: this mirrors NumPy's
    # nan-reducers and the caller's final nan_to_num policy.
    valid = masks & ~np.isnan(values)

    if combine_method in {"Mean", "Sum"}:
        if profile is EngineProfile.FAST:
            total, count = masked_sum_count(values, valid, parallel=kernel_parallel)
            if combine_method == "Sum":
                return total
            return np.divide(
                total,
                count,
                out=np.full(total.shape, np.nan, dtype=np.float32),
                where=count > 0,
            )
        total = np.sum(np.where(valid, values, np.float32(0.0)), axis=0)
        if combine_method == "Sum":
            return total
        count = np.sum(valid, axis=0, dtype=np.intp)
        return np.divide(
            total,
            count,
            out=np.full(total.shape, np.nan, dtype=np.float32),
            where=count > 0,
        )
    if combine_method == "Maximum":
        return masked_extrema(
            values,
            valid,
            True,
            parallel=kernel_parallel and profile is EngineProfile.FAST,
        )
    if combine_method == "Minimum":
        return masked_extrema(
            values,
            valid,
            False,
            parallel=kernel_parallel and profile is EngineProfile.FAST,
        )
    if combine_method == "Median":
        return _nanmedian_axis0_no_warning(np.where(valid, values, np.nan))
    raise ValueError(f"Unsupported combine method: {combine_method}")


def _stable_stack_reducer(values: np.ndarray, masks: np.ndarray, method: str) -> np.ndarray:
    return _combine_masked_impl(values, masks, method, EngineProfile.STABLE)


def _fast_stack_reducer(
    values: np.ndarray, masks: np.ndarray, method: str, kernel_parallel: bool = False
) -> np.ndarray:
    return _combine_masked_impl(values, masks, method, EngineProfile.FAST, kernel_parallel)


def _register_stack_engines() -> None:
    registry.register(
        EngineDescriptor("stable-numpy", "stack.reducer", "Stable NumPy", frozenset({EngineProfile.STABLE, EngineProfile.FAST})),
        _stable_stack_reducer,
    )
    registry.register(
        EngineDescriptor("fast-numba", "stack.reducer", "Fast Numba", frozenset({EngineProfile.FAST}), capabilities=frozenset({"mean", "sum", "minimum", "maximum"})),
        _fast_stack_reducer,
    )


def _combine_masked_cpu(
    values: np.ndarray,
    masks: np.ndarray,
    combine_method: str,
    engine_profile: str = "Stable",
    reducer_engine: str | None = None,
    kernel_parallel: bool = False,
) -> np.ndarray:
    _register_stack_engines()
    profile = EngineProfile.coerce(engine_profile)
    eligible_fast = combine_method in {"Mean", "Sum", "Minimum", "Maximum"}
    engine_id = reducer_engine or ("fast-numba" if profile is EngineProfile.FAST and eligible_fast else "stable-numpy")
    reducer = registry.resolve("stack.reducer", engine_id, profile)
    if engine_id == "fast-numba":
        return reducer(values, masks, combine_method, kernel_parallel)
    return reducer(values, masks, combine_method)


def _reject_cpu_all_valid(
    values: np.ndarray,
    method: str,
    low: float,
    high: float,
) -> np.ndarray:
    """Fast rejection path for the common fully valid, finite image interior."""
    if method == "SigmaClip":
        center = np.median(values, axis=0)
        std = np.std(values, axis=0)
        valid = (values >= center - low * std) & (values <= center + high * std)
        valid |= ~(std > 1e-10)
        if np.all(valid):
            return np.asarray(center, dtype=np.float32)
        return np.asarray(
            np.nanmedian(np.where(valid, values, np.nan), axis=0),
            dtype=np.float32,
        )

    if method == "MAD":
        center = np.median(values, axis=0)
        mad = np.median(np.abs(values - center), axis=0)
        sigma = np.float32(1.4826) * mad
        valid = (values >= center - low * sigma) & (values <= center + high * sigma)
        valid |= ~(mad > 1e-10)
        if np.all(valid):
            return np.asarray(center, dtype=np.float32)
        return np.asarray(
            np.nanmedian(np.where(valid, values, np.nan), axis=0),
            dtype=np.float32,
        )

    if method == "Winsorized":
        p_low = np.percentile(values, min(49.0, max(0.0, low * 10.0)), axis=0)
        p_high = np.percentile(
            values,
            max(51.0, min(100.0, 100.0 - high * 10.0)),
            axis=0,
        )
        return np.asarray(
            np.median(np.clip(values, p_low, p_high), axis=0), dtype=np.float32
        )

    raise ValueError(f"Unsupported rejection method: {method}")


def _reject_cpu(
    values: np.ndarray,
    masks: np.ndarray,
    method: str,
    low: float,
    high: float,
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if np.all(masks) and np.isfinite(values).all():
        return _reject_cpu_all_valid(values, method, low, high)

    masked = values.astype(np.float32, copy=True)
    masked[~masks] = np.nan

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)

        if method == "SigmaClip":
            center = _nanmedian_axis0_no_warning(masked)
            std = np.nanstd(masked, axis=0)

            lower = center - low * std
            upper = center + high * std

            valid = (masked >= lower) & (masked <= upper)

            stable = np.isfinite(std) & (std > 1e-10)
            valid |= ~stable
            masked[~valid] = np.nan

        elif method == "MAD":
            center = _nanmedian_axis0_no_warning(masked)
            mad = _nanmedian_axis0_no_warning(np.abs(masked - center))

            sigma = np.float32(1.4826) * mad
            lower = center - low * sigma
            upper = center + high * sigma

            valid = (masked >= lower) & (masked <= upper)

            stable = np.isfinite(mad) & (mad > 1e-10)
            valid |= ~stable
            masked[~valid] = np.nan

        elif method == "Winsorized":
            p_low = _nanpercentile_axis0(
                masked,
                min(49.0, max(0.0, low * 10.0)),
            )
            p_high = _nanpercentile_axis0(
                masked,
                max(51.0, min(100.0, 100.0 - high * 10.0)),
            )

            valid_range = np.isfinite(p_low) & np.isfinite(p_high) & (p_high >= p_low)

            masked = np.where(
                valid_range,
                np.clip(masked, p_low, p_high),
                masked,
            )

        if method in {"SigmaClip", "MAD"}:
            return np.asarray(
                _nanmedian_axis0_no_warning(masked),
                dtype=np.float32,
            )

        return np.asarray(
            _nanmedian_axis0_no_warning(masked),
            dtype=np.float32,
        )


def _weighted_rejection_values(
    values: np.ndarray,
    masks: np.ndarray,
    method: str,
    low: float,
    high: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return values and an acceptance mask for ``QualityWeightedMean``.

    The center/dispersion calculation intentionally ignores quality weights;
    otherwise one high-weight artifact could influence the rejection boundary
    that is supposed to protect the stack.  Weighted accumulation happens
    only after this deterministic Stable mask has been computed.
    """

    values = np.asarray(values, dtype=np.float32)
    masked = values.astype(np.float32, copy=True)
    valid_input = np.asarray(masks, dtype=bool) & ~np.isnan(masked)
    masked[~valid_input] = np.nan
    if method == "None" or values.shape[0] <= 3:
        return masked, valid_input
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        if method == "SigmaClip":
            center = _nanmedian_axis0_no_warning(masked)
            spread = np.nanstd(masked, axis=0)
            lower = center - np.float32(low) * spread
            upper = center + np.float32(high) * spread
            accepted = (masked >= lower) & (masked <= upper)
            accepted |= ~(np.isfinite(spread) & (spread > 1e-10))
        elif method == "MAD":
            center = _nanmedian_axis0_no_warning(masked)
            mad = _nanmedian_axis0_no_warning(np.abs(masked - center))
            spread = np.float32(1.4826) * mad
            lower = center - np.float32(low) * spread
            upper = center + np.float32(high) * spread
            accepted = (masked >= lower) & (masked <= upper)
            accepted |= ~(np.isfinite(mad) & (mad > 1e-10))
        elif method == "Winsorized":
            p_low = _nanpercentile_axis0(masked, min(49.0, max(0.0, low * 10.0)))
            p_high = _nanpercentile_axis0(masked, max(51.0, min(100.0, 100.0 - high * 10.0)))
            valid_range = np.isfinite(p_low) & np.isfinite(p_high) & (p_high >= p_low)
            clipped = np.where(valid_range, np.clip(masked, p_low, p_high), masked)
            return clipped, valid_input
        else:
            raise ValueError(f"Unsupported rejection method: {method}")
    accepted &= valid_input
    masked[~accepted] = np.nan
    return masked, accepted


def _quality_weighted_mean(
    values: np.ndarray,
    masks: np.ndarray,
    rejection_method: str,
    low: float,
    high: float,
    frame_weights: np.ndarray,
) -> np.ndarray:
    result, _denominator = _quality_weighted_reduce(
        values,
        masks,
        rejection_method,
        low,
        high,
        frame_weights,
    )
    return result


def _quality_weighted_reduce(
    values: np.ndarray,
    masks: np.ndarray,
    rejection_method: str,
    low: float,
    high: float,
    frame_weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return a weighted mean and its per-pixel effective weight sum."""

    if values.ndim != 3 or masks.shape != values.shape:
        raise ValueError("Bloco inválido para média ponderada.")
    weights = np.asarray(frame_weights, dtype=np.float32).reshape(-1)
    if weights.size != values.shape[0]:
        raise ValueError("Número de pesos incompatível com o bloco.")
    weights = np.clip(np.nan_to_num(weights, nan=0.0, posinf=2.0, neginf=0.0), 0.0, 2.0)
    reduced, accepted = _weighted_rejection_values(
        values,
        masks,
        rejection_method,
        low,
        high,
    )
    finite = accepted & ~np.isnan(reduced)
    expanded_weights = weights[:, None, None]
    weighted = np.where(finite, reduced, np.float32(0.0)) * expanded_weights
    denominator = np.sum(np.where(finite, expanded_weights, np.float32(0.0)), axis=0, dtype=np.float32)
    numerator = np.sum(weighted, axis=0, dtype=np.float32)
    result = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=np.float32),
        where=denominator > 0,
    ).astype(np.float32, copy=False)
    return result, denominator.astype(np.float32, copy=False)


def _quality_weighted_merge(
    values: np.ndarray,
    masks: np.ndarray,
    weight_sums: np.ndarray,
    rejection_method: str,
    low: float,
    high: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Merge weighted child nodes while retaining deterministic rejection."""

    if values.ndim != 3 or masks.shape != values.shape or weight_sums.shape != values.shape:
        raise ValueError("Nós incompatíveis para média ponderada.")
    reduced, accepted = _weighted_rejection_values(
        values,
        masks,
        rejection_method,
        low,
        high,
    )
    finite = accepted & ~np.isnan(reduced)
    effective = np.where(
        finite,
        np.asarray(weight_sums, dtype=np.float32),
        np.float32(0.0),
    )
    numerator = np.sum(
        np.where(finite, reduced * effective, np.float32(0.0)),
        axis=0,
        dtype=np.float32,
    )
    denominator = np.sum(effective, axis=0, dtype=np.float32)
    result = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=np.float32),
        where=denominator > 0,
    )
    return result.astype(np.float32, copy=False), denominator.astype(np.float32, copy=False)


def reject_and_combine_block(
    values: np.ndarray,
    masks: np.ndarray,
    combine_method: str,
    rejection_method: str,
    low: float,
    high: float,
    cancel_event: threading.Event | None = None,
    engine_profile: str = "Stable",
    reducer_engine: str | None = None,
    kernel_parallel: bool = False,
    frame_weights: np.ndarray | list[float] | tuple[float, ...] | None = None,
) -> np.ndarray:
    check_cancel(cancel_event)

    if values.ndim != 3 or masks.shape != values.shape:
        raise ValueError("Bloco inválido para rejeição.")

    if combine_method == "QualityWeightedMean":
        if frame_weights is None:
            frame_weights = np.ones(values.shape[0], dtype=np.float32)
        return _quality_weighted_mean(
            values,
            masks,
            rejection_method if rejection_method != "None" else "None",
            low,
            high,
            np.asarray(frame_weights, dtype=np.float32),
        )

    if rejection_method == "None" or values.shape[0] <= 3:
        rejection_method = "None"

    if rejection_method == "None":
        result = _combine_masked_cpu(
            values, masks, combine_method, engine_profile, reducer_engine, kernel_parallel
        )
        return np.asarray(
            np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0),
            dtype=np.float32,
        )

    return _reject_cpu(values, masks, rejection_method, low, high)


def _hierarchical_worker_count(config: StackingConfig, n_frames: int) -> int:
    """Use a few independent I/O streams, never one task per frame/block."""
    return max(1, min(8, config.worker_count, n_frames))


def _leaf_band_rows(
    config: StackingConfig,
    group_size: int,
    width: int,
    channels: int,
    height: int,
) -> int:
    # Values, masks and rejection temporaries need several copies of a band.
    # The calibrated factor leaves headroom for the reducer while avoiding the
    # excessive number of tiny FITS sections seen in the real 3856x2180 RGB
    # capture.  The budget is still divided by the bounded leaf worker count.
    bytes_per_row = max(
        1,
        group_size
        * width
        * max(1, channels)
        * STACK_BAND_BYTES_PER_PIXEL,
    )
    budget = max(
        64, config.memory_budget_mb // _hierarchical_worker_count(config, group_size)
    )
    return max(32, min(height, int((budget * 1024 * 1024) / bytes_per_row)))


def _partition_frames(items: list[Any], groups: int) -> list[list[Any]]:
    groups = max(1, min(groups, len(items)))
    return [items[index::groups] for index in range(groups)]


def _read_open_band(
    hdu: Any,
    frame: FrameInfo,
    geometry: FrameGeometry,
    y1: int,
    y2: int,
    channel: int | None,
    factor: float,
    shift: tuple[float, float] | None,
) -> BlockRead:
    # The open HDU is kept by the leaf worker for its entire group.  Reads are
    # full-width, monotonically increasing bands rather than random tiles.
    return read_frame_block(
        frame, geometry, y1, y2, 0, geometry.width, channel, factor, shift
    )


def _open_streaming_fits(
    path: Path,
    geometry: FrameGeometry | None = None,
) -> fits.HDUList:
    """Open a science stream with mmap whenever the geometry permits it.

    ``inspect_fits`` already reads the header once per frame.  Re-probing that
    header for every leaf is pure I/O overhead, especially on the 1,253-frame
    dedicated-camera capture.  Cached raw-storage geometry is guaranteed to
    have no active FITS scaling, while an uncached source carries the explicit
    ``requires_scaling`` flag.  The probe remains as a compatibility fallback
    for callers that do not have geometry available.
    """
    if path.suffix.casefold() in TIFF_SUFFIXES:
        try:
            import tifffile
            data = tifffile.memmap(path)
        except Exception:
            data, _ = read_tiff(path)
        return fits.HDUList([fits.PrimaryHDU(data=np.asarray(data))])

    if geometry is None:
        with fits.open(
            path,
            memmap=True,
            lazy_load_hdus=True,
            do_not_scale_image_data=True,
        ) as probe:
            _, image_hdu = _find_primary_image_hdu(probe)
            science_compressed = _is_compressed_hdu(image_hdu)
    else:
        science_compressed = bool(geometry.science_compressed)

    return fits.open(
        path,
        # Astropy cannot mmap a scaled image when it materializes ``.data``;
        # asking for raw storage avoids that conversion and lets the caller
        # restore BZERO/BSCALE on each contiguous band.  Tile-compressed HDUs
        # still decode in memory, but they use the same raw path after the
        # optional persistent cache.
        memmap=not science_compressed,
        lazy_load_hdus=True,
        do_not_scale_image_data=True,
    )


def _process_substack(
    frames: list[FrameInfo],
    geometries: dict[Path, FrameGeometry],
    factors: list[float],
    shifts: list[tuple[float, float] | None],
    config: StackingConfig,
    output_path: Path,
    height: int,
    width: int,
    channels: int,
    cancel_event: threading.Event | None,
    leaf_index: int = 0,
    progress_queue: queue.SimpleQueue[tuple[int, int, str]] | None = None,
    frame_weights: list[float] | None = None,
) -> SubstackInfo:
    """Create one leaf using wide row bands and a bounded number of files."""
    group_size = len(frames)
    band_rows = _leaf_band_rows(config, group_size, width, channels, height)
    total_bands = math.ceil(height / band_rows)
    shape = (height, width) if channels == 1 else (channels, height, width)
    result = np.zeros(shape, dtype=np.float32)
    counts = np.zeros((height, width), dtype=np.uint32)
    weight_sums = np.zeros(shape, dtype=np.float32) if config.method == "QualityWeightedMean" else None
    local_weights = np.asarray(frame_weights if frame_weights is not None else [1.0] * group_size, dtype=np.float32)
    if local_weights.size != group_size:
        raise ValueError("Número de pesos incompatível com a folha.")

    def report(increment: int, message: str) -> None:
        if progress_queue is not None:
            progress_queue.put((leaf_index, increment, message))

    # Opening each FITS once is the key change.  The previous spatial loop
    # reopened/decompressed each source for every x/y block.
    handles: list[fits.HDUList] = []
    try:
        for index, frame in enumerate(frames, start=1):
            check_cancel(cancel_event)
            report(
                0,
                f"Leaf {leaf_index + 1}: opening {index}/{group_size} ({frame.name})",
            )
            handles.append(
                _open_streaming_fits(frame.path, geometries[frame.path])
            )
    except Exception:
        for handle in handles:
            handle.close()
        raise
    try:
        last_frame_report = 0.0
        has_dither = any(shift is not None for shift in shifts)
        for band_index, y1 in enumerate(range(0, height, band_rows), start=1):
            check_cancel(cancel_event)
            y2 = min(height, y1 + band_rows)
            report(
                0,
                f"Leaf {leaf_index + 1}: reading band {band_index}/{total_bands}",
            )
            # Full VALID_MASK arrays are optional.  When the caller kept the
            # masks streamed to respect the RAM budget, read one mask section
            # per frame/band and reuse it for every RGB channel.
            band_masks: list[np.ndarray | None] | None = None
            if any(shift is None for shift in shifts):
                band_masks = []
                for frame, hdul in zip(frames, handles, strict=True):
                    frame_index = len(band_masks)
                    if shifts[frame_index] is not None:
                        # ``read_frame_block`` applies the translated mask
                        # itself; keep a placeholder for positional lookup.
                        band_masks.append(None)
                        continue
                    geometry = geometries[frame.path]
                    if frame.valid_mask is not None:
                        band_masks.append(
                            np.asarray(frame.valid_mask[y1:y2, :], dtype=bool)
                        )
                    elif geometry.mask_hdu_index is not None:
                        band_masks.append(
                            _read_mask_section(
                                hdul[geometry.mask_hdu_index],
                                y1,
                                y2,
                                0,
                                width,
                            )
                        )
                    else:
                        band_masks.append(None)
            # RGB FITS are commonly stored as [C,H,W].  Reading one full
            # channel stack per frame/band cuts the number of section calls by
            # 3x (or 4x for RGBA) while keeping the rejection/reduction order
            # identical: each channel is still combined independently below.
            # Dithered frames retain the established per-channel path because
            # their translated source window is not one contiguous section.
            batch_rgb = channels > 1 and not any(shift is not None for shift in shifts)
            if batch_rgb:
                values = np.empty(
                    (group_size, channels, y2 - y1, width), dtype=np.float32
                )
                masks = np.empty(values.shape, dtype=bool)
                for index, (frame, hdul) in enumerate(
                    zip(frames, handles, strict=True)
                ):
                    now = time.monotonic()
                    if index == 0 or now - last_frame_report >= 0.5:
                        report(
                            0,
                            f"Leaf {leaf_index + 1}: band {band_index}/{total_bands}, "
                            f"frame {index + 1}/{group_size} ({frame.name})",
                        )
                        last_frame_report = now
                    geometry = geometries[frame.path]
                    hdu = hdul[geometry.hdu_index]
                    channels_first = frame.path.suffix.casefold() not in TIFF_SUFFIXES
                    if channels_first:
                        raw = _read_hdu_section(
                            hdu,
                            (
                                slice(0, channels),
                                slice(y1, y2),
                                slice(0, width),
                            ),
                        )
                    else:
                        raw = _read_hdu_section(
                            hdu,
                            (
                                slice(y1, y2),
                                slice(0, width),
                                slice(0, channels),
                            ),
                        )
                        raw = np.moveaxis(raw, -1, 0)
                    raw = _restore_streaming_physical_values(raw, geometry)
                    mask = np.isfinite(raw)
                    if band_masks is not None and band_masks[index] is not None:
                        mask &= band_masks[index]
                    values[index] = raw
                    for channel in range(channels):
                        if EngineProfile.coerce(config.engine_profile) is EngineProfile.FAST:
                            apply_scale_and_mask_inplace(
                                values[index, channel], mask[channel], factors[index]
                            )
                        else:
                            if factors[index] != 1.0:
                                values[index, channel] *= np.float32(factors[index])
                            values[index, channel][~mask[channel]] = np.float32(0.0)
                    masks[index] = mask
                report(
                    0,
                    f"Leaf {leaf_index + 1}: combining band {band_index}/{total_bands}",
                )
                for channel in range(channels):
                    if config.method == "QualityWeightedMean":
                        combined, effective = _quality_weighted_reduce(
                            values[:, channel],
                            masks[:, channel],
                            config.rejection_method,
                            config.rejection_low,
                            config.rejection_high,
                            local_weights,
                        )
                        weight_sums[channel, y1:y2] = effective
                    else:
                        combined = reject_and_combine_block(
                            values[:, channel],
                            masks[:, channel],
                            config.method,
                            config.rejection_method,
                            config.rejection_low,
                            config.rejection_high,
                            cancel_event,
                            config.engine_profile,
                            config.reducer_engine,
                            ExecutionBudget.for_pipeline(config.worker_count).kernel_parallel,
                        )
                    result[channel, y1:y2] = combined
                counts[y1:y2] = masks[:, 0].sum(axis=0, dtype=np.uint32)
            else:
                for channel in range(channels):
                    values = np.empty((group_size, y2 - y1, width), dtype=np.float32)
                    masks = np.empty(values.shape, dtype=bool)
                    for index, (frame, hdul) in enumerate(
                        zip(frames, handles, strict=True)
                    ):
                        now = time.monotonic()
                        if index == 0 or now - last_frame_report >= 0.5:
                            report(
                                0,
                                f"Leaf {leaf_index + 1}: band {band_index}/{total_bands}, "
                                f"frame {index + 1}/{group_size} ({frame.name})",
                            )
                            last_frame_report = now
                        geometry = geometries[frame.path]
                        if shifts[index] is not None:
                            # Dithered bands need pixels outside their nominal row
                            # range.  Preserve the established translation path.
                            block = read_frame_block(
                                frame,
                                geometry,
                                y1,
                                y2,
                                0,
                                width,
                                channel,
                                factors[index],
                                shifts[index],
                            )
                            values[index] = block.data
                            masks[index] = block.mask
                            continue
                        # Inline the image section using the already-open HDU.
                        hdu = hdul[geometry.hdu_index]
                        if geometry.image_kind == "Mono":
                            raw = _read_hdu_section(hdu, (slice(y1, y2), slice(0, width)))
                        else:
                            channels_first = frame.path.suffix.casefold() not in TIFF_SUFFIXES
                            if channels_first:
                                raw = _read_hdu_section(
                                    hdu,
                                    (
                                        slice(channel, channel + 1),
                                        slice(y1, y2),
                                        slice(0, width),
                                    ),
                                )[0]
                            else:
                                raw = _read_hdu_section(
                                    hdu,
                                    (
                                        slice(y1, y2),
                                        slice(0, width),
                                        slice(channel, channel + 1),
                                    ),
                                )[:, :, 0]
                        raw = _restore_streaming_physical_values(raw, geometry)
                        mask = np.isfinite(raw)
                        if band_masks is not None and band_masks[index] is not None:
                            mask &= band_masks[index]
                        # Reuse the preallocated contiguous band slot instead of
                        # allocating a second full np.where result per frame.
                        values[index] = raw
                        if EngineProfile.coerce(config.engine_profile) is EngineProfile.FAST:
                            apply_scale_and_mask_inplace(values[index], mask, factors[index])
                        else:
                            if factors[index] != 1.0:
                                values[index] *= np.float32(factors[index])
                            values[index][~mask] = np.float32(0.0)
                        masks[index] = mask
                    report(
                        0,
                        f"Leaf {leaf_index + 1}: combining band {band_index}/{total_bands}",
                    )
                    if config.method == "QualityWeightedMean":
                        combined, effective = _quality_weighted_reduce(
                            values,
                            masks,
                            config.rejection_method,
                            config.rejection_low,
                            config.rejection_high,
                            local_weights,
                        )
                        if channels == 1:
                            weight_sums[y1:y2] = effective
                        else:
                            weight_sums[channel, y1:y2] = effective
                    else:
                        combined = reject_and_combine_block(
                            values,
                            masks,
                            config.method,
                            config.rejection_method,
                            config.rejection_low,
                            config.rejection_high,
                            cancel_event,
                            config.engine_profile,
                            config.reducer_engine,
                            ExecutionBudget.for_pipeline(config.worker_count).kernel_parallel,
                        )
                    if channels == 1:
                        result[y1:y2] = combined
                    else:
                        result[channel, y1:y2] = combined
                    if channel == 0:
                        counts[y1:y2] = masks.sum(axis=0, dtype=np.uint32)
            report(
                1,
                f"Leaf {leaf_index + 1}: completed band {band_index}/{total_bands}",
            )
    finally:
        for handle in handles:
            handle.close()

    mask = counts > 0
    hdus = [
        fits.PrimaryHDU(result),
        fits.ImageHDU(mask.astype(np.uint8), name="VALID_MASK"),
        fits.ImageHDU(counts, name="SUB_COUNT"),
    ]
    if weight_sums is not None:
        hdus.append(fits.ImageHDU(np.asarray(weight_sums, dtype=np.float32), name="SUB_WEIGHT"))
    fits.HDUList(hdus).writeto(output_path, overwrite=True)
    return SubstackInfo(output_path, group_size)


def _combine_substacks(
    substacks: list[SubstackInfo],
    config: StackingConfig,
    total_frames: int,
    cancel_event: threading.Event | None,
    return_weight: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data_list: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    counts: list[np.ndarray] = []
    weights_list: list[np.ndarray] = []
    for substack in substacks:
        check_cancel(cancel_event)
        # This reducer already materializes every child in memory. Avoid mmap
        # so Windows can remove the temporary leaf files immediately afterward.
        with fits.open(substack.path, memmap=False) as hdul:
            data_list.append(np.array(hdul[0].data, dtype=np.float32, copy=True))
            masks.append(np.array(hdul["VALID_MASK"].data, dtype=bool, copy=True))
            counts.append(np.array(hdul["SUB_COUNT"].data, dtype=np.uint32, copy=True))
            if config.method == "QualityWeightedMean":
                if "SUB_WEIGHT" in hdul:
                    weights_list.append(np.array(hdul["SUB_WEIGHT"].data, dtype=np.float32, copy=True))
                else:
                    child_data = data_list[-1]
                    child_count = counts[-1].astype(np.float32)
                    if child_data.ndim == 3:
                        child_count = np.broadcast_to(child_count[None, :, :], child_data.shape)
                    weights_list.append(np.array(child_count, dtype=np.float32, copy=True))

    # Materialize each child exactly once.  The previous reducer called
    # ``np.stack(counts)`` repeatedly for coverage and weighted merge, keeping
    # duplicate full-frame arrays alive during the expensive reduction.
    values = np.stack(data_list)
    masks_stack = np.stack(masks)
    counts_stack = np.stack(counts)
    del data_list, masks, counts
    weights_stack = np.stack(weights_list) if weights_list else None

    # Se a saída for RGB (4D: [N_substacks, Canais, Altura, Largura])
    # a máscara e a contagem que estão em 3D [N_substacks, Altura, Largura] precisam se expandir.
    if values.ndim == 4:
        valid = masks_stack
        valid = np.broadcast_to(valid[:, None, :, :], values.shape)
        coverage_base = np.sum(counts_stack, axis=0, dtype=np.uint32)
        # O coverage de saída deve permanecer 2D (Altura, Largura) para uso posterior na referência
        coverage = coverage_base
    else:
        valid = masks_stack
        coverage = np.sum(counts_stack, axis=0, dtype=np.uint32)

    threshold = max(1, math.ceil(total_frames * 0.70))
    final_mask = coverage >= threshold

    if config.method == "QualityWeightedMean":
        if weights_stack is None:
            raise ValueError("SUB_WEIGHT ausente na média ponderada.")
        if values.ndim == 3:
            result, result_weights = _quality_weighted_merge(
                values,
                valid,
                weights_stack,
                config.rejection_method,
                config.rejection_low,
                config.rejection_high,
            )
        else:
            result = np.empty(values.shape[1:], dtype=np.float32)
            result_weights = np.empty(values.shape[1:], dtype=np.float32)
            for channel in range(values.shape[1]):
                result[channel], result_weights[channel] = _quality_weighted_merge(
                    values[:, channel],
                    valid[:, channel],
                    weights_stack[:, channel],
                    config.rejection_method,
                    config.rejection_low,
                    config.rejection_high,
                )
    elif config.method == "Mean" and config.rejection_method == "None":
        if EngineProfile.coerce(config.engine_profile) is EngineProfile.FAST:
            result = weighted_merge(values, counts_stack)
        else:
            weights = counts_stack.astype(np.float32)
            if values.ndim == 4:
                weights = weights[:, None, :, :]
            result = np.sum(values * weights, axis=0) / np.maximum(
                coverage if values.ndim == 3 else coverage[None, :, :], 1
            )
    elif config.method == "Sum" and config.rejection_method == "None":
        result = np.sum(np.where(valid, values, 0.0), axis=0)
    else:
        if values.ndim == 3:
            result = reject_and_combine_block(
                values,
                valid,
                config.method,
                config.rejection_method,
                config.rejection_low,
                config.rejection_high,
                cancel_event,
                config.engine_profile,
                config.reducer_engine,
                ExecutionBudget.for_pipeline(config.worker_count).kernel_parallel,
            )
        else:
            result = np.empty(values.shape[1:], dtype=np.float32)
            for channel in range(values.shape[1]):
                check_cancel(cancel_event)
                result[channel] = reject_and_combine_block(
                    values[:, channel],
                    valid[:, channel],
                    config.method,
                    config.rejection_method,
                    config.rejection_low,
                    config.rejection_high,
                    cancel_event,
                    config.engine_profile,
                    config.reducer_engine,
                    ExecutionBudget.for_pipeline(config.worker_count).kernel_parallel,
                )
    result_tuple = (
        np.asarray(result, dtype=np.float32),
        final_mask,
        coverage,
    )
    if config.method == "QualityWeightedMean" and return_weight:
        return result_tuple + (np.asarray(result_weights, dtype=np.float32),)
    return result_tuple


def _write_substack(
    path: Path,
    data: np.ndarray,
    valid_mask: np.ndarray,
    coverage: np.ndarray,
    weight_sums: np.ndarray | None = None,
) -> None:
    hdus = [
        fits.PrimaryHDU(np.asarray(data, dtype=np.float32)),
        fits.ImageHDU(np.asarray(valid_mask, dtype=np.uint8), name="VALID_MASK"),
        fits.ImageHDU(np.asarray(coverage, dtype=np.uint32), name="SUB_COUNT"),
    ]
    if weight_sums is not None:
        hdus.append(fits.ImageHDU(np.asarray(weight_sums, dtype=np.float32), name="SUB_WEIGHT"))
    fits.HDUList(hdus).writeto(path, overwrite=True)


def _process_branch(
    children: list[SubstackInfo],
    config: StackingConfig,
    output_path: Path,
    cancel_event: threading.Event | None,
) -> SubstackInfo:
    total_frames = sum(child.frame_count for child in children)
    combined = _combine_substacks(
        children,
        config,
        total_frames,
        cancel_event,
        return_weight=config.method == "QualityWeightedMean",
    )
    if config.method == "QualityWeightedMean":
        data, valid_mask, coverage, weight_sums = combined
    else:
        data, valid_mask, coverage = combined
        weight_sums = None
    _write_substack(output_path, data, valid_mask, coverage, weight_sums)
    return SubstackInfo(output_path, total_frames)


def _reduce_substacks_tree(
    leaves: list[SubstackInfo],
    config: StackingConfig,
    temp_dir: Path,
    worker_count: int,
    progress_callback: Callable[[int, int, str], None] | None,
    cancel_event: threading.Event | None,
) -> list[SubstackInfo]:
    """Reduce leaves in parallel binary layers without increasing workers."""
    current = leaves
    level = 0
    # Always reduce to one root.  Stopping at ``worker_count`` made the final
    # grouping depend on the number of threads and changed rejection results
    # when a user only adjusted concurrency.
    while len(current) > 1:
        groups = [current[index : index + 2] for index in range(0, len(current), 2)]
        next_level: list[SubstackInfo] = []
        with ThreadPoolExecutor(
            max_workers=worker_count,
            thread_name_prefix=f"AstroBranch{level}",
            initializer=partial(
                configure_worker_runtime,
                ExecutionBudget.for_pipeline(worker_count).kernel_threads,
            ),
        ) as executor:
            futures = {
                executor.submit(
                    _process_branch,
                    group,
                    config,
                    temp_dir / f"branch_{level:02d}_{index:02d}.fits",
                    cancel_event,
                ): index
                for index, group in enumerate(groups)
            }
            for completed, future in enumerate(as_completed(futures), start=1):
                check_cancel(cancel_event)
                result = future.result()
                result.order = futures[future]
                next_level.append(result)
                if progress_callback:
                    progress_callback(
                        completed,
                        len(futures),
                        f"Branch level {level + 1}: {completed}/{len(futures)}",
                    )
        # ``as_completed`` is intentionally used for progress latency, but
        # reduction inputs must be restored to their logical left-to-right
        # order before the next binary layer.
        next_level.sort(key=lambda item: item.order)
        current = next_level
        level += 1
    return current


def _create_substacks(
    selected_frames: list[FrameInfo],
    geometries: dict[Path, FrameGeometry],
    normalization_factors: list[float],
    dither_shifts: list[tuple[float, float] | None],
    config: StackingConfig,
    height: int,
    width: int,
    channels: int,
    temp_dir: Path,
    progress_callback: Callable[[int, int, str], None] | None,
    status_callback: Callable[[str], None] | None,
    cancel_event: threading.Event | None,
    frame_weights: list[float] | None = None,
) -> list[SubstackInfo]:

    workers = _hierarchical_worker_count(config, len(selected_frames))
    leaf_count = min(len(selected_frames), DETERMINISTIC_LEAF_COUNT)
    groups = _partition_frames(selected_frames, leaf_count)
    factor_groups = _partition_frames(normalization_factors, leaf_count)
    shift_groups = _partition_frames(dither_shifts, leaf_count)
    weight_groups = _partition_frames(frame_weights or [1.0] * len(selected_frames), leaf_count)
    results: list[SubstackInfo] = []
    progress_queue: queue.SimpleQueue[tuple[int, int, str]] = queue.SimpleQueue()
    band_totals = [
        math.ceil(height / _leaf_band_rows(config, len(group), width, channels, height))
        for group in groups
    ]
    band_done = [0] * len(groups)
    total_bands = sum(band_totals)
    if status_callback:
        status_callback(
            f"Starting substack work | Leafs {leaf_count} | bands {total_bands}"
        )
    leaf_kernel_threads = ExecutionBudget.for_pipeline(workers).kernel_threads
    with ThreadPoolExecutor(
        max_workers=workers,
        thread_name_prefix="AstroLeaf_substack",
        initializer=partial(configure_worker_runtime, leaf_kernel_threads),
    ) as executor:
        futures = {
            executor.submit(
                _process_substack,
                group,
                geometries,
                factors,
                shifts,
                config,
                temp_dir / f"leaf_{index:02d}.fits",
                height,
                width,
                channels,
                cancel_event,
                index,
                progress_queue,
                weights,
            ): index
            for index, (group, factors, shifts, weights) in enumerate(
                zip(groups, factor_groups, shift_groups, weight_groups, strict=True)
            )
        }
        pending = set(futures)
        while pending:
            check_cancel(cancel_event)
            done, pending = wait(pending, timeout=0.1, return_when=FIRST_COMPLETED)
            while True:
                try:
                    leaf_index, increment, message = progress_queue.get_nowait()
                except queue.Empty:
                    break
                band_done[leaf_index] += increment
                if progress_callback:
                    progress_callback(
                        sum(band_done),
                        total_bands,
                        message,
                    )
            for future in done:
                result = future.result()
                result.order = futures[future]
                results.append(result)
        if progress_callback:
            progress_callback(total_bands, total_bands, "Leaf substacks complete")
    results.sort(key=lambda item: item.order)
    return results


def _extract_flow_shift(
    frame: FrameInfo,
    flow_cache: dict[str, dict[str, Any]],
) -> tuple[float, float] | None:
    flow = flow_cache.get(frame.batch, {})

    if not isinstance(flow, dict):
        return None

    frame_data = flow.get("frames", {}).get(
        frame.name,
        {},
    )

    if not isinstance(frame_data, dict):
        return None

    candidates = [
        frame_data.get("shift"),
        frame_data.get("translation"),
        frame_data.get("motion"),
        frame_data.get("phase_shift"),
    ]

    for candidate in candidates:
        if isinstance(candidate, (list, tuple)) and len(candidate) >= 2:
            dx = _safe_float(candidate[0], 0.0)
            dy = _safe_float(candidate[1], 0.0)
            return dx, dy

        if isinstance(candidate, dict):
            dx = candidate.get("dx", candidate.get("x"))
            dy = candidate.get("dy", candidate.get("y"))

            if dx is not None and dy is not None:
                return (
                    _safe_float(dx, 0.0),
                    _safe_float(dy, 0.0),
                )

    dx = frame_data.get("dx")
    dy = frame_data.get("dy")

    if dx is not None and dy is not None:
        return (
            _safe_float(dx, 0.0),
            _safe_float(dy, 0.0),
        )

    return None


def build_dither_shifts(
    selected_frames: list[FrameInfo],
    flow_cache: dict[str, dict[str, Any]],
    config: StackingConfig,
    app_print: Callable[[str], None],
) -> list[tuple[float, float] | None]:
    if not config.apply_dither_correction:
        return [None] * len(selected_frames)

    shifts = [_extract_flow_shift(frame, flow_cache) for frame in selected_frames]

    valid = [shift for shift in shifts if shift is not None]

    if not valid:
        app_print(
            "⚠️ Correção de dither solicitada, mas nenhum "
            "deslocamento compatível foi encontrado no Flow."
        )
        return [None] * len(selected_frames)

    reference = valid[0]
    normalized: list[tuple[float, float] | None] = []

    for shift in shifts:
        if shift is None:
            normalized.append(None)
        else:
            normalized.append(
                (
                    reference[0] - shift[0],
                    reference[1] - shift[1],
                )
            )

    app_print(
        f"↔️ Correção de dither habilitada: "
        f"{len(valid)}/{len(selected_frames)} frames com deslocamento."
    )

    return normalized


def flatten_background(
    data: np.ndarray,
    valid_mask: np.ndarray,
) -> np.ndarray:
    sigma_clip = SigmaClip(sigma=3.0)
    bkg_estimator = MedianBackground()
    bkg_mask = ~valid_mask

    try:
        if data.ndim == 3:
            flattened = np.empty_like(data)

            for channel in range(data.shape[0]):
                bkg = Background2D(
                    data[channel],
                    box_size=(50, 50),
                    filter_size=(3, 3),
                    sigma_clip=sigma_clip,
                    bkg_estimator=bkg_estimator,
                    mask=bkg_mask,
                    exclude_percentile=50.0,
                )

                flattened[channel] = data[channel] - bkg.background

            return flattened.astype(
                np.float32,
                copy=False,
            )

        bkg = Background2D(
            data,
            box_size=(50, 50),
            filter_size=(3, 3),
            sigma_clip=sigma_clip,
            bkg_estimator=bkg_estimator,
            mask=bkg_mask,
            exclude_percentile=50.0,
        )

        return (data - bkg.background).astype(np.float32, copy=False)

    except Exception:
        return data.astype(
            np.float32,
            copy=False,
        )


def sanitize_ascii(text: str) -> str:
    replacements = {
        "á": "a",
        "à": "a",
        "ã": "a",
        "â": "a",
        "ä": "a",
        "é": "e",
        "è": "e",
        "ê": "e",
        "ë": "e",
        "í": "i",
        "ì": "i",
        "î": "i",
        "ï": "i",
        "ó": "o",
        "ò": "o",
        "õ": "o",
        "ô": "o",
        "ö": "o",
        "ú": "u",
        "ù": "u",
        "û": "u",
        "ü": "u",
        "ç": "c",
        "Ç": "C",
        "ñ": "n",
        "Ñ": "N",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    return "".join(char for char in text if 32 <= ord(char) <= 126)


def prepare_output_header(
    source_header: fits.Header,
    config: StackingConfig,
    n_frames: int,
    n_frames_total: int,
    n_batches: int,
    avg_quality: float,
    avg_star_count: float,
    avg_fwhm: float,
) -> fits.Header:
    header = source_header.copy()

    cards = {
        "TRAILFLT": (config.trail_filter_enabled, "Reject elongated or unmeasured stars"),
        "MINROUND": (config.min_roundness, "Minimum second-moment axis ratio b/a"),
        "MINSHAPE": (config.min_shape_stars, "Minimum measured stars for trail filter"),
        "STACKING": (config.method, "AstroStack combine method"),
        "STACK_NFR": (n_frames, "Frames combined"),
        "STACK_TOT": (n_frames_total, "Frames available"),
        "STACK_SEL": (
            config.selection_mode,
            "Frame selection mode",
        ),
        "STACK_SELP": (
            config.selection_percentage
            if config.selection_mode == "BestPercentage"
            else 100.0,
            "Percentage selected",
        ),
        "STACK_METR": (
            config.selection_metric,
            "Selection metric",
        ),
        "STACK_PROF": (
            config.feature_profile,
            "Feature policy profile",
        ),
        "STACK_STOR": (
            config.reduction_storage,
            "Reduction storage policy",
        ),
        "STACK_WGTP": (
            config.selection_profile,
            "Quality weight profile",
        ),
        "STACK_REJ": (
            config.rejection_method if config.rejection_method != "None" else "NONE",
            "Outlier rejection",
        ),
        "STACK_REJ_L": (
            config.rejection_low,
            "Low rejection parameter",
        ),
        "STACK_REJ_H": (
            config.rejection_high,
            "High rejection parameter",
        ),
        "STACK_NORM": (
            bool(config.normalize),
            "Frame normalization",
        ),
        "STACK_NORMM": (
            config.normalize_method if config.normalize else "NONE",
            "Normalization method",
        ),
        "STACK_BITS": (
            config.output_bit_depth,
            "Output precision",
        ),
        "STACK_CHNK": (
            config.effective_chunk_size,
            "Processing block size",
        ),
        "STACK_MEM": (
            config.memory_budget_mb,
            "Memory budget MB",
        ),
        "STACK_STRM": (
            True,
            "Block streaming enabled",
        ),
        "STACK_BG": (
            bool(config.remove_background),
            "Background subtraction",
        ),
        "AVG_QUAL": (
            round(avg_quality, 3),
            "Average frame quality",
        ),
        "AVG_STARS": (
            round(avg_star_count, 1),
            "Average detected stars",
        ),
        "AVG_FWHM": (
            round(avg_fwhm, 2) if avg_fwhm > 0 else 0.0,
            "Average FWHM",
        ),
        "STACK_BATS": (
            n_batches,
            "Batches represented",
        ),
    }

    for key, value in cards.items():
        try:
            header[key] = value
        except Exception:
            pass

    header["HISTORY"] = sanitize_ascii(f"Stacking by AstroStack ({config.method})")
    header["HISTORY"] = sanitize_ascii(
        f"{n_frames} frames combinados de {n_frames_total} disponiveis"
    )

    if config.selection_mode == "BestPercentage":
        header["HISTORY"] = sanitize_ascii(
            f"Selecionados {config.selection_percentage}% por {config.selection_metric}"
        )

    if config.normalize:
        header["HISTORY"] = sanitize_ascii(f"Normalizacao: {config.normalize_method}")

    if config.rejection_method != "None":
        header["HISTORY"] = sanitize_ascii(
            f"Rejeicao: {config.rejection_method} "
            f"({config.rejection_low}, {config.rejection_high})"
        )

    if config.remove_background:
        header["HISTORY"] = sanitize_ascii("Background subtraction habilitado")

    return header


def convert_output_dtype(
    result: np.ndarray,
    config: StackingConfig,
) -> np.ndarray:
    result = np.asarray(result)

    clean = np.nan_to_num(
        result,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).astype(np.float32, copy=False)

    valid = clean[np.isfinite(clean)]

    if valid.size == 0:
        return np.zeros_like(
            clean,
            dtype=np.uint16,
        )

    data_min = float(np.min(valid))
    data_max = float(np.max(valid))

    if data_max <= data_min:
        return np.zeros_like(
            clean,
            dtype=np.uint16,
        )

    scale = np.float32(65535.0 / (data_max - data_min))

    scaled = (clean - np.float32(data_min)) * scale

    return np.clip(
        scaled,
        0.0,
        65535.0,
    ).astype(np.uint16)


def write_stack_output(
    output_path: Path,
    data_to_save: np.ndarray,
    mask: np.ndarray,
    header: fits.Header,
    config: StackingConfig,
) -> None:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if output_path.suffix.casefold() in TIFF_SUFFIXES:
        # TIFF has no FITS extension HDUs.  Keep the visible product Siril-
        # compatible and publish the validity plane as a compact sidecar.
        tiff_data = np.asarray(data_to_save)
        if tiff_data.ndim == 3 and tiff_data.shape[0] in (3, 4) and tiff_data.shape[-1] not in (3, 4):
            tiff_data = np.moveaxis(tiff_data, 0, -1)
        write_tiff(
            output_path,
            tiff_data,
            metadata={
                "schema_version": 1,
                "format": "TIFF",
                "bit_depth": 16,
                "linear": True,
            },
        )
        write_npz_sidecar(
            output_path.with_suffix(output_path.suffix + ".astrobatch.npz"),
            valid_mask=np.asarray(mask, dtype=np.uint8),
        )
        write_sidecar_json(
            output_path.with_suffix(output_path.suffix + ".json"),
            {
                "schema_version": 1,
                "format": "TIFF",
                "bit_depth": 16,
                "linear": True,
                "valid_mask_sidecar": output_path.name + ".astrobatch.npz",
                "header": {str(k): str(v) for k, v in header.items()},
            },
        )
        return

    temporary = output_path.with_name(output_path.name + ".tmp")

    if temporary.exists():
        try:
            temporary.unlink()
        except OSError:
            pass

    try:
        primary = fits.PrimaryHDU(
            data=data_to_save,
            header=header,
        )

        if config.compress_output:
            mask_hdu = fits.CompImageHDU(
                data=mask.astype(
                    np.uint8,
                    copy=False,
                ),
                name="VALID_MASK",
                compression_type="RICE_1",
            )
        else:
            mask_hdu = fits.ImageHDU(
                data=mask.astype(
                    np.uint8,
                    copy=False,
                ),
                name="VALID_MASK",
            )

        hdul = fits.HDUList([primary, mask_hdu])

        try:
            hdul.writeto(
                temporary,
                overwrite=True,
                output_verify="ignore",
            )
        finally:
            hdul.close()

        os.replace(
            temporary,
            output_path,
        )

    finally:
        if temporary.exists():
            try:
                temporary.unlink()
            except OSError:
                pass


def load_source_header(filepath: Path) -> fits.Header:
    if filepath.suffix.casefold() in TIFF_SUFFIXES:
        _data, header = read_tiff(filepath)
        return header.copy()
    with fits.open(
        filepath,
        memmap=False,
        lazy_load_hdus=True,
    ) as hdul:
        _, hdu = _find_primary_image_hdu(hdul)
        return hdu.header.copy()


def _validate_config(config: StackingConfig) -> None:
    if not math.isfinite(config.min_roundness) or not 0 <= config.min_roundness <= 1:
        raise ValueError("A circularidade mínima deve estar entre 0 e 1.")
    if not isinstance(config.min_shape_stars, int) or not 1 <= config.min_shape_stars <= 64:
        raise ValueError("O mínimo de estrelas medidas deve ser um inteiro entre 1 e 64.")
    if config.feature_profile not in {"Legacy", "Intelligent"}:
        raise ValueError(f"Perfil de features inválido: {config.feature_profile}")
    if config.method not in {
        "Median",
        "Mean",
        "QualityWeightedMean",
        "Sum",
        "Maximum",
        "Minimum",
    }:
        raise ValueError(f"Método de stacking inválido: {config.method}")

    if config.rejection_method not in {
        "None",
        "SigmaClip",
        "Winsorized",
        "MAD",
    }:
        raise ValueError(f"Método de rejeição inválido: {config.rejection_method}")

    if config.selection_mode not in {
        "All",
        "BestPercentage",
        "MultiMetric",
    }:
        raise ValueError(f"Modo de seleção inválido: {config.selection_mode}")
    if config.selection_profile not in {"Sharpness", "Balanced", "Signal", "Custom"}:
        raise ValueError(f"Perfil de seleção inválido: {config.selection_profile}")
    parsed_weights = parse_selection_weights(config.selection_weights)
    if config.selection_profile == "Custom" and not parsed_weights:
        raise ValueError(
            "selection_weights deve conter pelo menos um peso positivo e finito."
        )
    if config.trail_policy not in {"off", "exclude_severe", "weight_only", "report"}:
        raise ValueError(f"Política de trailing inválida: {config.trail_policy}")
    if config.reduction_storage not in {"disk_legacy", "ram", "ram_spill"}:
        raise ValueError(f"Armazenamento de redução inválido: {config.reduction_storage}")
    if config.reduction_storage == "ram_spill":
        if config.spill_directory is None:
            raise ValueError("spill_directory é obrigatório quando o spill está habilitado.")
        if int(config.spill_limit_mb) < 256:
            raise ValueError("spill_limit_mb deve ser pelo menos 256 MiB.")
    elif int(config.spill_limit_mb) < 0:
        raise ValueError("spill_limit_mb não pode ser negativo.")

    if config.output_bit_depth != "16-bit":
        raise ValueError(f"Profundidade inválida: {config.output_bit_depth}")
    if config.output_format not in {"auto", "fits", "tiff"}:
        raise ValueError(f"Formato de saída inválido: {config.output_format}")

    if config.selection_percentage <= 0:
        raise ValueError("selection_percentage deve ser > 0.")

    if config.chunk_size <= 0:
        raise ValueError("chunk_size deve ser > 0.")
    if int(config.memory_budget_mb) < 64:
        raise ValueError("memory_budget_mb deve ser pelo menos 64 MiB.")
    if config.workers is not None and int(config.workers) < 1:
        raise ValueError("workers deve ser pelo menos 1.")


def process_stacking(
    config: StackingConfig,
    progress_callback: Callable[
        [int, int, str],
        None,
    ]
    | None = None,
    status_callback: Callable[
        [str],
        None,
    ]
    | None = None,
    cancel_event: threading.Event | None = None,
) -> dict[str, Any]:

    profiler = None
    if HAS_PYINSTRUMENT:
        profiler = Profiler(interval=0.01)
        profiler.start()

    if cancel_event is None:
        cancel_event = threading.Event()

    # Keep Numba's global pool bounded when a direct library caller bypasses
    # the GUI warm-up.  A multi-worker leaf scheduler uses serial kernels so
    # frame workers remain the owners of the CPU.
    from cpu_runtime import configure_numba_threads

    pipeline_budget = ExecutionBudget.for_pipeline(config.worker_count)
    configure_opencv_threads(
        1 if pipeline_budget.worker_count > 1 else pipeline_budget.kernel_threads
    )
    configure_numba_threads(pipeline_budget.kernel_threads)

    def log(message: str) -> None:
        if status_callback:
            status_callback(message + "\n")

    def progress(
        current: int,
        total: int,
        message: str = "",
    ) -> None:
        if progress_callback:
            progress_callback(
                current,
                total,
                message,
            )

    try:
        _validate_config(config)
        check_cancel(cancel_event)

        log("🔍 Procurando frames alinhados...")

        if not config.input_dir.exists():
            log(f"❌ Pasta de entrada não encontrada: {config.input_dir}")
            return {
                "status": "error",
                "reason": "input_dir_not_found",
            }

        # Compact Align publishes one manifest per acquisition batch.  When
        # present, consume those scientific states directly instead of
        # treating each master as a single exposure.
        from batch_compaction import combine_bundles, discover_batch_bundles

        batch_manifests = discover_batch_bundles(config.input_dir)
        if batch_manifests:
            log(f"[Stack] {len(batch_manifests)} pré-stacks por batch encontrados.")
            return combine_bundles(
                batch_manifests,
                method=config.method,
                output_path=config.output_dir / config.output_name,
                compress_output=config.compress_output,
                rejection_method=config.rejection_method,
                rejection_low=config.rejection_low,
                rejection_high=config.rejection_high,
                output_format=config.output_format,
            )

        fits_files, batch_metadata = discover_aligned_frames(config.input_dir)

        check_cancel(cancel_event)

        if not fits_files:
            log(f"❌ Nenhum arquivo FITS/TIFF encontrado em: {config.input_dir}")
            return {
                "status": "error",
                "reason": "no_fits_files",
            }

        try:
            source_format = validate_single_format(fits_files)
        except ValueError as exc:
            log(f"[Stack] ERRO: {exc}")
            return {"status": "error", "reason": "mixed_input_format", "error": str(exc)}
        if config.output_format != "auto" and config.output_format != source_format.value:
            message = (
                f"Formato de saída incompatível: a sessão usa {source_format.value.upper()}, "
                f"mas foi solicitado {config.output_format.upper()}."
            )
            log(f"[Stack] ERRO: {message}")
            return {"status": "error", "reason": "output_format_mismatch", "error": message}
        expected_suffixes = {".fit", ".fits", ".fts"} if source_format.value == "fits" else {".tif", ".tiff"}
        if Path(config.output_name).suffix.casefold() not in expected_suffixes:
            config = replace(config, output_name=Path(config.output_name).with_suffix(source_format.default_suffix).name)

        log(f"📁 Total de {len(fits_files)} arquivos FITS/TIFF encontrados.")

        flow_cache = load_flow_cache(
            config.base_dir,
            batch_metadata,
        )

        if flow_cache:
            log(f"🧭 Flows locais em cache: {len(flow_cache)}")
        else:
            log(
                "[Stack] Aviso: Nenhum arquivo de flow encontrado. "
                "O Stacking usará apenas os metadados disponíveis."
            )

        log("🔎 Analisando metadados dos frames...")

        all_frames = build_frame_infos(
            fits_files,
            batch_metadata,
            flow_cache,
            log,
            cancel_event,
        )

        check_cancel(cancel_event)

        if not all_frames:
            log("❌ Nenhum frame válido encontrado.")
            return {
                "status": "error",
                "reason": "no_valid_frames",
            }

        # Build the Intelligent feature snapshot once and share it between
        # selection and weighting.  Both operations are pure policy over the
        # inspected metadata; repeating the trailing/MAD pass here only adds
        # latency and can create avoidable temporary mappings on large
        # sessions.
        feature_scores = _annotate_scores(all_frames, config)
        selected_frames = select_frames(
            all_frames,
            config,
            scores=feature_scores,
        )

        frame_weights = compute_quality_weights(
            all_frames,
            selected_frames,
            config,
            scores=feature_scores,
        )
        report_path = write_selection_report(all_frames, selected_frames, config)
        log(f"[Stack] Relatório de seleção: {report_path}")
        if config.trail_filter_enabled:
            excluded = [trail_exclusion_reason(frame, config) for frame in all_frames]
            log(f"[Stack] Sem guiagem: {excluded.count('elongated_stars')} frames alongados; "
                f"{excluded.count('missing_shape_measurement')} sem medição confiável. "
                "Execute Flow novamente para medir dados antigos.")

        log(
            f"✅ Selecionados {len(selected_frames)} "
            f"frames de {len(all_frames)} disponíveis."
        )

        if len(selected_frames) < 3:
            log("❌ Número insuficiente de frames selecionados (mínimo: 3).")
            return {
                "status": "error",
                "reason": "insufficient_frames",
                "selection_report": str(report_path),
            }

        check_cancel(cancel_event)

        log("📐 Validando geometria dos frames...")

        geometries: dict[
            Path,
            FrameGeometry,
        ] = {}

        reference_frame = selected_frames[0]
        reference_geometry = inspect_fits(reference_frame.path)

        for frame in selected_frames:
            check_cancel(cancel_event)

            try:
                geometry = inspect_fits(frame.path)
            except Exception as exc:
                log(f"❌ Falha de geometria em {frame.name}: {exc}")
                return {
                    "status": "error",
                    "reason": "geometry_error",
                    "file": frame.name,
                }

            if (
                geometry.height != reference_geometry.height
                or geometry.width != reference_geometry.width
                or geometry.image_kind != reference_geometry.image_kind
                or geometry.channels != reference_geometry.channels
            ):
                log(
                    f"❌ Geometria incompatível em "
                    f"{frame.name}: "
                    f"{geometry.height}x{geometry.width}, "
                    f"{geometry.image_kind}/"
                    f"{geometry.channels}"
                )

                return {
                    "status": "error",
                    "reason": "incompatible_geometry",
                    "file": frame.name,
                }

            geometries[frame.path] = geometry

        # The Intelligent RAM reducer reads source FITS directly in bounded
        # row bands. Building the legacy persistent decompression cache here
        # would defeat the RAM-first contract and can consume tens of GiB on
        # a dedicated-camera session, so it is disabled for in-memory modes.
        if config.cache_decompressed_fits and config.reduction_storage == "disk_legacy":
            cache_dir = config.input_dir / FITS_CACHE_DIR_NAME
            cache_candidates = [
                frame
                for frame in selected_frames
                if geometries[frame.path].science_compressed
            ]
            bypassed_uncompressed = len(selected_frames) - len(cache_candidates)
            cache_stats = FitsCacheStats()
            if cache_candidates:
                cached_geometries, cache_stats = cache_decompressed_frames(
                    cache_candidates,
                    geometries,
                    cache_dir,
                    log,
                    cancel_event,
                )
                geometries.update(cached_geometries)
            reference_geometry = geometries[reference_frame.path]
            log(
                "[Stack] Raw FITS cache: "
                f"{len(cache_candidates)} compressed candidates | "
                f"{cache_stats.hits} hits | "
                f"{cache_stats.rebuilt} rebuilt | "
                f"{cache_stats.direct} direct | "
                f"{cache_stats.scaled} uncompressed-scaled | "
                f"{bypassed_uncompressed} uncompressed mmap/raw | "
                f"{cache_stats.skipped} skipped"
            )

        height = reference_geometry.height
        width = reference_geometry.width
        channels = reference_geometry.channels

        log(
            f"📐 Dimensão: {width}×{height} | "
            f"{reference_geometry.image_kind} | "
            f"canais={channels}"
        )

        # VALID_MASK is independent of the image channel and output block.
        # Keep small masks resident because that avoids repeated section reads;
        # for a real dedicated-camera session, however, one bool array per
        # frame can consume several GiB (8.4 MiB × 1,242 Lagoon frames).  In
        # that case the leaf reader streams one mask band and reuses it across
        # RGB channels, preserving the memory budget.
        masks_loaded = 0
        mask_bytes = sum(
            geometries[frame.path].height * geometries[frame.path].width
            for frame in selected_frames
            if frame.has_valid_mask
        )
        # RAM-first reductions account for every resident allocation inside
        # their own guard.  Do not pre-load a full VALID_MASK per frame in
        # that path: even a seemingly small 25% slice of the configured
        # budget can become a multi-gigabyte, untracked allocation on a long
        # dedicated-camera session.  The bounded reducer reads mask bands
        # from its already-open FITS handles instead.
        if config.reduction_storage in {"ram", "ram_spill"}:
            mask_budget = 0
        else:
            mask_budget = max(
                16 * 1024 * 1024,
                int(config.memory_budget_mb * 1024 * 1024 * 0.25),
            )
        if mask_bytes <= mask_budget:
            for frame in selected_frames:
                if not frame.has_valid_mask:
                    continue

                frame.valid_mask = _load_full_valid_mask(
                    frame,
                    geometries[frame.path],
                )
                if frame.valid_mask is not None:
                    masks_loaded += 1

            if masks_loaded:
                log(f"VALID_MASK cached: {masks_loaded}")
        elif mask_bytes:
            log(
                "VALID_MASK streaming: "
                f"{mask_bytes / 1024**3:.2f} GiB exceeds "
                f"{mask_budget / 1024**3:.2f} GiB resident budget"
            )

        quality_values = [
            frame.quality for frame in selected_frames if math.isfinite(frame.quality)
        ]

        star_counts = [
            frame.star_count
            for frame in selected_frames
            if math.isfinite(frame.star_count)
        ]

        fwhm_values = [
            frame.fwhm
            for frame in selected_frames
            if frame.fwhm > 0 and math.isfinite(frame.fwhm)
        ]

        avg_quality = float(np.mean(quality_values) if quality_values else 0.0)

        avg_star_count = float(np.mean(star_counts) if star_counts else 0.0)

        avg_fwhm = float(np.mean(fwhm_values) if fwhm_values else 0.0)

        normalization_factors = calculate_normalization_factors(
            selected_frames,
            geometries,
            config,
            log,
            cancel_event,
        )

        if normalization_factors is None:
            return {"status": "cancelled"}

        check_cancel(cancel_event)

        dither_shifts = build_dither_shifts(
            selected_frames,
            flow_cache,
            config,
            log,
        )

        if config.reduction_storage in {"ram", "ram_spill"}:
            from ram_stack_logic import run_ram_stack

            log(
                "[Stack] Redução limitada pela RAM ativada; FITS de origem serão "
                "somente leitura e nenhum substack FITS será criado."
            )
            return run_ram_stack(
                selected_frames=selected_frames,
                all_frames=all_frames,
                geometries=geometries,
                normalization_factors=normalization_factors,
                dither_shifts=dither_shifts,
                frame_weights=frame_weights,
                config=config,
                report_path=report_path,
                total_frames=len(selected_frames),
                n_batches=len(batch_metadata),
                avg_quality=avg_quality,
                avg_star_count=avg_star_count,
                avg_fwhm=avg_fwhm,
                progress_callback=progress,
                status_callback=log,
                cancel_event=cancel_event,
            )

        leaf_workers = _hierarchical_worker_count(config, len(selected_frames))
        log(f"Hierarchical stacking: {leaf_workers} bounded leaf workers")
        config.output_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix=".astrostack-", dir=config.output_dir
        ) as temp_dir:
            substacks = _create_substacks(
                selected_frames,
                geometries,
                normalization_factors,
                dither_shifts,
                config,
                height,
                width,
                channels,
                Path(temp_dir),
                progress,
                log,
                cancel_event,
                frame_weights,
            )
            substacks = _reduce_substacks_tree(
                substacks,
                config,
                Path(temp_dir),
                leaf_workers,
                progress,
                cancel_event,
            )
            progress(0, 1, "Combining substacks at root")
            output_result, validity_mask, _ = _combine_substacks(
                substacks,
                config,
                len(selected_frames),
                cancel_event,
            )
            progress(1, 1, "Root stack complete")

        check_cancel(cancel_event)
        if config.remove_background:
            output_result = flatten_background(output_result, validity_mask)
        if output_result.ndim == 3:
            output_result[:, ~validity_mask] = 0.0
        else:
            output_result[~validity_mask] = 0.0
        output_result = np.nan_to_num(
            output_result, nan=0.0, posinf=0.0, neginf=0.0
        ).astype(np.float32, copy=False)

        header = prepare_output_header(
            load_source_header(reference_frame.path),
            config,
            len(selected_frames),
            len(all_frames),
            len(batch_metadata),
            avg_quality,
            avg_star_count,
            avg_fwhm,
        )
        data_to_save = convert_output_dtype(output_result, config)
        output_path = config.output_dir / config.output_name
        write_stack_output(output_path, data_to_save, validity_mask, header, config)
        return {
            "status": "success",
            "output_path": str(output_path),
            "selection_report": str(report_path),
            "n_frames": len(selected_frames),
            "n_frames_total": len(all_frames),
            "n_batches": len(batch_metadata),
            "method": config.method,
            "rejection": config.rejection_method,
            "selection_mode": config.selection_mode,
            "selection_percentage": config.selection_percentage,
            "selection_metric": config.selection_metric,
            "shape": tuple(output_result.shape),
            "dtype": str(data_to_save.dtype),
            "bit_depth": config.output_bit_depth,
            "min": float(np.min(output_result)),
            "max": float(np.max(output_result)),
            "mean": float(np.mean(output_result)),
            "std": float(np.std(output_result)),
            "avg_quality": avg_quality,
            "avg_star_count": avg_star_count,
            "avg_fwhm": avg_fwhm,
            "workers": leaf_workers,
            "chunk_size": "row-band",
            "memory_budget_mb": config.memory_budget_mb,
            "streaming": True,
            "hierarchical": True,
            "background_removed": bool(config.remove_background),
            "dither_correction": bool(config.apply_dither_correction),
            "feature_profile": config.feature_profile,
            "reduction_storage": config.reduction_storage,
            "quality_weighted": config.method == "QualityWeightedMean",
            "weight_min": min(frame_weights) if frame_weights else 1.0,
            "weight_median": float(np.median(frame_weights)) if frame_weights else 1.0,
            "weight_max": max(frame_weights) if frame_weights else 1.0,
            "batches": {
                name: meta["frame_count"] for name, meta in batch_metadata.items()
            },
        }

    except StackingCancelled:
        log("\n🛑 Stacking cancelado pelo usuário.")
        return {
            "status": "cancelled",
            "reason": "user_cancelled",
        }

    except Exception as exc:
        log(f"\n❌ Erro no Stacking: {type(exc).__name__}: {exc}")
        return {
            "status": "error",
            "reason": "stacking_exception",
            "error": str(exc),
        }

    finally:
        if profiler is not None:
            profiler.stop()
            prof_file = config.output_dir / "astrostack_profile.html"
            prof_file.parent.mkdir(parents=True, exist_ok=True)
            with open(prof_file, "w", encoding="utf-8") as f:
                f.write(profiler.output_html())
            log("\n=== PROFILER RESULT ===")
            log(f"Perfil interativo salvo em: {prof_file}")


def _build_config_from_dict(
    input_dir: Path,
    config_dict: dict[str, Any],
) -> StackingConfig:
    shape_count = float(config_dict.get("min_shape_stars", 5))
    if not math.isfinite(shape_count) or not shape_count.is_integer() or not 1 <= shape_count <= 64:
        raise ValueError("O mínimo de estrelas medidas deve ser um inteiro entre 1 e 64.")
    workers = None

    if config_dict.get("workers") is not None:
        try:
            workers = int(config_dict["workers"])
        except (TypeError, ValueError):
            workers = None

    feature_profile = str(config_dict.get("feature_profile", "Legacy"))
    if feature_profile not in {"Legacy", "Intelligent"}:
        feature_profile = "Legacy"
    default_selection_mode = "MultiMetric" if feature_profile == "Intelligent" else "BestPercentage"
    default_method = "QualityWeightedMean" if feature_profile == "Intelligent" else "Median"
    default_storage = "ram" if feature_profile == "Intelligent" else "disk_legacy"
    default_trail_policy = "exclude_severe" if feature_profile == "Intelligent" else "off"
    spill_directory = config_dict.get("spill_directory")
    spill_path = Path(spill_directory).expanduser().resolve() if spill_directory else None
    raw_weights = config_dict.get("selection_weights", {})
    selection_weights = parse_selection_weights(raw_weights)

    return StackingConfig(
        base_dir=Path(config_dict.get("base_dir", "")),
        input_dir=Path(input_dir),
        output_dir=Path(config_dict.get("output_dir", "")),
        feature_profile=feature_profile,
        selection_mode=config_dict.get("selection_mode", default_selection_mode),
        selection_percentage=float(
            config_dict.get(
                "selection_percentage",
                80.0,
            )
        ),
        selection_metric=config_dict.get(
            "selection_metric",
            "quality",
        ),
        selection_profile=config_dict.get("selection_profile", "Balanced"),
        selection_weights=selection_weights,
        trail_filter_enabled=bool(config_dict.get("trail_filter_enabled", False)),
        min_roundness=float(config_dict.get("min_roundness", 0.65)),
        min_shape_stars=int(shape_count),
        trail_policy=config_dict.get("trail_policy", default_trail_policy),
        method=config_dict.get("method", default_method),
        rejection_method=config_dict.get(
            "rejection_method",
            "SigmaClip",
        ),
        rejection_low=float(
            config_dict.get(
                "rejection_low",
                3.0,
            )
        ),
        rejection_high=float(
            config_dict.get(
                "rejection_high",
                3.0,
            )
        ),
        normalize=bool(
            config_dict.get(
                "normalize",
                True,
            )
        ),
        normalize_method=config_dict.get(
            "normalize_method",
            "Median",
        ),
        apply_dither_correction=bool(
            config_dict.get(
                "apply_dither_correction",
                False,
            )
        ),
        remove_background=bool(
            config_dict.get(
                "remove_background",
                False,
            )
        ),
        output_name=config_dict.get(
            "output_name",
            "stacked_image.fits",
        ),
        output_format=str(config_dict.get("output_format", "auto")),
        output_bit_depth="16-bit",
        compress_output=bool(
            config_dict.get(
                "compress_output",
                True,
            )
        ),
        workers=workers,
        chunk_size=int(
            config_dict.get(
                "chunk_size",
                DEFAULT_CHUNK_SIZE,
            )
        ),
        memory_budget_mb=int(
            config_dict.get(
                "memory_budget_mb",
                DEFAULT_MEMORY_BUDGET_MB,
            )
        ),
        normalization_max_samples=int(
            config_dict.get(
                "normalization_max_samples",
                DEFAULT_NORMALIZATION_MAX_SAMPLES,
            )
        ),
        io_queue_factor=int(
            config_dict.get(
                "io_queue_factor",
                2,
            )
        ),
        cache_decompressed_fits=bool(config_dict.get("cache_decompressed_fits", True)),
        reduction_storage=config_dict.get("reduction_storage", default_storage),
        spill_directory=spill_path,
        spill_limit_mb=int(config_dict.get("spill_limit_mb", 0)),
        engine_profile=str(config_dict.get("engine_profile", "Stable")),
        reducer_engine=config_dict.get("reducer_engine") or None,
    )


def process_all_stacking(
    input_dir: Path,
    config_dict: dict[str, Any],
    progress_callback: Callable | None = None,
    status_callback: Callable | None = None,
    cancel_event: threading.Event | None = None,
) -> dict[str, Any]:
    config_dict = config_dict if isinstance(config_dict, dict) else {}

    config = _build_config_from_dict(
        Path(input_dir),
        config_dict,
    )

    return process_stacking(
        config,
        progress_callback,
        status_callback,
        cancel_event,
    )


__all__ = [
    "BlockRead",
    "FitsCacheStats",
    "FrameGeometry",
    "FrameInfo",
    "StackingCancelled",
    "StackingConfig",
    "discover_aligned_frames",
    "inspect_fits",
    "process_all_stacking",
    "process_stacking",
    "select_frames",
]
