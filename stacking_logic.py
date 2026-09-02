"""
stacking_logic.py

AstroStack - empilhamento otimizado para o AstroProcessManager.
"""

from __future__ import annotations

import json
import math
import os
import threading
import warnings
from collections.abc import Callable, Iterator
from concurrent.futures import (
    FIRST_COMPLETED,
    Future,
    ThreadPoolExecutor,
    as_completed,
    wait,
)
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
from astropy.io import fits
from astropy.stats import SigmaClip
from astropy.utils.exceptions import AstropyWarning
from photutils.background import Background2D, MedianBackground

try:
    from pyinstrument import Profiler

    HAS_PYINSTRUMENT = True
except ImportError:
    HAS_PYINSTRUMENT = False

warnings.simplefilter("ignore", category=AstropyWarning)

try:
    import cupy as cp

    HAS_CUPY = True
except Exception:
    cp = None
    HAS_CUPY = False


FITS_SUFFIXES = {".fits", ".fit", ".fts"}
DEFAULT_CHUNK_SIZE = 2048
DEFAULT_MEMORY_BUDGET_MB = 4096
DEFAULT_NORMALIZATION_MAX_SAMPLES = 1_000_000
MAX_WORKERS = 16


class StackingCancelled(RuntimeError):
    """Cancelamento cooperativo solicitado pelo usuário."""


@dataclass
class StackingConfig:
    base_dir: Path = field(default_factory=Path)
    input_dir: Path = field(default_factory=Path)
    output_dir: Path = field(default_factory=Path)

    selection_mode: Literal["All", "BestPercentage"] = "BestPercentage"
    selection_percentage: float = 80.0
    selection_metric: Literal["quality", "fwhm", "star_count", "snr"] = "quality"

    method: Literal["Median", "Mean", "Sum", "Maximum", "Minimum"] = "Median"

    rejection_method: Literal["None", "SigmaClip", "Winsorized", "MAD"] = "SigmaClip"
    rejection_low: float = 3.0
    rejection_high: float = 3.0

    normalize: bool = True
    normalize_method: Literal["Median", "Mode"] = "Median"
    apply_dither_correction: bool = False
    remove_background: bool = False

    output_name: str = "stacked_image.fits"
    output_bit_depth: Literal["16-bit", "32-bit"] = "32-bit"
    compress_output: bool = True

    workers: int | None = None
    chunk_size: int = DEFAULT_CHUNK_SIZE
    memory_budget_mb: int = DEFAULT_MEMORY_BUDGET_MB
    normalization_max_samples: int = DEFAULT_NORMALIZATION_MAX_SAMPLES
    io_queue_factor: int = 2

    @property
    def worker_count(self) -> int:
        try:
            cpu_count = getattr(os, "process_cpu_count", os.cpu_count)() or 1
        except Exception:
            cpu_count = os.cpu_count() or 1

        if self.workers is not None:
            try:
                return max(1, min(MAX_WORKERS, int(self.workers)))
            except (TypeError, ValueError):
                pass

        return max(1, min(MAX_WORKERS, cpu_count))

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


@dataclass(slots=True)
class BlockRead:
    data: np.ndarray
    mask: np.ndarray


def get_optimal_worker_count() -> int:
    try:
        cpu_count = getattr(os, "process_cpu_count", os.cpu_count)() or 1
    except Exception:
        cpu_count = os.cpu_count() or 1
    return max(1, min(MAX_WORKERS, cpu_count))


def check_cancel(cancel_event: threading.Event | None) -> None:
    if cancel_event is not None and cancel_event.is_set():
        raise StackingCancelled("Operação cancelada pelo usuário.")


def estimate_block_memory_bytes(
    n_frames: int,
    chunk_size: int,
    channels: int,
    rejection: bool,
) -> int:
    pixels = chunk_size * chunk_size
    base = n_frames * pixels * max(1, channels) * 4

    if rejection:
        return int(base * 4.0)

    return int(base * 0.35 + pixels * 16)


def adapt_chunk_size(
    requested: int,
    n_frames: int,
    channels: int,
    memory_budget_mb: int,
    rejection: bool,
) -> int:
    chunk = max(32, int(requested))
    budget = max(64, int(memory_budget_mb)) * 1024 * 1024

    while chunk > 32:
        if estimate_block_memory_bytes(n_frames, chunk, channels, rejection) <= budget:
            break
        chunk //= 2

    return max(32, chunk)


def discover_aligned_frames(
    input_dir: Path,
) -> tuple[list[Path], dict[str, dict[str, Any]]]:
    if not input_dir.exists():
        return [], {}

    found: list[Path] = []
    batch_metadata: dict[str, dict[str, Any]] = {}

    for root, _, filenames in os.walk(input_dir):
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

    for batch_name in batch_metadata:
        flow_path = base_dir / batch_name / "flow_local.json"
        if not flow_path.is_file():
            continue

        try:
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
        )


def load_frame_metrics(filepath: Path) -> dict[str, Any]:
    metrics_path = filepath.parent / f"{filepath.stem}_metrics.json"

    if not metrics_path.exists():
        return {}

    try:
        return load_json(metrics_path)
    except Exception:
        return {}


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
    with ThreadPoolExecutor(
        max_workers=get_optimal_worker_count(), thread_name_prefix="AstroInspect"
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


def select_frames(
    all_frames: list[FrameInfo],
    config: StackingConfig,
) -> list[FrameInfo]:
    if not all_frames:
        return []

    if config.selection_mode == "All":
        return list(all_frames)

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
        elif value <= 0:
            continue

        valid.append(frame)

    if not valid:
        valid = list(all_frames)

    percentage = max(
        1.0,
        min(100.0, float(config.selection_percentage)),
    )

    minimum = min(3, len(valid))
    n_select = max(minimum, int(len(valid) * percentage / 100.0))
    n_select = min(n_select, len(valid))

    return valid[:n_select]


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
        elif image_hdu.shape[0] in (3, 4):
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

        raw_data = np.asarray(raw_data, dtype=np.float32)

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
        elif shape[0] in (3, 4):
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

        return np.asarray(sample, dtype=np.float32), mask


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


def read_block_worker(
    args: tuple[
        FrameInfo,
        FrameGeometry,
        int,
        int,
        int,
        int,
        int | None,
        float,
        tuple[float, float] | None,
    ],
) -> BlockRead:
    return read_frame_block(*args)


def _bounded_parallel_reads(
    args_list: list[
        tuple[
            FrameInfo,
            FrameGeometry,
            int,
            int,
            int,
            int,
            int | None,
            float,
            tuple[float, float] | None,
        ]
    ],
    executor: ThreadPoolExecutor,
    max_inflight: int,
    cancel_event: threading.Event | None,
) -> Iterator[tuple[int, BlockRead]]:
    total = len(args_list)
    next_index = 0
    pending: dict[Future[BlockRead], int] = {}

    while next_index < total or pending:
        check_cancel(cancel_event)

        while next_index < total and len(pending) < max_inflight:
            check_cancel(cancel_event)

            future = executor.submit(
                read_block_worker,
                args_list[next_index],
            )
            pending[future] = next_index
            next_index += 1

        if not pending:
            break

        done, _ = wait(
            pending,
            return_when=FIRST_COMPLETED,
        )

        for future in done:
            index = pending.pop(future)

            if cancel_event is not None and cancel_event.is_set():
                future.cancel()
                raise StackingCancelled("Operação cancelada pelo usuário.")

            yield index, future.result()


def load_channel_block_parallel(
    selected_frames: list[FrameInfo],
    geometries: dict[Path, FrameGeometry],
    y1: int,
    y2: int,
    x1: int,
    x2: int,
    channel: int | None,
    normalization_factors: list[float],
    executor: ThreadPoolExecutor | None,
    dither_shifts: list[tuple[float, float] | None] | None,
    cancel_event: threading.Event | None,
    io_queue_factor: int = 2,
) -> tuple[np.ndarray, np.ndarray]:

    args_list = [
        (
            frame,
            geometries[frame.path],
            y1,
            y2,
            x1,
            x2,
            channel,
            normalization_factors[index],
            dither_shifts[index] if dither_shifts else None,
        )
        for index, frame in enumerate(selected_frames)
    ]

    count = len(args_list)

    if count == 0:
        raise ValueError("Nenhum frame selecionado.")

    first = read_block_worker(args_list[0])
    block_h, block_w = first.data.shape

    values = np.empty(
        (count, block_h, block_w),
        dtype=np.float32,
    )
    masks = np.empty(
        (count, block_h, block_w),
        dtype=bool,
    )

    values[0] = first.data
    masks[0] = first.mask

    if count == 1:
        return values, masks

    if executor is None:
        for index in range(1, count):
            check_cancel(cancel_event)
            item = read_block_worker(args_list[index])
            values[index] = item.data
            masks[index] = item.mask
        return values, masks

    max_inflight = max(
        1,
        min(
            count - 1,
            executor._max_workers * max(1, int(io_queue_factor)),
        ),
    )

    for index, item in _bounded_parallel_reads(
        args_list[1:],
        executor,
        max_inflight,
        cancel_event,
    ):
        real_index = index + 1
        values[real_index] = item.data
        masks[real_index] = item.mask

    return values, masks


def _gpu_enabled() -> bool:
    return HAS_CUPY


def _to_gpu(array: np.ndarray):
    return cp.asarray(array) if HAS_CUPY else array


def _from_gpu(array) -> np.ndarray:
    return (
        cp.asnumpy(array)
        if HAS_CUPY and isinstance(array, cp.ndarray)
        else np.asarray(array)
    )


def _gpu_memory_cleanup() -> None:
    if not HAS_CUPY:
        return

    try:
        cp.get_default_memory_pool().free_all_blocks()
    except Exception:
        pass


def _streaming_cpu(
    selected_frames: list[FrameInfo],
    geometries: dict[Path, FrameGeometry],
    y1: int,
    y2: int,
    x1: int,
    x2: int,
    channel: int | None,
    normalization_factors: list[float],
    method: str,
    dither_shifts: list[tuple[float, float] | None] | None,
    cancel_event: threading.Event | None,
) -> np.ndarray:
    accumulator: np.ndarray | None = None
    counts: np.ndarray | None = None

    for index, frame in enumerate(selected_frames):
        check_cancel(cancel_event)

        block = read_frame_block(
            frame,
            geometries[frame.path],
            y1,
            y2,
            x1,
            x2,
            channel,
            normalization_factors[index],
            dither_shifts[index] if dither_shifts else None,
        )

        data = block.data
        mask = block.mask

        if accumulator is None:
            if method in {"Mean", "Sum"}:
                accumulator = np.zeros_like(
                    data,
                    dtype=np.float32,
                )
                counts = np.zeros(
                    data.shape,
                    dtype=np.uint32,
                )
            elif method == "Maximum":
                accumulator = np.full_like(
                    data,
                    -np.inf,
                    dtype=np.float32,
                )
            else:
                accumulator = np.full_like(
                    data,
                    np.inf,
                    dtype=np.float32,
                )

        if method in {"Mean", "Sum"}:
            accumulator[mask] += data[mask]
            counts[mask] += 1
        elif method == "Maximum":
            accumulator[mask] = np.maximum(
                accumulator[mask],
                data[mask],
            )
        elif method == "Minimum":
            accumulator[mask] = np.minimum(
                accumulator[mask],
                data[mask],
            )
        else:
            raise ValueError(f"Método streaming não suportado: {method}")

    if accumulator is None:
        raise ValueError("Bloco sem dados.")

    if method == "Mean":
        result = np.zeros_like(accumulator)
        valid = counts > 0
        result[valid] = accumulator[valid] / counts[valid]
        return result

    if method == "Sum":
        return accumulator

    accumulator[~np.isfinite(accumulator)] = 0.0
    return accumulator


def combine_block_streaming(
    selected_frames: list[FrameInfo],
    geometries: dict[Path, FrameGeometry],
    y1: int,
    y2: int,
    x1: int,
    x2: int,
    channel: int | None,
    normalization_factors: list[float],
    method: str,
    executor: ThreadPoolExecutor | None,
    dither_shifts: list[tuple[float, float] | None] | None,
    cancel_event: threading.Event | None,
    io_queue_factor: int,
) -> np.ndarray:

    if not HAS_CUPY:
        return _streaming_cpu(
            selected_frames,
            geometries,
            y1,
            y2,
            x1,
            x2,
            channel,
            normalization_factors,
            method,
            dither_shifts,
            cancel_event,
        )

    try:
        args_list = [
            (
                frame,
                geometries[frame.path],
                y1,
                y2,
                x1,
                x2,
                channel,
                normalization_factors[index],
                dither_shifts[index] if dither_shifts else None,
            )
            for index, frame in enumerate(selected_frames)
        ]

        accumulator = None
        counts = None
        executor_local = executor

        if executor_local is None:
            for index, args in enumerate(args_list):
                check_cancel(cancel_event)
                block = read_block_worker(args)

                data_gpu = cp.asarray(block.data)
                mask_gpu = cp.asarray(block.mask)

                if accumulator is None:
                    if method in {"Mean", "Sum"}:
                        accumulator = cp.zeros_like(
                            data_gpu,
                            dtype=cp.float32,
                        )
                        counts = cp.zeros(
                            data_gpu.shape,
                            dtype=cp.uint32,
                        )
                    elif method == "Maximum":
                        accumulator = cp.full_like(
                            data_gpu,
                            -cp.inf,
                            dtype=cp.float32,
                        )
                    elif method == "Minimum":
                        accumulator = cp.full_like(
                            data_gpu,
                            cp.inf,
                            dtype=cp.float32,
                        )

                if method in {"Mean", "Sum"}:
                    accumulator += cp.where(
                        mask_gpu,
                        data_gpu,
                        cp.float32(0),
                    )
                    counts += mask_gpu.astype(cp.uint32)
                elif method == "Maximum":
                    accumulator = cp.maximum(
                        accumulator,
                        cp.where(mask_gpu, data_gpu, -cp.inf),
                    )
                elif method == "Minimum":
                    accumulator = cp.minimum(
                        accumulator,
                        cp.where(mask_gpu, data_gpu, cp.inf),
                    )
                else:
                    raise ValueError(f"Método streaming não suportado: {method}")

        else:
            max_inflight = max(
                1,
                min(
                    len(args_list),
                    executor_local._max_workers * max(1, io_queue_factor),
                ),
            )

            for index, block in _bounded_parallel_reads(
                args_list,
                executor_local,
                max_inflight,
                cancel_event,
            ):
                del index
                data_gpu = cp.asarray(block.data)
                mask_gpu = cp.asarray(block.mask)

                if accumulator is None:
                    if method in {"Mean", "Sum"}:
                        accumulator = cp.zeros_like(
                            data_gpu,
                            dtype=cp.float32,
                        )
                        counts = cp.zeros(
                            data_gpu.shape,
                            dtype=cp.uint32,
                        )
                    elif method == "Maximum":
                        accumulator = cp.full_like(
                            data_gpu,
                            -cp.inf,
                            dtype=cp.float32,
                        )
                    elif method == "Minimum":
                        accumulator = cp.full_like(
                            data_gpu,
                            cp.inf,
                            dtype=cp.float32,
                        )

                if method in {"Mean", "Sum"}:
                    accumulator += cp.where(
                        mask_gpu,
                        data_gpu,
                        cp.float32(0),
                    )
                    counts += mask_gpu.astype(cp.uint32)
                elif method == "Maximum":
                    accumulator = cp.maximum(
                        accumulator,
                        cp.where(mask_gpu, data_gpu, -cp.inf),
                    )
                elif method == "Minimum":
                    accumulator = cp.minimum(
                        accumulator,
                        cp.where(mask_gpu, data_gpu, cp.inf),
                    )
                else:
                    raise ValueError(f"Método streaming não suportado: {method}")

                del data_gpu, mask_gpu

        check_cancel(cancel_event)

        if accumulator is None:
            raise ValueError("Bloco sem dados.")

        if method == "Mean":
            result_gpu = cp.zeros_like(
                accumulator,
                dtype=cp.float32,
            )
            valid = counts > 0
            result_gpu[valid] = accumulator[valid] / counts[valid]
        elif method == "Sum":
            result_gpu = accumulator
        else:
            result_gpu = cp.where(
                cp.isfinite(accumulator),
                accumulator,
                cp.float32(0),
            )

        result = cp.asnumpy(result_gpu)
        del accumulator, result_gpu
        if counts is not None:
            del counts

        return np.asarray(result, dtype=np.float32)

    except StackingCancelled:
        raise
    except Exception:
        _gpu_memory_cleanup()

        return _streaming_cpu(
            selected_frames,
            geometries,
            y1,
            y2,
            x1,
            x2,
            channel,
            normalization_factors,
            method,
            dither_shifts,
            cancel_event,
        )


def _reject_cpu(
    values: np.ndarray,
    masks: np.ndarray,
    method: str,
    low: float,
    high: float,
) -> np.ndarray:
    masked = values.astype(np.float32, copy=True)
    masked[~masks] = np.nan

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)

        if method == "SigmaClip":
            center = np.nanmedian(masked, axis=0)
            std = np.nanstd(masked, axis=0)

            lower = center - low * std
            upper = center + high * std

            valid = (masked >= lower) & (masked <= upper)

            stable = np.isfinite(std) & (std > 1e-10)
            valid |= ~stable
            masked[~valid] = np.nan

        elif method == "MAD":
            center = np.nanmedian(masked, axis=0)
            mad = np.nanmedian(
                np.abs(masked - center),
                axis=0,
            )

            sigma = np.float32(1.4826) * mad
            lower = center - low * sigma
            upper = center + high * sigma

            valid = (masked >= lower) & (masked <= upper)

            stable = np.isfinite(mad) & (mad > 1e-10)
            valid |= ~stable
            masked[~valid] = np.nan

        elif method == "Winsorized":
            p_low = np.nanpercentile(
                masked,
                min(49.0, max(0.0, low * 10.0)),
                axis=0,
            )
            p_high = np.nanpercentile(
                masked,
                max(51.0, min(100.0, 100.0 - high * 10.0)),
                axis=0,
            )

            valid_range = np.isfinite(p_low) & np.isfinite(p_high) & (p_high >= p_low)

            masked = np.where(
                valid_range,
                np.clip(masked, p_low, p_high),
                masked,
            )

        if method in {"SigmaClip", "MAD"}:
            return np.asarray(
                np.nanmedian(masked, axis=0),
                dtype=np.float32,
            )

        return np.asarray(
            np.nanmedian(masked, axis=0),
            dtype=np.float32,
        )


def reject_and_combine_block(
    values: np.ndarray,
    masks: np.ndarray,
    combine_method: str,
    rejection_method: str,
    low: float,
    high: float,
    cancel_event: threading.Event | None = None,
) -> np.ndarray:
    check_cancel(cancel_event)

    if values.ndim != 3 or masks.shape != values.shape:
        raise ValueError("Bloco inválido para rejeição.")

    if rejection_method == "None" or values.shape[0] <= 3:
        rejection_method = "None"

    if not HAS_CUPY:
        if rejection_method == "None":
            masked = values.astype(
                np.float32,
                copy=True,
            )
            masked[~masks] = np.nan

            with warnings.catch_warnings():
                warnings.simplefilter(
                    "ignore",
                    category=RuntimeWarning,
                )

                if combine_method == "Median":
                    result = np.nanmedian(
                        masked,
                        axis=0,
                    )
                elif combine_method == "Mean":
                    result = np.nanmean(
                        masked,
                        axis=0,
                    )
                elif combine_method == "Sum":
                    result = np.nansum(
                        masked,
                        axis=0,
                    )
                elif combine_method == "Maximum":
                    result = np.nanmax(
                        masked,
                        axis=0,
                    )
                elif combine_method == "Minimum":
                    result = np.nanmin(
                        masked,
                        axis=0,
                    )
                else:
                    raise ValueError(f"Método desconhecido: {combine_method}")

            return np.asarray(
                np.nan_to_num(
                    result,
                    nan=0.0,
                    posinf=0.0,
                    neginf=0.0,
                ),
                dtype=np.float32,
            )

        return _reject_cpu(
            values,
            masks,
            rejection_method,
            low,
            high,
        )

    try:
        masked_gpu = cp.asarray(values)
        masks_gpu = cp.asarray(masks)

        masked_gpu = cp.where(
            masks_gpu,
            masked_gpu,
            cp.nan,
        )

        if rejection_method == "SigmaClip":
            center = cp.nanmedian(
                masked_gpu,
                axis=0,
            )
            std = cp.nanstd(
                masked_gpu,
                axis=0,
            )

            lower = center - low * std
            upper = center + high * std

            valid = (masked_gpu >= lower) & (masked_gpu <= upper)

            stable = cp.isfinite(std) & (std > 1e-10)
            valid |= ~stable
            masked_gpu = cp.where(
                valid,
                masked_gpu,
                cp.nan,
            )

        elif rejection_method == "MAD":
            center = cp.nanmedian(
                masked_gpu,
                axis=0,
            )
            mad = cp.nanmedian(
                cp.abs(masked_gpu - center),
                axis=0,
            )

            sigma = cp.float32(1.4826) * mad
            lower = center - low * sigma
            upper = center + high * sigma

            valid = (masked_gpu >= lower) & (masked_gpu <= upper)

            stable = cp.isfinite(mad) & (mad > 1e-10)
            valid |= ~stable

            masked_gpu = cp.where(
                valid,
                masked_gpu,
                cp.nan,
            )

        elif rejection_method == "Winsorized":
            p_low = cp.nanpercentile(
                masked_gpu,
                min(49.0, max(0.0, low * 10.0)),
                axis=0,
            )
            p_high = cp.nanpercentile(
                masked_gpu,
                max(51.0, min(100.0, 100.0 - high * 10.0)),
                axis=0,
            )

            valid_range = cp.isfinite(p_low) & cp.isfinite(p_high) & (p_high >= p_low)

            clipped = cp.clip(
                masked_gpu,
                p_low,
                p_high,
            )

            masked_gpu = cp.where(
                valid_range,
                clipped,
                masked_gpu,
            )

        check_cancel(cancel_event)

        if combine_method == "Median":
            result_gpu = cp.nanmedian(
                masked_gpu,
                axis=0,
            )
        elif combine_method == "Mean":
            result_gpu = cp.nanmean(
                masked_gpu,
                axis=0,
            )
        elif combine_method == "Sum":
            result_gpu = cp.nansum(
                masked_gpu,
                axis=0,
            )
        elif combine_method == "Maximum":
            result_gpu = cp.nanmax(
                masked_gpu,
                axis=0,
            )
        elif combine_method == "Minimum":
            result_gpu = cp.nanmin(
                masked_gpu,
                axis=0,
            )
        else:
            raise ValueError(f"Método desconhecido: {combine_method}")

        result = cp.asnumpy(
            cp.nan_to_num(
                result_gpu,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
        )

        del masked_gpu, masks_gpu, result_gpu

        return np.asarray(
            result,
            dtype=np.float32,
        )

    except StackingCancelled:
        raise
    except Exception:
        _gpu_memory_cleanup()

        return _reject_cpu(
            values,
            masks,
            rejection_method,
            low,
            high,
        )


def combine_channel_blocks(
    selected_frames: list[FrameInfo],
    geometries: dict[Path, FrameGeometry],
    normalization_factors: list[float],
    height: int,
    width: int,
    channel: int | None,
    config: StackingConfig,
    executor: ThreadPoolExecutor | None,
    cancel_event: threading.Event | None,
    progress_callback: Callable[[int, int, str], None] | None,
    channel_label: str,
    dither_shifts: list[tuple[float, float] | None] | None,
) -> np.ndarray | None:
    check_cancel(cancel_event)

    rejection_enabled = config.rejection_method != "None"

    use_streaming = not rejection_enabled and config.method in {
        "Mean",
        "Sum",
        "Maximum",
        "Minimum",
    }

    chunk = adapt_chunk_size(
        config.effective_chunk_size,
        len(selected_frames),
        1,
        config.memory_budget_mb,
        rejection_enabled,
    )

    result = np.zeros(
        (height, width),
        dtype=np.float32,
    )

    n_blocks_y = math.ceil(height / chunk)
    n_blocks_x = math.ceil(width / chunk)
    total_blocks = n_blocks_y * n_blocks_x
    completed_blocks = 0

    for y1 in range(0, height, chunk):
        check_cancel(cancel_event)
        y2 = min(y1 + chunk, height)

        for x1 in range(0, width, chunk):
            check_cancel(cancel_event)
            x2 = min(x1 + chunk, width)

            if use_streaming:
                block_result = combine_block_streaming(
                    selected_frames=selected_frames,
                    geometries=geometries,
                    y1=y1,
                    y2=y2,
                    x1=x1,
                    x2=x2,
                    channel=channel,
                    normalization_factors=normalization_factors,
                    method=config.method,
                    executor=executor,
                    dither_shifts=dither_shifts,
                    cancel_event=cancel_event,
                    io_queue_factor=config.effective_io_queue,
                )
            else:
                values, masks = load_channel_block_parallel(
                    selected_frames=selected_frames,
                    geometries=geometries,
                    y1=y1,
                    y2=y2,
                    x1=x1,
                    x2=x2,
                    channel=channel,
                    normalization_factors=normalization_factors,
                    executor=executor,
                    dither_shifts=dither_shifts,
                    cancel_event=cancel_event,
                    io_queue_factor=config.effective_io_queue,
                )

                check_cancel(cancel_event)

                block_result = reject_and_combine_block(
                    values,
                    masks,
                    combine_method=config.method,
                    rejection_method=config.rejection_method,
                    low=config.rejection_low,
                    high=config.rejection_high,
                    cancel_event=cancel_event,
                )

                del values, masks

            result[y1:y2, x1:x2] = block_result
            del block_result

            completed_blocks += 1

            if progress_callback:
                progress_callback(
                    completed_blocks,
                    total_blocks,
                    (f"{channel_label}: bloco {completed_blocks}/{total_blocks}"),
                )

    return result


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


def build_reference_mask(
    selected_frames: list[FrameInfo],
    geometries: dict[Path, FrameGeometry],
    normalization_factors: list[float],
    height: int,
    width: int,
    config: StackingConfig,
    dither_shifts: list[tuple[float, float] | None] | None,
    executor: ThreadPoolExecutor | None,
    cancel_event: threading.Event | None,
) -> np.ndarray:
    chunk = adapt_chunk_size(
        config.effective_chunk_size,
        len(selected_frames),
        1,
        min(config.memory_budget_mb, 512),
        False,
    )

    coverage_map = np.zeros(
        (height, width),
        dtype=np.uint16,
    )

    for y1 in range(0, height, chunk):
        check_cancel(cancel_event)
        y2 = min(y1 + chunk, height)

        for x1 in range(0, width, chunk):
            check_cancel(cancel_event)
            x2 = min(x1 + chunk, width)

            coverage = np.zeros(
                (y2 - y1, x2 - x1),
                dtype=np.uint16,
            )

            args_list = [
                (
                    frame,
                    geometries[frame.path],
                    y1,
                    y2,
                    x1,
                    x2,
                    0 if frame.image_kind == "RGB" else None,
                    normalization_factors[index],
                    dither_shifts[index] if dither_shifts else None,
                )
                for index, frame in enumerate(selected_frames)
            ]

            if executor is None:
                iterator = (
                    (i, read_block_worker(args)) for i, args in enumerate(args_list)
                )
            else:
                iterator = _bounded_parallel_reads(
                    args_list,
                    executor,
                    max(
                        1,
                        min(
                            len(args_list),
                            executor._max_workers * config.effective_io_queue,
                        ),
                    ),
                    cancel_event,
                )

            for _, item in iterator:
                check_cancel(cancel_event)
                coverage += item.mask.astype(
                    np.uint16,
                    copy=False,
                )

            coverage_map[y1:y2, x1:x2] = coverage

    max_coverage = int(np.max(coverage_map))

    if max_coverage <= 0:
        return np.zeros(
            (height, width),
            dtype=bool,
        )

    threshold = max(
        1,
        int(math.ceil(max_coverage * 0.70)),
    )

    return coverage_map >= threshold


def flatten_background(
    data: np.ndarray,
    valid_mask: np.ndarray,
) -> np.ndarray:
    try:
        from cupyx.scipy.ndimage import median_filter, uniform_filter

        data_gpu = cp.asarray(data)
        mask_gpu = cp.asarray(valid_mask)

        if data_gpu.ndim == 3:
            output_gpu = cp.empty_like(data_gpu)

            for channel in range(data_gpu.shape[0]):
                channel_data = data_gpu[channel].copy()

                if bool(cp.any(mask_gpu)):
                    background_fill = cp.median(channel_data[mask_gpu])
                else:
                    background_fill = cp.float32(0.0)

                channel_data[~mask_gpu] = background_fill

                background = median_filter(
                    channel_data,
                    size=35,
                )
                background = uniform_filter(
                    background,
                    size=35,
                )

                output_gpu[channel] = data_gpu[channel] - background
        else:
            channel_data = data_gpu.copy()

            if bool(cp.any(mask_gpu)):
                background_fill = cp.median(channel_data[mask_gpu])
            else:
                background_fill = cp.float32(0.0)

            channel_data[~mask_gpu] = background_fill

            background = median_filter(
                channel_data,
                size=35,
            )
            background = uniform_filter(
                background,
                size=35,
            )

            output_gpu = data_gpu - background

        return cp.asnumpy(output_gpu).astype(np.float32, copy=False)

    except Exception:
        pass

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
        "STACK_GPU": (
            bool(HAS_CUPY),
            "CuPy available",
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

    if config.output_bit_depth == "32-bit":
        return np.nan_to_num(
            result,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).astype(np.float32, copy=False)

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
    with fits.open(
        filepath,
        memmap=False,
        lazy_load_hdus=True,
    ) as hdul:
        _, hdu = _find_primary_image_hdu(hdul)
        return hdu.header.copy()


def _validate_config(config: StackingConfig) -> None:
    if config.method not in {
        "Median",
        "Mean",
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
    }:
        raise ValueError(f"Modo de seleção inválido: {config.selection_mode}")

    if config.output_bit_depth not in {
        "16-bit",
        "32-bit",
    }:
        raise ValueError(f"Profundidade inválida: {config.output_bit_depth}")

    if config.selection_percentage <= 0:
        raise ValueError("selection_percentage deve ser > 0.")

    if config.chunk_size <= 0:
        raise ValueError("chunk_size deve ser > 0.")


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

        fits_files, batch_metadata = discover_aligned_frames(config.input_dir)

        check_cancel(cancel_event)

        if not fits_files:
            log(f"❌ Nenhum arquivo FITS encontrado em: {config.input_dir}")
            return {
                "status": "error",
                "reason": "no_fits_files",
            }

        log(f"📁 Total de {len(fits_files)} arquivos FITS encontrados.")

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

        selected_frames = select_frames(
            all_frames,
            config,
        )

        log(
            f"✅ Selecionados {len(selected_frames)} "
            f"frames de {len(all_frames)} disponíveis."
        )

        if len(selected_frames) < 3:
            log("❌ Número insuficiente de frames selecionados (mínimo: 3).")
            return {
                "status": "error",
                "reason": "insufficient_frames",
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

        height = reference_geometry.height
        width = reference_geometry.width
        channels = reference_geometry.channels

        log(
            f"📐 Dimensão: {width}×{height} | "
            f"{reference_geometry.image_kind} | "
            f"canais={channels}"
        )

        # VALID_MASK is independent of the image channel and output block.
        # Cache it once: compressed FITS mask tiles otherwise get decoded for
        # every block and once again for every RGB channel.
        masks_loaded = 0
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

        effective_chunk = adapt_chunk_size(
            config.effective_chunk_size,
            len(selected_frames),
            1,
            config.memory_budget_mb,
            config.rejection_method != "None",
        )

        log(
            f"🚀 Streaming: workers={config.worker_count} | "
            f"bloco={effective_chunk}×{effective_chunk} | "
            f"orçamento≈{config.memory_budget_mb} MB | "
            f"GPU={'CuPy' if HAS_CUPY else 'CPU'}"
        )

        if config.rejection_method == "None" and config.method in {
            "Mean",
            "Sum",
            "Maximum",
            "Minimum",
        }:
            log("⚡ Modo incremental ativo: não mantém stack global na RAM.")
        else:
            log("🧮 Mediana/rejeição: somente o bloco atual permanece na RAM.")

        with ThreadPoolExecutor(
            max_workers=config.worker_count,
            thread_name_prefix="AstroStackIO",
        ) as executor:
            if reference_geometry.image_kind == "Mono":
                result_2d = combine_channel_blocks(
                    selected_frames=selected_frames,
                    geometries=geometries,
                    normalization_factors=normalization_factors,
                    height=height,
                    width=width,
                    channel=None,
                    config=config,
                    executor=executor,
                    cancel_event=cancel_event,
                    progress_callback=progress,
                    channel_label="Mono",
                    dither_shifts=dither_shifts,
                )

                if result_2d is None:
                    return {"status": "cancelled"}

                output_result = result_2d

            else:
                output_result = np.zeros(
                    (channels, height, width),
                    dtype=np.float32,
                )

                for channel in range(channels):
                    check_cancel(cancel_event)

                    log(f"🎨 Processando canal {channel + 1}/{channels}...")

                    channel_result = combine_channel_blocks(
                        selected_frames=selected_frames,
                        geometries=geometries,
                        normalization_factors=normalization_factors,
                        height=height,
                        width=width,
                        channel=channel,
                        config=config,
                        executor=executor,
                        cancel_event=cancel_event,
                        progress_callback=progress,
                        channel_label=(f"Canal {channel + 1}/{channels}"),
                        dither_shifts=dither_shifts,
                    )

                    if channel_result is None:
                        return {"status": "cancelled"}

                    output_result[channel] = channel_result
                    del channel_result

            check_cancel(cancel_event)

            log("🛡️ Construindo máscara de validade...")

            validity_mask = build_reference_mask(
                selected_frames,
                geometries,
                normalization_factors,
                height,
                width,
                config,
                dither_shifts,
                executor,
                cancel_event,
            )

        check_cancel(cancel_event)

        if config.remove_background:
            log("🌌 Removendo gradiente/background (modo explicitamente habilitado)...")

            output_result = flatten_background(
                output_result,
                validity_mask,
            )
        else:
            log("🌌 Background preservado: remoção de gradiente desabilitada.")

        check_cancel(cancel_event)

        log("✂️ Aplicando máscara de cobertura...")

        if output_result.ndim == 3:
            output_result[:, ~validity_mask] = 0.0
        else:
            output_result[~validity_mask] = 0.0

        output_result = np.nan_to_num(
            output_result,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).astype(
            np.float32,
            copy=False,
        )

        check_cancel(cancel_event)

        source_header = load_source_header(reference_frame.path)

        header = prepare_output_header(
            source_header=source_header,
            config=config,
            n_frames=len(selected_frames),
            n_frames_total=len(all_frames),
            n_batches=len(batch_metadata),
            avg_quality=avg_quality,
            avg_star_count=avg_star_count,
            avg_fwhm=avg_fwhm,
        )

        log(f"💾 Preparando saída {config.output_bit_depth}...")

        data_to_save = convert_output_dtype(
            output_result,
            config,
        )

        config.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_path = config.output_dir / config.output_name

        check_cancel(cancel_event)

        log(f"💾 Salvando: {output_path}")

        write_stack_output(
            output_path,
            data_to_save,
            validity_mask,
            header,
            config,
        )

        check_cancel(cancel_event)

        stats = {
            "status": "success",
            "output_path": str(output_path),
            "n_frames": len(selected_frames),
            "n_frames_total": len(all_frames),
            "n_batches": len(batch_metadata),
            "method": config.method,
            "rejection": config.rejection_method,
            "selection_mode": config.selection_mode,
            "selection_percentage": (config.selection_percentage),
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
            "workers": config.worker_count,
            "chunk_size": effective_chunk,
            "memory_budget_mb": config.memory_budget_mb,
            "streaming": True,
            "gpu": bool(HAS_CUPY),
            "background_removed": bool(config.remove_background),
            "dither_correction": bool(config.apply_dither_correction),
            "batches": {
                name: meta["frame_count"] for name, meta in batch_metadata.items()
            },
        }

        log("\n✅ Stacking concluído!")
        log(f"📁 Resultado: {output_path}")
        log(
            f"📊 {len(selected_frames)} frames combinados "
            f"(de {len(all_frames)} disponíveis)"
        )
        log(f"📁 {len(batch_metadata)} batches processados")
        log(f"📈 Qualidade média: {avg_quality:.2f}")
        log(f"📈 FWHM médio: {avg_fwhm:.2f}px")
        log(f"💾 Profundidade: {config.output_bit_depth}")

        return stats

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
    workers = None

    if config_dict.get("workers") is not None:
        try:
            workers = int(config_dict["workers"])
        except (TypeError, ValueError):
            workers = None

    return StackingConfig(
        base_dir=Path(config_dict.get("base_dir", "")),
        input_dir=Path(input_dir),
        output_dir=Path(config_dict.get("output_dir", "")),
        selection_mode=config_dict.get(
            "selection_mode",
            "BestPercentage",
        ),
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
        method=config_dict.get(
            "method",
            "Median",
        ),
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
        output_bit_depth=config_dict.get(
            "output_bit_depth",
            "32-bit",
        ),
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
    "HAS_CUPY",
    "BlockRead",
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
