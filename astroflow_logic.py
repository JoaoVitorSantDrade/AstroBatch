import copy
import hashlib
import itertools
import json
import math
import os
import re
import tempfile
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import OrderedDict, deque
from functools import partial
from pathlib import Path

import cv2
import numpy as np
from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.utils.exceptions import AstropyWarning
from photutils.detection import DAOStarFinder
from scipy.spatial import KDTree

try:
    # DAO's public class is stable, while the cached-convolution path below
    # uses private helpers that vary between Photutils releases. Import the
    # optional module once on the main thread so frame workers do not contend
    # on Python's import lock for every detection attempt.
    from photutils.detection import daofinder as _DAOFINDER_MODULE
except Exception:  # pragma: no cover - exercised only by unsupported versions
    _DAOFINDER_MODULE = None

try:
    from photutils.detection.peakfinder import (
        _fast_circular_peaks as _DAO_FAST_CIRCULAR_PEAKS,
    )
except Exception:  # pragma: no cover - exercised only by unsupported versions
    _DAO_FAST_CIRCULAR_PEAKS = None


class _DAOFlowSources:
    """Minimal source-table contract consumed by :func:`detect_stars_dao`.

    Photutils' QTable materialization performs metadata/version work and
    computes many columns that Flow does not use. This private row container
    keeps the same ``sort``/``reverse``/column access contract for the four
    required arrays while preserving NumPy's default argsort ordering.
    """

    def __init__(self, columns: dict[str, np.ndarray]) -> None:
        self._columns = columns
        self.colnames = tuple(columns)

    def __len__(self) -> int:
        if not self._columns:
            return 0
        return len(next(iter(self._columns.values())))

    def __getitem__(self, key):
        if isinstance(key, str):
            return self._columns[key]
        return _DAOFlowSources({name: values[key] for name, values in self._columns.items()})

    def sort(self, key: str) -> None:
        if isinstance(key, (list, tuple)):
            if len(key) != 1:
                raise ValueError("Flow source sorting expects one column")
            key = key[0]
        order = np.argsort(self._columns[key])
        self._columns = {
            name: np.asarray(values)[order]
            for name, values in self._columns.items()
        }

    def reverse(self) -> None:
        self._columns = {
            name: np.asarray(values)[::-1].copy()
            for name, values in self._columns.items()
        }

    def sort_descending(self, key: str) -> None:
        """Sort once in the same order as ``sort`` followed by ``reverse``."""

        order = np.argsort(self._columns[key])[::-1]
        self._columns = {
            name: np.asarray(values)[order]
            for name, values in self._columns.items()
        }


def _dao_catalog_to_flow_table(catalog):
    """Materialize only the DAO columns consumed by Flow.

    ``DAOStarFinderCatalog.to_table()`` computes every public catalog column,
    including threshold-derived diagnostics that Flow never reads. Keeping
    this helper isolated makes the optimization safe across Photutils
    versions: unsupported private column selection falls back to the complete
    table, and the surrounding detector still has its public fallback.
    """

    attributes = {
        "xcentroid": ("x_centroid", "xcentroid"),
        "ycentroid": ("y_centroid", "ycentroid"),
        "flux": ("flux",),
        "sharpness": ("sharpness",),
    }
    try:
        columns = {}
        for public_name, candidates in attributes.items():
            for candidate in candidates:
                try:
                    columns[public_name] = np.asarray(getattr(catalog, candidate))
                    break
                except AttributeError:
                    continue
            else:
                raise AttributeError(public_name)
        return _DAOFlowSources(columns)
    except (AttributeError, KeyError, TypeError, ValueError):
        return catalog.to_table()


def _dao_find_stars_without_table(convolved, finder, threshold):
    """Return DAO peak coordinates without constructing Photutils' QTable.

    DAO's current configuration always uses a positive minimum separation and
    no mask/border exclusion. The direct path mirrors ``find_peaks``'s
    NaN/constant handling and row-major ``nonzero`` ordering; unsupported
    Photutils versions fall back to the public private helper unchanged.
    """

    fast_peaks = _DAO_FAST_CIRCULAR_PEAKS
    if fast_peaks is None or finder.min_separation <= 0:
        return finder._find_stars(
            convolved,
            finder.kernel,
            threshold,
            min_separation=finder.min_separation,
            mask=None,
            exclude_border=finder.exclude_border,
        )

    try:
        data = np.asanyarray(convolved)
        if data.size == 0 or np.all(data == data.flat[0]):
            return None
        if np.any(np.isnan(data)):
            data = np.array(data, copy=True)
            data[np.isnan(data)] = np.nanmin(data)
        peak_goodmask = fast_peaks(data, finder.min_separation)
        if finder.exclude_border:
            yborder = finder.kernel.y_radius
            xborder = finder.kernel.x_radius
            if yborder > 0:
                peak_goodmask[:yborder, :] = False
                peak_goodmask[-yborder:, :] = False
            if xborder > 0:
                peak_goodmask[:, :xborder] = False
                peak_goodmask[:, -xborder:] = False
        peak_goodmask &= data > threshold
        y_peaks, x_peaks = peak_goodmask.nonzero()
        if len(x_peaks) == 0:
            return None
        return np.transpose((x_peaks, y_peaks))
    except Exception:
        # Keep the established Photutils implementation as the compatibility
        # route if a future release changes the private peak helper contract.
        return finder._find_stars(
            convolved,
            finder.kernel,
            threshold,
            min_separation=finder.min_separation,
            mask=None,
            exclude_border=finder.exclude_border,
        )

from app.engines import EngineDescriptor, EngineProfile, EngineUnavailable, registry
from app.engines.execution import ExecutionBudget
from app.engines.astroalign_fallback import estimate_asterism_transform
from frame_quality import measure_star_shapes
from cpu_runtime import configure_opencv_threads, configure_worker_runtime, physical_core_count
from temporal_analysis import (
    DEFAULT_GAP_MINUTES,
    DEFAULT_SEEING_SIGMA,
    build_session_temporal_report,
    build_temporal_report,
    enrich_flow_frames,
    parse_date_obs,
    write_temporal_report,
)

warnings.simplefilter("ignore", category=AstropyWarning)


def max_science_frame_bytes(filepath: Path) -> int:
    """Estimate science image bytes from FITS headers without loading pixels."""
    with fits.open(filepath, memmap=False, lazy_load_hdus=True) as hdul:
        for hdu in hdul:
            if getattr(hdu, "is_image", False) and getattr(hdu, "shape", None):
                shape = tuple(int(v) for v in hdu.shape)
                return int(np.prod(shape, dtype=np.int64)) * 4
    return 0


# ============================================================
# Bayer / CFA (Contrato 100% compatível com [source: 17])
# ============================================================

BAYER_PATTERNS = {"RGGB", "BGGR", "GRBG", "GBRG"}


def get_bayer_pattern(header: fits.Header) -> str | None:
    for key in ["BAYERPAT", "BAYERPATTERN", "COLORTYP"]:
        if key in header:
            val = str(header[key]).strip().upper().strip("'")
            if val in BAYER_PATTERNS:
                return val
    return None


def split_cfa(data: np.ndarray, pattern: str):
    if pattern == "RGGB":
        return data[0::2, 0::2], data[0::2, 1::2], data[1::2, 0::2], data[1::2, 1::2]
    if pattern == "BGGR":
        return data[1::2, 1::2], data[0::2, 1::2], data[1::2, 0::2], data[0::2, 0::2]
    if pattern == "GRBG":
        return data[0::2, 1::2], data[0::2, 0::2], data[1::2, 1::2], data[1::2, 0::2]
    if pattern == "GBRG":
        return data[1::2, 0::2], data[1::2, 1::2], data[0::2, 0::2], data[0::2, 1::2]
    return data[0::2, 0::2], data[0::2, 1::2], data[1::2, 0::2], data[1::2, 1::2]


def extract_luminance(data: np.ndarray, header: fits.Header) -> np.ndarray:
    """Extrai a luminância retornando rigorosamente um array 2D float32 (essencial para o Card Preview)."""
    if data.ndim == 3:
        if data.shape[0] in (3, 4):
            img_hwc = np.moveaxis(data, 0, -1)
        else:
            img_hwc = data
        return (
            0.2126 * img_hwc[:, :, 0]
            + 0.7152 * img_hwc[:, :, 1]
            + 0.0722 * img_hwc[:, :, 2]
        ).astype(np.float32)

    if data.ndim == 2:
        pattern = get_bayer_pattern(header)
        if pattern:
            r, g1, g2, b = split_cfa(data, pattern)
            l_sub = 0.2126 * r + 0.3576 * g1 + 0.3576 * g2 + 0.0722 * b
            l_full = np.repeat(np.repeat(l_sub, 2, axis=0), 2, axis=1)
            return l_full.astype(np.float32)
        return data.astype(np.float32)

    raise ValueError(f"Dimensões não suportadas: {data.shape}")


# ============================================================
# FITS (Contrato compatível)
# ============================================================


def load_fits_data(filepath: Path) -> tuple[np.ndarray, fits.Header]:
    # Dedicated-camera uint16 FITS commonly use signed int16+BZERO.  Astropy
    # cannot memory-map the scaled ``.data`` property, so request raw storage
    # and restore the cards in-place in float32.  This keeps the large read
    # sequential and avoids an intermediate Astropy scaling allocation while
    # preserving the established physical values (including BLANK -> NaN).
    try:
        with fits.open(
            filepath,
            memmap=True,
            lazy_load_hdus=True,
            do_not_scale_image_data=True,
        ) as hdul:
            for hdu in hdul:
                if not hdu.is_image or getattr(hdu, "shape", None) is None:
                    continue
                if len(hdu.shape) not in (2, 3):
                    continue
                raw = np.asarray(hdu.data)
                header = hdu.header.copy()
                bscale = float(header.get("BSCALE", 1.0))
                bzero = float(header.get("BZERO", 0.0))
                blank = header.get("BLANK")
                blank_value = int(blank) if blank is not None else None
                if (
                    raw.dtype == np.float32
                    and bscale == 1.0
                    and bzero == 0.0
                    and blank_value is None
                ):
                    # Do not return a view into a file mapping after the HDU
                    # list closes; callers may retain this array through the
                    # detection/phase stages and Windows must be able to
                    # replace the source file later.
                    return np.array(raw, dtype=np.float32, copy=True), header
                values = np.asarray(raw, dtype=np.float32).copy()
                if bscale != 1.0:
                    np.multiply(values, np.float32(bscale), out=values)
                if bzero != 0.0:
                    np.add(values, np.float32(bzero), out=values)
                if blank_value is not None:
                    values[raw == blank_value] = np.nan
                return values, header
    except (OSError, ValueError, TypeError, OverflowError):
        # Keep the legacy path as a compatibility fallback for unusual FITS
        # files whose raw header cannot be interpreted by the explicit scaler.
        pass

    with fits.open(filepath, memmap=False) as hdul:
        for hdu in hdul:
            if hdu.is_image and hdu.data is not None and hdu.data.ndim in (2, 3):
                return np.asarray(hdu.data, dtype=np.float32), hdu.header.copy()
    raise ValueError(f"Imagem inválida: {filepath.name}")


# ============================================================
# Detecção de Estrelas
# ============================================================


def prepare_for_phase_correlation(data: np.ndarray) -> np.ndarray:
    finite_mask = np.isfinite(data)
    if not np.any(finite_mask):
        return np.zeros_like(data, dtype=np.float32)
    if finite_mask.all():
        d_min = float(np.min(data))
        d_max = float(np.max(data))
        if d_max > d_min:
            return ((data - d_min) / (d_max - d_min)).astype(
                np.float32, copy=False
            )
        return np.zeros_like(data, dtype=np.float32)
    finite_data = data[finite_mask]
    d_min, d_max = float(np.min(finite_data)), float(np.max(finite_data))
    if d_max > d_min:
        norm = (data - d_min) / (d_max - d_min)
    else:
        norm = np.zeros_like(data, dtype=np.float32)
    return np.nan_to_num(norm, nan=0.0, posinf=1.0, neginf=0.0).astype(
        np.float32, copy=False
    )


def calculate_anchor_quality(star_count: int, fwhm: float) -> float:
    if fwhm <= 0:
        return float(star_count)
    return float(star_count) / float(fwhm)


def _greedy_spatial_filter(coords: np.ndarray, min_dist: float, max_count: int) -> list:
    """
    Non-max suppression espacial usando grid hashing.
    Substitui a versão anterior O(N^2) (que recriava um np.array a cada
    iteração e comparava contra todos os pontos já aceitos) por uma
    versão O(N) médio, checando apenas as células vizinhas na grade.
    Assume que `coords` já está ordenado por prioridade (ex.: flux decrescente).
    """
    n = len(coords)
    if n == 0:
        return []

    cell_size = max(float(min_dist), 1e-3)
    min_dist_sq = float(min_dist) ** 2
    grid: dict[tuple[int, int], list[int]] = {}
    accepted: list[int] = []

    for idx in range(n):
        x, y = float(coords[idx][0]), float(coords[idx][1])
        cell_x, cell_y = int(x // cell_size), int(y // cell_size)

        too_close = False
        for gx in (cell_x - 1, cell_x, cell_x + 1):
            for gy in (cell_y - 1, cell_y, cell_y + 1):
                bucket = grid.get((gx, gy))
                if not bucket:
                    continue
                for other_idx in bucket:
                    ox, oy = coords[other_idx]
                    if (ox - x) ** 2 + (oy - y) ** 2 < min_dist_sq:
                        too_close = True
                        break
                if too_close:
                    break
            if too_close:
                break

        if not too_close:
            grid.setdefault((cell_x, cell_y), []).append(idx)
            accepted.append(idx)
            if len(accepted) >= max_count:
                break

    return accepted


def _dao_sources_with_cached_convolution(
    data: np.ndarray,
    background: float,
    background_std: float,
    threshold: float,
    fwhm: float,
    stats_cache: dict,
) -> tuple[bool, object | None]:
    """Run DAOStarFinder while reusing its invariant convolution.

    Photutils applies the same convolution every time an adaptive threshold is
    tried.  The public ``DAOStarFinder`` call intentionally remains the
    fallback, while this path mirrors its current private implementation and
    only runs for the per-frame cache used by Flow.  ``False`` means the
    installed Photutils version does not expose the compatible internals.
    """

    if stats_cache.get("dao_cache_disabled"):
        return False, None
    try:
        _daofinder = _DAOFINDER_MODULE
        if _daofinder is None:
            return False, None

        finder = stats_cache.get("dao_finder")
        if finder is None or float(getattr(finder, "fwhm", -1.0)) != float(fwhm):
            # The threshold is supplied again below for each retry; this
            # object owns only the fwhm-dependent kernel and filter settings.
            finder = DAOStarFinder(fwhm=fwhm, threshold=threshold)
            stats_cache["dao_finder"] = finder
            stats_cache.pop("dao_source_data", None)
            stats_cache.pop("dao_convolved", None)
            stats_cache.pop("dao_low_xypos", None)
            stats_cache.pop("dao_low_xypos_computed", None)
            stats_cache.pop("dao_catalog", None)
            stats_cache.pop("dao_catalog_table", None)
            stats_cache.pop("dao_catalog_ready", None)
            stats_cache.pop("dao_attempts", None)

        source_data = stats_cache.get("dao_source_data")
        if source_data is None:
            source_data = data - background
            stats_cache["dao_source_data"] = source_data
        convolved = stats_cache.get("dao_convolved")
        if convolved is None:
            convolved = _daofinder._filter_data(
                source_data,
                finder.kernel.data,
                mode="constant",
                fill_value=0.0,
                check_normalization=False,
            )
            stats_cache["dao_convolved"] = convolved

        threshold_eff = threshold * finder.kernel.rel_err
        # The first adaptive attempt keeps the established Photutils catalog
        # path.  When Flow asks for another threshold on the same image, build
        # the catalog once at the lowest search threshold and slice it for the
        # later attempts.  DAO's catalog filters (sharpness, roundness,
        # finiteness and peak limit) do not depend on the scalar threshold;
        # only ``daofind_mag`` does, and it is not consumed by Flow.  We still
        # refresh that field below for callers that inspect the private table.
        attempt_number = int(stats_cache.get("dao_attempts", 0))
        # Local Flow retries at 3.0 sigma and Global Flow at 2.8 sigma.  A
        # lower-threshold peak search is a superset of every higher-threshold
        # result for the same convolved image, so filter its coordinates by
        # the current threshold instead of rescanning the full image.  Calls
        # below 2.8 sigma retain the exact direct path as a compatibility
        # fallback.
        search_floor = 2.8 * float(background_std)
        if threshold >= search_floor:
            stats_cache["dao_attempts"] = attempt_number + 1
            if not stats_cache.get("dao_low_xypos_computed"):
                stats_cache["dao_low_xypos"] = _dao_find_stars_without_table(
                    convolved,
                    finder,
                    search_floor * finder.kernel.rel_err,
                )
                stats_cache["dao_low_xypos_computed"] = True
            low_xypos = stats_cache.get("dao_low_xypos")
            if low_xypos is None:
                xypos = None
            else:
                low_xypos = np.asarray(low_xypos)
                indices = low_xypos.astype(np.intp, copy=False)
                keep = convolved[indices[:, 1], indices[:, 0]] >= threshold_eff
                xypos = low_xypos[keep]
                if len(xypos) == 0:
                    xypos = None
        else:
            xypos = _dao_find_stars_without_table(
                convolved,
                finder,
                threshold_eff,
            )
        if xypos is None:
            return True, None

        if (
            threshold >= search_floor
            and attempt_number >= 1
            and not stats_cache.get("dao_catalog_ready")
        ):
            low_xypos = stats_cache.get("dao_low_xypos")
            if low_xypos is None:
                stats_cache["dao_catalog"] = None
            else:
                floor_catalog = _daofinder._DAOStarFinderCatalog(
                    source_data,
                    convolved,
                    np.asarray(low_xypos),
                    search_floor,
                    finder.kernel,
                    sharpness_range=finder.sharpness_range,
                    roundness_range=finder.roundness_range,
                    n_brightest=finder.n_brightest,
                    peak_max=finder.peak_max,
                    scale_threshold=finder.scale_threshold,
                )
                stats_cache["dao_catalog"] = floor_catalog.apply_all_filters()
            stats_cache["dao_catalog_ready"] = True

        cached_catalog = stats_cache.get("dao_catalog")
        if (
            threshold >= search_floor
            and stats_cache.get("dao_catalog_ready")
        ):
            if cached_catalog is None:
                return True, None
            catalog_xypos = np.asarray(cached_catalog.xypos)
            catalog_indices = catalog_xypos.astype(np.intp, copy=False)
            catalog_keep = (
                convolved[catalog_indices[:, 1], catalog_indices[:, 0]]
                >= threshold_eff
            )
            if not np.any(catalog_keep):
                return True, None
            # Flow consumes only these four columns. Cache the floor table so
            # later adaptive thresholds slice plain columns instead of
            # rebuilding every catalog property (including daofind_mag).
            flow_table = stats_cache.get("dao_catalog_table")
            if flow_table is None:
                flow_table = _dao_catalog_to_flow_table(cached_catalog)
                stats_cache["dao_catalog_table"] = flow_table
            return True, flow_table[catalog_keep]

        catalog = _daofinder._DAOStarFinderCatalog(
            source_data,
            convolved,
            xypos,
            threshold,
            finder.kernel,
            sharpness_range=finder.sharpness_range,
            roundness_range=finder.roundness_range,
            n_brightest=finder.n_brightest,
            peak_max=finder.peak_max,
            scale_threshold=finder.scale_threshold,
        )
        catalog = catalog.apply_all_filters()
        return True, None if catalog is None else _dao_catalog_to_flow_table(catalog)
    except Exception:
        # Private Photutils symbols may change independently of AstroBatch.
        # Keep a functional, scientifically conservative public fallback.
        stats_cache["dao_cache_disabled"] = True
        stats_cache.pop("dao_source_data", None)
        stats_cache.pop("dao_convolved", None)
        stats_cache.pop("dao_low_xypos", None)
        stats_cache.pop("dao_low_xypos_computed", None)
        stats_cache.pop("dao_catalog", None)
        stats_cache.pop("dao_catalog_table", None)
        stats_cache.pop("dao_catalog_ready", None)
        return False, None


def detect_stars_dao(
    data: np.ndarray,
    fwhm: float,
    sigma: float,
    max_stars: int,
    stats_cache: dict | None = None,
) -> tuple[np.ndarray, float, dict]:
    if stats_cache is not None:
        identity = (id(data), tuple(data.shape), str(data.dtype))
        if stats_cache.get("dao_data_identity") != identity:
            for key in tuple(stats_cache):
                if str(key).startswith("dao_"):
                    stats_cache.pop(key, None)
            stats_cache["dao_data_identity"] = identity
    # Adaptive detection tries several thresholds on the same luminance
    # image.  These image-only statistics are invariant to ``sigma`` and are
    # therefore safe to reuse for that bounded retry loop.  The cache is
    # created per prepared frame by ``_process_single_frame`` and never
    # escapes the call, so it cannot retain image buffers between frames.
    cached_stats = stats_cache.get("dao_stats") if stats_cache is not None else None
    if cached_stats is None:
        mean_val, median_val, std_val = (
            float(np.mean(data)),
            float(np.median(data)),
            float(np.std(data)),
        )
        _, bkg_median, bkg_std = sigma_clipped_stats(data, sigma=3.0)
        cached_stats = (
            mean_val,
            median_val,
            std_val,
            float(bkg_median),
            float(bkg_std),
        )
        if stats_cache is not None:
            stats_cache["dao_stats"] = cached_stats
    mean_val, median_val, std_val, bkg_median, bkg_std = cached_stats

    if not np.isfinite(bkg_std) or bkg_std <= 0:
        metrics = {
            "star_count": 0,
            "fwhm": 0.0,
            "mean": round(mean_val, 2),
            "median": round(median_val, 2),
            "std": round(std_val, 2),
            "background": round(bkg_median, 2),
            "snr": 0.0,
            "min_flux": 0.0,
            "max_flux": 0.0,
            "valid": False,
        }
        return (np.empty((0, 2), dtype=np.float32), 0.0, metrics)

    threshold = sigma * bkg_std
    used_cached_convolution = False
    sources = None
    if stats_cache is not None:
        used_cached_convolution, sources = _dao_sources_with_cached_convolution(
            data, bkg_median, bkg_std, threshold, fwhm, stats_cache
        )
    if not used_cached_convolution:
        daofind = DAOStarFinder(fwhm=fwhm, threshold=threshold)
        sources = daofind(data - bkg_median)

    if sources is not None and len(sources) > 0:
        if hasattr(sources, "sort_descending"):
            sources.sort_descending("flux")
        else:
            sources.sort("flux")
            sources.reverse()

        raw_coords = np.transpose((sources["xcentroid"], sources["ycentroid"])).astype(
            np.float32
        )
        raw_fluxes = np.asarray(sources["flux"], dtype=np.float32)
        has_sharpness = "sharpness" in sources.colnames
        raw_sharpness = (
            np.asarray(sources["sharpness"], dtype=np.float32)
            if has_sharpness
            else None
        )

        min_dist = fwhm * 1.5
        accepted_idx = _greedy_spatial_filter(raw_coords, min_dist, max_stars)

        coords = (
            raw_coords[accepted_idx]
            if accepted_idx
            else np.empty((0, 2), dtype=np.float32)
        )
        fluxes = (
            raw_fluxes[accepted_idx]
            if accepted_idx
            else np.empty((0,), dtype=np.float32)
        )
        star_count = len(coords)

        current_fwhm = (
            float(np.median(raw_sharpness[accepted_idx]) * fwhm)
            if has_sharpness and star_count > 0
            else float(fwhm)
        )
        min_flux = float(np.min(fluxes)) if star_count > 0 else 0.0
        max_flux = float(np.max(fluxes)) if star_count > 0 else 0.0
        snr = float(np.mean(fluxes) / bkg_std) if star_count > 0 else 0.0
    else:
        coords = np.empty((0, 2), dtype=np.float32)
        current_fwhm, star_count, min_flux, max_flux, snr = 0.0, 0, 0.0, 0.0, 0.0

    valid = bool(star_count > 10 and current_fwhm > 0 and current_fwhm < (fwhm * 2.0))
    metrics = {
        "star_count": star_count,
        "fwhm": round(current_fwhm, 2),
        "mean": round(mean_val, 2),
        "median": round(median_val, 2),
        "std": round(std_val, 2),
        "background": round(bkg_median, 2),
        "snr": round(snr, 2),
        "min_flux": round(min_flux, 2),
        "max_flux": round(max_flux, 2),
        "valid": valid,
    }
    return (coords, current_fwhm, metrics)


def detect_stars_opencv(
    data: np.ndarray, fwhm: float, sigma: float, max_stars: int
) -> tuple[np.ndarray, float, dict]:
    mean_val, median_val, std_val = (
        float(np.mean(data)),
        float(np.median(data)),
        float(np.std(data)),
    )
    norm_img = cv2.normalize(data, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    ksize = max(3, int(fwhm) | 1)
    blurred = cv2.GaussianBlur(norm_img, (ksize, ksize), 0)
    threshold_val = min(
        255.0, float(np.median(blurred)) + (sigma * float(np.std(blurred)))
    )

    _, thresh = cv2.threshold(blurred, threshold_val, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    valid_contours = sorted(
        (c for c in contours if 2 < cv2.contourArea(c) < 1000),
        key=cv2.contourArea,
        reverse=True,
    )[:max_stars]

    coords_list, areas, fluxes = [], [], []
    height, width = data.shape[:2]

    for cnt in valid_contours:
        moments = cv2.moments(cnt)
        if moments["m00"] == 0:
            continue
        cX = moments["m10"] / moments["m00"]
        cY = moments["m01"] / moments["m00"]
        px = int(np.clip(round(cX), 0, width - 1))
        py = int(np.clip(round(cY), 0, height - 1))
        coords_list.append([cX, cY])
        areas.append(cv2.contourArea(cnt))
        fluxes.append(float(data[py, px]))

    coords = (
        np.asarray(coords_list, dtype=np.float32)
        if coords_list
        else np.empty((0, 2), dtype=np.float32)
    )
    star_count = len(coords)
    current_fwhm = (
        float(np.mean([np.sqrt(area / np.pi) * 2.0 for area in areas]))
        if areas
        else 0.0
    )
    min_flux = float(np.min(fluxes)) if fluxes else 0.0
    max_flux = float(np.max(fluxes)) if fluxes else 0.0
    snr = float(np.mean(fluxes) / std_val) if std_val > 0 and fluxes else 0.0

    valid = bool(star_count > 10 and current_fwhm > 0 and current_fwhm < (fwhm * 2.5))
    metrics = {
        "star_count": star_count,
        "fwhm": round(current_fwhm, 2),
        "mean": round(mean_val, 2),
        "median": round(median_val, 2),
        "std": round(std_val, 2),
        "background": round(median_val, 2),
        "snr": round(snr, 2),
        "min_flux": round(min_flux, 2),
        "max_flux": round(max_flux, 2),
        "valid": valid,
    }
    return (coords, current_fwhm, metrics)


def detect_stars_opencv_components(
    data: np.ndarray, fwhm: float, sigma: float, max_stars: int
) -> tuple[np.ndarray, float, dict]:
    """Fast native connected-components detector.

    Unlike the compatibility OpenCV detector this does not create Python
    contour/moment objects for every candidate.
    """
    mean_val, median_val, std_val = (
        float(np.mean(data)), float(np.median(data)), float(np.std(data))
    )
    norm_img = cv2.normalize(data, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    ksize = max(3, int(fwhm) | 1)
    blurred = cv2.GaussianBlur(norm_img, (ksize, ksize), 0)
    threshold = min(255.0, float(np.median(blurred)) + sigma * float(np.std(blurred)))
    _, binary = cv2.threshold(blurred, threshold, 255, cv2.THRESH_BINARY)
    _, _, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    areas = stats[1:, cv2.CC_STAT_AREA].astype(np.float32, copy=False)
    centers = centroids[1:].astype(np.float32, copy=False)
    valid = (areas > 2) & (areas < 1000) & np.isfinite(centers).all(axis=1)
    areas, centers = areas[valid], centers[valid]
    if len(areas):
        order = np.argsort(areas)[::-1][:max_stars]
        areas, centers = areas[order], centers[order]
        height, width = data.shape[:2]
        pixels = np.rint(centers).astype(np.intp)
        pixels[:, 0] = np.clip(pixels[:, 0], 0, width - 1)
        pixels[:, 1] = np.clip(pixels[:, 1], 0, height - 1)
        fluxes = np.asarray(data[pixels[:, 1], pixels[:, 0]], dtype=np.float32)
    else:
        centers = np.empty((0, 2), dtype=np.float32)
        fluxes = np.empty((0,), dtype=np.float32)
    star_count = len(centers)
    current_fwhm = float(np.mean(np.sqrt(areas / np.pi) * 2.0)) if len(areas) else 0.0
    snr = float(np.mean(fluxes) / std_val) if len(fluxes) and std_val > 0 else 0.0
    metrics = {
        "star_count": star_count,
        "fwhm": round(current_fwhm, 2),
        "mean": round(mean_val, 2),
        "median": round(median_val, 2),
        "std": round(std_val, 2),
        "background": round(median_val, 2),
        "snr": round(snr, 2),
        "min_flux": round(float(np.min(fluxes)), 2) if len(fluxes) else 0.0,
        "max_flux": round(float(np.max(fluxes)), 2) if len(fluxes) else 0.0,
        "valid": bool(star_count > 10 and 0 < current_fwhm < fwhm * 2.5),
    }
    return centers, current_fwhm, metrics


def detect_stars_sep(
    data: np.ndarray, fwhm: float, sigma: float, max_stars: int
) -> tuple[np.ndarray, float, dict]:
    """Optional SEP detector with a spatially varying background model."""
    import sep

    image = np.ascontiguousarray(data, dtype=np.float32)
    background = sep.Background(image)
    sources = sep.extract(image - background.back(), sigma, err=background.rms())
    if len(sources):
        order = np.argsort(sources["flux"])[::-1][:max_stars]
        sources = sources[order]
        coords = np.column_stack((sources["x"], sources["y"])).astype(np.float32)
        fluxes = np.asarray(sources["flux"], dtype=np.float32)
        fwhm_values = 2.0 * np.sqrt(np.maximum(sources["a"] * sources["b"], 0.0))
        measured_fwhm = float(np.median(fwhm_values))
    else:
        coords, fluxes, measured_fwhm = np.empty((0, 2), np.float32), np.empty((0,), np.float32), 0.0
    std_val = float(np.std(image))
    metrics = {
        "star_count": len(coords), "fwhm": round(measured_fwhm, 2),
        "mean": round(float(np.mean(image)), 2), "median": round(float(np.median(image)), 2),
        "std": round(std_val, 2), "background": round(float(background.globalback), 2),
        "snr": round(float(np.mean(fluxes) / std_val), 2) if len(fluxes) and std_val > 0 else 0.0,
        "min_flux": round(float(np.min(fluxes)), 2) if len(fluxes) else 0.0,
        "max_flux": round(float(np.max(fluxes)), 2) if len(fluxes) else 0.0,
        "valid": bool(len(coords) > 10 and 0 < measured_fwhm < fwhm * 2.5),
    }
    return coords, measured_fwhm, metrics


def _register_flow_engines() -> None:
    profile_both = frozenset({EngineProfile.STABLE, EngineProfile.FAST})
    registry.register(EngineDescriptor("dao", "flow.detector", "DAO", profile_both), detect_stars_dao)
    registry.register(EngineDescriptor("opencv-contours", "flow.detector", "OpenCV contours", profile_both), detect_stars_opencv)
    registry.register(EngineDescriptor("opencv-components", "flow.detector", "OpenCV components", frozenset({EngineProfile.FAST})), detect_stars_opencv_components)
    registry.register(EngineDescriptor("sep", "flow.detector", "SEP / Source Extractor", frozenset({EngineProfile.FAST}), optional_dependency="sep"), detect_stars_sep)
    registry.register(EngineDescriptor("astroalign-asterism", "flow.transform_fallback", "Astroalign asterisms", profile_both, optional_dependency="astroalign"), estimate_asterism_transform)


def _flow_detector_choice(config: dict) -> str:
    choice = config.get("detector_engine") or config.get("engine", "DAO")
    if (
        EngineProfile.coerce(config.get("engine_profile", "Stable")) is EngineProfile.FAST
        and not config.get("detector_engine")
        and str(choice).upper() == "DAO"
    ):
        return "opencv-components"
    return str(choice)


def detect_stars(
    data: np.ndarray,
    fwhm: float,
    sigma: float,
    max_stars: int,
    engine: str = "DAO",
    profile: str = "Stable",
    stats_cache: dict | None = None,
) -> tuple[np.ndarray, float, dict]:
    """V1-compatible detector adapter resolved through the V2 registry."""
    _register_flow_engines()
    normalized = str(engine).strip().lower()
    aliases = {
        "dao": "dao", "opencv": "opencv-contours", "opencv-contours": "opencv-contours",
        "opencv-components": "opencv-components", "sep": "sep",
    }
    engine_id = aliases.get(normalized, "dao")
    selected_profile = EngineProfile.coerce(profile)
    detector = registry.resolve("flow.detector", engine_id, selected_profile)
    if engine_id == "dao" and stats_cache is not None:
        return detect_stars_dao(data, fwhm, sigma, max_stars, stats_cache=stats_cache)
    return detector(data, fwhm, sigma, max_stars)


# ============================================================
# Geometria
# ============================================================


def extract_geometric_properties(
    matrix_2x3: np.ndarray,
) -> tuple[float, float, float, float]:
    a, b, tx = matrix_2x3[0]
    c, d, ty = matrix_2x3[1]
    scale = float(np.sqrt(a**2 + c**2))
    rotation_deg = float(np.degrees(np.arctan2(c, a)))
    return (float(tx), float(ty), rotation_deg, scale)


def validate_transform(
    matrix_2x3: np.ndarray | None, metrics: dict, limits: dict
) -> tuple[bool, str]:
    if matrix_2x3 is None:
        if metrics.get("matches", 0) == 0:
            return (False, "phase_correlation_failed")
        return (False, "insufficient_matches")

    if metrics.get("inliers", 0) < limits["min_inliers"]:
        return (False, "insufficient_inliers")
    if metrics.get("inlier_ratio", 0.0) < limits["min_ratio"]:
        return (False, "low_inlier_ratio")
    if metrics.get("rms", 999.0) > limits["max_rms"]:
        return (False, "high_rms")

    tx, ty, rotation, scale = extract_geometric_properties(matrix_2x3)
    metrics["translation"] = [round(tx, 3), round(ty, 3)]
    metrics["translation_magnitude"] = round(float(np.hypot(tx, ty)), 3)
    metrics["rotation_deg"] = round(rotation, 4)
    metrics["scale"] = round(scale, 6)

    if np.hypot(tx, ty) > limits["max_translation"]:
        return (False, "high_translation")
    if abs(rotation) > limits["max_rotation"]:
        return (False, "high_rotation")
    if not (limits["min_scale"] <= scale <= limits["max_scale"]):
        return (False, "invalid_scale")

    return (True, "accepted")


def make_homogeneous(matrix_2x3: np.ndarray) -> np.ndarray:
    hom = np.eye(3, dtype=np.float64)
    hom[:2, :] = matrix_2x3
    return hom


def _match_incremental_stars(
    previous_stars: np.ndarray,
    current_stars: np.ndarray,
    shift: tuple[float, float],
    matching_radius: float,
) -> tuple[np.ndarray, np.ndarray]:
    if len(previous_stars) == 0 or len(current_stars) == 0:
        return [], []
    dx, dy = shift
    shifted_current = current_stars + np.array([dx, dy], dtype=np.float32)
    tree = KDTree(previous_stars)
    distances, indices = tree.query(
        shifted_current, distance_upper_bound=matching_radius
    )

    # KDTree.query already evaluates every candidate in compiled code. Keep the
    # original greedy rule (nearest current star wins for each previous star),
    # but choose winners with arrays instead of Python tuples and sets.
    current_indices = np.arange(len(current_stars), dtype=np.intp)
    valid = np.isfinite(distances) & (indices < len(previous_stars))
    if not np.any(valid):
        return [], []

    candidate_distances = distances[valid]
    candidate_previous = indices[valid].astype(np.intp, copy=False)
    candidate_current = current_indices[valid]
    # Stable ordering retains the previous tuple-sort tie break: input/current
    # star order wins when distances are equal.
    order = np.argsort(candidate_distances, kind="stable")
    ordered_previous = candidate_previous[order]
    # The same previous star can appear at non-adjacent distances, so obtain
    # each first occurrence globally, then restore distance ordering.
    _, first_occurrences = np.unique(ordered_previous, return_index=True)
    winner_order = order[np.sort(first_occurrences)]

    return (
        np.asarray(previous_stars, dtype=np.float32)[candidate_previous[winner_order]],
        np.asarray(current_stars, dtype=np.float32)[candidate_current[winner_order]],
    )


# ============================================================
# Estimativa de Transformação via USAC_MAGSAC
# ============================================================


def _estimate_incremental_transform(
    previous_stars: list | np.ndarray,
    current_stars: list | np.ndarray,
    ransac_thresh: float,
    min_stars: int,
):
    matches = min(len(previous_stars), len(current_stars))
    if matches < min_stars:
        return (
            None,
            {"matches": matches, "inliers": 0, "inlier_ratio": 0.0, "rms": 999.0},
        )

    previous = np.asarray(previous_stars, dtype=np.float32)
    current = np.asarray(current_stars, dtype=np.float32)

    matrix_2x3, inliers = cv2.estimateAffinePartial2D(
        current,
        previous,
        method=cv2.RANSAC,
        ransacReprojThreshold=ransac_thresh,
        maxIters=7000,
        confidence=0.99995,
        refineIters=25,
    )

    if matrix_2x3 is None or inliers is None:
        return (
            None,
            {"matches": len(previous), "inliers": 0, "inlier_ratio": 0.0, "rms": 999.0},
        )

    mask = inliers.ravel().astype(bool)
    inlier_count = int(mask.sum())
    if inlier_count == 0:
        return (
            None,
            {"matches": len(previous), "inliers": 0, "inlier_ratio": 0.0, "rms": 999.0},
        )

    transformed = cv2.transform(current.reshape(-1, 1, 2), matrix_2x3).reshape(-1, 2)
    errors = transformed[mask] - previous[mask]
    distances = np.linalg.norm(errors, axis=1)
    rms = float(np.sqrt(np.mean(distances**2)))
    inlier_ratio = inlier_count / len(previous)

    return (
        matrix_2x3,
        {
            "matches": len(previous),
            "inliers": inlier_count,
            "inlier_ratio": float(inlier_ratio),
            "rms": rms,
        },
    )


# ============================================================
# Local Flow
# ============================================================


def _process_single_frame(
    filepath: Path,
    fwhm_val: float,
    sigma_val: float,
    max_stars_val: int,
    min_stars: int,
    engine_val: str,
    engine_profile: str = "Stable",
) -> tuple[str, dict | None]:
    try:
        data, header = load_fits_data(filepath)
        timestamp_fields = _timestamp_fields_from_header(header)
        working_data = extract_luminance(data, header)

        current_sigma = sigma_val
        sigma_used = current_sigma
        detector_stats_cache: dict = {}
        best_stars, best_fwhm, best_metrics = [], 0.0, {}
        target_stars = max(20, min_stars * 2)

        while current_sigma >= 3.0:
            sigma_used = current_sigma
            stars, measured_fwhm, metrics = detect_stars(
                working_data, fwhm_val, current_sigma, max_stars_val, engine_val,
                engine_profile, stats_cache=detector_stats_cache,
            )
            best_stars, best_fwhm, best_metrics = stars, measured_fwhm, metrics
            if len(stars) >= target_stars:
                break
            current_sigma -= 0.5

        # Shape quality is derived once from the final detector selection. It
        # supplements (and does not replace) the detector's existing FWHM and
        # validity semantics.
        best_metrics = {**best_metrics, **measure_star_shapes(working_data, best_stars)}

        if len(best_stars) < min_stars:
            return (
                filepath.name,
                {
                    "path": filepath,
                    # The raw science array is not consumed after detection;
                    # dropping it keeps all prepared frames from retaining a
                    # full-resolution copy while phase_data remains available.
                    "data": None,
                    "phase_data": None,
                    "stars": best_stars,
                    "fwhm": best_fwhm,
                    "sigma_used": sigma_used,
                    "metrics": best_metrics,
                    **timestamp_fields,
                    "status": "rejected",
                    "reason": "insufficient_stars_in_detection",
                },
            )

        phase_data = prepare_for_phase_correlation(working_data)
        return (
            filepath.name,
            {
                "path": filepath,
                "data": None,
                "phase_data": phase_data,
                "stars": best_stars,
                "fwhm": best_fwhm,
                "sigma_used": sigma_used,
                "metrics": best_metrics,
                **timestamp_fields,
                "status": "prepared",
            },
        )
    except Exception as exc:
        return (
            filepath.name,
            {
                "status": "error",
                "reason": str(exc),
                "path": filepath,
                "data": None,
                "phase_data": None,
                "stars": np.empty((0, 2), dtype=np.float32),
                "fwhm": 0.0,
                "metrics": {},
            },
        )


def _shape_metrics_for_frame(frame: dict | None) -> dict:
    """Return shape measurements for persistence in flow frame metadata."""
    if not isinstance(frame, dict):
        return {}
    metrics = frame.get("metrics")
    if not isinstance(metrics, dict):
        return {}
    return {
        key: metrics[key]
        for key in ("roundness", "shape_star_count", "shape_fwhm", "elongation")
        if key in metrics
    }


_TIMESTAMP_KEYS = (
    "timestamp",
    "timestamp_utc",
    "timestamp_normalized",
    "timestamp_state",
    "timezone",
    "timezone_present",
    "epoch_s",
)


def _timestamp_fields_from_header(header) -> dict:
    """Normalize DATE-OBS from a header already read for pixel processing."""

    value = None
    try:
        if header is not None:
            value = header.get("DATE-OBS", header.get("DATEOBS"))
    except Exception:
        value = None
    parsed = parse_date_obs(value)
    return {key: parsed.get(key) for key in _TIMESTAMP_KEYS}


def _timestamp_fields_from_frame(frame: dict | None) -> dict:
    """Copy normalized timestamp fields from a prepared frame when present."""

    if not isinstance(frame, dict) or "timestamp_state" not in frame:
        return {}
    return {key: frame.get(key) for key in _TIMESTAMP_KEYS}


def _persisted_frame_metadata(frame: dict | None) -> dict:
    """Return scientific shape metrics plus cached temporal metadata."""

    return {
        **_shape_metrics_for_frame(frame),
        **_timestamp_fields_from_frame(frame),
    }


def _anchor_detection_metadata(
    anchor: dict,
    fwhm: float,
    sigma: float,
    max_stars: int,
    engine: str,
    engine_profile: str,
) -> dict:
    """Describe the exact local detector invocation used for the anchor.

    The Global Flow can reuse the in-memory result only when this metadata
    proves that its adaptive detector would stop on the same first pass.  It is
    deliberately persisted as a small compatibility hint; old flow JSONs do
    not have it and therefore keep the conservative reread path.
    """
    return {
        "fwhm": float(fwhm),
        "sigma": float(sigma),
        "sigma_used": float(anchor.get("sigma_used", sigma)),
        "max_stars": int(max_stars),
        "engine": str(engine),
        "engine_profile": str(engine_profile),
    }


def _classify_flow_confidence(metrics: dict, limits: dict) -> str:
    """Classify an accepted transform using observable registration evidence."""
    response = float(metrics.get("phase_response", 0.0))
    coverage = float(metrics.get("inlier_ratio", 0.0))
    rms = float(metrics.get("rms", 999.0))
    spatial = float(metrics.get("spatial_inlier_coverage", 0.0))
    if coverage >= max(float(limits.get("min_ratio", 0.15)) * 2.0, 0.5) and rms <= float(limits.get("max_rms", 4.0)) * 0.5 and response >= 0.15 and spatial >= 0.01:
        return "accepted"
    return "low_confidence"


def _flow_confidence_reason(metrics: dict, limits: dict) -> str:
    return (f"phase_response={float(metrics.get('phase_response', 0.0)):.4f}; "
            f"inlier_coverage={float(metrics.get('inlier_ratio', 0.0)):.3f}; "
            f"rms={float(metrics.get('rms', 999.0)):.3f}; "
            f"spatial_coverage={float(metrics.get('spatial_inlier_coverage', 0.0)):.4f}; "
            "motion=plausible")


def _spatial_inlier_coverage(ref_stars, current_stars, matrix, shape, residual_threshold=4.0) -> float:
    """Fraction of image area covered by inlier convex hull, as a quality cue."""
    if matrix is None or len(ref_stars) < 3 or len(current_stars) < 3:
        return 0.0
    transformed = cv2.transform(np.asarray(current_stars, np.float32).reshape(-1, 1, 2), matrix).reshape(-1, 2)
    residual = np.linalg.norm(transformed - np.asarray(ref_stars), axis=1)
    inliers = transformed[residual <= residual_threshold]
    if len(inliers) < 3:
        return 0.0
    hull = cv2.contourArea(cv2.convexHull(np.asarray(inliers, np.float32)))
    height, width = shape[:2]
    return float(np.clip(hull / max(float(height * width), 1.0), 0.0, 1.0))


def _persist_local_temporal_outputs(
    batch_dir: Path,
    flow_data: dict,
    config: dict,
    app_print,
) -> dict:
    """Attach timestamps and optionally publish the metadata-only report."""

    timestamp_cache: dict[str, dict] = {}
    flow_data = enrich_flow_frames(batch_dir, flow_data, timestamp_cache)
    enabled = config.get("temporal_analysis_enabled", True)
    if isinstance(enabled, str):
        enabled = enabled.strip().lower() not in {"0", "false", "no", "off"}
    if not enabled:
        return flow_data
    gap = config.get("temporal_gap_minutes", DEFAULT_GAP_MINUTES)
    seeing_sigma = config.get("temporal_seeing_sigma", DEFAULT_SEEING_SIGMA)
    try:
        report = build_temporal_report(
            batch_dir,
            flow_data,
            gap,
            seeing_sigma,
            timestamp_cache=timestamp_cache,
        )
        write_temporal_report(batch_dir / "temporal_analysis.json", report)
        flow_data["temporal_report"] = "temporal_analysis.json"
        flow_data["temporal_gap_minutes"] = report["gap_minutes"]
    except Exception as exc:
        # A report is a convenience sidecar.  A malformed header or a full
        # disk must not discard an otherwise valid Flow result.
        app_print(f"[{batch_dir.name}] Aviso: análise temporal indisponível: {exc}\n")
    return flow_data


def process_local_flow(batch_dir: Path, config: dict, app_print, cancellation_event=None) -> dict:
    """Build local flow metadata with bounded preparation and cooperative cancellation.

    ``cancellation_event`` is intentionally duck typed (``is_set``), keeping
    compatibility with threading.Event and the application's cancellation token.
    """
    def cancelled() -> bool:
        return bool(cancellation_event is not None and cancellation_event.is_set())
    files = sorted(
        [
            p
            for p in batch_dir.iterdir()
            if p.is_file() and p.suffix.lower() in {".fit", ".fits", ".fts"}
        ],
        key=_natural_frame_key,
    )
    if not files:
        return {}
    if not isinstance(config, dict):
        config = {}

    chosen_anchor_name = config.get("custom_anchors", {}).get(batch_dir.name)
    anchor_file = files[0]
    if chosen_anchor_name and any(p.name == chosen_anchor_name for p in files):
        anchor_file = next(p for p in files if p.name == chosen_anchor_name)
        app_print(
            f"[{batch_dir.name}] Frame Central (Batch Reference): {chosen_anchor_name}\n"
        )
    else:
        chosen_anchor_name = anchor_file.name
        app_print(
            f"[{batch_dir.name}] Nenhuma referência manual válida. Usando o 1º frame.\n"
        )

    fwhm_val = float(config.get("fwhm", 4.0))
    sigma_val = float(config.get("sigma", 5.0))
    max_stars_val = int(config.get("max_stars", 250))
    matching_radius = float(config.get("matching_radius", 25.0))
    ransac_thresh = float(config.get("ransac", 4.0))
    engine_val = _flow_detector_choice(config)
    engine_profile = str(config.get("engine_profile", "Stable"))
    fallback_engine = config.get("transform_fallback", "Disabled")

    limits = {
        "min_stars": int(config.get("min_stars", 4)),
        "min_inliers": int(config.get("min_inliers", 4)),
        "min_ratio": float(config.get("min_ratio", 0.15)),
        "max_rms": float(config.get("max_rms", 4.0)),
        "max_translation": float(config.get("max_translation", 1500.0)),
        "max_rotation": float(config.get("max_rotation", 10.0)),
        "min_scale": float(config.get("min_scale", 0.95)),
        "max_scale": float(config.get("max_scale", 1.05)),
    }

    try:
        cpu_count = getattr(os, "process_cpu_count", os.cpu_count)() or 1
    except Exception:
        cpu_count = os.cpu_count() or 1
    requested_workers = max(1, min(8, physical_core_count(), int(config.get("flow_workers", min(2, max(1, cpu_count // 4))))))
    # Header-only estimate: reserve roughly 40 bytes per pixel for detector,
    # luminance, phase and transient matching buffers.
    memory_budget_mb = max(64, int(config.get("memory_budget_mb", 512)))
    try:
        estimated = max(1, max(max_science_frame_bytes(p) for p in files)) * 10
        budget = ExecutionBudget.for_frame_pipeline(requested_workers, memory_budget_mb, estimated)
        worker_count = budget.worker_count
    except OSError:
        raise ValueError("Could not inspect FITS dimensions for the Flow memory budget")

    # Star detection and phase correlation call OpenCV from Python workers.
    # Bound its independent native pool before creating the executor so the
    # configured frame budget is not multiplied by OpenCV's default pool.
    native_threads = int(
        config.get("opencv_threads", 1 if worker_count > 1 else physical_core_count())
    )
    configure_opencv_threads(native_threads)

    registration_strategy = str(config.get("registration_strategy", "neighbor_bfs")).strip().lower()
    if registration_strategy not in {"neighbor_bfs", "legacy", "incremental_chain"}:
        raise ValueError("registration_strategy must be neighbor_bfs or legacy")
    if registration_strategy not in {"legacy", "incremental_chain"}:
        # The graph builder keeps image preparation on demand and bounded. The
        # legacy sequential implementation remains below as an explicit
        # comparison strategy for existing projects.
        graph_result = _build_local_registration_graph(
            files,
            anchor_file,
            config,
            limits,
            matching_radius,
            ransac_thresh,
            engine_val,
            engine_profile,
            fallback_engine,
            worker_count,
            budget,
            cancelled,
            app_print,
            batch_dir.name,
        )
        if graph_result is None or cancelled():
            return {}
        flow_data, anchor, anchor_quality, valid_count = graph_result
        flow_data["registration_strategy"] = "neighbor_bfs"
        flow_data = _persist_local_temporal_outputs(batch_dir, flow_data, config, app_print)
        output_path = batch_dir / "flow_local.json"
        from app.infrastructure.json_store import atomic_json_write
        atomic_json_write(output_path, flow_data)
        total_count = len(files)
        coverage = valid_count / max(total_count, 1)
        app_print(
            f"[{batch_dir.name}] Flow salvo: {valid_count}/{total_count} ({coverage:.1%})\n"
        )
        return {
            "batch_name": batch_dir.name,
            "anchor_path": anchor_file,
            "anchor_data": None,
            "anchor_stars": anchor["stars"],
            "star_count": len(anchor["stars"]),
            "fwhm": anchor["fwhm"],
            "anchor_quality": anchor_quality,
            "anchor_metrics": anchor.get("metrics", {}),
            "anchor_detection": flow_data.get("anchor_detection"),
            "anchor_shape": flow_data.get("anchor_shape"),
            "valid_frames": valid_count,
            "total_frames": total_count,
            "coverage": coverage,
            "local_transform_revision": flow_data.get("transform_revision"),
            "temporal_report": str(batch_dir / "temporal_analysis.json")
            if (batch_dir / "temporal_analysis.json").exists()
            else None,
        }

    # Prepare the anchor synchronously, then stream all other frames through a
    # bounded FIFO. No collection of phase images is built.
    _, anchor = _process_single_frame(anchor_file, fwhm_val, sigma_val, max_stars_val,
                                      limits["min_stars"], engine_val, engine_profile)

    def bounded_prepare():
        others = (p for p in files if p.name != anchor_file.name)
        with ThreadPoolExecutor(max_workers=worker_count,
                                thread_name_prefix="astroflow",
                                initializer=partial(configure_worker_runtime, 1)) as executor:
            pending = deque()
            for _ in range(budget.max_in_flight):
                try: p = next(others)
                except StopIteration: break
                pending.append((executor.submit(_process_single_frame, p, fwhm_val, sigma_val,
                    max_stars_val, limits["min_stars"], engine_val, engine_profile), p))
            try:
                while pending:
                    if cancelled(): return
                    future, path = pending.popleft()
                    try: yield path, future.result()[1]
                    except Exception as exc:
                        app_print(f"[{path.name}] Erro no worker: {exc}\n")
                    try: p = next(others)
                    except StopIteration: continue
                    pending.append((executor.submit(_process_single_frame, p, fwhm_val, sigma_val,
                        max_stars_val, limits["min_stars"], engine_val, engine_profile), p))
            finally:
                for future, _ in pending: future.cancel()
    if (
        anchor is None
        or anchor.get("status") == "error"
        or len(anchor["stars"]) < limits["min_stars"]
    ):
        app_print(f"[{batch_dir.name}] ERRO: Falha na âncora.\n")
        return {}

    anchor_quality = calculate_anchor_quality(len(anchor["stars"]), anchor["fwhm"])

    flow_data = {
        "schema_version": 2,
        "batch_anchor": anchor_file.name,
        "selected_reference": chosen_anchor_name,
        "mode": "incremental_chain",
        "registration_strategy": "legacy",
        "workers": worker_count,
        "engine": engine_val,
        "engine_profile": str(config.get("engine_profile", "Stable")),
        "transform_fallback": fallback_engine,
        "parameters": {
            "fwhm": fwhm_val,
            "sigma": sigma_val,
            "max_stars": max_stars_val,
            "matching_radius": matching_radius,
            "ransac": ransac_thresh,
            **limits,
        },
        "anchor_shape": list(anchor["phase_data"].shape) if anchor.get("phase_data") is not None else None,
        "anchor_stars": _flow_json_value(anchor["stars"]),
        "anchor_detection": _anchor_detection_metadata(
            anchor, fwhm_val, sigma_val, max_stars_val, engine_val, engine_profile,
        ),
        "anchor_metrics": {
            "star_count": len(anchor["stars"]),
            "fwhm": anchor["fwhm"],
            "quality": anchor_quality,
            **anchor["metrics"],
        },
        "frames": {},
    }

    flow_data["frames"][anchor_file.name] = {
        "status": "accepted",
        "confidence": "reference",
        "confidence_reason": "Coordinate reference; not an independently verified registration",
        "recovery_method": "reference",
        "matrix": np.eye(3).tolist(),
        "relative_to": None,
        "matches": len(anchor["stars"]),
        "inliers": len(anchor["stars"]),
        "inlier_ratio": 1.0,
        "rms": 0.0,
        "translation": [0.0, 0.0],
        "translation_magnitude": 0.0,
        "rotation_deg": 0.0,
        "scale": 1.0,
        "cumulative_rms": 0.0,
        "star_count": len(anchor["stars"]),
        "fwhm": anchor["fwhm"],
        **_persisted_frame_metadata(anchor),
    }

    previous_name = anchor_file.name
    previous_frame = anchor
    temporal_history: list[tuple[float, float, float]] = []

    # Start at the first file so a manual anchor in the middle cannot silently
    # drop the leading frames. The anchor remains an identity observation.
    for current_file, current_frame in bounded_prepare():
        if cancelled(): return {}
        current_name = current_file.name

        if (
            current_frame is None
            or current_frame.get("status") == "error"
            or len(current_frame["stars"]) < limits["min_stars"]
        ):
            flow_data["frames"][current_name] = {
                "status": "rejected",
                "confidence": "rejected",
                "confidence_reason": "insufficient_stars_or_error",
                "reason": "insufficient_stars_or_error",
                **_persisted_frame_metadata(current_frame),
            }
            continue

        phase_shift = (0.0, 0.0)
        phase_response = 0.0
        if previous_frame.get("phase_data") is not None and current_frame.get("phase_data") is not None:
            phase_shift, phase_response = cv2.phaseCorrelate(previous_frame["phase_data"], current_frame["phase_data"])
        attempts = [
            ("normal", previous_frame, matching_radius, previous_name),
            ("relaxed_radius", previous_frame, matching_radius * 2.0, previous_name),
        ]
        if previous_name != anchor_file.name:
            attempts.append(("anchor", anchor, matching_radius * 2.0, anchor_file.name))
        accepted = False
        best_metrics = {
            "reason": "phase_correlation_failed",
            "matches": 0,
            "inliers": 0,
            "inlier_ratio": 0.0,
            "rms": 999.0,
        }

        for attempt_name, ref_frame, radius, ref_name in attempts:
            if (
                ref_frame.get("phase_data") is None
                or current_frame.get("phase_data") is None
            ):
                continue

            if attempt_name == "anchor":
                anchor_shift, anchor_response = cv2.phaseCorrelate(anchor["phase_data"], current_frame["phase_data"])
                dx, dy, response = anchor_shift[0], anchor_shift[1], anchor_response
            else:
                dx, dy, response = phase_shift[0], phase_shift[1], phase_response
            m_ref, m_cur = _match_incremental_stars(
                ref_frame["stars"], current_frame["stars"], (dx, dy), radius
            )
            rel_matrix, metrics = _estimate_incremental_transform(
                m_ref, m_cur, ransac_thresh, limits["min_stars"]
            )
            metrics["spatial_inlier_coverage"] = _spatial_inlier_coverage(
                m_ref, m_cur, rel_matrix,
                current_frame["phase_data"].shape if current_frame.get("phase_data") is not None else (1, 1),
                residual_threshold=ransac_thresh,
            )

            metrics["phase_shift"] = [round(float(dx), 3), round(float(dy), 3)]
            metrics["phase_response"] = round(float(response), 5)
            valid, reason = validate_transform(rel_matrix, metrics, limits)

            # Validação Temporal por Inércia
            if valid and rel_matrix is not None and attempt_name != "anchor":
                tx, ty, rot, sc = extract_geometric_properties(rel_matrix)
                if len(temporal_history) >= 4:
                    recent = temporal_history[-8:]
                    med_dx = np.median([h[0] for h in recent])
                    med_dy = np.median([h[1] for h in recent])
                    med_rot = np.median([h[2] for h in recent])
                    if (
                        abs(tx - med_dx) > 120.0
                        or abs(ty - med_dy) > 120.0
                        or abs(rot - med_rot) > 2.0
                    ):
                        valid = False
                        reason = "temporal_validation_failed"

            if valid:
                accepted = True
                relative_homogeneous = make_homogeneous(rel_matrix)
                ref_matrix = np.asarray(
                    flow_data["frames"][ref_name]["matrix"], dtype=np.float64
                )
                cumulative_matrix = ref_matrix @ relative_homogeneous

                prev_cum_rms = float(
                    flow_data["frames"][ref_name].get("cumulative_rms", 0.0)
                )
                curr_rms = float(metrics.get("rms", 0.0))
                cumulative_rms = float(np.sqrt(prev_cum_rms**2 + curr_rms**2))

                flow_data["frames"][current_name] = {
                    "status": "accepted",
                    "confidence": _classify_flow_confidence(metrics, limits),
                    "confidence_reason": _flow_confidence_reason(metrics, limits),
                    "matrix": cumulative_matrix.tolist(),
                    "relative_matrix": relative_homogeneous.tolist(),
                    "relative_to": ref_name,
                    "recovery_method": attempt_name,
                    "cumulative_rms": cumulative_rms,
                    "fwhm": current_frame["fwhm"],
                    "star_count": len(current_frame["stars"]),
                    **_persisted_frame_metadata(current_frame),
                    **metrics,
                }

                tx, ty, rot, _ = extract_geometric_properties(rel_matrix)
                if attempt_name != "anchor":
                    temporal_history.append((tx, ty, rot))
                if len(temporal_history) > 8:
                    del temporal_history[:-8]

                app_print(
                    f"[{current_name}] OK ({attempt_name}) <- {ref_name} | "
                    f"{metrics['inliers']}/{metrics['matches']} inliers | "
                    f"ratio={metrics['inlier_ratio']:.1%} | RMS={metrics['rms']:.3f}px\n"
                )
                break

            metrics["reason"] = reason
            if metrics.get("inliers", 0) > best_metrics.get("inliers", 0) or (
                metrics.get("inliers", 0) == best_metrics.get("inliers", 0)
                and metrics.get("rms", 999.0) < best_metrics.get("rms", 999.0)
            ):
                best_metrics = metrics

        if not accepted and str(fallback_engine).lower() in {"astroalign", "astroalign-asterism"}:
            try:
                _register_flow_engines()
                fallback = registry.resolve("flow.transform_fallback", "astroalign-asterism")
                relative_homogeneous, metrics = fallback(current_frame["stars"], previous_frame["stars"])
                rel_matrix = np.asarray(relative_homogeneous[:2, :], dtype=np.float64)
                valid, reason = validate_transform(rel_matrix, metrics, limits)
                if valid:
                    accepted = True
                    ref_matrix = np.asarray(flow_data["frames"][previous_name]["matrix"], dtype=np.float64)
                    cumulative_matrix = ref_matrix @ relative_homogeneous
                    previous_rms = float(flow_data["frames"][previous_name].get("cumulative_rms", 0.0))
                    metrics["cumulative_rms"] = float(np.sqrt(previous_rms**2 + float(metrics["rms"]) ** 2))
                    flow_data["frames"][current_name] = {
                        "status": "accepted", "matrix": cumulative_matrix.tolist(),
                        "confidence": "low_confidence",
                        "confidence_reason": "Asterism transform passed geometry checks; spatial/phase confidence not calibrated",
                        "recovery_method": "astroalign_asterism",
                        "relative_matrix": relative_homogeneous.tolist(), "relative_to": previous_name,
                        "fwhm": current_frame["fwhm"], "star_count": len(current_frame["stars"]), **metrics,
                        **_persisted_frame_metadata(current_frame),
                    }
                    app_print(f"[{current_name}] OK (astroalign_asterism) <- {previous_name} | "
                              f"{metrics['inliers']}/{metrics['matches']} asterism matches | RMS={metrics['rms']:.3f}px\n")
                else:
                    best_metrics = {**metrics, "reason": reason}
            except Exception as exc:
                # Optional engines must never stop a normal Flow execution.
                app_print(f"[{current_name}] Astroalign fallback unavailable: {exc}\n")

        if accepted:
            # Release the predecessor's full phase image once it is no longer
            # needed; this bounds phase memory to the anchor and current frame.
            if previous_name != anchor_file.name:
                previous_frame["phase_data"] = None
            previous_name = current_name
            previous_frame = current_frame
        else:
            flow_data["frames"][current_name] = {
                "status": "rejected",
                "confidence": "rejected",
                "confidence_reason": best_metrics.get("reason", "unknown"),
                "reason": best_metrics.get("reason", "unknown"),
                "matches": best_metrics.get("matches", 0),
                "inliers": best_metrics.get("inliers", 0),
                "inlier_ratio": best_metrics.get("inlier_ratio", 0.0),
                "rms": best_metrics.get("rms", 999.0),
                "phase_shift": best_metrics.get("phase_shift", [0.0, 0.0]),
                "phase_response": best_metrics.get("phase_response", 0.0),
                "fwhm": current_frame["fwhm"],
                "star_count": len(current_frame["stars"]),
                **_persisted_frame_metadata(current_frame),
            }
            app_print(
                f"[{current_name}] REJEITADO: {best_metrics.get('reason', 'unknown')} | "
                f"inliers={best_metrics.get('inliers', 0)} | RMS={best_metrics.get('rms', 999.0):.3f}px\n"
            )

    if cancelled():
        return {}
    # Recentralização se a âncora manual não for o 1º frame
    selected_reference_info = flow_data["frames"].get(chosen_anchor_name)
    if selected_reference_info and selected_reference_info.get("status") == "accepted":
        reference_matrix = np.asarray(
            selected_reference_info["matrix"], dtype=np.float64
        )
        try:
            inverse_reference = np.linalg.inv(reference_matrix)
            for frame_data in flow_data["frames"].values():
                if frame_data.get("status") == "accepted":
                    old_matrix = np.asarray(frame_data["matrix"], dtype=np.float64)
                    frame_data["matrix"] = (inverse_reference @ old_matrix).tolist()
            flow_data["batch_anchor"] = chosen_anchor_name
        except np.linalg.LinAlgError:
            pass

    accepted_frames = [
        f for f in flow_data["frames"].values() if f.get("status") == "accepted"
    ]
    rejected_frames = [
        f for f in flow_data["frames"].values() if f.get("status") == "rejected"
    ]
    valid_count = len(accepted_frames)
    total_count = len(files)
    coverage = valid_count / total_count if total_count else 0.0

    flow_data["statistics"] = {
        "total_frames": total_count,
        "accepted_frames": valid_count,
        "rejected_frames": len(rejected_frames),
        "coverage": coverage,
        "first_frame": files[0].name,
        "last_frame": files[-1].name,
        "first_frame_valid": (
            flow_data["frames"].get(files[0].name, {}).get("status") == "accepted"
        ),
        "last_frame_valid": (
            flow_data["frames"].get(files[-1].name, {}).get("status") == "accepted"
        ),
        "chain_segments": _count_chain_segments(flow_data),
    }
    flow_data["geometry_revision"] = _geometry_revision(flow_data)

    flow_data = _persist_local_temporal_outputs(batch_dir, flow_data, config, app_print)

    output_path = batch_dir / "flow_local.json"
    from app.infrastructure.json_store import atomic_json_write
    atomic_json_write(output_path, flow_data)

    app_print(
        f"[{batch_dir.name}] Flow salvo: {valid_count}/{total_count} ({coverage:.1%})\n"
    )

    return {
        "batch_name": batch_dir.name,
        "anchor_path": anchor_file,
        "anchor_data": None,
        "anchor_stars": anchor["stars"],
        "star_count": len(anchor["stars"]),
        "fwhm": anchor["fwhm"],
        "anchor_quality": anchor_quality,
        "anchor_metrics": anchor["metrics"],
        "anchor_detection": flow_data.get("anchor_detection"),
        "anchor_shape": flow_data.get("anchor_shape"),
        "valid_frames": valid_count,
        "total_frames": total_count,
        "coverage": coverage,
        "temporal_report": str(batch_dir / "temporal_analysis.json")
        if (batch_dir / "temporal_analysis.json").exists()
        else None,
    }


def _count_chain_segments(flow_data: dict) -> int:
    segments, previous_accepted = 0, False
    for frame_data in flow_data.get("frames", {}).values():
        accepted = frame_data.get("status") == "accepted"
        if accepted and not previous_accepted:
            segments += 1
        previous_accepted = accepted
    return segments


# ============================================================
# Geometric Quad Asterism Hashing
# ============================================================


def _build_quad_hash(
    points_4: np.ndarray,
) -> tuple[tuple[float, float, float, float], list[int]] | None:
    """
    Normaliza 4 pontos em um sistema de coordenadas invariante a escala e rotação:
    A e B tornam-se (0,0) e (1,1). C e D geram o hash 4D.
    """
    # This fixed six-pair calculation is intentionally scalar: allocating
    # temporary NumPy arrays here is slower than this tiny hot loop.
    best_dist = -1.0
    best_pair = (0, 1)
    for i in range(4):
        for j in range(i + 1, 4):
            d = math.hypot(
                points_4[i, 0] - points_4[j, 0], points_4[i, 1] - points_4[j, 1]
            )
            if d > best_dist:
                best_dist = d
                best_pair = (i, j)

    if best_dist < 1e-4:
        return None

    i_a, i_b = best_pair
    pt_a, pt_b = points_4[i_a], points_4[i_b]

    # Ordena para garantir orientação consistente
    if pt_a[0] > pt_b[0] or (pt_a[0] == pt_b[0] and pt_a[1] > pt_b[1]):
        i_a, i_b = i_b, i_a
        pt_a, pt_b = points_4[i_a], points_4[i_b]

    # Vetores de transformação para base normalizada
    dx = float(pt_b[0] - pt_a[0])
    dy = float(pt_b[1] - pt_a[1])
    scale_sq = dx * dx + dy * dy

    # Os outros dois pontos restantes
    others = [idx for idx in range(4) if idx != i_a and idx != i_b]
    pt_c = points_4[others[0]]
    pt_d = points_4[others[1]]

    # Projeção Afim 2D
    def normalize_pt(pt):
        px = float(pt[0] - pt_a[0])
        py = float(pt[1] - pt_a[1])
        nx = (px * dx + py * dy) / scale_sq
        ny = (py * dx - px * dy) / scale_sq
        return nx, ny

    cx, cy = normalize_pt(pt_c)
    dx_, dy_ = normalize_pt(pt_d)

    # Ordenação canônica entre C e D
    if cx > dx_ or (cx == dx_ and cy > dy_):
        cx, dx_ = dx_, cx
        cy, dy_ = dy_, cy
        others[0], others[1] = others[1], others[0]

    hash_key = (float(cx), float(cy), float(dx_), float(dy_))
    ordering = [i_a, i_b, others[0], others[1]]
    return hash_key, ordering


def _extract_asterism_database(
    stars: np.ndarray, max_stars: int = 50
) -> tuple[list[tuple], list[list[int]]]:
    """Extrai quads locais usando K-Nearest Neighbors para manter complexidade O(N)."""
    if len(stars) < 4:
        return [], []

    subset = stars[:max_stars]
    tree = KDTree(subset)
    hashes = []
    star_quads = []
    seen_quads = set()

    k_neighbors = min(8, len(subset))
    for i in range(len(subset)):
        _, neighbors = tree.query(subset[i], k=k_neighbors)
        # Gera combinações de 4 estrelas dentro do grupo de vizinhos
        for combo in itertools.combinations(neighbors, 4):
            quad_key = tuple(sorted(combo))
            if quad_key in seen_quads:
                continue
            seen_quads.add(quad_key)

            quad_pts = subset[list(combo)]
            result = _build_quad_hash(quad_pts)
            if result is not None:
                h_key, ordering = result
                hashes.append(h_key)
                star_quads.append([combo[idx] for idx in ordering])

    return hashes, star_quads


def _match_quad_asterisms(
    ref_stars: np.ndarray, tgt_stars: np.ndarray, tolerance: float = 0.02
) -> tuple[np.ndarray, np.ndarray]:
    """Pareia estrelas comparando as distâncias euclidianas no espaço de hash 4D."""
    ref_hashes, ref_quads = _extract_asterism_database(ref_stars, max_stars=60)
    tgt_hashes, tgt_quads = _extract_asterism_database(tgt_stars, max_stars=60)

    if not ref_hashes or not tgt_hashes:
        return np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32)

    ref_tree = KDTree(ref_hashes)
    match_pairs = {}

    for t_idx, t_hash in enumerate(tgt_hashes):
        dist, r_idx = ref_tree.query(t_hash)
        if dist <= tolerance:
            # Associa os 4 vértices correspondentes
            r_star_indices = ref_quads[r_idx]
            t_star_indices = tgt_quads[t_idx]

            for r_s, t_s in zip(r_star_indices, t_star_indices):
                match_pairs[r_s] = match_pairs.get(r_s, set())
                match_pairs[r_s].add(t_s)

    pts_ref = []
    pts_tgt = []
    for r_s, t_set in match_pairs.items():
        # Filtra pareamentos consistentes (1 para 1)
        if len(t_set) == 1:
            t_s = list(t_set)[0]
            pts_ref.append(ref_stars[r_s])
            pts_tgt.append(tgt_stars[t_s])

    if not pts_ref:
        return np.empty((0, 2), dtype=np.float32), np.empty((0, 2), dtype=np.float32)

    return np.asarray(pts_ref, dtype=np.float32), np.asarray(pts_tgt, dtype=np.float32)


# ============================================================
# Global Flow
# ============================================================


def _global_limits(config: dict) -> dict:
    user_inliers = int(config.get("global_min_inliers", config.get("min_inliers", 4)))
    return {
        "min_inliers": max(3, user_inliers),
        "min_ratio": 0.1,
        "max_rms": float(config.get("global_max_rms", 4.0)),
        "max_translation": float(config.get("global_max_translation", 3000.0)),
        "max_rotation": float(config.get("global_max_rotation", 20.0)),
        "min_scale": max(0.96, float(config.get("global_min_scale", 0.95))),
        "max_scale": min(1.04, float(config.get("global_max_scale", 1.05))),
    }


def _estimate_global_pair(
    ref_info: dict,
    target_info: dict,
    matching_radius: float,
    ransac_thresh: float,
    limits: dict,
):
    stars_ref = ref_info.get("anchor_stars")
    stars_tgt = target_info.get("anchor_stars")

    if (
        stars_ref is None
        or stars_tgt is None
        or len(stars_ref) < 4
        or len(stars_tgt) < 4
    ):
        return (
            None,
            {
                "status": "rejected",
                "reason": "missing_data",
                "matches": 0,
                "inliers": 0,
                "inlier_ratio": 0.0,
                "rms": 999.0,
            },
        )

    # 1. Pareamento por Hashing de Quads Geométricos (Invariante a Rotação e Translação)
    pts_ref, pts_tgt = _match_quad_asterisms(stars_ref, stars_tgt, tolerance=0.02)

    # 2. Resolução Robusta com MAGSAC++
    matrix_2x3, metrics = _estimate_incremental_transform(
        pts_ref, pts_tgt, ransac_thresh, limits["min_inliers"]
    )

    metrics["phase_shift"] = [0.0, 0.0]
    metrics["phase_response"] = 1.0

    # 3. Validação Limítrofe dos Parâmetros Geométricos
    valid, reason = validate_transform(matrix_2x3, metrics, limits)
    if not valid:
        metrics["status"] = "rejected"
        metrics["reason"] = reason
        return (None, metrics)

    metrics["status"] = "accepted"
    return (make_homogeneous(matrix_2x3), metrics)


def _matrix_difference_score(first: np.ndarray, second: np.ndarray) -> float:
    return float(np.sqrt(np.mean((first - second) ** 2)))


def _detect_anchor_stars_task(
    info: dict, fwhm_val: float, base_sigma: float, engine_val: str, engine_profile: str = "Stable"
):
    """Worker independente por batch: usado para paralelizar a fase de detecção de
    estrelas-âncora do Global Flow (anteriormente sequencial)."""
    cached = _cached_anchor_for_global(info, fwhm_val, base_sigma, engine_val, engine_profile)
    if cached is not None:
        stars, shape, measured_fwhm = cached
        info["_anchor_detection_reused"] = True
        return info, shape, stars, measured_fwhm, base_sigma, None

    data, header = load_fits_data(info["anchor_path"])
    working_data = extract_luminance(data, header)

    current_sigma = base_sigma
    detector_stats_cache: dict = {}
    best_stars, best_fwhm = [], fwhm_val

    while current_sigma >= 2.8:
        g_stars, g_fwhm, _ = detect_stars(
            working_data, fwhm_val, current_sigma, 250, engine_val, engine_profile,
            stats_cache=detector_stats_cache,
        )
        best_stars, best_fwhm = g_stars, g_fwhm
        if len(g_stars) >= 35:
            break
        current_sigma -= 0.2

    phase_data = prepare_for_phase_correlation(working_data)
    return info, working_data.shape, best_stars, best_fwhm, current_sigma, phase_data


def _cached_anchor_for_global(
    info: dict,
    fwhm_val: float,
    base_sigma: float,
    engine_val: str,
    engine_profile: str,
) -> tuple[np.ndarray, tuple[int, ...], float] | None:
    """Return a local anchor catalogue only when Global would be identical.

    Global detection has an adaptive sigma loop and a fixed 250-star cap.  A
    local catalogue is therefore safe to reuse only when it stopped on the
    first pass with the same detector inputs and at least the Global stop
    threshold (35 stars).  Missing metadata intentionally falls back to the
    legacy FITS read, keeping old JSONs scientifically conservative.
    """
    detection = info.get("anchor_detection")
    if not isinstance(detection, dict):
        return None
    try:
        if float(detection.get("fwhm")) != float(fwhm_val):
            return None
        if float(detection.get("sigma")) != float(base_sigma):
            return None
        if float(detection.get("sigma_used")) != float(base_sigma):
            return None
        if int(detection.get("max_stars")) != 250:
            return None
        if str(detection.get("engine")) != str(engine_val):
            return None
        if str(detection.get("engine_profile")) != str(engine_profile):
            return None
        stars = np.asarray(info.get("anchor_stars"), dtype=np.float32)
        if stars.ndim != 2 or stars.shape[1] != 2 or len(stars) < 35:
            return None
        if not np.isfinite(stars).all():
            return None
        shape_value = info.get("anchor_shape", info.get("shape"))
        if shape_value is None:
            shape = (0, 0)
        else:
            shape = tuple(int(value) for value in shape_value)
        measured_fwhm = float(info.get("fwhm", detection.get("fwhm", fwhm_val)))
    except (TypeError, ValueError, OverflowError):
        return None
    return np.ascontiguousarray(stars, dtype=np.float32), shape, measured_fwhm


def _local_flow_batch_plan(
    batch_folders: list[Path], config: dict, requested_workers: int
) -> tuple[int, int, int]:
    """Split the Flow CPU/memory budget between independent batches.

    A session with several batches can opt into the Siril-style policy of
    distributing a global budget across independent images first.  It is
    intentionally opt-in because the best split depends on camera size and
    storage latency; the default keeps the established per-batch executor.
    The returned tuple is ``(batch_workers, frame_workers,
    memory_mb_per_batch)``.
    """
    requested = max(1, min(8, physical_core_count(), int(requested_workers)))
    batch_count = len(batch_folders)
    if batch_count <= 1 or requested <= 1:
        return 1, requested, max(64, int(config.get("memory_budget_mb", 512)))

    try:
        requested_outer = int(config.get("flow_batch_workers", 1) or 1)
    except (TypeError, ValueError):
        requested_outer = 1
    if requested_outer <= 1:
        return 1, requested, max(64, int(config.get("memory_budget_mb", 512)))

    memory_mb = max(64, int(config.get("memory_budget_mb", 512)))
    # Match the per-frame reservation used by process_local_flow.  If one
    # frame already consumes the whole budget, do not add an outer pool.
    estimated_per_batch = 64 * 1024 * 1024
    for batch in batch_folders:
        try:
            frame_estimates = [
                max_science_frame_bytes(path) * 10
                for path in batch.iterdir()
                if path.is_file() and path.suffix.lower() in {".fit", ".fits", ".fts"}
            ]
            if frame_estimates:
                estimated_per_batch = max(estimated_per_batch, max(frame_estimates))
        except (OSError, ValueError):
            continue
    memory_limited_batches = max(1, (memory_mb * 1024 * 1024) // estimated_per_batch)
    batch_workers = min(batch_count, requested, memory_limited_batches, requested_outer)
    batch_workers = max(1, batch_workers)
    frame_workers = max(1, requested // batch_workers)
    per_batch_memory = max(64, memory_mb // batch_workers)
    return batch_workers, frame_workers, per_batch_memory


def process_all_flows(
    base_dir: Path, config: dict, app_print, app_progress, cancel_event
):
    if not isinstance(config, dict):
        config = {}

    batch_folders = sorted(
        [d for d in base_dir.iterdir() if d.is_dir() and "batch" in d.name.lower()],
        key=_natural_frame_key,
    )
    total_batches = len(batch_folders)

    if not batch_folders:
        app_print(f"Nenhuma subpasta de Batch encontrada em {base_dir}\n")
        return

    app_progress(0, total_batches, "Iniciando AstroFlow...")
    anchors_info = []
    skip_local = bool(config.get("skip_local_flow", False))

    if skip_local:
        app_print("\n[GLOBAL] Recarregando Flows Locais existentes...\n")
        from astroalign_logic import load_local_flow
        for batch_folder in batch_folders:
            local_data = load_local_flow(batch_folder)
            if not isinstance(local_data, dict):
                app_print(
                    f"[{batch_folder.name}] AVISO: flow_local.json ausente. Ignorando.\n"
                )
                continue

            try:
                anchor_name = local_data.get("batch_anchor")
                if not anchor_name:
                    continue
                if not _persisted_local_flow_is_current(batch_folder, local_data):
                    app_print(
                        f"[{batch_folder.name}] AVISO: Flow Local antigo ou desatualizado; "
                        "reexecute o Flow Local antes do Global.\n"
                    )
                    continue

                anchor_metrics = local_data.get("anchor_metrics", {})
                info = {
                    "batch_name": batch_folder.name,
                    "anchor_path": batch_folder / anchor_name,
                    "anchor_data": None,
                    "anchor_stars": local_data.get("anchor_stars", []),
                    "shape": local_data.get("anchor_shape"),
                    "anchor_shape": local_data.get("anchor_shape"),
                    "anchor_detection": local_data.get("anchor_detection"),
                    "star_count": anchor_metrics.get("star_count", 0),
                    "fwhm": anchor_metrics.get("fwhm", 0.0),
                    "anchor_quality": anchor_metrics.get("quality", 0.0),
                    "anchor_metrics": anchor_metrics,
                    "valid_frames": local_data.get("statistics", {}).get(
                        "accepted_frames", 0
                    ),
                    "total_frames": local_data.get("statistics", {}).get(
                        "total_frames", 0
                    ),
                    "coverage": local_data.get("statistics", {}).get("coverage", 0.0),
                    "local_transform_revision": local_data.get("transform_revision"),
                }
                anchors_info.append(info)
            except Exception as exc:
                app_print(f"[{batch_folder.name}] Erro ao ler json: {exc}\n")

    else:
        try:
            cpu_count = getattr(os, "process_cpu_count", os.cpu_count)() or 1
        except Exception:
            cpu_count = os.cpu_count() or 1

        requested_local_workers = max(
            1,
            min(
                8,
                physical_core_count(),
                int(config.get("flow_workers", min(2, cpu_count))),
            ),
        )
        batch_workers, local_workers, batch_memory_mb = _local_flow_batch_plan(
            batch_folders, config, requested_local_workers
        )
        if batch_workers > 1:
            app_print(
                f"[FLOW] {batch_workers} batches em paralelo, "
                f"{local_workers} worker(s) por batch, "
                f"{batch_memory_mb} MiB/batch.\n"
            )

        def run_local_batch(batch_folder: Path):
            messages: list[str] = []
            batch_config = {
                **config,
                "flow_workers": local_workers,
                "memory_budget_mb": batch_memory_mb,
            }
            if batch_workers > 1:
                # A concurrent batch is itself the outer worker.  Do not let
                # the single-frame path restore OpenCV's full native pool.
                batch_config["opencv_threads"] = 1
            try:
                info = process_local_flow(
                    batch_folder,
                    batch_config,
                    messages.append,
                    cancellation_event=cancel_event,
                )
                return batch_folder, info, messages, None
            except Exception as exc:
                return batch_folder, None, messages, exc

        if batch_workers == 1:
            batch_results = [run_local_batch(batch_folder) for batch_folder in batch_folders]
        else:
            with ThreadPoolExecutor(
                max_workers=batch_workers,
                thread_name_prefix="astroflow-batch",
                initializer=partial(configure_worker_runtime, 1),
            ) as executor:
                futures = [executor.submit(run_local_batch, batch_folder) for batch_folder in batch_folders]
                # Consume in natural batch order so logs and publication stay
                # deterministic even though the work overlaps.
                batch_results = [future.result() for future in futures]

        for batch_folder, info, messages, error in batch_results:
            for message in messages:
                app_print(message)
            if error is not None:
                app_print(f"Erro em {batch_folder.name}: {error}\n")
                continue
            if info:
                anchors_info.append(info)
            app_print(f"Flow Local Finalizado: {batch_folder.name}\n")

    if not anchors_info:
        app_print("Nenhum Flow Local válido foi produzido ou encontrado.\n")
        return {"status": "failed", "reason": "no_current_local_flow"}

    anchors_info.sort(key=lambda x: _natural_frame_key(x["batch_name"]))

    base_sigma = float(config.get("sigma", 5.0))
    engine_val = _flow_detector_choice(config)
    engine_profile = str(config.get("engine_profile", "Stable"))
    fwhm_val = float(config.get("fwhm", 4.0))
    anchor_cache_hits = 0

    app_print("\n[GLOBAL] Gerando imagens sintéticas para Pareamento Global...\n")

    try:
        cpu_count = getattr(os, "process_cpu_count", os.cpu_count)() or 1
    except Exception:
        cpu_count = os.cpu_count() or 1
    global_workers = max(1, min(8, physical_core_count(), cpu_count))

    # Global matching is independent across batches.  It uses OpenCV from
    # each worker, so only the single-worker case may use a native pool.
    configure_opencv_threads(1 if global_workers > 1 else physical_core_count())

    # Anchor detection is independent across batches. Keep the executor
    # bounded and consume futures in natural batch order so the resulting
    # metadata/log publication remains deterministic. The submission window
    # is bounded so at most ``anchor_workers`` small phase images are retained.
    anchor_workers = max(1, min(global_workers, len(anchors_info)))
    with ThreadPoolExecutor(
        max_workers=anchor_workers,
        thread_name_prefix="astroflow-anchor",
        initializer=partial(configure_worker_runtime, 1),
    ) as anchor_executor:
        pending_anchors = []
        next_anchor = 0
        for _ in range(anchor_workers):
            if next_anchor >= len(anchors_info):
                break
            pending_anchors.append(
                anchor_executor.submit(
                    _detect_anchor_stars_task,
                    anchors_info[next_anchor],
                    fwhm_val,
                    base_sigma,
                    engine_val,
                    engine_profile,
                )
            )
            next_anchor += 1

        while pending_anchors:
            if cancel_event.is_set():
                for pending in pending_anchors:
                    pending.cancel()
                return
            future = pending_anchors.pop(0)
            info, shape, best_stars, best_fwhm, stopped_sigma, phase_data = future.result()
            del stopped_sigma
            if info.pop("_anchor_detection_reused", False):
                anchor_cache_hits += 1
            info.update(shape=shape, anchor_stars=best_stars,
                        star_count=len(best_stars), fwhm=best_fwhm, anchor_data=None)
            del phase_data
            app_print(f"  -> {info['batch_name']}: {len(best_stars):02d} estrelas base\n")
            if next_anchor < len(anchors_info):
                pending_anchors.append(
                    anchor_executor.submit(
                        _detect_anchor_stars_task,
                        anchors_info[next_anchor],
                        fwhm_val,
                        base_sigma,
                        engine_val,
                        engine_profile,
                    )
                )
                next_anchor += 1

    if anchor_cache_hits:
        app_print(
            f"[GLOBAL] Detecção de âncora reutilizada em {anchor_cache_hits}/{len(anchors_info)} batches.\n"
        )

    global_master_cfg = config.get("global_master", "Auto")
    if str(global_master_cfg).lower() == "auto":
        master_info = max(
            anchors_info,
            key=lambda item: (
                item.get("anchor_quality", 0.0),
                item.get("star_count", 0),
            ),
        )
        app_print(
            f"\n[GLOBAL] Master Automático eleito: {master_info['batch_name']} | estrelas={master_info['star_count']} | FWHM={master_info['fwhm']:.2f}\n"
        )
    else:
        master_info = next(
            (item for item in anchors_info if item["batch_name"] == global_master_cfg),
            None,
        )
        if master_info is None:
            master_info = max(
                anchors_info,
                key=lambda item: (
                    item.get("anchor_quality", 0.0),
                    item.get("star_count", 0),
                ),
            )
            app_print(
                f"\n[GLOBAL] Master especificado não encontrado. Fallback: {master_info['batch_name']}\n"
            )

    matching_radius = float(
        config.get("global_matching_radius", config.get("matching_radius", 100.0))
    )
    ransac_thresh = float(config.get("global_ransac", config.get("ransac", 5.0)))
    limits = _global_limits(config)

    global_flow = {
        "schema_version": 3,
        "mode": "global_neighbor_bfs",
        "global_master_batch": master_info["batch_name"],
        "transform_revision": _registration_input_fingerprint(
            [Path(item["anchor_path"]) for item in anchors_info],
            {"master": master_info["batch_name"], "matching_radius": matching_radius,
             "ransac": ransac_thresh, **limits},
        )[:16],
        "parameters": {
            "matching_radius": matching_radius,
            "ransac": ransac_thresh,
            **limits,
        },
        "batches": {},
        "quality": {},
        "cross_checks": [],
        "anchor_detection_cache_hits": anchor_cache_hits,
    }

    global_flow["batches"][master_info["batch_name"]] = {
        "status": "accepted",
        "matrix": np.eye(3).tolist(),
        "relative_to": None,
        "strategy": "master",
        "matches": master_info["star_count"],
        "inliers": master_info["star_count"],
        "inlier_ratio": 1.0,
        "rms": 0.0,
        "translation": [0.0, 0.0],
        "translation_magnitude": 0.0,
        "rotation_deg": 0.0,
        "scale": 1.0,
        "phase_response": 1.0,
        "local_transform_revision": master_info.get("local_transform_revision"),
    }
    global_flow["source_revisions"] = {
        info["batch_name"]: info.get("local_transform_revision")
        for info in anchors_info
        if info.get("local_transform_revision")
    }

    # O pareamento direto de cada batch contra o Master é independente entre
    # batches: paraleliza (antes era um loop sequencial).
    direct_results = {}
    targets_for_direct = [
        info for info in anchors_info if info["batch_name"] != master_info["batch_name"]
    ]

    with ThreadPoolExecutor(
        max_workers=global_workers, thread_name_prefix="astroflow-direct",
        initializer=partial(configure_worker_runtime, 1),
    ) as executor:
        futures = {
            executor.submit(
                _estimate_global_pair,
                master_info,
                target_info,
                matching_radius,
                ransac_thresh,
                limits,
            ): target_info
            for target_info in targets_for_direct
        }
        completed = 0
        for future in as_completed(futures):
            if cancel_event.is_set():
                return
            target_info = futures[future]
            batch_name = target_info["batch_name"]
            matrix, metrics = future.result()
            direct_results[batch_name] = (matrix, metrics)
            completed += 1
            app_progress(
                completed, total_batches, f"Alinhando {batch_name} ao Master Global..."
            )

    ordered_global_matrices = {master_info["batch_name"]: np.eye(3, dtype=np.float64)}
    global_graph_edges = []

    # Every successful direct match is a seed before neighbor expansion.  The
    # old loop re-ran these matches one at a time and could prevent an already
    # validated distant batch from serving as a propagation source.
    for target_info in sorted(targets_for_direct, key=lambda item: _natural_frame_key(item["batch_name"])):
        target_name = target_info["batch_name"]
        direct_matrix, direct_metrics = direct_results.get(target_name, (None, {}))
        if direct_matrix is None or direct_metrics.get("status") != "accepted":
            continue
        ordered_global_matrices[target_name] = np.asarray(direct_matrix, dtype=np.float64)
        global_flow["batches"][target_name] = {
            "status": "accepted",
            "matrix": np.asarray(direct_matrix, dtype=np.float64).tolist(),
            "relative_matrix": np.asarray(direct_matrix, dtype=np.float64).tolist(),
            "relative_to": master_info["batch_name"],
            "strategy": "master_direct",
            "hop_count": 1,
            "local_transform_revision": target_info.get("local_transform_revision"),
            **direct_metrics,
        }
        global_graph_edges.append({
            "source": master_info["batch_name"],
            "target": target_name,
            "relative_matrix": np.asarray(direct_matrix, dtype=np.float64).tolist(),
            "method": "master_direct",
            "hop_count": 1,
            "metrics": _flow_json_value(direct_metrics),
        })

    for distance in range(1, len(anchors_info)):
        progress_made = False
        for index, target_info in enumerate(anchors_info):
            target_name = target_info["batch_name"]
            if target_name in ordered_global_matrices:
                continue

            candidate_indices = []
            if index > 0:
                candidate_indices.append(index - 1)
            if index + 1 < len(anchors_info):
                candidate_indices.append(index + 1)

            for ref_index in candidate_indices:
                ref_info = anchors_info[ref_index]
                ref_name = ref_info["batch_name"]
                if ref_name not in ordered_global_matrices:
                    continue

                matrix, metrics = _estimate_global_pair(
                    ref_info, target_info, matching_radius, ransac_thresh, limits
                )
                if matrix is not None and metrics.get("status") == "accepted":
                    reference_matrix = ordered_global_matrices[ref_name]
                    absolute_matrix = reference_matrix @ matrix
                    ordered_global_matrices[target_name] = absolute_matrix

                    global_flow["batches"][target_name] = {
                        "status": "accepted",
                        "matrix": absolute_matrix.tolist(),
                        "relative_matrix": matrix.tolist(),
                        "relative_to": ref_name,
                        "strategy": (
                            "master_direct"
                            if ref_name == master_info["batch_name"]
                            else "neighbor_chain"
                        ),
                        "hop_count": int(global_flow["batches"].get(ref_name, {}).get("hop_count", 0)) + 1,
                        "local_transform_revision": target_info.get("local_transform_revision"),
                        **metrics,
                    }
                    global_graph_edges.append({
                        "source": ref_name,
                        "target": target_name,
                        "relative_matrix": matrix.tolist(),
                        "method": "neighbor_chain",
                        "hop_count": int(global_flow["batches"][target_name]["hop_count"]),
                        "metrics": _flow_json_value(metrics),
                    })
                    progress_made = True
                    break
        if not progress_made:
            break

    for target_info in anchors_info:
        batch_name = target_info["batch_name"]
        if batch_name == master_info["batch_name"]:
            continue
        if batch_name in global_flow["batches"]:
            continue

        direct_matrix, direct_metrics = direct_results.get(batch_name, (None, {}))
        if direct_matrix is not None and direct_metrics.get("status") == "accepted":
            global_flow["batches"][batch_name] = {
                "status": "accepted",
                "matrix": direct_matrix.tolist(),
                "relative_matrix": direct_matrix.tolist(),
                "relative_to": master_info["batch_name"],
                "strategy": "master_direct",
                "local_transform_revision": target_info.get("local_transform_revision"),
                **direct_metrics,
            }
            ordered_global_matrices[batch_name] = direct_matrix
        else:
            target_index = anchors_info.index(target_info)
            candidate_indices = []
            if target_index > 0:
                candidate_indices.append(target_index - 1)
            if target_index + 1 < len(anchors_info):
                candidate_indices.append(target_index + 1)

            progress_made = False
            for ref_index in candidate_indices:
                ref_info = anchors_info[ref_index]
                ref_name = ref_info["batch_name"]
                if ref_name not in ordered_global_matrices:
                    continue

                matrix, metrics = _estimate_global_pair(
                    ref_info, target_info, matching_radius, ransac_thresh, limits
                )
                if matrix is not None and metrics.get("status") == "accepted":
                    reference_matrix = ordered_global_matrices[ref_name]
                    absolute_matrix = reference_matrix @ matrix
                    ordered_global_matrices[batch_name] = absolute_matrix

                    global_flow["batches"][batch_name] = {
                        "status": "accepted",
                        "matrix": absolute_matrix.tolist(),
                        "relative_matrix": matrix.tolist(),
                        "relative_to": ref_name,
                        "strategy": "neighbor_chain",
                        "local_transform_revision": target_info.get("local_transform_revision"),
                        **metrics,
                    }
                    progress_made = True
                    break
            if not progress_made:
                global_flow["batches"][batch_name] = {
                    "status": "rejected",
                    "matrix": None,
                    "relative_to": None,
                    "strategy": None,
                    **{key: value for key, value in direct_metrics.items()},
                    "reason": direct_metrics.get("reason", "global_alignment_failed"),
                }

    for index, target_info in enumerate(anchors_info):
        batch_name = target_info["batch_name"]
        if batch_name == master_info["batch_name"]:
            continue
        entry = global_flow["batches"].get(batch_name)
        if not entry:
            continue

        direct_matrix, direct_metrics = direct_results.get(batch_name, (None, {}))
        final_matrix = entry.get("matrix")
        if direct_matrix is None or final_matrix is None:
            continue

        closure_error = _matrix_difference_score(
            direct_matrix, np.asarray(final_matrix, dtype=np.float64)
        )
        cross_check = {
            "target": batch_name,
            "direct_status": direct_metrics.get("status", "unknown"),
            "closure_error": round(closure_error, 6),
            "consistent": bool(
                closure_error <= float(config.get("global_closure_threshold", 0.05))
            ),
        }
        global_flow["cross_checks"].append(cross_check)
        entry["closure_error"] = cross_check["closure_error"]
        if not cross_check["consistent"]:
            entry["warning"] = "global_cross_check_inconsistent"

    accepted_batches = [
        e for e in global_flow["batches"].values() if e.get("status") == "accepted"
    ]
    rejected_batches = [
        e for e in global_flow["batches"].values() if e.get("status") == "rejected"
    ]

    global_flow["quality"] = {
        "total_batches": len(anchors_info),
        "accepted_batches": len(accepted_batches),
        "rejected_batches": len(rejected_batches),
        "coverage": (
            len(accepted_batches) / len(anchors_info) if anchors_info else 0.0
        ),
        "cross_checks": len(global_flow["cross_checks"]),
        "cross_check_failures": sum(
            1 for item in global_flow["cross_checks"] if not item["consistent"]
        ),
    }
    global_flow["registration_graph"] = {
        "version": 1,
        "strategy": "master_seed_neighbor_bfs",
        "root": master_info["batch_name"],
        "nodes": [
            {
                "id": info["batch_name"],
                "sequence_index": index,
                "status": global_flow["batches"].get(info["batch_name"], {}).get(
                    "status", "rejected"
                ),
            }
            for index, info in enumerate(anchors_info)
        ],
        "edges": global_graph_edges,
    }
    global_flow["master_metrics"] = {
        "batch": master_info["batch_name"],
        "star_count": master_info["star_count"],
        "fwhm": master_info["fwhm"],
        "anchor_quality": master_info.get("anchor_quality", 0.0),
    }
    global_flow["geometry_revision"] = _geometry_revision(global_flow)

    global_path = base_dir / "global_flow.json"
    if cancel_event.is_set():
        return {"status": "cancelled"}
    from app.infrastructure.json_store import atomic_json_write
    # Prepare an immutable revision for readers. The legacy JSON files remain
    # as compatibility copies, while the manifest switches all consumers to a
    # complete local/global set in one atomic operation.
    revision = str(global_flow.get("transform_revision") or hashlib.sha256(
        json.dumps(global_flow, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:16])
    snapshot_root = base_dir / ".flow_revisions" / revision
    local_flows = {}
    try:
        for batch_folder in batch_folders:
            local_path = batch_folder / "flow_local.json"
            if not local_path.exists():
                continue
            local_data = load_local_flow(batch_folder) if 'load_local_flow' in locals() else json.loads(local_path.read_text(encoding="utf-8"))
            if isinstance(local_data, dict):
                local_snapshot = snapshot_root / "batches" / batch_folder.name / "flow_local.json"
                atomic_json_write(local_snapshot, local_data)
                local_flows[batch_folder.name] = str(local_snapshot.relative_to(base_dir))
        global_snapshot = snapshot_root / "global_flow.json"
        atomic_json_write(global_snapshot, global_flow)
        manifest = {
            "schema_version": 2,
            "active_revision": revision,
            "local_flows": local_flows,
            "global_flow": str(global_snapshot.relative_to(base_dir)),
        }
        if cancel_event.is_set():
            return {"status": "cancelled"}
        atomic_json_write(base_dir / "flow_revision.json", manifest)
        # Keep the compatibility files available for older scripts. Readers
        # in this application use the manifest above.
        atomic_json_write(global_path, global_flow)
    except Exception as exc:
        app_print(f"[GLOBAL] Falha ao publicar revisão imutável: {exc}\n")
        return {"status": "failed", "reason": f"flow revision publish failed: {exc}"}

    temporal_enabled = config.get("temporal_analysis_enabled", True)
    if isinstance(temporal_enabled, str):
        temporal_enabled = temporal_enabled.strip().lower() not in {"0", "false", "no", "off"}
    if temporal_enabled:
        try:
            temporal_report = build_session_temporal_report(
                base_dir,
                config.get("temporal_gap_minutes", DEFAULT_GAP_MINUTES),
                config.get("temporal_seeing_sigma", DEFAULT_SEEING_SIGMA),
            )
            write_temporal_report(base_dir / "temporal_analysis.json", temporal_report)
            app_print("[GLOBAL] Relatório temporal salvo (revisão manual).\n")
        except Exception as exc:
            app_print(f"[GLOBAL] Aviso: análise temporal indisponível: {exc}\n")

    app_progress(total_batches, total_batches, "AstroFlow Finalizado.")
    app_print(
        f"\n>>> AstroFlow Finalizado. {len(accepted_batches)}/{len(anchors_info)} Batches aceitas no Global Flow. <<<\n"
    )
    return {
        "status": "partial" if rejected_batches else "success",
        "message": f"Flow: {len(accepted_batches)}/{len(anchors_info)} batches aceitas.",
        "output_path": str(global_path),
        "anchor_detection_cache_hits": anchor_cache_hits,
        "temporal_report": str(base_dir / "temporal_analysis.json")
        if (base_dir / "temporal_analysis.json").exists()
        else None,
    }


def _natural_frame_key(value: Path | str) -> tuple:
    """Return a deterministic human/numeric ordering for frame names."""
    name = Path(value).name.casefold()
    parts = re.split(r"(\d+)", name)
    return tuple((0, int(part)) if part.isdigit() else (1, part) for part in parts) + ((2, name),)


def _registration_input_fingerprint(files: list[Path], parameters: dict) -> str:
    """Fingerprint the registration inputs without decoding image pixels."""
    manifest = []
    for path in sorted(files, key=_natural_frame_key):
        try:
            stat = path.stat()
            manifest.append({
                "name": path.name,
                "size": int(stat.st_size),
                "mtime_ns": int(stat.st_mtime_ns),
            })
        except OSError:
            manifest.append({"name": path.name, "missing": True})
    payload = json.dumps(
        {"files": manifest, "parameters": parameters},
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _geometry_revision(value: dict, matrix_key: str = "matrix") -> str:
    """Stable revision for final geometry, independent of reference metadata."""
    matrices = []
    if isinstance(value, dict):
        entries = value.get("frames") or value.get("batches") or {}
        if isinstance(entries, dict):
            for name in sorted(entries, key=_natural_frame_key):
                matrix = entries[name].get(matrix_key) if isinstance(entries[name], dict) else None
                if matrix is not None:
                    matrices.append((name, np.asarray(matrix, dtype=np.float64).round(12).tolist()))
    payload = json.dumps(matrices, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _persisted_local_flow_is_current(batch_folder: Path, local_data: dict) -> bool:
    """Reject old graph results when files or registration settings changed."""
    if not isinstance(local_data, dict) or not local_data.get("input_fingerprint"):
        return False
    files = sorted(
        [
            path
            for path in batch_folder.iterdir()
            if path.is_file() and path.suffix.lower() in {".fit", ".fits", ".fts"}
        ],
        key=_natural_frame_key,
    )
    parameters = dict(local_data.get("parameters") or {})
    fingerprint = _registration_input_fingerprint(
        files,
        {
            **parameters,
            "registration_strategy": local_data.get("registration_strategy", "neighbor_bfs"),
            "engine": local_data.get("engine", "DAO"),
            "engine_profile": local_data.get("engine_profile", "Stable"),
            "fallback": local_data.get("transform_fallback", "Disabled"),
        },
    )
    return fingerprint == local_data.get("input_fingerprint")


def _flow_json_value(value):
    """Convert small numpy scalars/containers into JSON-safe values."""
    if isinstance(value, np.generic):
        return _flow_json_value(value.item())
    if isinstance(value, np.ndarray):
        return _flow_json_value(value.tolist())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _flow_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_flow_json_value(item) for item in value]
    return value


def _attempt_local_edge(
    ref_name: str,
    target_name: str,
    ref_frame: dict,
    target_frame: dict,
    anchor_name: str,
    matching_radius: float,
    ransac_thresh: float,
    limits: dict,
    fallback_engine: str,
    primary: bool,
    direction_history: list[tuple[float, float, float]] | None,
    sequence_gap: int,
):
    """Try one parent/target edge using the existing registration contract.

    ``primary`` retains the legacy normal/relaxed attempts for the nearest
    parent. A farther parent is a bounded recovery attempt and gets one wider
    probe. Keeping this distinction makes the graph deterministic while
    preserving the old rejection and fallback semantics.
    """
    if (
        not isinstance(ref_frame, dict)
        or not isinstance(target_frame, dict)
        or ref_frame.get("phase_data") is None
        or target_frame.get("phase_data") is None
    ):
        return None, {"reason": "missing_phase_data", "matches": 0, "inliers": 0,
                      "inlier_ratio": 0.0, "rms": 999.0}, "missing_data"

    try:
        phase_shift, phase_response = cv2.phaseCorrelate(
            ref_frame["phase_data"], target_frame["phase_data"]
        )
        dx, dy = float(phase_shift[0]), float(phase_shift[1])
        phase_response = float(phase_response)
    except Exception:
        dx, dy, phase_response = 0.0, 0.0, 0.0

    if primary:
        attempts = [("normal", matching_radius),
                    ("relaxed_radius", matching_radius * 2.0)]
    else:
        attempts = [("anchor" if ref_name == anchor_name else "neighbor",
                     matching_radius * 2.0)]

    best_metrics = {
        "reason": "phase_correlation_failed",
        "matches": 0,
        "inliers": 0,
        "inlier_ratio": 0.0,
        "rms": 999.0,
    }
    for attempt_name, radius in attempts:
        m_ref, m_cur = _match_incremental_stars(
            ref_frame["stars"], target_frame["stars"], (dx, dy), radius
        )
        rel_matrix, metrics = _estimate_incremental_transform(
            m_ref, m_cur, ransac_thresh, limits["min_stars"]
        )
        metrics = dict(metrics or {})
        metrics["spatial_inlier_coverage"] = _spatial_inlier_coverage(
            m_ref,
            m_cur,
            rel_matrix,
            target_frame["phase_data"].shape,
            residual_threshold=ransac_thresh,
        )
        metrics["phase_shift"] = [round(dx, 3), round(dy, 3)]
        metrics["phase_response"] = round(phase_response, 5)
        valid, reason = validate_transform(rel_matrix, metrics, limits)

        if valid and rel_matrix is not None and direction_history is not None:
            tx, ty, rot, _ = extract_geometric_properties(rel_matrix)
            if len(direction_history) >= 4:
                recent = direction_history[-8:]
                med_dx = float(np.median([item[0] for item in recent]))
                med_dy = float(np.median([item[1] for item in recent]))
                med_rot = float(np.median([item[2] for item in recent]))
                gap_scale = max(1, int(sequence_gap))
                if (
                    abs(tx - med_dx) > 120.0 * gap_scale
                    or abs(ty - med_dy) > 120.0 * gap_scale
                    or abs(rot - med_rot) > 2.0 * gap_scale
                ):
                    valid = False
                    reason = "temporal_validation_failed"

        if valid:
            return make_homogeneous(rel_matrix), metrics, attempt_name

        metrics["reason"] = reason
        if (
            int(metrics.get("inliers", 0)) > int(best_metrics.get("inliers", 0))
            or (
                int(metrics.get("inliers", 0)) == int(best_metrics.get("inliers", 0))
                and float(metrics.get("rms", 999.0)) < float(best_metrics.get("rms", 999.0))
            )
        ):
            best_metrics = metrics

    if str(fallback_engine).lower() in {"astroalign", "astroalign-asterism"}:
        try:
            _register_flow_engines()
            fallback = registry.resolve("flow.transform_fallback", "astroalign-asterism")
            relative_homogeneous, metrics = fallback(
                target_frame["stars"], ref_frame["stars"]
            )
            relative_homogeneous = np.asarray(relative_homogeneous, dtype=np.float64)
            rel_matrix = relative_homogeneous[:2, :]
            valid, reason = validate_transform(rel_matrix, metrics, limits)
            if valid:
                return relative_homogeneous, dict(metrics), "astroalign_asterism"
            best_metrics = {**dict(metrics), "reason": reason}
        except Exception:
            # Optional engines must never stop normal Flow execution.
            pass

    return None, best_metrics, best_metrics.get("reason", "registration_failed")


def _build_local_registration_graph(
    files: list[Path],
    anchor_file: Path,
    config: dict,
    limits: dict,
    matching_radius: float,
    ransac_thresh: float,
    engine_val: str,
    engine_profile: str,
    fallback_engine: str,
    worker_count: int,
    budget,
    cancelled,
    app_print,
    batch_name: str,
):
    """Build a bounded, deterministic local registration graph."""
    files = sorted(files, key=_natural_frame_key)
    name_to_index = {path.name: index for index, path in enumerate(files)}
    anchor_name = anchor_file.name
    anchor_index = name_to_index[anchor_name]
    cache_limit = max(3, min(16, int(config.get("flow_cache_frames", budget.max_in_flight + 4))))
    frame_cache: OrderedDict[str, dict | None] = OrderedDict()

    def prepare(name: str):
        if name in frame_cache:
            frame_cache.move_to_end(name)
            return frame_cache[name]
        path = files[name_to_index[name]]
        _, frame = _process_single_frame(
            path,
            float(config.get("fwhm", 4.0)),
            float(config.get("sigma", 5.0)),
            int(config.get("max_stars", 250)),
            limits["min_stars"],
            engine_val,
            engine_profile,
        )
        frame_cache[name] = frame
        frame_cache.move_to_end(name)
        while len(frame_cache) > cache_limit:
            old_name, old_frame = frame_cache.popitem(last=False)
            if old_name == anchor_name:
                frame_cache[old_name] = old_frame
                break
        return frame

    prefetch_executor = ThreadPoolExecutor(
        max_workers=max(1, worker_count),
        thread_name_prefix="astroflow-prepare",
        initializer=partial(configure_worker_runtime, 1),
    )
    # Prefetch is fully drained before a target's edge attempts begin. Reuse
    # its already-bounded pool instead of constructing and destroying a new
    # executor for every target with multiple candidate parents. Futures are
    # still consumed in candidate order below, so graph publication remains
    # deterministic.
    edge_executor_enabled = bool(
        config.get("flow_workers") is not None
        and worker_count > 1
    )

    def prefetch(names):
        """Prepare an independent bounded batch without changing graph order."""
        wanted = []
        seen = set()
        for name in names:
            if name in seen or name in frame_cache or name not in name_to_index:
                continue
            seen.add(name)
            wanted.append(name)
        if not wanted:
            return
        futures = {
            name: prefetch_executor.submit(
                _process_single_frame,
                files[name_to_index[name]],
                float(config.get("fwhm", 4.0)),
                float(config.get("sigma", 5.0)),
                int(config.get("max_stars", 250)),
                limits["min_stars"],
                engine_val,
                engine_profile,
            )
            for name in wanted
        }
        for name in wanted:
            try:
                _, frame = futures[name].result()
            except Exception as exc:
                app_print(f"[{files[name_to_index[name]].name}] Erro no worker: {exc}\n")
                frame = None
            frame_cache[name] = frame
            frame_cache.move_to_end(name)
            while len(frame_cache) > cache_limit:
                old_name, old_frame = frame_cache.popitem(last=False)
                if old_name == anchor_name:
                    frame_cache[old_name] = old_frame
                    break
    anchor = prepare(anchor_name)
    if (
        anchor is None
        or anchor.get("status") == "error"
        or len(anchor.get("stars", [])) < limits["min_stars"]
    ):
        prefetch_executor.shutdown(wait=True, cancel_futures=True)
        app_print(f"[{batch_name}] ERRO: Falha na âncora.\n")
        return None

    anchor_quality = calculate_anchor_quality(len(anchor["stars"]), anchor["fwhm"])
    registration_parameters = {
        "fwhm": float(config.get("fwhm", 4.0)),
        "sigma": float(config.get("sigma", 5.0)),
        "max_stars": int(config.get("max_stars", 250)),
        "matching_radius": matching_radius,
        "ransac": ransac_thresh,
        "neighbor_window": int(config.get("neighbor_window", 4)),
        **limits,
    }
    input_fingerprint = _registration_input_fingerprint(files, {
        **registration_parameters,
        "registration_strategy": "neighbor_bfs",
        "engine": engine_val,
        "engine_profile": engine_profile,
        "fallback": fallback_engine,
    })

    flow_data = {
        "schema_version": 3,
        "batch_anchor": anchor_name,
        "selected_reference": anchor_name,
        "mode": "neighbor_bfs",
        "registration_strategy": "neighbor_bfs",
        "workers": worker_count,
        "engine": engine_val,
        "engine_profile": engine_profile,
        "transform_fallback": fallback_engine,
        "input_fingerprint": input_fingerprint,
        "transform_revision": input_fingerprint[:16],
        "parameters": registration_parameters,
        "anchor_shape": list(anchor["phase_data"].shape) if anchor.get("phase_data") is not None else None,
        "anchor_stars": _flow_json_value(anchor["stars"]),
        "anchor_detection": _anchor_detection_metadata(
            anchor, float(config.get("fwhm", 4.0)), float(config.get("sigma", 5.0)),
            int(config.get("max_stars", 250)),
            engine_val, engine_profile,
        ),
        "anchor_metrics": {
            "star_count": len(anchor["stars"]),
            "fwhm": anchor["fwhm"],
            "quality": anchor_quality,
            **anchor.get("metrics", {}),
        },
        "frames": {},
    }

    flow_data["frames"][anchor_name] = {
        "status": "accepted",
        "confidence": "reference",
        "confidence_reason": "Coordinate reference; not an independently verified registration",
        "recovery_method": "reference",
        "matrix": np.eye(3).tolist(),
        "relative_to": None,
        "matches": len(anchor["stars"]),
        "inliers": len(anchor["stars"]),
        "inlier_ratio": 1.0,
        "rms": 0.0,
        "translation": [0.0, 0.0],
        "translation_magnitude": 0.0,
        "rotation_deg": 0.0,
        "scale": 1.0,
        "cumulative_rms": 0.0,
        "hop_count": 0,
        "sequence_index": anchor_index,
        "star_count": len(anchor["stars"]),
        "fwhm": anchor["fwhm"],
        **_persisted_frame_metadata(anchor),
    }

    accepted_names = {anchor_name}
    remaining_names = {path.name for path in files if path.name != anchor_name}
    direction_history: dict[int, list[tuple[float, float, float]]] = {-1: [], 1: []}
    edges: list[dict] = []
    last_failures: dict[str, dict] = {}
    edge_attempt_cache: dict[tuple[str, str, bool], tuple[np.ndarray | None, dict, str]] = {}
    neighbor_window = max(1, int(config.get("neighbor_window", 4)))

    def candidates_for(target_name: str):
        target_index = name_to_index[target_name]
        candidates = []
        for candidate_name in accepted_names:
            candidate_index = name_to_index[candidate_name]
            distance = abs(candidate_index - target_index)
            if candidate_name != anchor_name and distance > neighbor_window:
                continue
            parent_entry = flow_data["frames"].get(candidate_name, {})
            candidates.append((
                0 if distance == 1 else 1,
                int(parent_entry.get("hop_count", 0)),
                distance,
                _natural_frame_key(candidate_name),
                candidate_name,
            ))
        candidates.sort()
        return [item[-1] for item in candidates]

    while remaining_names and not cancelled():
        made_progress = False
        ordered_remaining = sorted(
            remaining_names,
            key=lambda name: (abs(name_to_index[name] - anchor_index),
                              _natural_frame_key(name)),
        )
        # Overlap FITS decoding and star detection for the next bounded
        # frontier. Results are still consumed and committed in natural order.
        frontier = ordered_remaining[:max(1, worker_count * 2)]
        # Prefetch only new targets. Prefetching every accepted parent here
        # caused cache churn and repeated FITS preparation on long batches.
        prefetch(frontier)
        for target_name in ordered_remaining:
            if cancelled():
                prefetch_executor.shutdown(wait=True, cancel_futures=True)
                return None
            target_frame = prepare(target_name)
            if (
                target_frame is None
                or target_frame.get("status") == "error"
                or len(target_frame.get("stars", [])) < limits["min_stars"]
            ):
                flow_data["frames"][target_name] = {
                    "status": "rejected",
                    "confidence": "rejected",
                    "confidence_reason": "insufficient_stars_or_error",
                    "reason": "insufficient_stars_or_error",
                    "sequence_index": name_to_index[target_name],
                    **_persisted_frame_metadata(target_frame),
                }
                remaining_names.remove(target_name)
                made_progress = True
                continue

            candidates = candidates_for(target_name)
            if not candidates:
                continue
            # The first candidates are the nearest, lowest-hop routes. Keep a
            # bounded recovery probe and retain the anchor as a last resort.
            max_candidates = max(2, int(config.get("max_edge_candidates", 3)))
            if len(candidates) > max_candidates:
                selected = candidates[:max_candidates]
                if anchor_name in candidates and anchor_name not in selected:
                    selected[-1] = anchor_name
                candidates = selected
            target_index = name_to_index[target_name]
            successful_edges = []
            edge_jobs = []
            edge_executor = (
                prefetch_executor
                if edge_executor_enabled and len(candidates) > 1
                else None
            )
            for candidate_position, parent_name in enumerate(candidates):
                parent_frame = prepare(parent_name)
                parent_index = name_to_index[parent_name]
                direction = 1 if target_index > parent_index else -1
                cache_key = (parent_name, target_name, candidate_position == 0)
                if cache_key in edge_attempt_cache:
                    cached_relative, cached_metrics, cached_method = edge_attempt_cache[cache_key]
                    relative = None if cached_relative is None else cached_relative.copy()
                    metrics = dict(cached_metrics)
                    method = cached_method
                else:
                    history = list(direction_history[direction])
                    if edge_executor is not None:
                        edge_jobs.append((candidate_position, parent_name, parent_index, cache_key,
                                          edge_executor.submit(
                            _attempt_local_edge, parent_name, target_name, parent_frame,
                            target_frame, anchor_name, matching_radius, ransac_thresh, limits,
                            fallback_engine, candidate_position == 0, history,
                            abs(target_index - parent_index))))
                        continue
                    relative, metrics, method = _attempt_local_edge(
                        parent_name, target_name, parent_frame, target_frame, anchor_name,
                        matching_radius, ransac_thresh, limits, fallback_engine,
                        primary=(candidate_position == 0), direction_history=history,
                        sequence_gap=abs(target_index - parent_index))
                    edge_attempt_cache[cache_key] = (None if relative is None else relative.copy(),
                                                     dict(metrics), method)
                if edge_executor is not None and cache_key not in edge_attempt_cache:
                    continue
                # Fall through using the cached or freshly computed result.
                if cache_key in edge_attempt_cache:
                    relative, metrics, method = edge_attempt_cache[cache_key]
                    relative = None if relative is None else relative.copy()
                    metrics = dict(metrics)
                if relative is not None:
                    parent_entry = flow_data["frames"][parent_name]
                    parent_matrix = np.asarray(parent_entry["matrix"], dtype=np.float64)
                    cumulative_matrix = parent_matrix @ relative
                    parent_rms = float(parent_entry.get("cumulative_rms", 0.0))
                    current_rms = float(metrics.get("rms", 0.0))
                    cumulative_rms = float(np.sqrt(parent_rms ** 2 + current_rms ** 2))
                    successful_edges.append((
                        int(parent_entry.get("hop_count", 0)) + 1,
                        cumulative_rms,
                        abs(target_index - parent_index),
                        _natural_frame_key(parent_name),
                        parent_name,
                        relative,
                        metrics,
                        method,
                        cumulative_matrix,
                    ))
                else:
                    last_failures[target_name] = metrics

            if edge_executor is not None:
                for candidate_position, parent_name, parent_index, cache_key, future in edge_jobs:
                    try:
                        relative, metrics, method = future.result()
                    except Exception:
                        relative, metrics, method = None, {"reason": "registration_failed"}, "registration_failed"
                    edge_attempt_cache[cache_key] = (
                        None if relative is None else relative.copy(), dict(metrics), method)
                    if relative is not None:
                        parent_entry = flow_data["frames"][parent_name]
                        parent_matrix = np.asarray(parent_entry["matrix"], dtype=np.float64)
                        parent_rms = float(parent_entry.get("cumulative_rms", 0.0))
                        current_rms = float(metrics.get("rms", 0.0))
                        successful_edges.append((
                            int(parent_entry.get("hop_count", 0)) + 1,
                            float(np.sqrt(parent_rms ** 2 + current_rms ** 2)),
                            abs(target_index - parent_index), _natural_frame_key(parent_name),
                            parent_name, relative, metrics, method, parent_matrix @ relative))
                    else:
                        last_failures[target_name] = metrics

            if not successful_edges:
                continue

            successful_edges.sort(key=lambda item: item[:4])
            route_warnings = []
            if len(successful_edges) > 1:
                best_candidate = successful_edges[0]
                best_matrix = best_candidate[8]
                height, width = target_frame["phase_data"].shape[:2]
                probe_points = np.asarray(
                    [[0.0, 0.0], [max(0, width - 1), 0.0],
                     [0.0, max(0, height - 1)],
                     [max(0, width - 1), max(0, height - 1)],
                     [width / 2.0, height / 2.0]],
                    dtype=np.float32,
                ).reshape(-1, 1, 2)
                best_probe = cv2.perspectiveTransform(
                    probe_points, np.asarray(best_matrix, dtype=np.float64)
                ).reshape(-1, 2)
                for alternate in successful_edges[1:]:
                    alternate_probe = cv2.perspectiveTransform(
                        probe_points, np.asarray(alternate[8], dtype=np.float64)
                    ).reshape(-1, 2)
                    probe_error = float(np.sqrt(np.mean((best_probe - alternate_probe) ** 2)))
                    if probe_error > ransac_thresh:
                        route_warnings.append({
                            "alternate_parent": alternate[4],
                            "probe_error": round(probe_error, 4),
                            "threshold": ransac_thresh,
                        })
            (
                hop_count,
                cumulative_rms,
                _distance,
                _parent_key,
                parent_name,
                relative,
                metrics,
                method,
                cumulative_matrix,
            ) = successful_edges[0]
            parent_index = name_to_index[parent_name]
            direction = 1 if target_index > parent_index else -1
            tx, ty, rot, _ = extract_geometric_properties(relative[:2, :])
            if method != "anchor":
                direction_history[direction].append((tx, ty, rot))
                if len(direction_history[direction]) > 8:
                    del direction_history[direction][:-8]

            flow_data["frames"][target_name] = {
                "status": "accepted",
                "confidence": _classify_flow_confidence(metrics, limits),
                "confidence_reason": _flow_confidence_reason(metrics, limits),
                "matrix": cumulative_matrix.tolist(),
                "relative_matrix": relative.tolist(),
                "relative_to": parent_name,
                "recovery_method": method,
                "cumulative_rms": cumulative_rms,
                "hop_count": hop_count,
                "sequence_index": target_index,
                "route_consistent": not route_warnings,
                "route_warnings": route_warnings,
                "fwhm": target_frame["fwhm"],
                "star_count": len(target_frame["stars"]),
                **_flow_json_value(metrics),
                **_persisted_frame_metadata(target_frame),
            }
            edge = {
                "source": parent_name,
                "target": target_name,
                "relative_matrix": relative.tolist(),
                "method": method,
                "hop_count": hop_count,
                "cumulative_rms": cumulative_rms,
                "metrics": _flow_json_value(metrics),
                "route_warnings": route_warnings,
            }
            edges.append(edge)
            accepted_names.add(target_name)
            remaining_names.remove(target_name)
            made_progress = True
            app_print(
                f"[{target_name}] OK ({method}) <- {parent_name} | "
                f"{metrics.get('inliers', 0)}/{metrics.get('matches', 0)} inliers | "
                f"ratio={float(metrics.get('inlier_ratio', 0.0)):.1%} | "
                f"RMS={float(metrics.get('rms', 999.0)):.3f}px\n"
            )

        if not made_progress:
            break

    for target_name in sorted(remaining_names, key=_natural_frame_key):
        metrics = last_failures.get(target_name, {})
        flow_data["frames"][target_name] = {
            "status": "rejected",
            "confidence": "rejected",
            "confidence_reason": metrics.get("reason", "disconnected_from_reference"),
            "reason": metrics.get("reason", "disconnected_from_reference"),
            "sequence_index": name_to_index[target_name],
            "matches": metrics.get("matches", 0),
            "inliers": metrics.get("inliers", 0),
            "inlier_ratio": metrics.get("inlier_ratio", 0.0),
            "rms": metrics.get("rms", 999.0),
            "phase_shift": metrics.get("phase_shift", [0.0, 0.0]),
            "phase_response": metrics.get("phase_response", 0.0),
            **_persisted_frame_metadata(prepare(target_name)),
        }
        app_print(
            f"[{target_name}] REJEITADO: {flow_data['frames'][target_name]['reason']} | "
            f"inliers={flow_data['frames'][target_name]['inliers']} | "
            f"RMS={float(flow_data['frames'][target_name]['rms']):.3f}px\n"
        )

    flow_data["registration_graph"] = {
        "version": 1,
        "strategy": "neighbor_bfs",
        "root": anchor_name,
        "neighbor_window": neighbor_window,
        "nodes": [
            {
                "id": path.name,
                "sequence_index": index,
                "status": flow_data["frames"].get(path.name, {}).get("status", "rejected"),
            }
            for index, path in enumerate(files)
        ],
        "edges": edges,
    }
    flow_data["statistics"] = {
        "total_frames": len(files),
        "accepted_frames": sum(1 for item in flow_data["frames"].values()
                                if item.get("status") == "accepted"),
        "rejected_frames": sum(1 for item in flow_data["frames"].values()
                                if item.get("status") == "rejected"),
        "coverage": sum(1 for item in flow_data["frames"].values()
                         if item.get("status") == "accepted") / max(len(files), 1),
        "first_frame": files[0].name,
        "last_frame": files[-1].name,
        "first_frame_valid": flow_data["frames"].get(files[0].name, {}).get("status") == "accepted",
        "last_frame_valid": flow_data["frames"].get(files[-1].name, {}).get("status") == "accepted",
        "chain_segments": _count_chain_segments(flow_data),
    }
    for node in flow_data["registration_graph"]["nodes"]:
        frame = flow_data["frames"].get(node["id"], {})
        node["parent"] = frame.get("relative_to")
        node["hop_count"] = int(frame.get("hop_count", 0))
    flow_data["geometry_revision"] = _geometry_revision(flow_data)
    flow_data = _flow_json_value(flow_data)
    accepted_count = int(flow_data["statistics"]["accepted_frames"])
    anchor_metrics = flow_data["anchor_metrics"]
    prefetch_executor.shutdown(wait=True, cancel_futures=True)
    return flow_data, anchor, anchor_quality, accepted_count


def _graph_adjacency(graph: dict) -> dict[str, list[tuple[str, np.ndarray]]]:
    """Build an undirected adjacency map from persisted graph edges."""
    adjacency: dict[str, list[tuple[str, np.ndarray]]] = {}
    for edge in graph.get("edges", []) if isinstance(graph, dict) else []:
        source = edge.get("source")
        target = edge.get("target")
        matrix = edge.get("relative_matrix")
        if not source or not target or matrix is None:
            continue
        try:
            relative = np.asarray(matrix, dtype=np.float64)
            if relative.shape != (3, 3):
                continue
            inverse = np.linalg.inv(relative)
        except (TypeError, ValueError, np.linalg.LinAlgError):
            continue
        # The stored relative matrix maps target pixels into source pixels.
        adjacency.setdefault(source, []).append((target, relative))
        adjacency.setdefault(target, []).append((source, inverse))
    return adjacency


def rebase_flow_reference(flow_data: dict, new_reference: str) -> dict:
    """Re-root a connected local Flow without decoding FITS or detecting stars.

    Frame matrices map a frame into the current local reference. Re-rooting
    therefore uses ``inverse(L_reference) @ L_frame``. When a registration
    graph is present, paths and ``relative_to`` fields are rebuilt so later
    propagation and provenance remain explicit.
    """
    if not isinstance(flow_data, dict):
        raise ValueError("Flow result is not a mapping")
    frames = flow_data.get("frames")
    if not isinstance(frames, dict) or new_reference not in frames:
        raise ValueError(f"Reference frame not found: {new_reference}")
    reference_entry = frames[new_reference]
    if reference_entry.get("status") != "accepted" or reference_entry.get("matrix") is None:
        raise ValueError(f"Reference frame is disconnected: {new_reference}")

    result = copy.deepcopy(flow_data)
    result_frames = result["frames"]
    reference_matrix = np.asarray(reference_entry["matrix"], dtype=np.float64)
    try:
        inverse_reference = np.linalg.inv(reference_matrix)
    except np.linalg.LinAlgError as exc:
        raise ValueError("Reference transform is singular") from exc

    graph = result.get("registration_graph")
    adjacency = _graph_adjacency(graph) if isinstance(graph, dict) else {}
    visited = {new_reference}
    queue = deque([new_reference])
    root_matrix = {new_reference: np.eye(3, dtype=np.float64)}
    parent_map: dict[str, tuple[str, np.ndarray, int]] = {}
    while queue:
        parent = queue.popleft()
        for child, relative in sorted(
            adjacency.get(parent, []),
            key=lambda item: _natural_frame_key(item[0]),
        ):
            if child in visited:
                continue
            if result_frames.get(child, {}).get("status") != "accepted":
                continue
            visited.add(child)
            root_matrix[child] = root_matrix[parent] @ relative
            parent_hops = parent_map[parent][2] if parent in parent_map else 0
            parent_map[child] = (parent, relative, int(parent_hops) + 1)
            queue.append(child)

    accepted_names = {
        name for name, item in result_frames.items() if item.get("status") == "accepted"
    }
    if adjacency:
        disconnected = sorted(accepted_names - visited, key=_natural_frame_key)
        if disconnected:
            raise ValueError(
                "Reference frame is not connected to accepted frames: " + ", ".join(disconnected)
            )
    else:
        # Legacy flow files predate the persisted graph. Their cumulative
        # matrices still provide the exact rebase operation, so retain them
        # without claiming a reconstructed path.
        visited = set(accepted_names)

    # Use the graph path when available; the matrix formula is the fallback
    # for legacy flow files without graph metadata.
    for name, item in result_frames.items():
        if item.get("status") != "accepted" or item.get("matrix") is None:
            continue
        old_matrix = np.asarray(item["matrix"], dtype=np.float64)
        # The matrix formula is authoritative for final geometry. The graph
        # traversal below only supplies the new provenance parent/path.
        rebased = inverse_reference @ old_matrix
        item["matrix"] = rebased.tolist()
        item["reference_rebased"] = True
        item["reference_rebased_from"] = flow_data.get("batch_anchor")
        if name == new_reference:
            item["relative_to"] = None
            item["relative_matrix"] = np.eye(3, dtype=np.float64).tolist()
            item["hop_count"] = 0
            item["recovery_method"] = "reference"
            item["confidence"] = "reference"
        elif name in parent_map:
            parent, relative, hop_count = parent_map[name]
            item["relative_to"] = parent
            item["relative_matrix"] = relative.tolist()
            item["hop_count"] = hop_count
            item["recovery_method"] = "reference_rebased"

    # Keep graph-level provenance in lockstep with the compatibility fields.
    if isinstance(graph, dict):
        for node in result["registration_graph"].get("nodes", []):
            if not isinstance(node, dict):
                continue
            name = node.get("id", node.get("name"))
            if name == new_reference:
                node["parent"] = None
                node["hop_count"] = 0
            elif name in parent_map:
                node["parent"] = parent_map[name][0]
                node["hop_count"] = parent_map[name][2]

    result["batch_anchor"] = new_reference
    result["selected_reference"] = new_reference
    result["reference_provenance"] = {
        "action": "rebase",
        "from": flow_data.get("batch_anchor"),
        "to": new_reference,
        "source_revision": flow_data.get("transform_revision"),
        "analysis_reused": True,
        "fits_decoded": False,
        "star_detection_reused": False,
        "input_fingerprint": flow_data.get("input_fingerprint"),
    }
    old_revision = str(flow_data.get("transform_revision", ""))
    result["transform_revision"] = hashlib.sha256(
        f"{old_revision}|reference:{new_reference}".encode("utf-8")
    ).hexdigest()[:16]
    if isinstance(graph, dict):
        result["registration_graph"]["root"] = new_reference
        result["registration_graph"]["rebase_from"] = graph.get("root")
        result["registration_graph"]["rebase_revision"] = result["transform_revision"]
    return _flow_json_value(result)


def rebase_global_transform(global_matrix, local_reference_matrix) -> np.ndarray:
    """Preserve final geometry when a batch reference is changed."""
    return np.asarray(global_matrix, dtype=np.float64) @ np.asarray(
        local_reference_matrix, dtype=np.float64
    )


def apply_reference_change(
    batch_dir: Path,
    new_reference: str,
    base_dir: Path | None = None,
    cancellation_event=None,
) -> dict:
    """Rebase a connected Flow graph and switch one immutable revision.

    Payloads are written below ``.flow_revisions/<revision>`` first. Readers
    consume the single manifest switch, so a failed write or cancellation
    leaves the previously active local/global pair selected.
    """
    def cancelled() -> bool:
        return bool(cancellation_event is not None and cancellation_event.is_set())

    batch_dir = Path(batch_dir).resolve()
    root_dir = Path(base_dir).resolve() if base_dir is not None else batch_dir.parent
    if cancelled():
        return {"status": "cancelled"}

    try:
        # Import lazily to keep the Flow module independent of Align at import
        # time while making all consumers honour the active manifest.
        from astroalign_logic import load_global_flow, load_local_flow
        local_flow = load_local_flow(batch_dir)
        if not isinstance(local_flow, dict):
            return {"status": "failed", "reason": "flow_local_missing"}
        if not _persisted_local_flow_is_current(batch_dir, local_flow):
            return {"status": "failed", "reason": "flow_local_stale_recompute_required"}
        old_reference_entry = local_flow.get("frames", {}).get(new_reference)
        if not isinstance(old_reference_entry, dict):
            return {"status": "failed", "reason": "reference_not_found"}
        old_reference_matrix = np.asarray(old_reference_entry.get("matrix"), dtype=np.float64)
        updated_local = rebase_flow_reference(local_flow, new_reference)
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        return {"status": "failed", "reason": str(exc)}
    if cancelled():
        return {"status": "cancelled"}

    updated_local["geometry_revision"] = local_flow.get("geometry_revision") or _geometry_revision(local_flow)
    updated_global = None
    try:
        from astroalign_logic import load_global_flow
        global_flow = load_global_flow(root_dir)
        if isinstance(global_flow, dict):
            batch_entry = global_flow.get("batches", {}).get(batch_dir.name)
            source_revision = (global_flow.get("source_revisions", {}) or {}).get(batch_dir.name)
            if source_revision is None and isinstance(batch_entry, dict):
                source_revision = batch_entry.get("local_transform_revision")
            if source_revision and source_revision != local_flow.get("transform_revision"):
                return {"status": "failed", "reason": "global_flow_stale_recompute_required"}
            if isinstance(batch_entry, dict) and batch_entry.get("matrix") is not None:
                updated_global = copy.deepcopy(global_flow)
                batch_entry = updated_global["batches"][batch_dir.name]
                batch_entry["matrix"] = rebase_global_transform(
                    batch_entry["matrix"], old_reference_matrix
                ).tolist()
                if batch_entry.get("relative_matrix") is not None:
                    batch_entry["relative_matrix"] = (
                        np.asarray(batch_entry["relative_matrix"], dtype=np.float64)
                        @ old_reference_matrix
                    ).tolist()
                batch_entry["reference_rebased_from"] = local_flow.get("batch_anchor")
                batch_entry["reference_rebased_to"] = new_reference
                batch_entry["local_transform_revision"] = updated_local["transform_revision"]
                updated_global.setdefault("source_revisions", {})[batch_dir.name] = updated_local["transform_revision"]
                try:
                    inverse_reference = np.linalg.inv(old_reference_matrix)
                except np.linalg.LinAlgError as exc:
                    return {"status": "failed", "reason": "reference_transform_singular"}
                for edge in updated_global.get("registration_graph", {}).get("edges", []):
                    if edge.get("relative_matrix") is None:
                        continue
                    relative = np.asarray(edge["relative_matrix"], dtype=np.float64)
                    if edge.get("target") == batch_dir.name:
                        edge["relative_matrix"] = (relative @ old_reference_matrix).tolist()
                    elif edge.get("source") == batch_dir.name:
                        edge["relative_matrix"] = (inverse_reference @ relative).tolist()
                    if edge.get("source") == batch_dir.name or edge.get("target") == batch_dir.name:
                        edge["method"] = "reference_rebased"
                updated_global["geometry_revision"] = global_flow.get("geometry_revision") or _geometry_revision(global_flow, "matrix")
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        return {"status": "failed", "reason": str(exc)}
    if cancelled():
        return {"status": "cancelled"}

    from app.infrastructure.json_store import atomic_json_write
    revision = updated_local["transform_revision"]
    snapshot_root = root_dir / ".flow_revisions" / revision
    local_snapshot = snapshot_root / "batches" / batch_dir.name / "flow_local.json"
    global_snapshot = snapshot_root / "global_flow.json" if updated_global is not None else None
    manifest_path = root_dir / "flow_revision.json"
    current_manifest = {}
    if manifest_path.exists():
        try:
            current_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            current_manifest = {}

    local_flows = dict(current_manifest.get("local_flows", {})) if isinstance(current_manifest, dict) else {}
    for candidate in sorted(root_dir.iterdir(), key=lambda p: p.name.casefold()) if root_dir.is_dir() else []:
        if candidate.is_dir() and "batch" in candidate.name.lower() and candidate.name not in local_flows:
            legacy = candidate / "flow_local.json"
            if legacy.exists():
                local_flows[candidate.name] = str(legacy.relative_to(root_dir))
    local_flows[batch_dir.name] = str(local_snapshot.relative_to(root_dir))
    manifest = {
        "schema_version": 2,
        "active_revision": revision,
        "local_flows": local_flows,
        "global_flow": str(global_snapshot.relative_to(root_dir)) if global_snapshot is not None else current_manifest.get("global_flow"),
    }
    try:
        atomic_json_write(local_snapshot, updated_local)
        if global_snapshot is not None:
            atomic_json_write(global_snapshot, updated_global)
        if cancelled():
            return {"status": "cancelled"}
        # This is the only operation that changes the active revision.
        atomic_json_write(manifest_path, manifest)
    except Exception as exc:
        return {"status": "failed", "reason": f"reference revision publish failed: {exc}"}
    return {
        "status": "success",
        "revision": revision,
        "analysis_reused": True,
        "fits_decoded": False,
        "output_path": str(local_snapshot),
        "manifest_path": str(manifest_path),
    }


# ============================================================
# Preview (Contrato 100% compatível com [source: 17])
# ============================================================


def preview_star_detection(
    batch_dir: Path, config: dict
) -> tuple[np.ndarray | None, int, float]:
    """Gera o preview com as marcações de estrelas em conformidade com show_astroflow_preview em main.py."""
    files = sorted(
        [
            p
            for p in batch_dir.iterdir()
            if p.is_file() and p.suffix.lower() in {".fit", ".fits", ".fts"}
        ],
        key=_natural_frame_key,
    )
    if not files:
        return (None, 0, 0.0)

    anchor_name = config.get("custom_anchors", {}).get(batch_dir.name)
    anchor_file = next((p for p in files if p.name == anchor_name), files[0])

    data, header = load_fits_data(anchor_file)
    working_data = extract_luminance(data, header)

    fwhm_val = float(config.get("fwhm", 4.0))
    sigma_val = float(config.get("sigma", 5.0))
    max_stars_val = int(config.get("max_stars", 250))
    engine_val = _flow_detector_choice(config)

    stars, measured_fwhm, _ = detect_stars(
        working_data, fwhm_val, sigma_val, max_stars_val, engine_val,
        str(config.get("engine_profile", "Stable")),
    )
    _, median, std = sigma_clipped_stats(working_data, sigma=3.0)
    median, std = float(median), float(std)

    denominator = max((8.0 * std), 1e-5)
    vmin, vmax = median, median + denominator
    norm_data = np.clip((working_data - vmin) / max(vmax - vmin, 1e-5), 0, 1) * 255.0

    img_color = cv2.cvtColor(norm_data.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    radius = max(3, int(max(measured_fwhm, fwhm_val) * 1.5))

    for x, y in stars:
        cv2.circle(img_color, (int(round(x)), int(round(y))), radius, (0, 0, 255), 1)

    return (img_color, len(stars), float(measured_fwhm))


def save_frame_metrics(image_path: Path | str, metrics: dict):
    base_path, _ = os.path.splitext(str(image_path))
    json_path = f"{base_path}_metrics.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False)
    return json_path
