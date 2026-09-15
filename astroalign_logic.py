import hashlib
import json
import os
import shutil
import tempfile
import threading
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed, wait, FIRST_COMPLETED
from dataclasses import dataclass
from collections import OrderedDict
from functools import partial
from pathlib import Path

import colour_demosaicing
import cv2
import numpy as np
from astropy.io import fits
from astropy.utils.exceptions import AstropyWarning
from skimage.transform import AffineTransform, warp

from app.engines import EngineProfile, ExecutionBudget, registry
from app.engines.align import register_align_engines
from cpu_runtime import configure_opencv_threads, configure_worker_runtime, physical_core_count

# Suprime todos os avisos de verificação de cabeçalho do Astropy
warnings.simplefilter("ignore", category=AstropyWarning)

FITS_SUFFIXES = {".fit", ".fits", ".fts"}

INTERPOLATION_MODES = {
    "Nearest": "nearest",
    "Bilinear": "bilinear",
    "Bicubic": "bicubic",
    "Lanczos": "lanczos",
}

CV2_INTERPOLATION_MODES = {
    "nearest": cv2.INTER_NEAREST,
    "bilinear": cv2.INTER_LINEAR,
}

# ============================================================
# Bayer
# ============================================================

BAYER_BASE_CODES = {
    "RGGB": True,
    "BGGR": True,
    "GRBG": True,
    "GBRG": True,
}


# ============================================================
# Configuração
# ============================================================


@dataclass(frozen=True)
class AlignConfig:
    base_dir: Path
    output_dir: Path

    # Debayer
    debayer_pattern: str
    debayer_method: str

    # Alignment
    interpolation: str

    # Advanced Chromatic Registration (Níveis 1 e 2)
    rgb_registration: bool

    # Storage / execution
    overwrite: bool
    dry_run: bool
    keep_header: bool
    delete_intermediates: bool
    compress_output: bool
    engine_profile: str = "Stable"
    warp_engine: str | None = None
    quality_gate: bool = False
    quality_min_confidence: float = 0.15
    quality_max_rms: float = 3.0
    quality_max_shift: float = 1.5
    quality_min_coverage: float = 0.20
    max_in_flight: int = 0
    workers: int = 0
    memory_budget_mb: int = 0
    # Internal diagnostic override.  Zero keeps the conservative single
    # writer; production callers can opt into a bounded pool after measuring
    # compression throughput on their storage.
    writer_workers: int = 0
    # ``translation`` preserves the historical correction. ``similarity``
    # additionally fits a small scale/rotation term from stellar centroids,
    # while ``hybrid`` uses the more conservative translation for R and the
    # similarity fit for B.  Both alternatives are opt-in and leave the
    # default Stable output unchanged.
    rgb_registration_mode: str = "translation"


def _build_align_config(
    base_dir: Path,
    output_dir: Path,
    config_dict: dict,
) -> AlignConfig:
    return AlignConfig(
        base_dir=base_dir,
        output_dir=output_dir,
        debayer_pattern=config_dict.get("debayer_pattern", "Auto"),
        debayer_method=config_dict.get("debayer_method", "Bilinear"),
        interpolation=config_dict.get("interpolation", "Lanczos"),
        rgb_registration=bool(config_dict.get("rgb_registration", True)),
        rgb_registration_mode=(
            str(config_dict.get("rgb_registration_mode", "translation")).strip().lower()
            if str(config_dict.get("rgb_registration_mode", "translation")).strip().lower()
            in {"translation", "similarity", "hybrid"}
            else "translation"
        ),
        overwrite=bool(config_dict.get("overwrite", False)),
        dry_run=bool(config_dict.get("dry_run", False)),
        keep_header=bool(config_dict.get("keep_header", True)),
        delete_intermediates=bool(config_dict.get("delete_intermediates", False)),
        compress_output=bool(config_dict.get("compress_output", True)),
        engine_profile=str(config_dict.get("engine_profile", "Stable")),
        warp_engine=config_dict.get("warp_engine") or None,
        quality_gate=bool(config_dict.get("quality_gate", False)),
        quality_min_confidence=float(config_dict.get("quality_min_confidence", 0.15)),
        quality_max_rms=float(config_dict.get("quality_max_rms", 3.0)),
        quality_max_shift=float(config_dict.get("quality_max_shift", 1.5)),
        quality_min_coverage=float(config_dict.get("quality_min_coverage", 0.20)),
        max_in_flight=max(0, int(config_dict.get("max_in_flight", 0))),
        workers=max(0, int(config_dict.get("workers", 0))),
        memory_budget_mb=max(0, int(config_dict.get("memory_budget_mb", 0))),
        writer_workers=max(0, int(config_dict.get("writer_workers", 0))),
    )


# ============================================================
# Workers
# ============================================================


def get_optimal_worker_count() -> int:
    try:
        cpu_count = (
            getattr(
                os,
                "process_cpu_count",
                os.cpu_count,
            )()
            or 1
        )
    except Exception:
        cpu_count = os.cpu_count() or 1

    try:
        import psutil

        available_ram = psutil.virtual_memory().available

        # Limita aproximadamente a 800 MB de RAM por worker.
        ram_workers = max(
            1,
            int(available_ram // (800 * 1024 * 1024)),
        )

    except ImportError:
        ram_workers = cpu_count

    return max(
        1,
        min(
            8,
            physical_core_count(),
            cpu_count,
            ram_workers,
        ),
    )


# ============================================================
# Descoberta de arquivos
# ============================================================


def find_batch_folders(
    base_dir: Path,
) -> list[Path]:
    return sorted(
        (d for d in base_dir.iterdir() if (d.is_dir() and "batch" in d.name.lower())),
        key=lambda p: p.name.lower(),
    )


# ============================================================
# JSON / Flow
# ============================================================


def load_json(
    filepath: Path,
) -> dict:
    with open(
        filepath,
        "r",
        encoding="utf-8",
    ) as f:
        return json.load(f)


def _flow_manifest(base_dir: Path) -> dict | None:
    """Read the active Flow revision manifest when one is present."""
    manifest_path = Path(base_dir) / "flow_revision.json"
    if not manifest_path.exists():
        return None
    try:
        value = load_json(manifest_path)
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _manifest_path(base_dir: Path, value: str | None) -> Path | None:
    if not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else Path(base_dir) / path


def load_local_flow(
    batch_dir: Path,
) -> dict | None:
    batch_dir = Path(batch_dir)
    root = batch_dir.parent
    manifest = _flow_manifest(root)
    flow_path = None
    if manifest:
        local_flows = manifest.get("local_flows", {})
        if isinstance(local_flows, dict):
            flow_path = _manifest_path(root, local_flows.get(batch_dir.name))
        if flow_path is None and manifest.get("batch") == batch_dir.name:
            flow_path = _manifest_path(root, manifest.get("local_flow"))
    if flow_path is None:
        # A batch-local manifest is supported for projects that do not have a
        # global Flow revision yet.
        local_manifest = _flow_manifest(batch_dir)
        if local_manifest:
            flow_path = _manifest_path(batch_dir, local_manifest.get("local_flow"))
    if flow_path is None:
        flow_path = batch_dir / "flow_local.json"

    if not flow_path.exists():
        return None

    return load_json(flow_path)


def load_global_flow(
    base_dir: Path,
) -> dict | None:
    base_dir = Path(base_dir)
    manifest = _flow_manifest(base_dir)
    flow_path = _manifest_path(base_dir, manifest.get("global_flow")) if manifest else None
    if flow_path is None:
        flow_path = base_dir / "global_flow.json"

    if not flow_path.exists():
        return None

    return load_json(flow_path)


def compute_final_matrix(
    local_matrix: list,
    global_matrix: list,
) -> np.ndarray:
    local = np.asarray(
        local_matrix,
        dtype=np.float64,
    )

    offset = np.asarray(
        global_matrix,
        dtype=np.float64,
    )

    return offset @ local


# ============================================================
# FITS
# ============================================================


def _load_fits_data_and_optional_masks(
    filepath: Path,
    include_masks: bool = False,
) -> tuple[np.ndarray, fits.Header, np.ndarray | None, np.ndarray | None]:
    """Load an alignment frame, optionally consuming masks in the same open.

    Alignment used to call ``load_fits_data`` and ``load_fits_masks``
    sequentially, reopening every source FITS.  Keeping the optional path here
    preserves the lightweight pixel-only API used by Flow and previews while
    giving the hot alignment worker one header/data/mask read.
    """

    from app.infrastructure.fits_masks import read_science_masks_from_hdul

    with fits.open(
        filepath,
        memmap=False,
        ignore_missing_end=True,
    ) as hdul:
        for hdu in hdul:
            if (hdu.is_image and hdu.name not in {"VALID_MASK", "SAT_MASK", "DISAGREE", "HDR_META"}
                    and hdu.shape is not None and len(hdu.shape) in (2,3)):
                header = hdu.header.copy(strip=False)

                data = np.asarray(
                    hdu.data,
                    dtype=np.float32,
                )
                if data.ndim == 3:
                    if data.shape[0] in (3,4):
                        spatial_shape = data.shape[1:]
                        data = np.moveaxis(data,0,-1)
                    elif data.shape[-1] in (3,4):
                        spatial_shape = data.shape[:2]
                    else:
                        raise ValueError(f"Unsupported RGB geometry: {data.shape}")
                else:
                    spatial_shape = data.shape

                if include_masks:
                    source_valid, source_sat = read_science_masks_from_hdul(
                        hdul, spatial_shape
                    )
                else:
                    source_valid = source_sat = None

                return (
                    data,
                    header,
                    source_valid,
                    source_sat,
                )

    raise ValueError(f"Imagem 2D não encontrada em {filepath.name}")


def load_fits_data(
    filepath: Path,
) -> tuple[np.ndarray, fits.Header]:
    data, header, _valid, _saturated = _load_fits_data_and_optional_masks(
        filepath, include_masks=False
    )
    return data, header


def load_fits_data_and_masks(
    filepath: Path,
) -> tuple[np.ndarray, fits.Header, np.ndarray, np.ndarray]:
    data, header, valid, saturated = _load_fits_data_and_optional_masks(
        filepath, include_masks=True
    )
    assert valid is not None and saturated is not None
    return data, header, valid, saturated


def load_fits_masks(filepath: Path, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Read optional validity/saturation masks without FITS memmapping."""
    from app.infrastructure.fits_masks import read_science_masks
    return read_science_masks(filepath, shape[:2])


# ============================================================
# Bayer / Header
# ============================================================


def get_bayer_pattern(
    header: fits.Header,
) -> str | None:

    for key in (
        "BAYERPAT",
        "BAYERPATTERN",
        "COLORTYP",
    ):
        if key not in header:
            continue

        value = str(header[key]).strip().upper().strip("'")

        if value in BAYER_BASE_CODES:
            return value

    return None


# ============================================================
# Debayer
# ============================================================


def process_in_memory_debayer(
    data: np.ndarray,
    header: fits.Header,
    pattern: str | None,
    method: str = "VNG",
) -> tuple[np.ndarray, fits.Header]:

    if data.ndim == 3 or not pattern:
        return data, header

    pattern = str(pattern).upper()
    valid_patterns = ["RGGB", "BGGR", "GRBG", "GBRG"]

    if pattern not in valid_patterns:
        raise ValueError(f"Padrão Bayer inválido: {pattern}")

    # Normalize 16-bit data to [0.0, 1.0] float32 for colour-demosaicing algorithms
    data_normalized = np.clip(data, 0, 65535).astype(np.float32) / 65535.0

    # Route to the requested demosaicing algorithm natively
    if method == "VNG":
        rgb_float = colour_demosaicing.demosaicing_CFA_Bayer_Malvar2004(
            data_normalized, pattern
        )
    elif method == "Bilinear":
        rgb_float = colour_demosaicing.demosaicing_CFA_Bayer_bilinear(
            data_normalized, pattern
        )
    elif method == "Menon2007":
        # DDFAPD (Edge-Aware alternative that is excellent for astrophotography)
        rgb_float = colour_demosaicing.demosaicing_CFA_Bayer_Menon2007(
            data_normalized, pattern
        )
    else:
        raise ValueError(f"Método de debayer não suportado: {method}")

    # Scale back to 16-bit range as float32 for downstream compatibility
    rgb_data = (np.clip(rgb_float, 0.0, 1.0) * 65535.0).astype(np.float32)

    # --------------------------------------------------------
    # Update Header
    # --------------------------------------------------------
    for key in [
        "BAYERPAT",
        "BAYERPATTERN",
        "COLORTYP",
        "BZERO",
        "BSCALE",
    ]:
        header.remove(key, ignore_missing=True)

    header["DEBAYER"] = pattern
    header["DEBMETHOD"] = method
    header["CTYPE3"] = "RGB"

    return rgb_data, header


# ============================================================
# Warping (Scikit-Image Refactored)
# ============================================================


def _warp_affine_cpu(
    data: np.ndarray,
    matrix: np.ndarray,
    interpolation_mode: str,
    engine_profile: str = "Stable",
    warp_engine: str | None = None,
) -> np.ndarray:
    """Apply one affine transform to a mono or interleaved RGB image.

    OpenCV processes an HxWxC image in one native call, replacing the former
    Python loop that invoked scikit-image once for every colour channel.  The
    matrix maps source coordinates to output coordinates, matching
    ``warp(..., tform.inverse)`` used previously.
    """
    register_align_engines()
    profile = EngineProfile.coerce(engine_profile)
    engine_id = warp_engine or (
        "opencv-fast" if profile is EngineProfile.FAST else "opencv-stable"
    )
    engine = registry.resolve("align.warp", engine_id, profile)
    return engine(data, matrix, interpolation_mode)


def _weighted_channel_centroid(
    image: np.ndarray,
    x: int,
    y: int,
    radius: int = 4,
) -> tuple[float, float, float] | None:
    """Return a bounded, background-subtracted centroid around one star."""

    height, width = image.shape
    if (
        x - radius < 0
        or y - radius < 0
        or x + radius >= width
        or y + radius >= height
    ):
        return None
    cutout = np.asarray(image[y - radius : y + radius + 1, x - radius : x + radius + 1], dtype=np.float32)
    finite = np.isfinite(cutout)
    if int(finite.sum()) < max(9, cutout.size // 2):
        return None
    border = np.concatenate((cutout[0], cutout[-1], cutout[:, 0], cutout[:, -1]))
    border = border[np.isfinite(border)]
    if border.size < 4:
        return None
    background = float(np.median(border))
    weights = np.where(finite, cutout - np.float32(background), 0.0)
    weights = np.maximum(weights, 0.0)
    total = float(np.sum(weights, dtype=np.float64))
    if not np.isfinite(total) or total <= 0.0:
        return None
    yy, xx = np.indices(weights.shape, dtype=np.float32)
    cx = float(np.sum(weights * xx, dtype=np.float64) / total) + x - radius
    cy = float(np.sum(weights * yy, dtype=np.float64) / total) + y - radius
    peak = float(np.max(weights))
    # Hot pixels and clipped single samples do not provide a stable color
    # centroid.  The shape detector already applies the same conservative
    # principle to Flow metrics; keep the RGB correction bounded as well.
    if peak <= 0.0 or peak / total > 0.88:
        return None
    return cx, cy, total


def _estimate_rgb_similarity_warp(
    reference: np.ndarray,
    channel: np.ndarray,
    *,
    max_shift: float,
    max_points: int = 96,
) -> tuple[np.ndarray, float] | None:
    """Fit a small channel-to-green similarity transform from star centroids.

    A single phase-correlation translation cannot model the scale component
    of longitudinal chromatic aberration.  This estimator uses the green
    channel as the optical anchor, measures the same local stellar centroids
    in the candidate channel, and fits a bounded similarity transform with
    RANSAC.  It is deliberately opt-in so existing Stable products remain
    unchanged unless the user selects the stronger correction.
    """

    ref = np.asarray(reference, dtype=np.float32)
    candidate = np.asarray(channel, dtype=np.float32)
    if ref.ndim != 2 or candidate.shape != ref.shape or ref.size == 0:
        return None
    finite = np.isfinite(ref) & np.isfinite(candidate)
    if int(finite.sum()) < 64:
        return None

    # High-pass the reference so nebulosity and gradients do not become
    # corners.  The operation remains in native OpenCV code and is bounded by
    # ``max_points`` for every frame.
    ref_work = np.where(finite, ref, 0.0).astype(np.float32, copy=False)
    # Feature discovery does not need native sensor resolution.  Keeping the
    # centroid measurement on the original arrays while downsampling this
    # bounded probe cuts the per-frame cost on an 8 MP Uranus-C capture by an
    # order of magnitude and does not change the fitted full-resolution warp.
    probe_scale = max(1, int(np.ceil(max(ref.shape) / 1024.0)))
    if probe_scale > 1:
        probe = cv2.resize(
            ref_work,
            (max(16, ref.shape[1] // probe_scale), max(16, ref.shape[0] // probe_scale)),
            interpolation=cv2.INTER_AREA,
        )
    else:
        probe = ref_work
    blur = cv2.GaussianBlur(probe, (0, 0), 1.0)
    high = np.maximum(probe - blur, 0.0)
    probe_finite = np.isfinite(probe)
    positive = high[probe_finite & (high > 0.0)]
    if positive.size < 16:
        return None
    threshold = float(np.percentile(positive, 99.2))
    if not np.isfinite(threshold) or threshold <= 0.0:
        return None
    response = np.clip(high / np.float32(threshold), 0.0, 1.0)
    points = cv2.goodFeaturesToTrack(
        response,
        maxCorners=int(max_points),
        qualityLevel=0.03,
        minDistance=max(3.0, 18.0 / float(probe_scale)),
        blockSize=7,
        useHarrisDetector=False,
    )
    if points is None or len(points) < 6:
        return None

    reference_points: list[tuple[float, float]] = []
    channel_points: list[tuple[float, float]] = []
    for point in points.reshape(-1, 2):
        x = int(np.rint(float(point[0]) * probe_scale))
        y = int(np.rint(float(point[1]) * probe_scale))
        ref_centroid = _weighted_channel_centroid(ref, x, y)
        channel_centroid = _weighted_channel_centroid(candidate, x, y)
        if ref_centroid is None or channel_centroid is None:
            continue
        rx, ry, ref_flux = ref_centroid
        cx, cy, channel_flux = channel_centroid
        if ref_flux <= 0.0 or channel_flux <= 0.0:
            continue
        displacement = float(np.hypot(cx - rx, cy - ry))
        # Permit a little more than the eventual acceptance threshold for the
        # RANSAC fit, but reject unrelated peaks early.
        if not np.isfinite(displacement) or displacement > max(4.0, max_shift * 3.0):
            continue
        reference_points.append((rx, ry))
        channel_points.append((cx, cy))

    if len(reference_points) < 6:
        return None
    source = np.asarray(channel_points, dtype=np.float32)
    target = np.asarray(reference_points, dtype=np.float32)
    matrix, inliers = cv2.estimateAffinePartial2D(
        source,
        target,
        method=cv2.RANSAC,
        ransacReprojThreshold=1.5,
        maxIters=1000,
        confidence=0.995,
        refineIters=10,
    )
    if matrix is None or inliers is None or not np.isfinite(matrix).all():
        return None
    inlier_mask = inliers.ravel().astype(bool)
    inlier_count = int(inlier_mask.sum())
    if inlier_count < 5:
        return None
    transformed = cv2.transform(source.reshape(-1, 1, 2), matrix).reshape(-1, 2)
    residual = np.linalg.norm(transformed - target, axis=1)
    confidence = float(np.clip(inlier_count / max(1, len(source)), 0.0, 1.0))
    if not np.isfinite(residual[inlier_mask]).all():
        return None
    if float(np.median(residual[inlier_mask])) > 1.25:
        return None
    center = np.asarray([(ref.shape[1] - 1) * 0.5, (ref.shape[0] - 1) * 0.5], dtype=np.float32)
    center_target = cv2.transform(center.reshape(1, 1, 2), matrix).reshape(2)
    if float(np.linalg.norm(center_target - center)) > max(0.5, max_shift):
        return None
    # The center check alone cannot reject a rotation around the optical
    # center: a pathological fit could leave the center fixed while moving
    # stars at the field edge by tens of pixels.  Bound the actual footprint
    # displacement at every corner as well, keeping the opt-in correction
    # genuinely sub-pixel/small on the 5800X capture.
    corners = np.asarray(
        [[0.0, 0.0], [ref.shape[1] - 1.0, 0.0],
         [0.0, ref.shape[0] - 1.0],
         [ref.shape[1] - 1.0, ref.shape[0] - 1.0]],
        dtype=np.float32,
    )
    transformed_corners = cv2.transform(corners.reshape(-1, 1, 2), matrix).reshape(-1, 2)
    if not np.isfinite(transformed_corners).all():
        return None
    if float(np.max(np.linalg.norm(transformed_corners - corners, axis=1))) > max_shift:
        return None
    return np.asarray(matrix, dtype=np.float32), confidence


def warp_frame(
    data: np.ndarray,
    final_matrix: np.ndarray,
    interpolation_mode: str,
    rgb_registration: bool = False,
    engine_profile: str = "Stable",
    warp_engine: str | None = None,
    diagnostics: dict | None = None,
    rgb_max_shift: float = 2.0,
    rgb_registration_mode: str = "translation",
) -> np.ndarray:
    """
    Aplica a transformação afim global e, opcionalmente, executa
    um micro-registro sub-pixel dos canais R e B usando o canal G como âncora
    (correção de dispersão atmosférica / color fringing).
    """
    matrix_3x3 = np.eye(3, dtype=np.float64)
    matrix_3x3[:2, :] = final_matrix[:2, :].astype(np.float64)
    tform = AffineTransform(matrix=matrix_3x3)
    order_map = {
        "nearest": 0,
        "bilinear": 1,
        "bicubic": 3,
        "lanczos": 3,
    }
    order = order_map.get(interpolation_mode, 3)
    profile = EngineProfile.coerce(engine_profile)
    has_native_path = (
        interpolation_mode in CV2_INTERPOLATION_MODES or profile is EngineProfile.FAST
    )

    if data.ndim == 3:
        if has_native_path:
            # OpenCV handles interleaved channels in one native call.
            output = _warp_affine_cpu(
                data, final_matrix, interpolation_mode, engine_profile, warp_engine
            )
        else:
            # OpenCV's cubic kernel is not numerically compatible with the
            # prior skimage order=3 kernel, so retain it for these modes.
            output = np.empty(data.shape, dtype=np.float32)
            for i in range(data.shape[2]):
                output[:, :, i] = warp(
                    data[:, :, i],
                    tform.inverse,
                    order=order,
                    mode="constant",
                    cval=0.0,
                    preserve_range=True,
                )

        # 2. Nível 1: Micro-Registro RGB pós-warp
        if rgb_registration and data.shape[2] >= 3:
            registration_mode = str(rgb_registration_mode or "translation").strip().lower()
            if registration_mode not in {"translation", "similarity", "hybrid"}:
                registration_mode = "translation"
            if profile is EngineProfile.FAST:
                reference = output[:, :, 1]
                height, width = reference.shape
                for channel in (0, 2):
                    if registration_mode == "similarity" or (
                        registration_mode == "hybrid" and channel == 2
                    ):
                        similarity = _estimate_rgb_similarity_warp(
                            reference,
                            output[:, :, channel],
                            max_shift=rgb_max_shift,
                        )
                        if similarity is not None:
                            matrix, confidence = similarity
                            tx = float(matrix[0, 2])
                            ty = float(matrix[1, 2])
                            if diagnostics is not None:
                                diagnostics.setdefault("rgb_models", {})[channel] = {
                                    "matrix": np.asarray(matrix, dtype=np.float64).tolist(),
                                    "confidence": float(confidence),
                                }
                                diagnostics.setdefault("rgb_shifts", {})[channel] = (
                                    tx,
                                    ty,
                                    float(confidence),
                                )
                            output[:, :, channel] = cv2.warpAffine(
                                output[:, :, channel],
                                np.asarray(matrix, dtype=np.float32),
                                (width, height),
                                flags=cv2.INTER_LINEAR,
                                borderMode=cv2.BORDER_CONSTANT,
                                borderValue=0.0,
                            )
                            continue
                    dx, dy, confidence = rgb_registration_shift(reference, output[:, :, channel])
                    if confidence < 0.15 or not np.isfinite(dx + dy) or np.hypot(dx, dy) > rgb_max_shift:
                        continue
                    if diagnostics is not None:
                        diagnostics.setdefault("rgb_shifts", {})[channel] = (dx, dy, confidence)
                    matrix = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]])
                    output[:, :, channel] = cv2.warpAffine(
                        output[:, :, channel], matrix, (width, height),
                        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0,
                    )
                return np.asarray(output, dtype=np.float32)
            from skimage.registration import phase_cross_correlation

            # O Canal Verde (índice 1) é nossa referência fixa e opticamente mais nítida
            ref_channel = output[:, :, 1]

            for c in [0, 2]:  # Processa o Vermelho (0) e o Azul (2)
                if registration_mode == "similarity" or (
                    registration_mode == "hybrid" and c == 2
                ):
                    similarity = _estimate_rgb_similarity_warp(
                        ref_channel,
                        output[:, :, c],
                        max_shift=rgb_max_shift,
                    )
                    if similarity is not None:
                        matrix, confidence = similarity
                        tx = float(matrix[0, 2])
                        ty = float(matrix[1, 2])
                        if diagnostics is not None:
                            diagnostics.setdefault("rgb_models", {})[c] = {
                                "matrix": np.asarray(matrix, dtype=np.float64).tolist(),
                                "confidence": float(confidence),
                            }
                            diagnostics.setdefault("rgb_shifts", {})[c] = (
                                tx,
                                ty,
                                float(confidence),
                            )
                        output[:, :, c] = cv2.warpAffine(
                            output[:, :, c],
                            np.asarray(matrix, dtype=np.float32),
                            (output.shape[1], output.shape[0]),
                            flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT,
                            borderValue=0.0,
                        )
                        continue
                # Calcula o desvio sub-pixel exato do canal em relação ao verde
                shift_vector, error, diffphase = phase_cross_correlation(
                    ref_channel,
                    output[:, :, c],
                    upsample_factor=10,  # Precisão de 0.1 pixel
                    normalization=None,
                )

                # shift_vector retorna (y, x). O AffineTransform espera translação em (x, y)
                dx, dy = float(shift_vector[1]), float(shift_vector[0])
                _, _, confidence = rgb_registration_shift(ref_channel, output[:, :, c])
                if confidence < 0.15 or not np.isfinite(dx + dy) or np.hypot(dx, dy) > rgb_max_shift:
                    continue
                micro_tform = AffineTransform(translation=(dx, dy))

                # Realinha o canal com o desvio cromático corrigido
                if diagnostics is not None:
                    diagnostics.setdefault("rgb_shifts", {})[c] = (
                        dx, dy, confidence)
                output[:, :, c] = warp(
                    output[:, :, c],
                    micro_tform.inverse,
                    order=order,
                    mode="constant",
                    cval=0.0,
                    preserve_range=True,
                )

        return output
    else:
        if has_native_path:
            return _warp_affine_cpu(
                data, final_matrix, interpolation_mode, engine_profile, warp_engine
            )
        return warp(
            data,
            tform.inverse,
            order=order,
            mode="constant",
            cval=0.0,
            preserve_range=True,
        ).astype(np.float32)


def estimate_alignment_quality(reference: np.ndarray, aligned: np.ndarray,
                               valid_mask: np.ndarray | None = None,
                               max_samples: int = 200_000) -> dict[str, float]:
    """Return inexpensive residual shift, RMS and coverage diagnostics."""
    ref = np.asarray(reference, dtype=np.float32)
    cur = np.asarray(aligned, dtype=np.float32)
    if ref.ndim == 3:
        ref = np.mean(ref, axis=2)
    if cur.ndim == 3:
        cur = np.mean(cur, axis=2)
    finite = np.isfinite(ref) & np.isfinite(cur)
    if valid_mask is not None:
        finite &= np.asarray(valid_mask, dtype=bool)
    coverage = float(np.mean(finite)) if finite.size else 0.0
    if not np.any(finite):
        return {"shift_x": 0.0, "shift_y": 0.0, "rms": float("inf"), "confidence": 0.0, "coverage": coverage}
    # Bound both the residual calculation and phase correlation memory/time.
    stride = max(1, int(np.ceil(np.sqrt(ref.size / max(1, max_samples)))))
    if stride > 1:
        ref = ref[::stride, ::stride]; cur = cur[::stride, ::stride]
        finite = finite[::stride, ::stride]
    a, b = ref.copy(), cur.copy()
    # Remove photometric offset and normalize scale before residual evaluation.
    av = a[finite]; bv = b[finite]
    if av.size:
        asd = float(np.std(av)); bsd = float(np.std(bv))
        if bsd > 1e-6:
            b = (b - float(np.mean(bv))) * (asd / bsd) + float(np.mean(av))
    a[~finite] = 0.0; b[~finite] = 0.0
    shift, response = cv2.phaseCorrelate(a, b)
    diff = (a - b)[finite]
    if diff.size > max_samples:
        diff = diff[::max(1, diff.size // max_samples)]
    scale = float(np.std(a[finite]) + 1e-6)
    return {"shift_x": float(shift[0]) * stride, "shift_y": float(shift[1]) * stride,
            "rms": float(np.sqrt(np.mean(diff * diff)) / scale),
            "confidence": float(max(0.0, min(1.0, response))), "coverage": coverage}


def rgb_registration_shift(reference: np.ndarray, channel: np.ndarray) -> tuple[float, float, float]:
    """Estimate channel displacement and return the correction (x, y, confidence)."""
    shift, response = cv2.phaseCorrelate(np.asarray(reference, np.float32), np.asarray(channel, np.float32))
    return float(-shift[0]), float(-shift[1]), float(max(0.0, min(1.0, response)))


def prepare_reference_preview(filepath: Path, matrix: np.ndarray,
                              interpolation_mode: str = "bilinear",
                              max_size: int = 512) -> tuple[np.ndarray, np.ndarray]:
    """Load and warp an alignment anchor once for bounded quality checks."""
    data, header, source_valid, source_sat = load_fits_data_and_masks(filepath)
    finite_source = np.isfinite(data)
    source_valid &= np.all(finite_source, axis=2) if data.ndim == 3 else finite_source
    saturation = header.get("SATURATE", header.get("SATLEVEL"))
    if saturation is not None:
        clipped = data >= float(saturation)
        source_sat |= np.any(clipped, axis=2) if data.ndim == 3 else clipped
    pattern = get_bayer_pattern(header)
    data, _ = process_in_memory_debayer(data, header, pattern, "Bilinear")
    warped = warp_frame(data, matrix, interpolation_mode, rgb_registration=False)
    mask = generate_valid_mask(data.shape, matrix).astype(bool)
    source_valid = _warp_affine_cpu(source_valid.astype(np.float32), matrix, "nearest") > 0.5
    source_sat = _warp_affine_cpu(source_sat.astype(np.float32), matrix, "nearest") > 0.5
    mask &= source_valid
    mask &= np.isfinite(np.mean(warped, axis=2) if warped.ndim == 3 else warped)
    mask &= ~source_sat
    stride = max(1, int(np.ceil(max(data.shape[:2]) / max_size)))
    if stride > 1:
        warped = warped[::stride, ::stride]
        mask = mask[::stride, ::stride]
    return np.asarray(warped, dtype=np.float32), np.asarray(mask, dtype=np.uint8)


class ReferencePreviewCache:
    """Thread-safe, bounded, demand-driven quality preview cache.

    The cache stores only preview arrays. FITS decoding and warping happen on
    the worker that first requests a graph neighbour, so large jobs do not
    eagerly retain every batch reference.
    """

    def __init__(self, targets: dict[str, tuple[Path, str, dict, np.ndarray, str]], max_size: int = 8):
        self.targets = dict(targets)
        self.max_size = max(1, int(max_size))
        self._values: OrderedDict[str, tuple[np.ndarray, np.ndarray]] = OrderedDict()
        self._inflight: dict[str, threading.Event] = {}
        self._lock = threading.RLock()

    def keys(self):
        return self.targets.keys()

    def __contains__(self, label: str) -> bool:
        """Support the mapping protocol used by the alignment worker."""
        return label in self.targets

    def get(self, label: str):
        with self._lock:
            cached = self._values.get(label)
            if cached is not None:
                self._values.move_to_end(label)
                return cached
            target = self.targets.get(label)
            pending = self._inflight.get(label)
            if pending is None:
                pending = threading.Event()
                self._inflight[label] = pending
                owner = True
            else:
                owner = False
        if target is None:
            if owner:
                with self._lock:
                    self._inflight.pop(label, None)
                    pending.set()
            raise KeyError(label)
        if not owner:
            # Another alignment worker is already decoding/warping this
            # preview. Waiting avoids duplicate full FITS reads and warps.
            pending.wait()
            return self.get(label)
        batch_folder, frame_name, _info, matrix, interpolation_mode = target
        try:
            value = prepare_reference_preview(
                batch_folder / frame_name,
                matrix,
                interpolation_mode,
                max_size=512,
            )
            with self._lock:
                self._values[label] = value
                self._values.move_to_end(label)
                while len(self._values) > self.max_size:
                    self._values.popitem(last=False)
            return value
        finally:
            with self._lock:
                self._inflight.pop(label, None)
                pending.set()


def _alignment_transform_revision(
    frame_info: dict,
    interpolation_mode: str | None = None,
    rgb_registration_mode: str | None = None,
) -> str:
    geometry_revision = frame_info.get("_geometry_revision")
    if geometry_revision:
        source = str(geometry_revision)
    else:
        source = (
            f"{frame_info.get('_local_transform_revision', '')}|"
            f"{frame_info.get('_global_transform_revision', '')}"
        )
    return hashlib.sha256(
        f"{source}|{interpolation_mode or ''}|{rgb_registration_mode or 'translation'}".encode("utf-8")
    ).hexdigest()[:16]


def generate_valid_mask(
    shape: tuple,
    final_matrix: np.ndarray,
) -> np.ndarray:
    """
    Gera a máscara de pixels válidos com o mesmo caminho afim nativo.
    """
    height = shape[0]
    width = shape[1]

    mask = np.ones((height, width), dtype=np.float32)
    warped_mask = _warp_affine_cpu(mask, final_matrix, "nearest")

    return (warped_mask > 0.5).astype(np.uint8)


# ============================================================
# FITS output
# ============================================================


def _clean_structural_keywords(header: fits.Header | None) -> fits.Header | None:
    """Remove cartas estruturais e de compressão para evitar corrupção ao reescrever o FITS."""
    if header is None:
        return None

    output_header = header.copy()
    structural = {
        "XTENSION",
        "BITPIX",
        "NAXIS",
        "NAXIS1",
        "NAXIS2",
        "NAXIS3",
        "PCOUNT",
        "GCOUNT",
        "THEAP",
        "BSCALE",
        "BZERO",
        "BLANK",
    }
    for keyword in list(output_header):
        if keyword in structural or keyword.startswith("Z"):
            del output_header[keyword]
    return output_header


def save_aligned_fits(
    data: np.ndarray,
    mask: np.ndarray,
    header: fits.Header | None,
    output_path: Path,
    compress_output: bool = True,
    metadata: dict[str, float | str] | None = None,
    cancel_event: threading.Event | None = None,
    sat_mask: np.ndarray | None = None,
) -> None:
    """Write aligned science data and its valid-pixel mask in the chosen layout."""

    clean_header = _clean_structural_keywords(header)

    if clean_header is not None:
        clean_header["BITPIX"] = 16
        clean_header["BZERO"] = 32768
        clean_header["BSCALE"] = 1
        if metadata:
            for key, value in metadata.items():
                card = str(key).upper()[:8]
                try:
                    clean_header[card] = value
                except (ValueError, KeyError):
                    pass

    data_uint16 = np.clip(
        data,
        0,
        65535,
    ).astype(np.uint16)

    # FITS RGB:
    # OpenCV/NumPy -> H, W, C
    # FITS -> C, H, W
    if data_uint16.ndim == 3:
        data_uint16 = np.moveaxis(
            data_uint16,
            -1,
            0,
        )

    if compress_output:
        hdu_data = fits.CompImageHDU(
            data=data_uint16,
            header=clean_header,
            compression_type="RICE_1",
        )
        hdu_mask = fits.CompImageHDU(
            data=np.asarray(mask, dtype=np.uint8),
            name="VALID_MASK",
            compression_type="PLIO_1",
        )
        hdu_sat = fits.CompImageHDU(data=np.asarray(sat_mask, dtype=np.uint8), name="SAT_MASK",
                                    compression_type="PLIO_1") if sat_mask is not None else None
    else:
        hdu_data = fits.ImageHDU(
            data=data_uint16,
            header=clean_header,
        )
        hdu_mask = fits.ImageHDU(
            data=np.asarray(mask, dtype=np.uint8),
            name="VALID_MASK",
        )
        hdu_sat = fits.ImageHDU(data=np.asarray(sat_mask, dtype=np.uint8), name="SAT_MASK") if sat_mask is not None else None

    hdul = fits.HDUList([fits.PrimaryHDU(), hdu_data, hdu_mask] + ([hdu_sat] if hdu_sat is not None else []))

    fd, temporary_name = tempfile.mkstemp(prefix=f".{output_path.name}.", suffix=".tmp",
                                          dir=str(output_path.parent))
    os.close(fd)
    temporary_path = Path(temporary_name)
    try:
        if cancel_event is not None and cancel_event.is_set():
            raise InterruptedError("Alignment cancelled before writing")
        hdul.writeto(str(temporary_path), overwrite=True, output_verify="ignore")
        if cancel_event is not None and cancel_event.is_set():
            raise InterruptedError("Alignment cancelled before commit")
        os.replace(temporary_path, output_path)
    finally:
        hdul.close()
        try:
            temporary_path.unlink(missing_ok=True)
        except OSError:
            pass


# ============================================================
# Frame individual
# ============================================================


def _process_single_alignment(
    frame_name: str,
    frame_info: dict,
    batch_dir: Path,
    output_dir: Path,
    global_matrix: list,
    interpolation_mode: str,
    config: AlignConfig,
    cancel_event: threading.Event | None = None,
    reference_preview: tuple[np.ndarray, np.ndarray] | None = None,
    writer_executor: ThreadPoolExecutor | None = None,
    reference_previews: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
    reference_preview_sources: dict[str, tuple[str, str]] | None = None,
) -> tuple[str, str | None]:

    try:
        if cancel_event is not None and cancel_event.is_set():
            return frame_name, "CANCELLED"
        # ----------------------------------------------------
        # Validação do arquivo
        # ----------------------------------------------------

        filepath = batch_dir / frame_name

        if not filepath.exists():
            return (
                frame_name,
                (f"Aviso: arquivo original não encontrado: {filepath}"),
            )

        output_path = output_dir / frame_name

        if output_path.exists() and not config.overwrite:
            sidecar_path = output_path.with_suffix(output_path.suffix + ".align.json")
            try:
                if sidecar_path.exists():
                    previous_quality = json.loads(sidecar_path.read_text(encoding="utf-8"))
                    previous_revision = previous_quality.get("alignment_revision")
                    current_revision = _alignment_transform_revision(
                        frame_info,
                        interpolation_mode,
                        config.rgb_registration_mode,
                    )
                    if previous_revision and previous_revision != current_revision:
                        return frame_name, (
                            "STALE_OUTPUT_REQUIRES_REGENERATION "
                            f"(revision {previous_revision} -> {current_revision})"
                        )
            except (OSError, ValueError, TypeError, json.JSONDecodeError):
                pass
            return frame_name, f"ERRO: destino já existe, arquivo ignorado: {output_path}"

        # ----------------------------------------------------
        # Matriz
        # ----------------------------------------------------

        if "matrix" not in frame_info:
            return (
                frame_name,
                "ERRO: frame sem matriz de alinhamento.",
            )

        final_matrix = compute_final_matrix(
            frame_info["matrix"],
            global_matrix,
        )

        if not np.all(np.isfinite(final_matrix)):
            return (
                frame_name,
                "ERRO: matriz final contém valores inválidos.",
            )

        # ----------------------------------------------------
        # Dry Run
        # ----------------------------------------------------

        if config.dry_run:
            return (
                frame_name,
                None,
            )

        # ----------------------------------------------------
        # Leitura
        # ----------------------------------------------------

        raw_data, raw_header, source_valid, source_sat = load_fits_data_and_masks(filepath)
        finite_source = np.isfinite(raw_data)
        source_valid &= np.all(finite_source, axis=2) if raw_data.ndim == 3 else finite_source
        saturation = raw_header.get("SATURATE", raw_header.get("SATLEVEL"))
        if saturation is not None:
            clipped = raw_data >= float(saturation)
            source_sat |= np.any(clipped, axis=2) if raw_data.ndim == 3 else clipped

        # ----------------------------------------------------
        # Debayer
        # ----------------------------------------------------

        if config.debayer_pattern == "Nenhum":
            pattern = None

        elif config.debayer_pattern == "Auto":
            pattern = get_bayer_pattern(raw_header)

        else:
            pattern = config.debayer_pattern

        radius = {"nearest": 0, "bilinear": 1, "bicubic": 2,
                  "lanczos": 4 if config.engine_profile == "Fast" else 2}.get(interpolation_mode, 2)
        if pattern and raw_data.ndim == 2:
            radius += 1 if config.debayer_method == "Bilinear" else 3
        if radius:
            kernel = np.ones((2*radius+1, 2*radius+1), np.uint8)
            source_valid = cv2.erode(source_valid.astype(np.uint8), kernel, borderType=cv2.BORDER_CONSTANT, borderValue=0).astype(bool)
            source_sat = cv2.dilate(source_sat.astype(np.uint8), kernel).astype(bool)
        raw_data = np.nan_to_num(raw_data, nan=0., posinf=0., neginf=0.)
        rgb_data, updated_header = process_in_memory_debayer(
            raw_data,
            raw_header,
            pattern,
            config.debayer_method,
        )

        # ----------------------------------------------------
        # Warping (Scikit-Image)
        # ----------------------------------------------------

        warp_diagnostics: dict = {}
        warped_data = warp_frame(
            rgb_data,
            final_matrix,
            interpolation_mode,
            rgb_registration=config.rgb_registration,
            engine_profile=config.engine_profile,
            warp_engine=config.warp_engine,
            diagnostics=warp_diagnostics,
            rgb_registration_mode=config.rgb_registration_mode,
        )

        # ----------------------------------------------------
        # Máscara
        #
        # A geometria é baseada em H x W, independentemente
        # de a imagem ser Mono ou RGB.
        # ----------------------------------------------------

        mask = generate_valid_mask(
            raw_data.shape,
            final_matrix,
        )
        source_valid = _warp_affine_cpu(source_valid.astype(np.float32), final_matrix, "nearest") > 0.5
        source_sat = _warp_affine_cpu(source_sat.astype(np.float32), final_matrix, "nearest") > 0.5
        mask = (mask.astype(bool) & source_valid).astype(np.uint8)
        if warped_data.ndim == 3:
            # RGB channels can have distinct valid footprints after correction.
            # Intensity is deliberately not used: black and signed science
            # pixels remain valid.  The geometric mask is the authority here.
            mask = (mask.astype(bool) & np.all(np.isfinite(warped_data), axis=2)).astype(np.uint8)
            original_mask = mask.copy()
            original_sat = source_sat.copy()
            # Apply the exact per-channel geometry to the common valid footprint.
            # Similarity registration can include a tiny scale/rotation; reducing it
            # to a translation would leave invalid edge pixels marked as usable.
            rgb_shifts = warp_diagnostics.get("rgb_shifts", {})
            rgb_models = warp_diagnostics.get("rgb_models", {})
            for channel, shift in rgb_shifts.items():
                dx, dy, _confidence = shift
                model = rgb_models.get(channel)
                matrix = None
                if isinstance(model, dict):
                    candidate = np.asarray(model.get("matrix"), dtype=np.float32)
                    if candidate.shape == (2, 3) and np.all(np.isfinite(candidate)):
                        matrix = candidate
                if matrix is None:
                    matrix = np.float32([[1, 0, dx], [0, 1, dy]])
                shifted = cv2.warpAffine(original_mask.astype(np.float32),
                    matrix,
                    (mask.shape[1], mask.shape[0]), flags=cv2.INTER_NEAREST,
                    borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                mask = (mask.astype(bool) & (shifted > 0.5)).astype(np.uint8)
                shifted_sat = cv2.warpAffine(original_sat.astype(np.uint8),
                    matrix, (mask.shape[1], mask.shape[0]), flags=cv2.INTER_NEAREST,
                    borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                source_sat |= shifted_sat > 0
        quality_candidates: list[tuple[str, tuple[np.ndarray, np.ndarray]]] = []
        if reference_previews:
            preferred_labels = frame_info.get("_quality_reference_labels", [])
            seen_labels = set()
            # The task carries an ordered parent/reference shortlist. Do not
            # enumerate every batch preview for every frame: that turns the
            # bounded cache into an O(frames*batches) workload and repeatedly
            # evicts previews on large jobs. Four candidates cover the direct
            # graph parent, batch reference, global parent and master fallback
            # while retaining the configured bounded cache.
            candidate_labels = list(preferred_labels)
            if not candidate_labels:
                candidate_labels = list(reference_previews.keys())[:1]
            for label in candidate_labels[:4]:
                if label in seen_labels or label not in reference_previews:
                    continue
                seen_labels.add(label)
                source = (reference_preview_sources or {}).get(label)
                current = frame_info.get("_quality_frame_identity")
                if source is not None and current is not None and tuple(source) == tuple(current):
                    # A frame must never validate itself. This is especially
                    # important for batch anchors, which otherwise compare a
                    # warped image against the same cached preview.
                    continue
                try:
                    candidate = reference_previews.get(label)
                except (KeyError, OSError, ValueError) as exc:
                    candidate = None
                    frame_info.setdefault("_quality_preview_errors", {})[label] = str(exc)
                if candidate is not None:
                    quality_candidates.append((label, candidate))
        elif reference_preview is not None:
            quality_candidates.append(("legacy_reference", reference_preview))

        best_quality = None
        best_label = None
        overlap_found = False
        for label, candidate in quality_candidates:
            ref_data, ref_mask = candidate
            # Compare on the candidate preview grid; this also keeps quality
            # work bounded for large sensors and allows a graph parent with a
            # different valid footprint to be tried next.
            sy = max(1, int(np.ceil(warped_data.shape[0] / ref_data.shape[0])))
            sx = max(1, int(np.ceil(warped_data.shape[1] / ref_data.shape[1])))
            preview = warped_data[::sy, ::sx]
            preview_mask = mask[::sy, ::sx]
            h, w = ref_data.shape[:2]
            preview = preview[:h, :w]
            preview_mask = preview_mask[:h, :w]
            ref_cmp = ref_data[:preview.shape[0], :preview.shape[1]]
            refm_cmp = np.asarray(ref_mask, dtype=bool)[:preview.shape[0], :preview.shape[1]]
            sat_cmp = source_sat[::sy, ::sx][:preview.shape[0], :preview.shape[1]]
            common = preview_mask.astype(bool) & refm_cmp
            if np.any(common & ~sat_cmp):
                overlap_found = True
            candidate_quality = estimate_alignment_quality(
                ref_cmp,
                preview,
                common & ~sat_cmp,
            )
            candidate_quality["shift_x"] *= sx
            candidate_quality["shift_y"] *= sy
            if best_quality is None:
                best_quality, best_label = candidate_quality, label
                continue
            best_score = (
                float(best_quality.get("coverage", 0.0)) >= config.quality_min_coverage,
                float(best_quality.get("confidence", 0.0)),
                -float(best_quality.get("rms", float("inf"))),
            )
            candidate_score = (
                float(candidate_quality.get("coverage", 0.0)) >= config.quality_min_coverage,
                float(candidate_quality.get("confidence", 0.0)),
                -float(candidate_quality.get("rms", float("inf"))),
            )
            if candidate_score > best_score:
                best_quality, best_label = candidate_quality, label

        if not quality_candidates:
            quality = {"shift_x": 0.0, "shift_y": 0.0, "rms": 0.0,
                       "confidence": 0.0, "coverage": float(np.mean(mask))}
        else:
            quality = best_quality or {
                "shift_x": 0.0,
                "shift_y": 0.0,
                "rms": float("inf"),
                "confidence": 0.0,
                "coverage": 0.0,
            }
        accepted = (quality["confidence"] >= config.quality_min_confidence and
                    quality["rms"] <= config.quality_max_rms and
                    quality["coverage"] >= config.quality_min_coverage and
                    float(np.hypot(quality["shift_x"], quality["shift_y"])) <= config.quality_max_shift)
        if not quality_candidates:
            decision = "unverified"
        elif not overlap_found or float(quality.get("coverage", 0.0)) < config.quality_min_coverage:
            decision = "insufficient_overlap"
        elif accepted:
            decision = "verified"
        else:
            decision = "failed_quality"
        if config.quality_gate and not accepted:
            return frame_name, f"QUALITY_REJECTED ({decision})"

        # ----------------------------------------------------
        # Escrita
        # ----------------------------------------------------

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        output_header = updated_header if config.keep_header else fits.Header()
        if not config.keep_header:
            # Retain calibration and exposure semantics needed to decode the
            # persisted uint16 science plane even with cosmetic headers off.
            for key in ("CALNORM", "CALMIN", "CALMAX", "EXPTIME", "GAIN", "FILTER",
                        "BAYERPAT", "BINNING", "SATURATE", "SATLEVEL", "EXPOSURE", "BUNIT",
                        "CALHDR", "EGAIN", "ISOSPEED", "XBINNING", "YBINNING", "SATKNOWN"):
                if key in updated_header:
                    output_header[key] = updated_header[key]
        alignment_revision = _alignment_transform_revision(
            frame_info,
            interpolation_mode,
            config.rgb_registration_mode,
        )
        save_args = (warped_data, mask, output_header, output_path,
                     config.compress_output,
                     {"ALNSTAT": decision, "ALNRMS": quality["rms"],
                      "ALNCONF": quality["confidence"], "ALNCOV": quality["coverage"],
                      "ALNSHIX": quality["shift_x"], "ALNSHIY": quality["shift_y"],
                      "RGBMODE": config.rgb_registration_mode,
                      "ALNREV": alignment_revision}, cancel_event, source_sat)
        if cancel_event is not None and cancel_event.is_set():
            return frame_name, "CANCELLED"
        if writer_executor is None:
            save_aligned_fits(*save_args)
        else:
            # Keep this frame's payload alive until the single writer commits.
            writer_executor.submit(save_aligned_fits, *save_args).result()
        try:
            from app.infrastructure.json_store import atomic_json_write
            safe_quality = {k: v if np.isfinite(v) else None for k,v in quality.items()}
            atomic_json_write(output_path.with_suffix(output_path.suffix + ".align.json"),
                              {"frame": frame_name, "decision": decision,
                               "quality": safe_quality, "quality_reference": best_label,
                               "rgb_registration_mode": config.rgb_registration_mode,
                               "rgb_models": warp_diagnostics.get("rgb_models", {}),
                               "alignment_revision": alignment_revision,
                               "local_transform_revision": frame_info.get("_local_transform_revision"),
                               "global_transform_revision": frame_info.get("_global_transform_revision"),
                               "source_matrix_revision": frame_info.get("_local_transform_revision")})
        except OSError as exc:
            return frame_name, f"Science FITS saved; quality sidecar failed: {exc}"

        return (
            frame_name,
            None,
        )

    except Exception as exc:
        return (
            frame_name,
            f"Erro ao alinhar {frame_name}: {exc}",
        )


# ============================================================
# Batch
# ============================================================


def process_batch_alignment(
    batch_dir: Path,
    global_flow: dict,
    config: AlignConfig,
    app_print,
    app_progress,
    cancel_event: threading.Event,
    progress_state: dict,
) -> tuple[int, int]:

    local_flow = load_local_flow(batch_dir)

    if local_flow is None:
        app_print(f"[{batch_dir.name}] flow_local.json não encontrado.\n")
        return (
            0,
            0,
        )

    batch_entry = global_flow.get("batches", {}).get(batch_dir.name)

    if batch_entry is None:
        app_print(f"[{batch_dir.name}] Batch não encontrada no Global Flow.\n")
        return (
            0,
            0,
        )

    if batch_entry.get(
        "status",
        "accepted",
    ) not in {
        "accepted",
        "master",
    }:
        app_print(
            f"[{batch_dir.name}] "
            f"Batch rejeitada pelo Global Flow: "
            f"{batch_entry.get('reason', 'unknown')}\n"
        )
        return (
            0,
            0,
        )

    global_matrix = batch_entry.get("matrix")

    if global_matrix is None:
        app_print(f"[{batch_dir.name}] Matriz Global ausente.\n")
        return (
            0,
            0,
        )

    frames = local_flow.get(
        "frames",
        {},
    )

    # Somente frames aceitos possuem transformação válida.
    valid_frames = {
        fname: finfo
        for fname, finfo in frames.items()
        if finfo.get(
            "status",
            "accepted",
        )
        == "accepted"
        and finfo.get("matrix") is not None
    }

    total_frames = len(valid_frames)

    if total_frames == 0:
        app_print(f"[{batch_dir.name}] Nenhum frame válido no Flow Local.\n")
        return (
            0,
            0,
        )

    output_dir = config.output_dir / batch_dir.name

    interpolation_mode = INTERPOLATION_MODES.get(
        config.interpolation,
        "lanczos",
    )

    worker_count = get_optimal_worker_count()

    processed = 0
    failed = 0

    # Alignment uses OpenCV/scikit-image in each frame worker.  Multiple frame
    # workers own parallelism; the single-worker path may use the physical
    # native pool for the large warp.
    configure_opencv_threads(1 if worker_count > 1 else physical_core_count())
    with ThreadPoolExecutor(
        max_workers=worker_count,
        thread_name_prefix="astroalign",
        initializer=partial(configure_worker_runtime, 1),
    ) as executor:
        futures = {
            executor.submit(
                _process_single_alignment,
                fname,
                finfo,
                batch_dir,
                output_dir,
                global_matrix,
                interpolation_mode,
                config,
            ): fname
            for fname, finfo in valid_frames.items()
        }

        for future in as_completed(futures):
            if cancel_event.is_set():
                for pending in futures:
                    pending.cancel()

                break

            try:
                frame_name, error = future.result()

            except Exception as exc:
                frame_name = futures[future]

                error = f"Erro inesperado: {exc}"

            progress_state["done"] += 1

            if error:
                app_print(f"  [{frame_name}] {error}\n")
                failed += 1

            else:
                processed += 1

            done = progress_state["done"]

            if done % 10 == 0 or done == progress_state["total"]:
                app_progress(
                    done,
                    progress_state["total"],
                    (f"Alinhando frames ({done}/{progress_state['total']})..."),
                )

    # --------------------------------------------------------
    # Limpeza dos intermediários
    # --------------------------------------------------------

    if (
        config.delete_intermediates
        and failed == 0
        and not config.dry_run
        and not cancel_event.is_set()
    ):
        try:
            shutil.rmtree(batch_dir)

            app_print(f"[{batch_dir.name}] Batch original limpo com sucesso.\n")

        except Exception as exc:
            app_print(f"[{batch_dir.name}] Erro ao apagar intermediários: {exc}\n")

    app_print(
        f"[{batch_dir.name}] Concluído: {processed} alinhados, {failed} falhas.\n"
    )

    return (
        processed,
        failed,
    )


# ============================================================
# Pipeline completo
# ============================================================


def process_all_alignments(
    base_dir: Path,
    output_dir: Path,
    config_dict: dict,
    app_print,
    app_progress,
    cancel_event: threading.Event,
) -> tuple[int, int]:

    if not isinstance(
        config_dict,
        dict,
    ):
        config_dict = {}

    align_config = _build_align_config(base_dir, output_dir, config_dict)

    # --------------------------------------------------------
    # Global Flow
    # --------------------------------------------------------

    global_flow = load_global_flow(base_dir)

    if global_flow is None:
        app_print("ERRO: global_flow.json não encontrado.\n")
        return (
            0,
            0,
        )

    # --------------------------------------------------------
    # Batches
    # --------------------------------------------------------

    batch_folders = find_batch_folders(base_dir)

    if not batch_folders:
        app_print("Nenhuma Batch encontrada.\n")
        return (
            0,
            0,
        )

    # --------------------------------------------------------
    # Conta frames realmente utilizáveis
    # --------------------------------------------------------

    total_frames = 0
    batches_with_flow = []

    for batch_folder in batch_folders:
        local_flow = load_local_flow(batch_folder)

        if not local_flow:
            continue

        batch_entry = global_flow.get("batches", {}).get(batch_folder.name)

        if not batch_entry:
            continue

        if batch_entry.get(
            "status",
            "accepted",
        ) not in {
            "accepted",
            "master",
        }:
            continue

        local_frames = local_flow.get(
            "frames",
            {},
        )

        valid_count = sum(
            1
            for frame_data in local_frames.values()
            if (
                frame_data.get(
                    "status",
                    "accepted",
                )
                == "accepted"
                and frame_data.get("matrix") is not None
            )
        )

        if valid_count <= 0:
            continue

        batches_with_flow.append(batch_folder)

        total_frames += valid_count

    if total_frames == 0:
        app_print("Nenhum frame válido para alinhamento.\n")
        return (
            0,
            0,
        )

    progress_state = {
        "done": 0,
        "total": total_frames,
    }

    if not align_config.dry_run:
        align_config.output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    total_processed = 0
    total_failed = 0
    preflight_failed = 0

    # V2 scheduler: flatten accepted frames from every batch into one bounded
    # executor. This avoids leaving CPU idle when individual batches are small
    # and prevents the old nested batch/frame thread pools.
    tasks: list[tuple[Path, str, dict, np.ndarray, Path, str]] = []
    task_counts: dict[Path, int] = {}
    batch_failures: dict[Path, int] = {}
    for batch_folder in batches_with_flow:
        local_flow = load_local_flow(batch_folder) or {}
        batch_entry = global_flow.get("batches", {}).get(batch_folder.name, {})
        global_matrix = batch_entry.get("matrix")
        if global_matrix is None:
            continue
        local_revision = local_flow.get("transform_revision")
        global_revision = (global_flow.get("source_revisions", {}) or {}).get(batch_folder.name)
        if global_revision is None:
            global_revision = batch_entry.get("local_transform_revision")
        if local_revision and global_revision and local_revision != global_revision:
            preflight_failed += sum(
                1 for info in local_flow.get("frames", {}).values()
                if info.get("status", "accepted") == "accepted" and info.get("matrix") is not None
            )
            app_print(
                f"[{batch_folder.name}] ERRO: Flow Local e Global usam revisões diferentes; "
                "reexecute o Flow antes do Align.\n"
            )
            continue
        valid_frames = {
            fname: finfo
            for fname, finfo in local_flow.get("frames", {}).items()
            if finfo.get("status", "accepted") == "accepted" and finfo.get("matrix") is not None
        }
        output_dir = align_config.output_dir / batch_folder.name
        interpolation_mode = INTERPOLATION_MODES.get(align_config.interpolation, "lanczos")
        task_counts[batch_folder] = len(valid_frames)
        batch_failures[batch_folder] = 0
        for fname, finfo in valid_frames.items():
            task_info = dict(finfo)
            task_info["_local_transform_revision"] = local_revision
            task_info["_global_transform_revision"] = global_flow.get("transform_revision")
            task_info["_geometry_revision"] = (
                f"{local_flow.get('geometry_revision', '')}|"
                f"{global_flow.get('geometry_revision', '')}"
            )
            labels = [batch_folder.name]
            local_parent = finfo.get("relative_to")
            if local_parent and local_parent in local_flow.get("frames", {}):
                labels.insert(0, f"{batch_folder.name}/{local_parent}")
            parent_batch = batch_entry.get("relative_to")
            if parent_batch and parent_batch != batch_folder.name:
                labels.append(str(parent_batch))
            master_batch = global_flow.get("global_master_batch")
            if master_batch and master_batch not in labels:
                labels.append(str(master_batch))
            task_info["_quality_reference_labels"] = labels
            task_info["_quality_frame_identity"] = (batch_folder.name, fname)
            tasks.append((batch_folder, fname, task_info, global_matrix, output_dir, interpolation_mode))

    if len(tasks) != total_frames:
        total_frames = len(tasks)
        progress_state["total"] = total_frames
    if not tasks:
        app_print("Nenhum frame valido para alinhamento.\n")
        return (0, preflight_failed)

    total_failed = preflight_failed

    # Build a bounded, demand-driven cache of graph reference previews in the
    # same output coordinate system used by Align. Workers prefer their own
    # batch, then the global parent/master; unrelated cached previews are
    # fallbacks only when overlap is insufficient.
    preview_targets: dict[str, tuple[Path, str, dict, np.ndarray, str]] = {}
    preview_sources: dict[str, tuple[str, str]] = {}
    for batch_folder in batches_with_flow:
        local_flow = load_local_flow(batch_folder) or {}
        batch_entry = global_flow.get("batches", {}).get(batch_folder.name, {})
        global_matrix = batch_entry.get("matrix")
        if global_matrix is None:
            continue
        local_revision = local_flow.get("transform_revision")
        global_revision = (global_flow.get("source_revisions", {}) or {}).get(batch_folder.name)
        if global_revision is None:
            global_revision = batch_entry.get("local_transform_revision")
        if local_revision and global_revision and local_revision != global_revision:
            continue
        anchor_name = local_flow.get("batch_anchor")
        anchor_info = (local_flow.get("frames", {}) or {}).get(anchor_name, {})
        if anchor_info.get("status", "accepted") != "accepted":
            anchor_name, anchor_info = next(
                (
                    (name, info)
                    for name, info in sorted(
                        (local_flow.get("frames", {}) or {}).items(),
                        key=lambda item: item[0].casefold(),
                    )
                    if info.get("status", "accepted") == "accepted" and info.get("matrix") is not None
                ),
                (None, None),
            )
        if anchor_name and isinstance(anchor_info, dict) and anchor_info.get("matrix") is not None:
            preview_targets[batch_folder.name] = (
                batch_folder,
                anchor_name,
                anchor_info,
                np.asarray(global_matrix, dtype=np.float64),
                INTERPOLATION_MODES.get(align_config.interpolation, "lanczos"),
            )
            preview_sources[batch_folder.name] = (batch_folder.name, anchor_name)

        # A local graph parent often overlaps a distant frame better than the
        # batch reference. Keep these targets available to the bounded cache;
        # the task label order above makes the parent the first candidate.
        for frame_name, frame_info in (local_flow.get("frames", {}) or {}).items():
            parent_name = frame_info.get("relative_to") if isinstance(frame_info, dict) else None
            if not parent_name or parent_name not in (local_flow.get("frames", {}) or {}):
                continue
            parent_info = local_flow["frames"].get(parent_name, {})
            if parent_info.get("status", "accepted") != "accepted" or parent_info.get("matrix") is None:
                continue
            preview_targets.setdefault(
                f"{batch_folder.name}/{parent_name}",
                (
                    batch_folder,
                    parent_name,
                    parent_info,
                    np.asarray(global_matrix, dtype=np.float64),
                    INTERPOLATION_MODES.get(align_config.interpolation, "lanczos"),
                ),
            )
            preview_sources.setdefault(f"{batch_folder.name}/{parent_name}", (batch_folder.name, parent_name))

    for label, target in list(preview_targets.items()):
        batch_folder, frame_name, frame_info, global_matrix, interpolation_mode = target
        preview_targets[label] = (
            batch_folder,
            frame_name,
            frame_info,
            compute_final_matrix(frame_info["matrix"], global_matrix),
            interpolation_mode,
        )
    reference_previews = ReferencePreviewCache(
        preview_targets,
        max_size=max(1, int(config_dict.get("quality_preview_cache", 8))),
    )

    from app.engines.execution import science_frame_bytes
    requested_workers = align_config.workers or get_optimal_worker_count()
    frame_bytes = 1 if align_config.dry_run else max(
        science_frame_bytes(batch / name) for batch, name, *_ in tasks) * 10
    budget = ExecutionBudget.for_frame_pipeline(
        requested_workers, align_config.memory_budget_mb or 512,
        frame_bytes=frame_bytes)
    worker_count = budget.worker_count
    limit = min(align_config.max_in_flight or budget.max_in_flight, budget.max_in_flight)
    # Keep writes bounded.  The default remains one writer (conservative for
    # spinning disks and compressed FITS); diagnostics may request up to four
    # writers to measure whether storage/compression can overlap productively.
    writer_workers = max(1, min(worker_count, 4, align_config.writer_workers or 1))
    configure_opencv_threads(1 if worker_count > 1 else physical_core_count())
    with ThreadPoolExecutor(max_workers=writer_workers,
                            thread_name_prefix="astroalign-writer",
                            initializer=partial(configure_worker_runtime, 1)) as writer_executor, \
         ThreadPoolExecutor(max_workers=worker_count,
                            thread_name_prefix="astroalign-v2",
                            initializer=partial(configure_worker_runtime, budget.kernel_threads)) as executor:
        iterator = iter(tasks)
        futures = {}
        canceled = False
        while futures or not cancel_event.is_set():
            while not cancel_event.is_set() and len(futures) < limit:
                try:
                    batch_folder, fname, finfo, global_matrix, output_dir, interpolation_mode = next(iterator)
                except StopIteration:
                    break
                future = executor.submit(_process_single_alignment, fname, finfo, batch_folder, output_dir,
                                         global_matrix, interpolation_mode, align_config, cancel_event,
                                         None, writer_executor, reference_previews, preview_sources)
                futures[future] = batch_folder
            if not futures:
                break
            done_set, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in done_set:
                if cancel_event.is_set():
                    for pending in futures:
                        pending.cancel()
                    canceled = True
                    break
                batch_folder = futures.pop(future)
                try:
                    frame_name, error = future.result()
                except Exception as exc:
                    frame_name, error = "unknown", f"Erro inesperado: {exc}"
                progress_state["done"] += 1
                if error:
                    batch_failures[batch_folder] += 1
                    total_failed += 1
                    app_print(f"  [{frame_name}] {error}\n")
                else:
                    total_processed += 1
                done = progress_state["done"]
                if done % 10 == 0 or done == progress_state["total"]:
                    app_progress(done, progress_state["total"], f"Alinhando frames ({done}/{progress_state['total']})...")
            if canceled:
                # Futures that did not start are discarded; running workers
                # observe cancel_event and return before writing.
                futures.clear()
                break

    if align_config.delete_intermediates and not align_config.dry_run and not cancel_event.is_set():
        for batch_folder, failed in batch_failures.items():
            if failed == 0:
                try:
                    shutil.rmtree(batch_folder)
                    app_print(f"[{batch_folder.name}] Batch original limpo com sucesso.\n")
                except Exception as exc:
                    app_print(f"[{batch_folder.name}] Erro ao apagar intermediarios: {exc}\n")

    if not cancel_event.is_set():
        app_progress(total_frames, total_frames, "Concluido.")
    app_print(f"\n>>> AstroAlign Finalizado! {total_processed} frames alinhados, {total_failed} falhas. <<<\n")
    return (total_processed, total_failed)

    # --------------------------------------------------------
    # Processamento das Batches
    # --------------------------------------------------------

    for batch_folder in batches_with_flow:
        if cancel_event.is_set():
            break

        app_print(f"\nAlinhando Batch: {batch_folder.name}\n")

        processed, failed = process_batch_alignment(
            batch_folder,
            global_flow,
            align_config,
            app_print,
            app_progress,
            cancel_event,
            progress_state,
        )

        total_processed += processed
        total_failed += failed

    # --------------------------------------------------------
    # Finalização
    # --------------------------------------------------------

    if not cancel_event.is_set():
        app_progress(
            total_frames,
            total_frames,
            "Concluído.",
        )

    app_print(
        f"\n>>> AstroAlign Finalizado! "
        f"{total_processed} frames alinhados, "
        f"{total_failed} falhas. <<<\n"
    )

    return (
        total_processed,
        total_failed,
    )
