import json
import os
import shutil
import tempfile
import threading
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed, wait, FIRST_COMPLETED
from dataclasses import dataclass
from pathlib import Path

import colour_demosaicing
import cv2
import numpy as np
from astropy.io import fits
from astropy.utils.exceptions import AstropyWarning
from skimage.transform import AffineTransform, warp

from app.engines import EngineProfile, ExecutionBudget, registry
from app.engines.align import register_align_engines

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
            16,
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


def load_local_flow(
    batch_dir: Path,
) -> dict | None:
    flow_path = batch_dir / "flow_local.json"

    if not flow_path.exists():
        return None

    return load_json(flow_path)


def load_global_flow(
    base_dir: Path,
) -> dict | None:
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


def load_fits_data(
    filepath: Path,
) -> tuple[np.ndarray, fits.Header]:

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
                        data = np.moveaxis(data,0,-1)
                    elif data.shape[-1] not in (3,4):
                        raise ValueError(f"Unsupported RGB geometry: {data.shape}")

                return (
                    data,
                    header,
                )

    raise ValueError(f"Imagem 2D não encontrada em {filepath.name}")


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


def warp_frame(
    data: np.ndarray,
    final_matrix: np.ndarray,
    interpolation_mode: str,
    rgb_registration: bool = False,
    engine_profile: str = "Stable",
    warp_engine: str | None = None,
    diagnostics: dict | None = None,
    rgb_max_shift: float = 2.0,
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
            if profile is EngineProfile.FAST:
                reference = output[:, :, 1]
                height, width = reference.shape
                for channel in (0, 2):
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
    data, header = load_fits_data(filepath)
    pattern = get_bayer_pattern(header)
    data, _ = process_in_memory_debayer(data, header, pattern, "Bilinear")
    warped = warp_frame(data, matrix, interpolation_mode, rgb_registration=False)
    mask = generate_valid_mask(data.shape, matrix).astype(bool)
    stride = max(1, int(np.ceil(max(data.shape[:2]) / max_size)))
    if stride > 1:
        warped = warped[::stride, ::stride]
        mask = mask[::stride, ::stride]
    return np.asarray(warped, dtype=np.float32), np.asarray(mask, dtype=np.uint8)


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
            return (
                frame_name,
                (f"ERRO: destino já existe, arquivo ignorado: {output_path}"),
            )

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

        raw_data, raw_header = load_fits_data(filepath)
        source_valid, source_sat = load_fits_masks(filepath, raw_data.shape)
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
            for dx, dy, confidence in warp_diagnostics.get("rgb_shifts", {}).values():
                shifted = cv2.warpAffine(original_mask.astype(np.float32),
                    np.float32([[1, 0, dx], [0, 1, dy]]),
                    (mask.shape[1], mask.shape[0]), flags=cv2.INTER_NEAREST,
                    borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                mask = (mask.astype(bool) & (shifted > 0.5)).astype(np.uint8)
                shifted_sat = cv2.warpAffine(original_sat.astype(np.uint8),
                    np.float32([[1,0,dx],[0,1,dy]]), (mask.shape[1],mask.shape[0]), flags=cv2.INTER_NEAREST)
                source_sat |= shifted_sat > 0
        if reference_preview is None:
            quality = {"shift_x": 0.0, "shift_y": 0.0, "rms": 0.0,
                       "confidence": 0.0, "coverage": float(np.mean(mask))}
        else:
            ref_data, ref_mask = reference_preview
            # Compare on the anchor preview grid; this also keeps quality work
            # bounded for large sensors.
            sy = max(1, int(np.ceil(warped_data.shape[0] / ref_data.shape[0])))
            sx = max(1, int(np.ceil(warped_data.shape[1] / ref_data.shape[1])))
            preview = warped_data[::sy, ::sx]
            preview_mask = mask[::sy, ::sx]
            h, w = ref_data.shape[:2]
            preview = preview[:h, :w]
            preview_mask = preview_mask[:h, :w]
            ref_cmp = ref_data[:preview.shape[0], :preview.shape[1]]
            refm_cmp = np.asarray(ref_mask, dtype=bool)[:preview.shape[0], :preview.shape[1]]
            common = preview_mask.astype(bool) & refm_cmp
            quality = estimate_alignment_quality(ref_cmp, preview, common & ~source_sat[::sy,::sx][:h,:w])
            # The preview shift is measured in preview pixels; report and gate
            # it in source/output pixels for a stable user-facing threshold.
            quality["shift_x"] *= sx
            quality["shift_y"] *= sy
        accepted = (quality["confidence"] >= config.quality_min_confidence and
                    quality["rms"] <= config.quality_max_rms and
                    quality["coverage"] >= config.quality_min_coverage and
                    float(np.hypot(quality["shift_x"], quality["shift_y"])) <= config.quality_max_shift)
        decision = "unverified" if reference_preview is None else "accepted" if accepted else "low_confidence"
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
        save_args = (warped_data, mask, output_header, output_path,
                     config.compress_output,
                     {"ALNSTAT": decision, "ALNRMS": quality["rms"],
                      "ALNCONF": quality["confidence"], "ALNCOV": quality["coverage"],
                      "ALNSHIX": quality["shift_x"], "ALNSHIY": quality["shift_y"]}, cancel_event, source_sat)
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
                              {"frame": frame_name, "decision": decision, "quality": safe_quality})
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

    with ThreadPoolExecutor(
        max_workers=worker_count,
        thread_name_prefix="astroalign",
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
            tasks.append((batch_folder, fname, finfo, global_matrix, output_dir, interpolation_mode))

    if len(tasks) != total_frames:
        total_frames = len(tasks)
        progress_state["total"] = total_frames
    if not tasks:
        app_print("Nenhum frame valido para alinhamento.\n")
        return (0, 0)

    # Establish one immutable quality anchor in the output coordinate system.
    # A missing/unreadable anchor leaves advisory metrics unverified and makes
    # quality_gate fail closed in the worker.
    reference_preview = None
    try:
        anchor_batch, anchor_name, anchor_info, anchor_global, _, anchor_interp = tasks[0]
        anchor_matrix = compute_final_matrix(anchor_info["matrix"], anchor_global)
        reference_preview = prepare_reference_preview(
            anchor_batch / anchor_name, anchor_matrix, anchor_interp, max_size=512
        )
    except Exception as exc:
        app_print(f"Aviso: referência de qualidade indisponível ({exc}).\n")

    from app.engines.execution import science_frame_bytes
    requested_workers = align_config.workers or get_optimal_worker_count()
    frame_bytes = 1 if align_config.dry_run else max(
        science_frame_bytes(batch / name) for batch, name, *_ in tasks) * 10
    budget = ExecutionBudget.for_frame_pipeline(
        requested_workers, align_config.memory_budget_mb or 512,
        frame_bytes=frame_bytes)
    worker_count = budget.worker_count
    limit = min(align_config.max_in_flight or budget.max_in_flight, budget.max_in_flight)
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="astroalign-writer") as writer_executor, \
         ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="astroalign-v2") as executor:
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
                                         reference_preview, writer_executor)
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
