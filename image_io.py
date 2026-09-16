"""Small common image-input adapter for FITS and TIFF/TIF captures."""

from __future__ import annotations

import json
import os
import tempfile
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits


FITS_SUFFIXES = {".fit", ".fits", ".fts"}
TIFF_SUFFIXES = {".tif", ".tiff"}
IMAGE_SUFFIXES = FITS_SUFFIXES | TIFF_SUFFIXES


class ImageFormat(str, Enum):
    """Scientific image families supported by the pipeline."""

    FITS = "fits"
    TIFF = "tiff"

    @property
    def default_suffix(self) -> str:
        return ".fits" if self is ImageFormat.FITS else ".tif"


def format_from_suffix(path: Path | str) -> ImageFormat | None:
    suffix = Path(path).suffix.casefold()
    if suffix in FITS_SUFFIXES:
        return ImageFormat.FITS
    if suffix in TIFF_SUFFIXES:
        return ImageFormat.TIFF
    return None


def detect_format(path: Path | str) -> ImageFormat:
    """Detect a source format from suffix and, when needed, its magic bytes."""

    filepath = Path(path)
    by_suffix = format_from_suffix(filepath)
    if by_suffix is not None:
        return by_suffix
    try:
        with filepath.open("rb") as stream:
            magic = stream.read(8)
    except OSError as exc:
        raise ValueError(f"Imagem não encontrada: {filepath}") from exc
    if magic[:4] in {b"II*\x00", b"MM\x00*", b"II+\x00", b"MM\x00+"}:
        return ImageFormat.TIFF
    if magic.startswith(b"SIMPLE  "):
        return ImageFormat.FITS
    raise ValueError(f"Formato de imagem não suportado: {filepath.name}")


def validate_single_format(paths: list[Path] | tuple[Path, ...]) -> ImageFormat:
    """Require a non-empty collection containing one image family only."""

    if not paths:
        raise ValueError("Nenhuma imagem encontrada.")
    formats = {detect_format(path) for path in paths}
    if len(formats) != 1:
        names = ", ".join(sorted(fmt.value for fmt in formats))
        raise ValueError(
            "Sessão mista não suportada: escolha somente TIF/TIFF ou somente FIT/FITS "
            f"(detectado: {names})."
        )
    return formats.pop()


def ensure_output_suffix(path: Path | str, image_format: ImageFormat) -> Path:
    """Validate/normalize a user-facing output path for the session format."""

    target = Path(path)
    actual = format_from_suffix(target)
    if actual is None:
        return target.with_suffix(image_format.default_suffix)
    if actual is not image_format:
        raise ValueError(
            f"Formato de saída incompatível: a sessão usa {image_format.value.upper()}, "
            f"mas o destino termina em {target.suffix}."
        )
    return target


def is_supported_image(path: Path | str) -> bool:
    return Path(path).suffix.casefold() in IMAGE_SUFFIXES


def _tiff_header(path: Path) -> fits.Header:
    """Expose the small amount of TIFF metadata useful to the pipeline."""

    header = fits.Header()
    try:
        from PIL import Image

        with Image.open(path) as image:
            tags = getattr(image, "tag_v2", {})
            # TIFF DateTime is ``YYYY:MM:DD HH:MM:SS``.  Converting it here
            # lets the existing DATE-OBS normalizer consume it uniformly.
            value = tags.get(306)
            if value:
                text = str(value).strip()
                if len(text) >= 19 and text[4] == ":" and text[7] == ":":
                    text = f"{text[:4]}-{text[5:7]}-{text[8:10]}T{text[11:19]}"
                header["DATE-OBS"] = text
            description = tags.get(270)
            if description:
                header["COMMENT"] = str(description)[:68]
    except Exception:
        pass
    # AstroBatch sidecars carry FITS-like cards that TIFF cannot represent
    # portably. Merge only the documented header object; unrelated JSON next
    # to a user image is ignored.
    sidecar = path.with_suffix(path.suffix + ".json")
    try:
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
        cards = payload.get("header") if isinstance(payload, dict) else None
        if isinstance(cards, dict):
            for key, value in cards.items():
                card = str(key).upper()[:8]
                parsed: Any = value
                text = str(value)
                if text.casefold() in {"true", "false"}:
                    parsed = text.casefold() == "true"
                else:
                    try:
                        parsed = float(text) if any(ch in text for ch in ".eE") else int(text)
                    except (TypeError, ValueError):
                        parsed = value
                try:
                    header[card] = parsed
                except (ValueError, KeyError):
                    continue
    except (OSError, ValueError, TypeError):
        pass
    return header


def read_tiff(path: Path | str) -> tuple[np.ndarray, fits.Header]:
    """Read the first TIFF page while preserving its native sample depth."""

    filepath = Path(path)
    data: Any
    try:
        import tifffile

        data = tifffile.imread(filepath)
    except Exception:
        from PIL import Image

        with Image.open(filepath) as image:
            data = np.asarray(image).copy()
    array = np.asarray(data)
    if array.ndim not in (2, 3):
        raise ValueError(f"Imagem TIFF precisa ser 2D ou RGB: {filepath.name} ({array.shape})")
    return np.array(array, copy=True), _tiff_header(filepath)


def tiff_shape(path: Path | str) -> tuple[int, ...]:
    """Inspect TIFF geometry without decoding the pixel plane."""

    filepath = Path(path)
    try:
        import tifffile

        with tifffile.TiffFile(filepath) as tif:
            if not tif.series:
                raise ValueError("TIFF sem série de imagem")
            return tuple(int(value) for value in tif.series[0].shape)
    except Exception:
        # Pillow parses the directory/photometric tags lazily.  ``size`` and
        # ``mode`` therefore remain a low-RAM fallback even when the TIFF is
        # compressed and cannot be memory-mapped by tifffile.
        from PIL import Image

        with Image.open(filepath) as image:
            channels = len(image.getbands())
            return (int(image.height), int(image.width)) if channels <= 1 else (
                int(image.height), int(image.width), int(channels)
            )


def read_tiff_header(path: Path | str) -> fits.Header:
    """Read TIFF tags without materializing pixel data."""
    return _tiff_header(Path(path))


def read_tiff_sidecar_arrays(path: Path | str) -> dict[str, np.ndarray]:
    """Load optional validity/saturation arrays written beside a TIFF."""

    filepath = Path(path)
    candidates = [filepath.with_suffix(filepath.suffix + ".astrobatch.npz")]
    try:
        payload = json.loads(
            filepath.with_suffix(filepath.suffix + ".json").read_text(encoding="utf-8")
        )
        name = payload.get("science_sidecar") or payload.get("sidecar")
        if name:
            candidates.insert(0, filepath.with_name(str(name)))
    except (OSError, ValueError, TypeError):
        pass
    for candidate in candidates:
        try:
            with np.load(candidate, allow_pickle=False) as loaded:
                return {name: np.array(loaded[name], copy=True) for name in loaded.files}
        except (OSError, ValueError, KeyError):
            continue
    return {}


def write_tiff(
    path: Path | str,
    data: np.ndarray,
    *,
    metadata: dict[str, Any] | None = None,
    compression: str = "deflate",
) -> None:
    """Write a linear uint16 TIFF atomically, preserving mono/RGB layout."""

    filepath = Path(path)
    array = np.asarray(data)
    if array.ndim not in (2, 3):
        raise ValueError(f"Imagem TIFF precisa ser 2D ou RGB: {array.shape}")
    if array.ndim == 3 and array.shape[-1] not in (3, 4):
        raise ValueError(f"Imagem RGB precisa ter 3 ou 4 canais: {array.shape}")
    if array.dtype != np.uint16:
        array = np.clip(array, 0, 65535).astype(np.uint16, copy=False)
    array = np.ascontiguousarray(array)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    temporary = filepath.with_name(f".{filepath.name}.{os.getpid()}.tmp")
    description = json.dumps(metadata, ensure_ascii=False, allow_nan=False) if metadata else None
    try:
        try:
            import tifffile

            kwargs: dict[str, Any] = {
                "photometric": "rgb" if array.ndim == 3 else "minisblack",
                "metadata": None,
                "bigtiff": bool(array.nbytes + 4096 > 0xFFFFFFFF),
            }
            if compression:
                kwargs["compression"] = compression
            if description:
                kwargs["description"] = description
            tifffile.imwrite(temporary, array, **kwargs)
        except ImportError:
            from PIL import Image

            if array.ndim == 3 and array.dtype != np.uint8:
                raise RuntimeError("tifffile é necessário para TIFF RGB uint16")
            Image.fromarray(array).save(temporary, format="TIFF", compression="tiff_deflate")
        os.replace(temporary, filepath)
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


def write_sidecar_json(path: Path | str, payload: dict[str, Any]) -> Path:
    """Atomically publish metadata next to an image."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent)
    )
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False),
            encoding="utf-8",
        )
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def write_npz_sidecar(path: Path | str, **arrays: np.ndarray) -> Path:
    """Atomically persist compact scientific arrays next to a non-FITS image."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def read_image(path: Path | str) -> tuple[np.ndarray, fits.Header]:
    filepath = Path(path)
    if filepath.suffix.casefold() in TIFF_SUFFIXES:
        return read_tiff(filepath)
    with fits.open(filepath, memmap=False, ignore_missing_end=True) as hdul:
        for hdu in hdul:
            if getattr(hdu, "is_image", False) and getattr(hdu, "data", None) is not None:
                data = np.asarray(hdu.data)
                if data.ndim in (2, 3):
                    return np.array(data, copy=True), hdu.header.copy()
    raise ValueError(f"Imagem não encontrada em {filepath.name}")


__all__ = [
    "FITS_SUFFIXES", "TIFF_SUFFIXES", "IMAGE_SUFFIXES", "ImageFormat",
    "detect_format", "format_from_suffix", "validate_single_format", "ensure_output_suffix",
    "is_supported_image", "read_image", "read_tiff", "read_tiff_header",
    "tiff_shape",
    "read_tiff_sidecar_arrays",
    "write_npz_sidecar", "write_sidecar_json", "write_tiff",
]
