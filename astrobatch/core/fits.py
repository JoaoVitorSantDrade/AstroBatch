from __future__ import annotations

from pathlib import Path

from astropy.io import fits
from image_io import IMAGE_SUFFIXES, read_tiff_header, tiff_shape

FITS_SUFFIXES = IMAGE_SUFFIXES


def discover_fits(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(
        (
            path
            for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() in FITS_SUFFIXES
        ),
        key=lambda path: path.name.casefold(),
    )


def inspect_fits(path: Path) -> tuple[tuple[int, ...], fits.Header]:
    if path.suffix.casefold() in {".tif", ".tiff"}:
        return tiff_shape(path), read_tiff_header(path)
    with fits.open(path, memmap=False, ignore_missing_end=True) as hdul:
        for hdu in hdul:
            if hdu.is_image and hdu.data is not None:
                return tuple(hdu.data.shape), hdu.header.copy()
    raise ValueError(f"No image HDU found in {path.name}")
