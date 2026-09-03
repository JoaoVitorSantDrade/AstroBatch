from __future__ import annotations

from pathlib import Path

from astropy.io import fits

FITS_SUFFIXES = {".fit", ".fits", ".fts"}


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
    with fits.open(path, memmap=False, ignore_missing_end=True) as hdul:
        for hdu in hdul:
            if hdu.is_image and hdu.data is not None:
                return tuple(hdu.data.shape), hdu.header.copy()
    raise ValueError(f"No image HDU found in {path.name}")
