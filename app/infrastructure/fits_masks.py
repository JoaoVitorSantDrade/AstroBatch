"""Conservative science masks shared by calibration and alignment."""
import numpy as np
from astropy.io import fits


def read_science_masks(path, spatial_shape):
    spatial_shape = tuple(spatial_shape[:2])
    valid = np.ones(spatial_shape, bool)
    saturated = np.zeros(spatial_shape, bool)
    with fits.open(path, memmap=False, lazy_load_hdus=True) as hdul:
        for hdu in hdul:
            name = hdu.name.upper()
            if name not in {"VALID_MASK", "VALIDMASK", "SAT_MASK"}:
                continue
            if hdu.shape is None:
                raise ValueError(f"{name} has no pixels")
            shape = tuple(hdu.shape)
            if shape == spatial_shape:
                mask = np.asarray(hdu.data, bool)
            elif len(shape) == 3 and shape[1:] == spatial_shape:
                pixels = np.asarray(hdu.data, bool)
                mask = np.any(pixels,axis=0) if name == "SAT_MASK" else np.all(pixels,axis=0)
            else:
                raise ValueError(f"{name} geometry {shape} differs from {spatial_shape}")
            if name == "SAT_MASK":
                saturated |= mask
            else:
                valid &= mask
    return valid, saturated
