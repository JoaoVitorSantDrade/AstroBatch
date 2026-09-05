"""CPU HDR fusion for linear, calibrated, already aligned FITS frames.

The persisted product is deliberately uint16.  Fusion remains float32/64 in
RAM and records the radiometric range so callers can restore CALNORM values.
"""
from __future__ import annotations

import json
import os
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
from astropy.io import fits

from calibration_logic import _restore_normalized_values, _sanitize_float_header
from app.infrastructure.json_store import atomic_json_write


@dataclass(frozen=True)
class HDRConfig:
    input_paths: tuple[Path, ...]
    output_path: Path
    saturation: float | None = None
    row_band: int = 256
    noise_floor: float = 1.0
    exposure_override: float | None = None
    disagreement_sigma: float = 5.0
    overwrite: bool = True

def validate_hdr_config(cfg: HDRConfig) -> HDRConfig:
    if not cfg.input_paths or not cfg.output_path: raise ValueError("HDR input/output required")
    if not isinstance(cfg.row_band, int) or isinstance(cfg.row_band, bool) or cfg.row_band < 1: raise ValueError("row_band must be a positive integer")
    for name, value in (("noise_floor", cfg.noise_floor), ("disagreement_sigma", cfg.disagreement_sigma)):
        if isinstance(value, bool) or not np.isfinite(value) or value <= 0: raise ValueError(f"{name} must be finite and positive")
    for name, value in (("saturation", cfg.saturation), ("exposure_override", cfg.exposure_override)):
        if value is not None and (isinstance(value, bool) or not np.isfinite(value) or value <= 0): raise ValueError(f"{name} must be finite and positive")
    return cfg


def _config(raw: dict[str, Any]) -> HDRConfig:
    paths = raw.get("input_paths", raw.get("paths", []))
    if isinstance(paths, (str, Path)):
        paths = [paths]
    if not paths:
        raise ValueError("HDR requires input_paths")
    output = raw.get("output_path")
    if not output:
        raise ValueError("HDR requires output_path")
    for key in ("noise_floor", "disagreement_sigma", "saturation", "exposure_override"):
        if isinstance(raw.get(key), bool):
            raise ValueError(f"{key} must be numeric, not boolean")
    noise = float(raw.get("noise_floor", 1.0)); rb = raw.get("row_band", 256)
    if isinstance(rb, bool) or int(rb) != rb: raise ValueError("row_band must be a positive integer")
    return validate_hdr_config(HDRConfig(tuple(Path(p) for p in paths), Path(output),
        None if raw.get("saturation") in (None, "") else float(raw["saturation"]),
        int(rb), noise,
        None if raw.get("exposure_override") in (None, "") else float(raw["exposure_override"]),
        float(raw.get("disagreement_sigma", 5.0)), bool(raw.get("overwrite", True))))


build_hdr_config = _config


def _inspect(path: Path, override: float | None) -> tuple[tuple[int, ...], fits.Header, float, int]:
    with fits.open(path, memmap=False, ignore_missing_end=True) as hdul:
        for index, hdu in enumerate(hdul):
            if hdu.is_image and hdu.shape is not None and len(hdu.shape) in (2, 3) and hdu.name not in {"VALID_MASK", "SAT_MASK", "DISAGREE"}:
                header = _sanitize_float_header(hdu.header)
                aliases = [float(header[k]) for k in ("EXPTIME", "EXPOSURE") if k in header]
                if override is None and len(aliases) == 2 and not np.isclose(*aliases, rtol=1e-6, atol=0):
                    raise ValueError(f"{path.name}: conflicting EXPTIME and EXPOSURE")
                exposure = override if override is not None else (aliases[0] if aliases else None)
                if exposure is None or not np.isfinite(float(exposure)) or float(exposure) <= 0:
                    raise ValueError(f"{path.name}: EXPTIME missing or invalid")
                if header.get("CALNORM"):
                    lo, hi = float(header.get("CALMIN", np.nan)), float(header.get("CALMAX", np.nan))
                    if not np.isfinite(lo+hi) or hi <= lo:
                        raise ValueError(f"{path.name}: invalid calibration range")
                header["EXPTIME"] = float(exposure)
                return tuple(hdu.shape), header, float(exposure), index
    raise ValueError(f"{path.name}: no image found")

def _band(path: Path, index: int, y: int, y2: int, shape: tuple[int, ...], header: fits.Header | None = None, restore: bool = True) -> np.ndarray:
    with fits.open(path, memmap=False, ignore_missing_end=True) as hdul:
        section = hdul[index].section
        raw = section[y:y2, :] if len(shape) == 2 else section[:, y:y2, :]
        out = np.ascontiguousarray(raw, dtype=np.float32)
        if restore and header is not None and bool(header.get("CALNORM", False)):
            out = _restore_normalized_values(out, header)
        return out

def _mask_band(path: Path, y: int, y2: int, shape: tuple[int, ...]) -> np.ndarray | None:
    valid = None
    with fits.open(path, memmap=False, ignore_missing_end=True) as hdul:
        for hdu in hdul:
            name = str(hdu.name).upper()
            if name not in {"VALID_MASK", "VALIDMASK", "SAT_MASK"} or hdu.shape is None:
                continue
            if tuple(hdu.shape) not in (tuple(shape[-2:]), tuple(shape)):
                raise ValueError(f"{path.name}: {name} geometry mismatch")
            pixels = np.asarray(hdu.section[y:y2, :] if len(hdu.shape) == 2 else hdu.section[:, y:y2, :], dtype=bool)
            if name == "SAT_MASK":
                pixels = ~pixels
            if pixels.ndim == 3:
                pixels = np.all(pixels, axis=0)
            valid = pixels if valid is None else valid & pixels
    return valid


def _atomic_write(data: np.ndarray, valid: np.ndarray, disagree: np.ndarray,
                  header: fits.Header, output: Path, lo: float, hi: float, nframes: int = 0,
                  cancel: threading.Event | None = None, provenance: dict | None = None) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    scale = 65535.0 / (hi - lo) if hi > lo else 1.0
    encoded = np.clip(np.nan_to_num((data - lo) * scale, nan=0.0), 0, 65535).astype(np.uint16)
    h = _sanitize_float_header(header)
    unit = str(header.get("BUNIT", "ADU")).strip()
    unit = unit if unit.lower().endswith("/s") else unit + "/s"
    h.update({"CALNORM": (True, "HDR radiometric normalization"), "CALHDR": (True, "HDR radiance product"), "EXPTIME": (1.0, "Radiance normalized to one second"), "BUNIT": ("ADU/s", "Exposure normalized radiance"), "HDRNFRM": (int(nframes), "Input frame count"), "CALMIN": (float(lo), "HDR minimum"),
              "CALMAX": (float(hi), "HDR maximum"), "HDRBITS": (16, "Persisted HDR bit depth"),
              "HDRQERR": (float((hi - lo) / 65535.0), "Maximum quantization step"),
              "HDRWARN": ("16-bit quantization is not arbitrary lossless", "HDR precision note")})
    h["BUNIT"] = unit
    for key in ("SATURATE", "SATLEVEL", "EXPOSURE"):
        h.remove(key, ignore_missing=True)
    hdus = fits.HDUList([fits.PrimaryHDU(encoded, h), fits.ImageHDU(valid.astype(np.uint8), name="VALID_MASK"),
                         fits.ImageHDU(disagree.astype(np.uint8), name="DISAGREE")])
    if provenance is not None:
        hdus[0].header["HDRQWARN"] = bool(provenance["precision_warning"])
        payload = json.dumps(provenance, ensure_ascii=False, allow_nan=False).encode("utf-8")
        hdus.append(fits.ImageHDU(np.frombuffer(payload, dtype=np.uint8).copy(), name="HDR_META"))
    fd, tmp = tempfile.mkstemp(prefix=f".{output.name}.", suffix=".tmp", dir=str(output.parent)); os.close(fd)
    try:
        hdus.writeto(tmp, overwrite=True, output_verify="ignore")
        if cancel is not None and cancel.is_set():
            raise InterruptedError("HDR cancelled before commit")
        os.replace(tmp, output)
    finally:
        hdus.close()
        if os.path.exists(tmp): os.unlink(tmp)


def run_hdr_pipeline(config: dict[str, Any] | HDRConfig, log: Callable[[str], None],
                     progress: Callable[[int, int, str], None], cancel: threading.Event | None) -> dict[str, Any]:
    """Fuse linear frames; returns a typed, JSON-serializable operation result."""
    try:
        cfg = validate_hdr_config(config) if isinstance(config, HDRConfig) else _config(config)
        for key in ("noise_floor", "disagreement_sigma", "row_band"):
            value = getattr(cfg, key)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{key} must be finite and positive")
        if int(cfg.row_band) != cfg.row_band:
            raise ValueError("row_band must be an integer")
        for key in ("saturation", "exposure_override"):
            value = getattr(cfg, key)
            if value is not None and (not np.isfinite(value) or value <= 0):
                raise ValueError(f"{key} must be finite and positive")
        if cancel is not None and cancel.is_set(): return {"status": "cancelled", "output_path": None}
        paths = tuple(dict.fromkeys(p.resolve() for p in cfg.input_paths))
        if len(paths) < 2: raise ValueError("HDR requires at least two distinct input paths")
        if cfg.output_path.resolve() in paths: raise ValueError("output_path cannot overwrite an input")
        if cfg.output_path.exists() and not cfg.overwrite: raise FileExistsError(str(cfg.output_path))
        meta = [_inspect(p, cfg.exposure_override) for p in paths]
        shape = meta[0][0]
        if any(x[0] != shape for x in meta): raise ValueError("incompatible image geometry")
        headers = [x[1] for x in meta]
        # Refuse incompatible optical metadata when present.
        for key in ("FILTER", "XBINNING", "YBINNING", "BAYERPAT", "BUNIT", "EGAIN", "ISOSPEED"):
            vals = {str(h.get(key)).strip() if h.get(key) is not None else None for h in headers}
            if len(vals) > 1: raise ValueError(f"incompatible {key} metadata")
        gains = {str(h.get("GAIN")) if h.get("GAIN") is not None else None for h in headers}
        if len(gains) > 1: raise ValueError("incompatible GAIN metadata")
        height, width = shape[-2:]; n = len(paths); channels = shape[0] if len(shape) == 3 else 1
        if len(shape) == 3 and channels not in (3, 4): raise ValueError("unsupported RGB channel count")
        divisors = [1.0 if str(h.get("BUNIT", "")).strip().lower().endswith("/s") else m[2]
                    for h, m in zip(headers, meta)]
        result = np.full(shape, np.nan, dtype=np.float32); valid = np.zeros(shape, bool); disagreement = np.zeros(shape, bool)
        sat_values = [cfg.saturation if cfg.saturation is not None else h.get("SATURATE", h.get("SATLEVEL")) for h in headers]
        if any(s is not None and (not np.isfinite(float(s)) or float(s) <= 0) for s in sat_values):
            raise ValueError("Invalid saturation metadata")
        unknown_saturation = any(s is None and not h.get("SATKNOWN", False) for s,h in zip(sat_values,headers))
        if unknown_saturation:
            log("Saturation is unknown for some inputs; clipped pixels cannot be inferred from CALNORM maxima.")
        total = max(1, (height + cfg.row_band - 1) // cfg.row_band)
        for band, y in enumerate(range(0, height, cfg.row_band), 1):
            if cancel is not None and cancel.is_set(): return {"status": "cancelled", "output_path": None}
            y2 = min(height, y + cfg.row_band); band_shape = (channels, y2-y, width)
            weighted = np.zeros(band_shape, np.float64); weights = np.zeros_like(weighted)
            sum_r = np.zeros(band_shape, np.float64); sum_r2 = np.zeros(band_shape, np.float64); count_r = np.zeros(band_shape, np.int32)
            for index, (path, (fshape, header, exposure, hdu_index)) in enumerate(zip(paths, meta)):
                if cancel is not None and cancel.is_set():
                    return {"status": "cancelled", "output_path": None}
                raw = _band(path, hdu_index, y, y2, fshape, header, restore=False)
                if len(shape) == 2: raw = raw[None, ...]
                ok = np.isfinite(raw)
                source_mask = _mask_band(path, y, y2, fshape)
                if source_mask is not None:
                    if source_mask.shape != (y2-y, width):
                        raise ValueError(f"{path.name}: VALID_MASK geometry mismatch")
                    ok &= source_mask
                sat = sat_values[index]
                if sat is not None: ok &= raw < float(sat)
                if bool(header.get("CALNORM", False)): raw = _restore_normalized_values(raw, header)
                divisor = divisors[index]
                radiance = raw / divisor
                ok &= np.isfinite(radiance)
                w = np.where(ok, divisor * divisor / (cfg.noise_floor ** 2), 0.0)
                clean = np.where(ok, radiance, 0.0)
                weighted += clean * w; weights += w
                sum_r += clean; sum_r2 += clean * clean; count_r += ok
            with np.errstate(invalid="ignore", divide="ignore"): out = weighted / weights
            if len(shape) == 2: out, band_valid = out[0], (weights > 0)[0]
            else: band_valid = weights > 0
            if len(shape) == 3:
                result[:, y:y2, :] = out; valid[:, y:y2, :] = band_valid
            else:
                result[y:y2, :] = out; valid[y:y2, :] = band_valid
            if n > 1:
                mean_r = sum_r / np.maximum(count_r, 1)
                variance_r = np.maximum(sum_r2 / np.maximum(count_r, 1) - mean_r * mean_r, 0.0)
                # Conservative scatter flag, not an automatic rejection or a
                # significance probability; use the noisiest exposure's noise.
                noise_r = cfg.noise_floor / min(divisors)
                bad = (count_r >= 2) & (np.sqrt(variance_r) > cfg.disagreement_sigma * noise_r)
                if len(shape) == 3: disagreement[:, y:y2, :] = bad
                else: disagreement[y:y2, :] = bad[0]
            progress(band, total, f"HDR fusion ({band}/{total})")
        finite = result[np.isfinite(result) & valid]
        if finite.size == 0: raise ValueError("all pixels are invalid or saturated; no HDR output written")
        lo, hi = (float(np.min(finite)), float(np.max(finite))) if finite.size else (0.0, 1.0)
        if hi <= lo: hi = lo + 1.0
        if cancel is not None and cancel.is_set(): return {"status": "cancelled", "output_path": None}
        valid_output = valid if len(shape) == 2 else np.all(valid, axis=0)
        disagreement_output = disagreement if len(shape) == 2 else np.any(disagreement, axis=0)
        if not np.any(valid_output):
            raise ValueError("No pixel has valid coverage in all output channels")
        summary = {"status": "success", "output_path": str(cfg.output_path), "frames": n,
                   "capability": "equal_exposure" if len({x[2] for x in meta}) == 1 else "multi_exposure",
                   "valid_pixels": int(valid_output.sum()), "disagreement_pixels": int(disagreement_output.sum()),
                   "quantization_step": (hi-lo)/65535.0,
                   "precision_warning": bool((hi-lo)/65535.0 > cfg.noise_floor / np.sqrt(sum(d**2 for d in divisors))),
                   "unknown_saturation": unknown_saturation,
                   "exposure_groups": [{"seconds": t, "count": sum(m[2] == t for m in meta)} for t in sorted({m[2] for m in meta})],
                   "radiance_min": lo, "radiance_max": hi,
                   "exposure_provenance": [{"path": str(p), "exptime": m[2], "weight_model": "constant noise floor"} for p, m in zip(paths, meta)]}
        _atomic_write(result, valid_output, disagreement_output, headers[0], cfg.output_path, lo, hi, n, cancel, summary)
        try:
            atomic_json_write(cfg.output_path.with_suffix(cfg.output_path.suffix + ".json"), summary)
        except OSError as exc:
            summary["provenance_warning"] = str(exc)
            log(f"Science output saved; could not write JSON provenance: {exc}")
        return summary
    except InterruptedError:
        return {"status": "cancelled", "output_path": None}
    except Exception as exc:
        log(f"HDR failed: {exc}"); return {"status": "error", "output_path": None, "error": str(exc)}
