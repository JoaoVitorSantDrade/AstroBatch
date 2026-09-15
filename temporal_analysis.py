"""Timestamp grouping and seeing-trend analysis for Flow outputs.

This module is deliberately metadata-only: it never moves frames, changes
selection, or feeds a weight into the scientific reducer.  A missing or bad
``DATE-OBS`` is represented explicitly and remains usable by legacy Flow
readers.
"""

from __future__ import annotations

from datetime import datetime, timezone
import math
from pathlib import Path
import re
from typing import Any, Iterable

from astropy.io import fits

from app.infrastructure.json_store import atomic_json_write


TIMESTAMP_SCHEMA_VERSION = 1
DEFAULT_GAP_MINUTES = 15.0
DEFAULT_SEEING_SIGMA = 3.0
FITS_SUFFIXES = {".fit", ".fits", ".fts"}


def _natural_key(path_or_name: Path | str) -> tuple[Any, ...]:
    text = Path(path_or_name).name if isinstance(path_or_name, Path) else str(path_or_name)
    return tuple(
        (0, int(piece)) if piece.isdigit() else (1, piece)
        for piece in re.split(r"(\d+)", text.casefold())
    )


def _finite(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def parse_date_obs(value: Any) -> dict[str, Any]:
    """Parse the explicitly supported FITS ISO timestamp forms.

    A timezone-less FITS timestamp is interpreted as UTC per the FITS
    convention, while ``timezone_present`` records that the original value
    carried no offset.  Invalid or absent values never raise.
    """

    if isinstance(value, (bytes, bytearray)):
        raw = value.decode("ascii", errors="replace").strip().strip("'")
    else:
        raw = str(value).strip().strip("'") if value is not None else ""
    if not raw:
        return {
            "timestamp": None,
            "timestamp_utc": None,
            "timestamp_normalized": None,
            "timestamp_state": "unknown",
            "timezone": None,
            "timezone_present": False,
            "epoch_s": None,
            "raw": None,
        }

    candidate = raw
    if candidate.endswith(("Z", "z")):
        candidate = candidate[:-1] + "+00:00"
    candidate = candidate.replace(" ", "T", 1)
    parsed: datetime | None = None
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError:
        # A few capture programs omit the ``T`` or fractional part in a way
        # older Python versions do not accept through fromisoformat.
        for pattern in (
            "%Y-%m-%dT%H:%M:%S.%f",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%d %H:%M:%S.%f",
            "%Y-%m-%d %H:%M:%S",
        ):
            try:
                parsed = datetime.strptime(raw, pattern)
                break
            except ValueError:
                continue

    if parsed is None:
        return {
            "timestamp": raw,
            "timestamp_utc": None,
            "timestamp_normalized": None,
            "timestamp_state": "unknown",
            "timezone": None,
            "timezone_present": False,
            "epoch_s": None,
            "raw": raw,
        }

    timezone_present = parsed.tzinfo is not None
    offset_name = None
    if timezone_present:
        offset = parsed.utcoffset()
        if offset == timezone.utc.utcoffset(parsed):
            offset_name = "UTC"
        elif offset is not None:
            total_minutes = int(offset.total_seconds() // 60)
            sign = "+" if total_minutes >= 0 else "-"
            absolute = abs(total_minutes)
            offset_name = f"{sign}{absolute // 60:02d}:{absolute % 60:02d}"
    normalized = parsed if timezone_present else parsed.replace(tzinfo=timezone.utc)
    utc_value = normalized.astimezone(timezone.utc)
    normalized_text = utc_value.isoformat(timespec="microseconds").replace("+00:00", "Z")
    try:
        epoch_s = float(utc_value.timestamp())
    except (OverflowError, OSError, ValueError):
        return {
            "timestamp": raw,
            "timestamp_utc": None,
            "timestamp_normalized": None,
            "timestamp_state": "unknown",
            "timezone": None,
            "timezone_present": timezone_present,
            "epoch_s": None,
            "raw": raw,
        }
    return {
        "timestamp": raw,
        "timestamp_utc": normalized_text,
        "timestamp_normalized": normalized_text,
        "timestamp_state": "valid",
        "timezone": offset_name if timezone_present else None,
        "timezone_present": timezone_present,
        "epoch_s": epoch_s,
        "raw": raw,
    }


def read_fits_timestamp(path: Path) -> dict[str, Any]:
    """Read ``DATE-OBS``/``DATEOBS`` from any FITS header without pixels."""

    try:
        with fits.open(path, memmap=False, lazy_load_hdus=True) as hdul:
            value = None
            for hdu in hdul:
                header = getattr(hdu, "header", None)
                if header is None:
                    continue
                for key in ("DATE-OBS", "DATEOBS"):
                    if key in header:
                        value = header[key]
                        break
                if value is not None:
                    break
        parsed = parse_date_obs(value)
    except Exception:
        parsed = parse_date_obs(None)
        parsed["timestamp_state"] = "unknown"
    return parsed


def _frame_quality(flow_frame: dict[str, Any] | None) -> dict[str, float | None]:
    frame = flow_frame if isinstance(flow_frame, dict) else {}
    metrics = frame.get("metrics") if isinstance(frame.get("metrics"), dict) else {}

    def first(*keys: str) -> float | None:
        for source in (frame, metrics):
            for key in keys:
                value = _finite(source.get(key))
                if value is not None:
                    return value
        return None

    fwhm = first("fwhm", "shape_fwhm")
    roundness = first("roundness")
    star_count = first("star_count", "shape_star_count")
    quality = first("quality")
    if quality is None and star_count is not None:
        quality = star_count / max(fwhm or 0.0, 0.1)
    return {
        "fwhm": fwhm,
        "roundness": roundness,
        "star_count": star_count,
        "quality": quality,
    }


def _median(values: Iterable[float | None]) -> float | None:
    usable = sorted(value for value in values if value is not None and math.isfinite(value))
    if not usable:
        return None
    middle = len(usable) // 2
    if len(usable) % 2:
        return float(usable[middle])
    return float((usable[middle - 1] + usable[middle]) / 2.0)


def _mad(values: Iterable[float | None], center: float | None) -> float | None:
    if center is None:
        return None
    return _median(abs(value - center) for value in values if value is not None)


def _quality_summary(frames: list[dict[str, Any]], global_quality: dict[str, float | None] | None = None) -> dict[str, Any]:
    quality = _median(frame.get("quality") for frame in frames)
    fwhm = _median(frame.get("fwhm") for frame in frames)
    roundness = _median(frame.get("roundness") for frame in frames)
    star_count = _median(frame.get("star_count") for frame in frames)
    summary = {
        "frame_count": len(frames),
        "timed_frame_count": sum(frame.get("timestamp_state") == "valid" for frame in frames),
        "accepted_frame_count": sum(frame.get("status") == "accepted" for frame in frames),
        "median_quality": quality,
        "median_fwhm": fwhm,
        "median_roundness": roundness,
        "median_star_count": star_count,
    }
    if global_quality and quality is not None and global_quality.get("median") is not None:
        spread = global_quality.get("robust_sigma") or 0.0
        threshold = float(global_quality["median"]) - DEFAULT_SEEING_SIGMA * float(spread)
        summary["review_suggested"] = bool(spread > 0 and quality < threshold)
        summary["review_reason"] = "quality_below_robust_session_baseline" if summary["review_suggested"] else None
    else:
        summary["review_suggested"] = False
        summary["review_reason"] = None
    return summary


def build_temporal_report(
    batch_dir: Path,
    flow_data: dict[str, Any] | None = None,
    gap_minutes: float = DEFAULT_GAP_MINUTES,
    seeing_sigma: float = DEFAULT_SEEING_SIGMA,
    timestamp_cache: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build a deterministic logical grouping and seeing report for a batch."""

    try:
        gap = float(gap_minutes)
    except (TypeError, ValueError, OverflowError):
        gap = DEFAULT_GAP_MINUTES
    if not math.isfinite(gap) or gap <= 0:
        gap = DEFAULT_GAP_MINUTES
    try:
        sigma = float(seeing_sigma)
    except (TypeError, ValueError, OverflowError):
        sigma = DEFAULT_SEEING_SIGMA
    if not math.isfinite(sigma) or sigma <= 0:
        sigma = DEFAULT_SEEING_SIGMA

    flow_frames = flow_data.get("frames", {}) if isinstance(flow_data, dict) else {}
    if not isinstance(flow_frames, dict):
        flow_frames = {}
    try:
        files = sorted(
            [path for path in Path(batch_dir).iterdir() if path.is_file() and path.suffix.lower() in FITS_SUFFIXES],
            key=_natural_key,
        )
    except OSError:
        files = []
    known_names = set(flow_frames)
    for name in known_names:
        path = Path(batch_dir) / str(name)
        if path.is_file() and path.suffix.lower() in FITS_SUFFIXES and path not in files:
            files.append(path)
    files.sort(key=_natural_key)

    records: list[dict[str, Any]] = []
    for path in files:
        quality = _frame_quality(flow_frames.get(path.name))
        frame_info = flow_frames.get(path.name) if isinstance(flow_frames.get(path.name), dict) else {}
        # New Flow JSON already carries the normalized timestamp.  Reuse it
        # instead of opening every FITS a second time while building the
        # sidecar; legacy JSON without ``timestamp_state`` falls back to the
        # original header reader for compatibility.
        timestamp = _timestamp_from_flow_frame(frame_info)
        if timestamp is None:
            timestamp = (timestamp_cache or {}).get(path.name) or read_fits_timestamp(path)
        records.append({
            "frame": path.name,
            "path": str(path),
            **{key: timestamp.get(key) for key in ("timestamp", "timestamp_utc", "timestamp_normalized", "timestamp_state", "timezone", "timezone_present", "epoch_s")},
            "status": str(frame_info.get("status", "unknown")),
            **quality,
        })

    timed = sorted(
        (record for record in records if record.get("timestamp_state") == "valid"),
        key=lambda record: (float(record["epoch_s"]), _natural_key(record["frame"])),
    )
    untimed = sorted(
        (record for record in records if record.get("timestamp_state") != "valid"),
        key=lambda record: _natural_key(record["frame"]),
    )
    ordered = timed + untimed
    qualities = [record.get("quality") for record in timed]
    median_quality = _median(qualities)
    mad_quality = _mad(qualities, median_quality)
    robust_sigma = 1.4826 * mad_quality if mad_quality is not None else None
    global_quality = {"median": median_quality, "mad": mad_quality, "robust_sigma": robust_sigma}

    groups: list[dict[str, Any]] = []
    current: list[dict[str, Any]] = []
    previous_epoch: float | None = None
    gap_seconds = gap * 60.0
    # Epoch conversion is floating point; tolerate sub-microsecond noise so a
    # timestamp exactly at the configured boundary remains in the same group.
    gap_tolerance = max(1e-9, abs(gap_seconds) * 1e-12)
    for record in timed:
        epoch = float(record["epoch_s"])
        if current and previous_epoch is not None and epoch - previous_epoch > gap_seconds + gap_tolerance:
            groups.append(current)
            current = []
        current.append(record)
        previous_epoch = epoch
    if current:
        groups.append(current)
    if untimed:
        groups.append(untimed)

    serialized_groups: list[dict[str, Any]] = []
    for index, group_frames in enumerate(groups, start=1):
        timed_frames = [frame for frame in group_frames if frame.get("timestamp_state") == "valid"]
        first = timed_frames[0] if timed_frames else None
        last = timed_frames[-1] if timed_frames else None
        gap_before = None
        if first is not None and serialized_groups:
            previous_end = serialized_groups[-1].get("end_epoch_s")
            if previous_end is not None:
                gap_before = float(first["epoch_s"] - previous_end)
        summary = _quality_summary(group_frames, global_quality)
        # Use the caller's sigma for the review decision, not a hard-coded
        # value.  A zero MAD intentionally produces no automatic suggestion.
        if summary.get("median_quality") is not None and median_quality is not None and robust_sigma and robust_sigma > 0:
            threshold = median_quality - sigma * robust_sigma
            summary["review_suggested"] = bool(summary["median_quality"] < threshold)
            summary["review_reason"] = "quality_below_robust_session_baseline" if summary["review_suggested"] else None
        serialized_groups.append({
            "id": f"group_{index:03d}" if timed_frames else "unknown",
            "frames": [frame["frame"] for frame in group_frames],
            "start": first.get("timestamp_utc") if first else None,
            "end": last.get("timestamp_utc") if last else None,
            "start_epoch_s": first.get("epoch_s") if first else None,
            "end_epoch_s": last.get("epoch_s") if last else None,
            "duration_seconds": float(last["epoch_s"] - first["epoch_s"]) if first and last else None,
            "gap_before_seconds": gap_before,
            "timing_state": "valid" if timed_frames else "unknown",
            "quality": summary,
        })

    for record in ordered:
        record.pop("path", None)
        record.pop("epoch_s", None)

    return {
        "schema_version": TIMESTAMP_SCHEMA_VERSION,
        "kind": "batch_temporal_analysis",
        "batch": Path(batch_dir).name,
        "gap_minutes": gap,
        "seeing_sigma": sigma,
        "ordering": "timestamp_then_natural_name",
        "frames": ordered,
        "groups": serialized_groups,
        "seeing": {
            "median_quality": median_quality,
            "median_fwhm": _median(record.get("fwhm") for record in ordered),
            "median_roundness": _median(record.get("roundness") for record in ordered),
            "median_star_count": _median(record.get("star_count") for record in ordered),
            "mad_quality": mad_quality,
            "robust_sigma_quality": robust_sigma,
            "review_only": True,
        },
        "unknown_timestamp_frames": [record["frame"] for record in untimed],
        "files_unchanged": True,
    }


def _timestamp_from_flow_frame(frame_info: dict[str, Any] | None) -> dict[str, Any] | None:
    """Return a timestamp already normalized by Flow, if present.

    ``None`` means the caller is looking at an old Flow schema and should read
    the FITS header.  An explicit ``unknown`` state is returned as a valid
    cached result so malformed/missing headers are not reopened repeatedly.
    """

    if not isinstance(frame_info, dict) or "timestamp_state" not in frame_info:
        return None
    state = str(frame_info.get("timestamp_state") or "unknown")
    parsed: dict[str, Any] = {
        "timestamp": frame_info.get("timestamp"),
        "timestamp_utc": frame_info.get("timestamp_utc"),
        "timestamp_normalized": frame_info.get("timestamp_normalized"),
        "timestamp_state": state,
        "timezone": frame_info.get("timezone"),
        "timezone_present": bool(frame_info.get("timezone_present", False)),
        "epoch_s": frame_info.get("epoch_s"),
    }
    if state == "valid" and parsed.get("epoch_s") is None:
        reparsed = parse_date_obs(
            parsed.get("timestamp_normalized")
            or parsed.get("timestamp_utc")
            or parsed.get("timestamp")
        )
        if reparsed.get("timestamp_state") == "valid":
            parsed.update(reparsed)
        else:
            parsed["timestamp_state"] = "unknown"
            parsed["epoch_s"] = None
    return parsed


def enrich_flow_frames(
    batch_dir: Path,
    flow_data: dict[str, Any],
    timestamp_cache: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Add timestamp metadata to each Flow frame while preserving old fields."""

    if not isinstance(flow_data, dict):
        return flow_data
    frames = flow_data.get("frames")
    if not isinstance(frames, dict):
        return flow_data
    for name, frame in frames.items():
        if not isinstance(frame, dict):
            continue
        # Current Flow already parsed DATE-OBS while the FITS header was open
        # for pixel processing.  Reuse that value; only legacy JSONs without
        # ``timestamp_state`` need a second header-only read.
        parsed = _timestamp_from_flow_frame(frame)
        if parsed is None:
            parsed = read_fits_timestamp(Path(batch_dir) / str(name))
        if timestamp_cache is not None:
            timestamp_cache[str(name)] = parsed
        for key in (
            "timestamp",
            "timestamp_utc",
            "timestamp_normalized",
            "timestamp_state",
            "timezone",
            "timezone_present",
            "epoch_s",
        ):
            frame[key] = parsed.get(key)
    flow_data["temporal_schema_version"] = TIMESTAMP_SCHEMA_VERSION
    return flow_data


def write_temporal_report(path: Path, report: dict[str, Any]) -> None:
    """Persist a temporal report through the project's atomic JSON writer."""

    atomic_json_write(Path(path), report)


def _flatten_session_reports(
    reports: list[dict[str, Any]],
    gap_minutes: float,
    seeing_sigma: float,
) -> dict[str, Any]:
    """Build a session-wide timeline while retaining batch ownership."""

    flattened: list[dict[str, Any]] = []
    for report in reports:
        batch_name = str(report.get("batch", "batch"))
        for frame in report.get("frames", []) or []:
            if not isinstance(frame, dict):
                continue
            item = {"batch": batch_name, **frame}
            parsed = parse_date_obs(frame.get("timestamp_normalized") or frame.get("timestamp_utc"))
            if frame.get("timestamp_state") == "valid" and parsed.get("timestamp_state") == "valid":
                item["_epoch_s"] = parsed["epoch_s"]
            else:
                item["_epoch_s"] = None
            flattened.append(item)

    timed = sorted(
        (item for item in flattened if item.get("_epoch_s") is not None),
        key=lambda item: (float(item["_epoch_s"]), str(item["batch"]).casefold(), _natural_key(item["frame"])),
    )
    untimed = sorted(
        (item for item in flattened if item.get("_epoch_s") is None),
        key=lambda item: (str(item["batch"]).casefold(), _natural_key(item["frame"])),
    )
    median_quality = _median(item.get("quality") for item in timed)
    mad_quality = _mad((item.get("quality") for item in timed), median_quality)
    robust_sigma = 1.4826 * mad_quality if mad_quality is not None else None
    global_quality = {"median": median_quality, "robust_sigma": robust_sigma}

    groups: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    previous_epoch: float | None = None
    gap_seconds = gap_minutes * 60.0
    gap_tolerance = max(1e-9, abs(gap_seconds) * 1e-12)
    for item in timed:
        epoch = float(item["_epoch_s"])
        if current and previous_epoch is not None and epoch - previous_epoch > gap_seconds + gap_tolerance:
            groups.append(current)
            current = []
        current.append(item)
        previous_epoch = epoch
    if current:
        groups.append(current)
    if untimed:
        groups.append(untimed)

    serialized: list[dict[str, Any]] = []
    for index, group in enumerate(groups, start=1):
        timed_group = [item for item in group if item.get("_epoch_s") is not None]
        first = timed_group[0] if timed_group else None
        last = timed_group[-1] if timed_group else None
        gap_before = None
        if first is not None and serialized and serialized[-1].get("_end_epoch_s") is not None:
            gap_before = float(first["_epoch_s"] - serialized[-1]["_end_epoch_s"])
        summary = _quality_summary(group, global_quality)
        if summary.get("median_quality") is not None and median_quality is not None and robust_sigma and robust_sigma > 0:
            threshold = median_quality - seeing_sigma * robust_sigma
            summary["review_suggested"] = bool(summary["median_quality"] < threshold)
            summary["review_reason"] = "quality_below_robust_session_baseline" if summary["review_suggested"] else None
        serialized.append({
            "id": f"group_{index:03d}" if timed_group else "unknown",
            "frames": [f"{item['batch']}/{item['frame']}" for item in group],
            "start": first.get("timestamp_utc") if first else None,
            "end": last.get("timestamp_utc") if last else None,
            "duration_seconds": float(last["_epoch_s"] - first["_epoch_s"]) if first and last else None,
            "gap_before_seconds": gap_before,
            "timing_state": "valid" if timed_group else "unknown",
            "quality": summary,
            "_end_epoch_s": last.get("_epoch_s") if last else None,
        })

    for group in serialized:
        group.pop("_end_epoch_s", None)
    flat_output = []
    for item in timed + untimed:
        output = {key: value for key, value in item.items() if not key.startswith("_")}
        flat_output.append(output)
    return {
        "frames": flat_output,
        "groups": serialized,
        "unknown_timestamp_frames": [f"{item['batch']}/{item['frame']}" for item in untimed],
        "seeing": {
            "median_quality": median_quality,
            "median_fwhm": _median(item.get("fwhm") for item in timed),
            "median_roundness": _median(item.get("roundness") for item in timed),
            "median_star_count": _median(item.get("star_count") for item in timed),
            "mad_quality": mad_quality,
            "robust_sigma_quality": robust_sigma,
            "review_only": True,
        },
    }


def build_session_temporal_report(
    base_dir: Path,
    gap_minutes: float = DEFAULT_GAP_MINUTES,
    seeing_sigma: float = DEFAULT_SEEING_SIGMA,
) -> dict[str, Any]:
    """Aggregate per-batch reports without flattening away batch identity."""

    try:
        session_gap = float(gap_minutes)
    except (TypeError, ValueError, OverflowError):
        session_gap = DEFAULT_GAP_MINUTES
    if not math.isfinite(session_gap) or session_gap <= 0:
        session_gap = DEFAULT_GAP_MINUTES
    try:
        session_sigma = float(seeing_sigma)
    except (TypeError, ValueError, OverflowError):
        session_sigma = DEFAULT_SEEING_SIGMA
    if not math.isfinite(session_sigma) or session_sigma <= 0:
        session_sigma = DEFAULT_SEEING_SIGMA

    reports: list[dict[str, Any]] = []
    try:
        batches = sorted(
            [path for path in Path(base_dir).iterdir() if path.is_dir() and "batch" in path.name.casefold()],
            key=_natural_key,
        )
    except OSError:
        batches = []
    for batch in batches:
        flow_path = batch / "flow_local.json"
        try:
            import json

            flow_data = json.loads(flow_path.read_text(encoding="utf-8")) if flow_path.exists() else None
        except Exception:
            flow_data = None
        reports.append(build_temporal_report(batch, flow_data, session_gap, session_sigma))
    session_view = _flatten_session_reports(reports, session_gap, session_sigma)
    return {
        "schema_version": TIMESTAMP_SCHEMA_VERSION,
        "kind": "session_temporal_analysis",
        "base_dir": str(Path(base_dir)),
        "gap_minutes": session_gap,
        "seeing_sigma": session_sigma,
        "batches": reports,
        **session_view,
        "review_only": True,
        "files_unchanged": True,
    }
