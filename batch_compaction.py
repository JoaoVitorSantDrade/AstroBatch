"""RAM-bounded per-batch integration for the Align -> Stack hand-off.

The compact path deliberately keeps the visible master separate from the
scientific state.  A TIFF is a convenient interchange image for Siril, while
the adjacent compressed NPZ retains the per-pixel accumulators needed by the
final Stack to weight batches by their actual contribution.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from astropy.io import fits

try:
    import psutil
except Exception:  # pragma: no cover - psutil is optional for library users
    psutil = None

from image_io import (
    ImageFormat,
    detect_format,
    read_image,
    write_npz_sidecar,
    write_sidecar_json,
    write_tiff,
)


SCHEMA_VERSION = 1
STATE_SUFFIX = ".astrobatch.npz"
MANIFEST_SUFFIX = ".json"


def _finite_weight(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return 1.0
    if not math.isfinite(result) or result <= 0:
        return 1.0
    return float(np.clip(result, 0.25, 2.0))


def _fingerprint(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {"path": str(path), "size": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)}


def _as_hwc(data: np.ndarray) -> np.ndarray:
    array = np.asarray(data)
    if array.ndim == 2:
        return array
    if array.ndim == 3 and array.shape[-1] in (3, 4):
        return array
    if array.ndim == 3 and array.shape[0] in (3, 4):
        return np.moveaxis(array, 0, -1)
    raise ValueError(f"Formato de imagem não suportado no pré-stack: {array.shape}")


def _visible_uint16(data: np.ndarray) -> np.ndarray:
    return np.clip(np.nan_to_num(data, nan=0.0, posinf=65535.0, neginf=0.0), 0, 65535).astype(
        np.uint16, copy=False
    )


def _atomic_fits_image(
    path: Path,
    data: np.ndarray,
    valid_mask: np.ndarray,
    sat_mask: np.ndarray,
    header: fits.Header | None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    clean = header.copy() if header is not None else fits.Header()
    for key in ("XTENSION", "PCOUNT", "GCOUNT", "CHECKSUM", "DATASUM"):
        clean.remove(key, ignore_missing=True)
    array = np.asarray(data, dtype=np.uint16)
    # The compact accumulator keeps RGB in the TIFF-friendly HWC layout,
    # while FITS conventionally stores the channel axis first.  Accept either
    # layout here so callers cannot accidentally publish a transposed RGB
    # product.
    if array.ndim == 3 and array.shape[-1] in (3, 4) and array.shape[0] not in (3, 4):
        array = np.moveaxis(array, -1, 0)
    output = fits.PrimaryHDU(data=array, header=clean)
    hdul = fits.HDUList([
        output,
        fits.ImageHDU(np.asarray(valid_mask, dtype=np.uint8), name="VALID_MASK"),
        fits.ImageHDU(np.asarray(sat_mask, dtype=np.uint8), name="SAT_MASK"),
    ])
    try:
        hdul.writeto(temporary, overwrite=True, output_verify="ignore")
        os.replace(temporary, path)
    finally:
        hdul.close()
        temporary.unlink(missing_ok=True)


@dataclass
class BatchAccumulator:
    """Deterministic, thread-safe accumulator for one acquisition batch."""

    batch_name: str
    output_dir: Path
    image_format: ImageFormat
    method: str = "Mean"
    rejection_method: str = "None"
    rejection_low: float = 3.0
    rejection_high: float = 3.0
    config_snapshot: dict[str, Any] = field(default_factory=dict)
    # ``None`` keeps the standalone accumulator useful outside Align.  The
    # pipeline passes its configured budget so a pathological sensor is
    # rejected before NumPy attempts an allocation large enough to destabilize
    # the host.
    memory_budget_mb: int | None = None
    expected_frames: int | None = None
    _next_sequence: int = 0
    _pending: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]] | None] = field(default_factory=dict)
    _data_shape: tuple[int, ...] | None = None
    _sum: np.ndarray | None = None
    _count: np.ndarray | None = None
    _weighted_sum: np.ndarray | None = None
    _weight_sum: np.ndarray | None = None
    _extreme: np.ndarray | None = None
    _extreme_valid: np.ndarray | None = None
    _sat_mask: np.ndarray | None = None
    _samples: list[np.ndarray] = field(default_factory=list, repr=False)
    _sample_masks: list[np.ndarray] = field(default_factory=list, repr=False)
    accepted: list[dict[str, Any]] = field(default_factory=list)
    rejected: list[dict[str, Any]] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def __post_init__(self) -> None:
        # Library callers historically passed plain strings in a few
        # integrations; normalize them once so the identity checks below are
        # reliable without changing the public constructor.
        if not isinstance(self.image_format, ImageFormat):
            self.image_format = ImageFormat(str(self.image_format).lower())

    def _memory_limit(self) -> int | None:
        available = None
        if psutil is not None:
            try:
                available = int(psutil.virtual_memory().available)
            except Exception:
                available = None
        if available is None:
            return None
        # Keep a substantial reserve for the OS, decoder, worker frames and
        # the final uint16/NPZ publication.  The explicit user budget remains
        # the tighter ceiling when one was supplied.
        limit = int(available * 0.60)
        if self.memory_budget_mb is not None and int(self.memory_budget_mb) > 0:
            limit = min(limit, int(self.memory_budget_mb) * 1024 * 1024 * 3 // 4)
        return max(0, limit)

    def _estimate_state_bytes(self, shape: tuple[int, ...]) -> int:
        elements = int(np.prod(shape, dtype=np.int64))
        pixels = int(np.prod(shape[:2], dtype=np.int64))
        # State plus the result/encoding temporaries used by ``finalize``.
        if self.method in {"Mean", "Sum"}:
            state = elements * np.dtype(np.float64).itemsize + pixels * np.dtype(np.uint32).itemsize
        elif self.method == "QualityWeightedMean":
            state = elements * np.dtype(np.float64).itemsize + pixels * np.dtype(np.float64).itemsize
        elif self.method in {"Maximum", "Minimum"}:
            state = elements * np.dtype(np.float32).itemsize
        elif self.method == "Median":
            # Exact median requires retaining the batch planes.  It remains a
            # valid compact option for bounded batches; the same RAM guard is
            # applied before the first frame so a large session fails cleanly
            # instead of growing an unbounded Python list.
            frames = max(1, int(self.expected_frames or 1))
            state = frames * (
                elements * np.dtype(np.float32).itemsize
                + pixels * np.dtype(bool).itemsize
            )
        else:
            raise ValueError(
                f"Método {self.method} não pode ser usado como acumulador compacto; "
                "use Mean/QualityWeightedMean e deixe a decisão hierárquica para o Stack."
            )
        masks = pixels * 2  # validity + saturation
        # A 1.5x headroom covers the float32 frame currently being consumed,
        # np.where's temporary and the uint16 visible image at publication.
        return int((state + masks) * 1.5)

    def _initialize(self, data: np.ndarray) -> None:
        shape = tuple(data.shape)
        if self._data_shape is not None:
            if self._data_shape != shape:
                raise ValueError(f"Geometria incompatível dentro do batch {self.batch_name}: {shape}")
            return
        estimated = self._estimate_state_bytes(shape)
        limit = self._memory_limit()
        if limit is not None and estimated > limit:
            available_mb = (limit / (1024 * 1024)) if limit else 0.0
            required_mb = estimated / (1024 * 1024)
            raise MemoryError(
                f"Pré-stack do batch {self.batch_name} exigiria ~{required_mb:.0f} MiB, "
                f"acima do limite seguro (~{available_mb:.0f} MiB). "
                "Reduza o lote/resolução ou aumente o orçamento de RAM."
            )
        spatial = shape[:2]
        sat_mask = np.zeros(spatial, dtype=bool)
        sum_state = count_state = weighted_state = weight_state = None
        extreme_state = extreme_valid_state = None
        if self.method in {"Mean", "Sum"}:
            sum_state = np.zeros(shape, dtype=np.float64)
            count_state = np.zeros(spatial, dtype=np.uint32)
        elif self.method == "QualityWeightedMean":
            weighted_state = np.zeros(shape, dtype=np.float64)
            weight_state = np.zeros(spatial, dtype=np.float64)
        elif self.method in {"Maximum", "Minimum"}:
            extreme_state = np.zeros(shape, dtype=np.float32)
            extreme_valid_state = np.zeros(spatial, dtype=bool)
        self._data_shape = shape
        self._sat_mask = sat_mask
        self._sum = sum_state
        self._count = count_state
        self._weighted_sum = weighted_state
        self._weight_sum = weight_state
        self._extreme = extreme_state
        self._extreme_valid = extreme_valid_state

    def _consume(
        self,
        sequence: int,
        payload: tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]] | None,
    ) -> None:
        if payload is None:
            return
        data, mask, sat_mask, info = payload
        self._initialize(data)
        valid = np.asarray(mask, dtype=bool)
        finite = np.all(np.isfinite(data), axis=2) if data.ndim == 3 else np.isfinite(data)
        valid &= finite
        weight = _finite_weight(info.get("weight", 1.0))
        if self._sat_mask is not None:
            self._sat_mask |= np.asarray(sat_mask, dtype=bool)
        if self.method in {"Mean", "Sum"}:
            assert self._sum is not None and self._count is not None
            if data.ndim == 3:
                self._sum += np.where(valid[:, :, None], data, 0.0)
            else:
                self._sum += np.where(valid, data, 0.0)
            self._count += valid.astype(np.uint32)
        elif self.method == "QualityWeightedMean":
            assert self._weighted_sum is not None and self._weight_sum is not None
            weighted = np.float64(weight) * data
            if data.ndim == 3:
                self._weighted_sum += np.where(valid[:, :, None], weighted, 0.0)
            else:
                self._weighted_sum += np.where(valid, weighted, 0.0)
            self._weight_sum += np.where(valid, weight, 0.0)
        else:
            if self.method == "Median":
                # ``np.nanmedian`` at finalize uses the exact frame order and
                # keeps invalid pixels out of the sample.  Copies are
                # deliberate: worker buffers are released after submit.
                projected = int((
                    (len(self._samples) + 1) * data.nbytes
                    + (len(self._sample_masks) + 1) * valid.nbytes
                    + self._estimate_state_bytes(data.shape) / 1.5
                ) * 1.5)
                limit = self._memory_limit()
                if limit is not None and projected > limit:
                    raise MemoryError(
                        f"A mediana do batch {self.batch_name} excederia o limite seguro de RAM; "
                        "reduza o tamanho do batch ou use Mean/QualityWeightedMean."
                    )
                self._samples.append(np.array(data, dtype=np.float32, copy=True))
                self._sample_masks.append(np.array(valid, dtype=bool, copy=True))
                self.accepted.append({"sequence": sequence, **info})
                return
            assert self._extreme is not None and self._extreme_valid is not None
            if data.ndim == 3:
                valid3 = valid[:, :, None]
                if self.method == "Maximum":
                    self._extreme = np.where(valid3 & (~self._extreme_valid[:, :, None] | (data > self._extreme)), data, self._extreme)
                else:
                    self._extreme = np.where(valid3 & (~self._extreme_valid[:, :, None] | (data < self._extreme)), data, self._extreme)
            else:
                if self.method == "Maximum":
                    self._extreme = np.where(valid & (~self._extreme_valid | (data > self._extreme)), data, self._extreme)
                else:
                    self._extreme = np.where(valid & (~self._extreme_valid | (data < self._extreme)), data, self._extreme)
            self._extreme_valid |= valid
        self.accepted.append({"sequence": sequence, **info})

    def submit(
        self,
        sequence: int,
        data: np.ndarray,
        mask: np.ndarray,
        sat_mask: np.ndarray,
        info: dict[str, Any] | None = None,
    ) -> None:
        payload = (_as_hwc(np.asarray(data, dtype=np.float32)), np.asarray(mask, bool), np.asarray(sat_mask, bool), dict(info or {}))
        with self._lock:
            self._pending[int(sequence)] = payload
            while self._next_sequence in self._pending:
                current = self._pending.pop(self._next_sequence)
                self._consume(self._next_sequence, current)
                self._next_sequence += 1

    def skip(self, sequence: int, reason: str) -> None:
        with self._lock:
            self._pending[int(sequence)] = None
            self.rejected.append({"sequence": int(sequence), "reason": str(reason)})
            while self._next_sequence in self._pending:
                current = self._pending.pop(self._next_sequence)
                self._consume(self._next_sequence, current)
                self._next_sequence += 1

    @property
    def frame_count(self) -> int:
        return len(self.accepted)

    def _result(self) -> tuple[np.ndarray, np.ndarray]:
        if self._data_shape is None:
            raise ValueError(f"Nenhum frame aceito no batch {self.batch_name}")
        if self.method in {"Mean", "Sum"}:
            assert self._sum is not None and self._count is not None
            if self.method == "Sum":
                result = self._sum
            else:
                denom = np.maximum(self._count, 1)
                result = self._sum / (denom[:, :, None] if self._sum.ndim == 3 else denom)
            return np.asarray(result, dtype=np.float32), self._count > 0
        if self.method == "QualityWeightedMean":
            assert self._weighted_sum is not None and self._weight_sum is not None
            denom = np.maximum(self._weight_sum, np.float64(1e-12))
            result = self._weighted_sum / (denom[:, :, None] if self._weighted_sum.ndim == 3 else denom)
            return np.asarray(result, dtype=np.float32), self._weight_sum > 0
        if self.method == "Median":
            if not self._samples:
                raise ValueError(f"Nenhum frame aceito no batch {self.batch_name}")
            values = np.stack(self._samples, axis=0).astype(np.float32, copy=False)
            masks = np.stack(self._sample_masks, axis=0).astype(bool, copy=False)
            if values.ndim == 4:
                values = np.where(masks[:, :, :, None], values, np.nan)
            else:
                values = np.where(masks, values, np.nan)
            with np.errstate(all="ignore"):
                result = np.nanmedian(values, axis=0).astype(np.float32, copy=False)
            valid = np.any(masks, axis=0)
            return result, valid
        assert self._extreme is not None and self._extreme_valid is not None
        return self._extreme, self._extreme_valid

    def finalize(self, header: fits.Header | None = None) -> dict[str, Any]:
        data, valid = self._result()
        sat = self._sat_mask if self._sat_mask is not None else np.zeros(valid.shape, bool)
        suffix = self.image_format.default_suffix
        image_path = self.output_dir / f"batch_stack{suffix}"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        visible = _visible_uint16(data)
        if self.image_format is ImageFormat.TIFF:
            write_tiff(image_path, visible, metadata={"schema_version": SCHEMA_VERSION, "linear": True, "batch": self.batch_name})
        else:
            _atomic_fits_image(image_path, visible, valid, sat, header)

        state_path = image_path.with_suffix(image_path.suffix + STATE_SUFFIX)
        state: dict[str, np.ndarray] = {"valid_mask": valid.astype(np.uint8), "sat_mask": sat.astype(np.uint8)}
        if self.method in {"Mean", "Sum"}:
            assert self._sum is not None and self._count is not None
            state.update({"sum": self._sum, "count": self._count})
        elif self.method == "QualityWeightedMean":
            assert self._weighted_sum is not None and self._weight_sum is not None
            state.update({"weighted_sum": self._weighted_sum, "weight_sum": self._weight_sum})
        elif self.method in {"Maximum", "Minimum"}:
            assert self._extreme is not None
            state["extreme"] = self._extreme
        write_npz_sidecar(state_path, **state)
        manifest_path = image_path.with_suffix(image_path.suffix + MANIFEST_SUFFIX)
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "kind": "astrobatch_batch_stack",
            "batch": self.batch_name,
            "format": self.image_format.value,
            "image": image_path.name,
            "state": state_path.name,
            "method": self.method,
            "rejection_method": self.rejection_method,
            "hierarchical": bool(self.rejection_method != "None" or self.method == "Median"),
            "frame_count": self.frame_count,
            "accepted": self.accepted,
            "rejected": self.rejected,
            "config": self.config_snapshot,
            "files": [entry.get("source") for entry in self.accepted if entry.get("source")],
            "header": {str(key): str(value) for key, value in (header or fits.Header()).items()},
        }
        write_sidecar_json(manifest_path, manifest)
        # Keep path objects for callers that need to open/log the files.  Do
        # not let the JSON-friendly manifest fields (which contain only file
        # names) overwrite these resolved paths.
        result = dict(manifest)
        result.update({"image": image_path, "state": state_path, "manifest": manifest_path})
        return result


def discover_batch_bundles(input_dir: Path) -> list[Path]:
    manifests = sorted(input_dir.rglob(f"*{MANIFEST_SUFFIX}"), key=lambda p: str(p).casefold())
    result: list[Path] = []
    for manifest in manifests:
        try:
            payload = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            continue
        if payload.get("kind") != "astrobatch_batch_stack":
            continue
        image = manifest.with_name(str(payload.get("image", "")))
        state = manifest.with_name(str(payload.get("state", "")))
        if image.exists() and state.exists():
            result.append(manifest)
    return result


def load_bundle(manifest_path: Path) -> tuple[dict[str, Any], dict[str, np.ndarray], np.ndarray, fits.Header]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    image_path = manifest_path.with_name(manifest["image"])
    state_path = manifest_path.with_name(manifest["state"])
    with np.load(state_path, allow_pickle=False) as loaded:
        state = {name: np.array(loaded[name], copy=True) for name in loaded.files}
    data, header = read_image(image_path)
    return manifest, state, _as_hwc(data), header


def _bundle_config_hash(manifest: dict[str, Any]) -> str:
    encoded = json.dumps(manifest.get("config", {}), sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def combine_bundles(
    manifests: list[Path],
    *,
    method: str,
    output_path: Path,
    compress_output: bool = True,
    rejection_method: str = "None",
    rejection_low: float = 3.0,
    rejection_high: float = 3.0,
    output_format: str = "auto",
) -> dict[str, Any]:
    """Combine batch states without assigning equal weight to each batch."""

    del compress_output  # TIFF/FITS writers own their lossless encoding.
    loaded = [load_bundle(path) for path in manifests]
    formats = {str(item[0].get("format", "")).strip().lower() for item in loaded}
    if len(formats) != 1:
        raise ValueError("Bundles de batch usam formatos diferentes.")
    if formats not in ({ImageFormat.FITS.value}, {ImageFormat.TIFF.value}):
        raise ValueError(f"Formato de bundle não suportado: {', '.join(sorted(formats))}")
    source_format = ImageFormat.TIFF if formats == {ImageFormat.TIFF.value} else ImageFormat.FITS
    requested_format = str(output_format or "auto").strip().lower()
    if requested_format not in {"auto", source_format.value}:
        raise ValueError(
            f"Formato de saída incompatível: a sessão usa {source_format.value.upper()}, "
            f"mas foi solicitado {requested_format.upper()}."
        )
    shape = loaded[0][2].shape
    method_mismatch = False
    for manifest, state, image, _header in loaded:
        if image.shape != shape:
            raise ValueError(f"Geometria incompatível no batch {manifest.get('batch')}")
        if manifest.get("method") != method:
            # A visible master is still a valid hierarchical input.  The
            # mismatch is recorded below and forces the explicit two-stage
            # path instead of silently claiming bit-exact equivalence.
            method_mismatch = True

    valid = np.zeros(shape[:2], dtype=bool)
    sat = np.zeros(shape[:2], dtype=bool)
    # A rejection choice made during Align is advisory for the compact
    # accumulator: it is intentionally not allowed to discard frames there.
    # Treat its manifest as a hierarchical request at the final Stack even
    # when the Stack UI itself is set to ``None``; otherwise an associative
    # state merge could silently claim that the requested rejection ran.
    batch_rejection_present = any(
        str(item[0].get("rejection_method", "None") or "None") != "None"
        for item in loaded
    )
    robust_two_stage = str(rejection_method or "None") != "None" or batch_rejection_present
    if (not robust_two_stage and not method_mismatch and method in {"Mean", "Sum"}
            and all("sum" in state and "count" in state for _, state, _, _ in loaded)):
        sums = np.zeros(shape, dtype=np.float64)
        counts = np.zeros(shape[:2], dtype=np.uint64)
        for _manifest, state, _image, _header in loaded:
            sums += state["sum"]
            counts += state["count"].astype(np.uint64)
            valid |= state["valid_mask"] > 0
            sat |= state.get("sat_mask", np.zeros_like(valid, dtype=np.uint8)) > 0
        if method == "Sum":
            result = sums
        else:
            denom = np.maximum(counts, 1)
            result = sums / (denom[:, :, None] if sums.ndim == 3 else denom)
    elif (not robust_two_stage and not method_mismatch and method == "QualityWeightedMean"
          and all("weighted_sum" in state and "weight_sum" in state for _, state, _, _ in loaded)):
        sums = np.zeros(shape, dtype=np.float64)
        weights = np.zeros(shape[:2], dtype=np.float64)
        for _manifest, state, _image, _header in loaded:
            sums += state["weighted_sum"]
            weights += state["weight_sum"]
            valid |= state["valid_mask"] > 0
            sat |= state.get("sat_mask", np.zeros_like(valid, dtype=np.uint8)) > 0
        denom = np.maximum(weights, 1e-12)
        result = sums / (denom[:, :, None] if sums.ndim == 3 else denom)
    elif (not robust_two_stage and not method_mismatch and method in {"Maximum", "Minimum"}
          and all("extreme" in state for _, state, _, _ in loaded)):
        # Extrema are associative too.  Merge the retained float32 state
        # rather than the visible uint16 masters, which would introduce an
        # avoidable quantization step before the final product.
        result = None
        for _manifest, state, _image, _header in loaded:
            candidate = np.asarray(state["extreme"], dtype=np.float32)
            candidate_valid = np.asarray(state.get("valid_mask"), dtype=bool)
            if result is None:
                result = np.array(candidate, copy=True)
                valid = candidate_valid.copy()
            else:
                comparison = candidate > result if method == "Maximum" else candidate < result
                if result.ndim == 3:
                    valid3 = valid[:, :, None]
                    candidate_valid3 = candidate_valid[:, :, None]
                    use = candidate_valid3 & (~valid3 | comparison)
                else:
                    use = candidate_valid & (~valid | comparison)
                result = np.where(use, candidate, result)
                valid |= candidate_valid
            sat |= np.asarray(state.get("sat_mask", np.zeros_like(valid, dtype=np.uint8)), dtype=bool)
        if result is None:  # pragma: no cover - manifests are non-empty by contract
            raise ValueError("Nenhum estado extremo encontrado nos batches")
    else:
        # Non-associative methods, explicit rejection, and method mismatches
        # operate on one master per batch.  The manifest records this
        # explicitly so the UI can show the warning.  ``rejection_low/high``
        # are retained in the report even though the compact two-stage pass
        # uses a deterministic median of batch masters; this is intentional
        # and prevents a false claim of per-frame sigma clipping.
        images = [image.astype(np.float32, copy=False) for _manifest, state, image, _header in loaded]
        stack = np.stack(images, axis=0)
        if method == "Maximum" and not robust_two_stage:
            result = np.max(stack, axis=0)
        elif method == "Minimum" and not robust_two_stage:
            result = np.min(stack, axis=0)
        else:
            result = np.median(stack, axis=0)
        for _manifest, state, _image, _header in loaded:
            valid |= state["valid_mask"] > 0
            sat |= state.get("sat_mask", np.zeros_like(valid, dtype=np.uint8)) > 0

    output_path = Path(output_path)
    expected_suffixes = {".tif", ".tiff"} if source_format is ImageFormat.TIFF else {".fit", ".fits", ".fts"}
    if output_path.suffix.casefold() not in expected_suffixes:
        output_path = output_path.with_suffix(source_format.default_suffix)
    if source_format is ImageFormat.TIFF:
        tiff_result = _visible_uint16(result)
        if tiff_result.ndim == 3 and tiff_result.shape[0] in (3, 4) and tiff_result.shape[-1] not in (3, 4):
            tiff_result = np.moveaxis(tiff_result, 0, -1)
        write_tiff(output_path, tiff_result, metadata={"schema_version": SCHEMA_VERSION, "linear": True, "batches": len(loaded)})
        write_npz_sidecar(output_path.with_suffix(output_path.suffix + STATE_SUFFIX), valid_mask=valid.astype(np.uint8), sat_mask=sat.astype(np.uint8))
    else:
        output_path = output_path.with_suffix(".fits") if output_path.suffix.casefold() not in {".fit", ".fits", ".fts"} else output_path
        _atomic_fits_image(output_path, _visible_uint16(result), valid, sat, loaded[0][3])
    report_path = output_path.with_suffix(output_path.suffix + ".json")
    first_header = loaded[0][3]
    report = {
        "schema_version": SCHEMA_VERSION,
        "kind": "astrobatch_final_stack",
        "format": source_format.value,
        "method": method,
        "rejection_method": str(rejection_method or "None"),
        "rejection_low": float(rejection_low),
        "rejection_high": float(rejection_high),
        "batches": [{"batch": item[0].get("batch"), "frames": item[0].get("frame_count", 0), "config_hash": _bundle_config_hash(item[0])} for item in loaded],
        "hierarchical": bool(
            method_mismatch or robust_two_stage or method == "Median" or not (
                method in {"Mean", "Sum", "QualityWeightedMean", "Maximum", "Minimum"}
                and all(method == item[0].get("method") and item[0].get("rejection_method") == "None" for item in loaded)
            )
        ),
        "warning": (
            "Resultado hierárquico: método/rejeição final foi aplicado sobre os mestres por batch; "
            "não equivale a uma redução por pixel de todos os frames."
            if (method_mismatch or robust_two_stage or method == "Median")
            else None
        ),
        "header": {str(key): str(value) for key, value in first_header.items()},
        "output": output_path.name,
    }
    write_sidecar_json(report_path, report)
    return {"status": "success", "output_path": str(output_path), "n_batches": len(loaded), "n_frames": int(sum(item[0].get("frame_count", 0) for item in loaded)), "hierarchical": report["hierarchical"], "report": str(report_path)}


__all__ = ["BatchAccumulator", "SCHEMA_VERSION", "combine_bundles", "discover_batch_bundles", "load_bundle"]
