"""Runtime CPU/SIMD policy for the CPU-first processing pipeline.

NumPy and Numba dispatch to the host instruction set when they compile a
kernel.  This module exposes the detected capabilities and applies one
conservative thread policy so an eight-core Ryzen 7 5800X is used effectively
without making a host-specific cache mandatory on another CPU.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import os
import re
import warnings
from typing import Mapping


@dataclass(frozen=True, slots=True)
class CPURuntimeInfo:
    """Serializable snapshot used by diagnostics and benchmark reports."""

    logical_cores: int
    physical_cores: int
    numba_threads: int
    avx2: bool
    avx512f: bool
    simd_level: str

    def as_dict(self) -> dict[str, int | bool | str]:
        return asdict(self)


def _cpu_counts() -> tuple[int, int]:
    logical = max(1, int(os.cpu_count() or 1))
    physical = logical
    try:
        import psutil

        physical = int(psutil.cpu_count(logical=False) or logical)
    except Exception:
        pass
    return logical, max(1, min(physical, logical))


def numpy_cpu_features() -> dict[str, bool]:
    """Return NumPy's host feature map across supported NumPy locations."""

    for module_name in (
        "numpy._core._multiarray_umath",
        "numpy.core._multiarray_umath",
    ):
        try:
            module = __import__(module_name, fromlist=["__cpu_features__"])
            features = getattr(module, "__cpu_features__", None)
            if isinstance(features, Mapping):
                return {
                    str(key).upper(): bool(value)
                    for key, value in features.items()
                }
        except Exception:
            continue
    return {}


def avx2_available(features: Mapping[str, bool] | None = None) -> bool:
    """Whether AVX2 is explicitly reported by NumPy for this process."""

    values = features if features is not None else numpy_cpu_features()
    return bool(values.get("AVX2", False))


def simd_level(features: Mapping[str, bool] | None = None) -> str:
    values = features if features is not None else numpy_cpu_features()
    if values.get("AVX512F", False):
        return "AVX-512"
    if values.get("AVX2", False):
        return "AVX2"
    if values.get("AVX", False):
        return "AVX"
    if values.get("SSE4_2", False) or values.get("SSE42", False):
        return "SSE4.2"
    return "baseline"


def recommended_numba_threads(requested: int | None = None) -> int:
    """Choose a physical-core ceiling suitable for image kernels."""

    _logical, physical = _cpu_counts()
    ceiling = max(1, min(8, physical))
    if requested is not None:
        try:
            ceiling = min(ceiling, max(1, int(requested)))
        except (TypeError, ValueError):
            pass
    return ceiling


def physical_core_count() -> int:
    """Return the detected physical-core count with a safe minimum."""

    return _cpu_counts()[1]


def configure_numba_threads(requested: int | None = None) -> int:
    """Apply the bounded Numba thread pool and return the active count."""

    target = recommended_numba_threads(requested)
    try:
        from numba import set_num_threads

        set_num_threads(target)
    except Exception:
        return 1
    return target


def configure_opencv_threads(requested: int = 1) -> int | None:
    """Bound OpenCV's process-wide worker pool.

    OpenCV owns a native pool which is independent from Python and Numba
    workers.  Leaving its default (16 on the Ryzen host) enabled inside a
    frame executor creates nested parallelism and makes the configured
    ``ExecutionBudget`` ineffective.  The function is deliberately optional:
    environments without OpenCV simply report ``None``.
    """

    try:
        import cv2

        target = max(1, int(requested))
        cv2.setNumThreads(target)
        return int(cv2.getNumThreads())
    except Exception:
        return None


def configure_worker_runtime(numba_threads: int = 1) -> int:
    """Initializer used by every image worker.

    Numba's thread count is thread-local on the supported runtimes, so calling
    :func:`configure_numba_threads` only before constructing an executor does
    not constrain its workers.  Keep the initializer tiny and side-effect
    free apart from the worker's own native runtime state.
    """

    return configure_numba_threads(numba_threads)


def runtime_info() -> CPURuntimeInfo:
    logical, physical = _cpu_counts()
    try:
        from numba import get_num_threads

        numba_threads = int(get_num_threads())
    except Exception:
        numba_threads = 1
    features = numpy_cpu_features()
    return CPURuntimeInfo(
        logical_cores=logical,
        physical_cores=physical,
        numba_threads=max(1, numba_threads),
        avx2=avx2_available(features),
        avx512f=bool(features.get("AVX512F", False)),
        simd_level=simd_level(features),
    )


def inspect_numba_assembly(dispatcher: object) -> dict[str, object]:
    """Inspect compiled Numba assembly for a diagnostic benchmark report.

    This is intentionally observational: it never changes compiler flags or
    assumes a mnemonic is available on another host.  An empty result means
    that the dispatcher has not yet been compiled or the backend does not
    expose assembly inspection.
    """

    signatures = getattr(dispatcher, "signatures", ())
    if not signatures:
        return {
            "compiled": False,
            "inspection_available": False,
            "inspection_unavailable": "dispatcher is not compiled",
            "has_ymm": False,
            "has_avx2_mnemonic": False,
        }
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            assembly = str(dispatcher.inspect_asm(signatures[0]))
    except Exception:
        return {
            "compiled": True,
            "inspection_available": False,
            "inspection_unavailable": "dispatcher assembly is unavailable",
            "has_ymm": False,
            "has_avx2_mnemonic": False,
        }
    if len(assembly) <= 128:
        return {
            "compiled": True,
            "inspection_available": False,
            "inspection_unavailable": "Numba cached code does not expose assembly",
            "has_ymm": False,
            "has_avx2_mnemonic": False,
        }
    # YMM plus a VEX vector arithmetic/load/mask instruction is the useful
    # portable signal here; exact mnemonic selection can vary by LLVM/Numba.
    avx2_mnemonics = re.compile(
        r"\b(?:v(?:add|sub|mul|div|max|min|fma|mov|blend|cmp|perm|broadcast|sqrt|and|or|xor|pmov|pblend)[a-z0-9]*)\b",
        re.I,
    )
    return {
        "compiled": True,
        "inspection_available": True,
        "signature": str(signatures[0]),
        "assembly_bytes": len(assembly),
        "has_ymm": "ymm" in assembly.lower(),
        "has_avx2_mnemonic": bool(avx2_mnemonics.search(assembly)),
    }
