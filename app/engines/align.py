"""OpenCV warp engines used by AstroAlign."""

from __future__ import annotations

import cv2
import numpy as np

from .contracts import EngineDescriptor, EngineProfile
from .registry import registry


_STABLE_INTERPOLATION = {
    "nearest": cv2.INTER_NEAREST,
    "bilinear": cv2.INTER_LINEAR,
}
_FAST_INTERPOLATION = {
    **_STABLE_INTERPOLATION,
    "bicubic": cv2.INTER_CUBIC,
    "lanczos": cv2.INTER_LANCZOS4,
}


def _warp(data: np.ndarray, matrix: np.ndarray, interpolation: str, modes: dict[str, int]) -> np.ndarray:
    mode = modes.get(interpolation)
    if mode is None:
        raise ValueError(f"No OpenCV {interpolation} warp is available.")
    height, width = data.shape[:2]
    return np.asarray(
        cv2.warpAffine(
            np.asarray(data, dtype=np.float32), np.asarray(matrix[:2, :], dtype=np.float64),
            (width, height), flags=mode, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0,
        ), dtype=np.float32,
    )


def warp_opencv_stable(data: np.ndarray, matrix: np.ndarray, interpolation: str) -> np.ndarray:
    return _warp(data, matrix, interpolation, _STABLE_INTERPOLATION)


def warp_opencv_fast(data: np.ndarray, matrix: np.ndarray, interpolation: str) -> np.ndarray:
    return _warp(data, matrix, interpolation, _FAST_INTERPOLATION)


def register_align_engines() -> None:
    registry.register(
        EngineDescriptor("opencv-stable", "align.warp", "OpenCV compatibility", frozenset({EngineProfile.STABLE, EngineProfile.FAST}), capabilities=frozenset(_STABLE_INTERPOLATION)),
        warp_opencv_stable,
    )
    registry.register(
        EngineDescriptor("opencv-fast", "align.warp", "OpenCV native fast", frozenset({EngineProfile.FAST}), capabilities=frozenset(_FAST_INTERPOLATION)),
        warp_opencv_fast,
    )
