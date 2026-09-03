"""Small, dependency-free contracts shared by all processing engines."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

import numpy as np


class EngineProfile(StrEnum):
    STABLE = "Stable"
    FAST = "Fast"

    @classmethod
    def coerce(cls, value: object) -> "EngineProfile":
        try:
            return cls(str(value).title())
        except ValueError:
            return cls.STABLE


@dataclass(frozen=True, slots=True)
class EngineDescriptor:
    id: str
    stage: str
    label: str
    profiles: frozenset[EngineProfile]
    optional_dependency: str | None = None
    capabilities: frozenset[str] = field(default_factory=frozenset)


@dataclass(frozen=True, slots=True)
class EngineSelection:
    profile: EngineProfile = EngineProfile.STABLE
    detector: str | None = None
    transform_fallback: str | None = None
    warp: str | None = None
    reducer: str | None = None


@runtime_checkable
class StarDetector(Protocol):
    def __call__(
        self, data: np.ndarray, fwhm: float, sigma: float, max_stars: int
    ) -> tuple[np.ndarray, float, dict[str, Any]]: ...


@runtime_checkable
class TransformEstimator(Protocol):
    def __call__(self, source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]: ...


@runtime_checkable
class WarpEngine(Protocol):
    def __call__(self, data: np.ndarray, matrix: np.ndarray, interpolation: str) -> np.ndarray: ...


@runtime_checkable
class ChannelRefiner(Protocol):
    def __call__(self, data: np.ndarray) -> np.ndarray: ...


@runtime_checkable
class StackReducer(Protocol):
    def __call__(self, values: np.ndarray, masks: np.ndarray, method: str) -> np.ndarray: ...
