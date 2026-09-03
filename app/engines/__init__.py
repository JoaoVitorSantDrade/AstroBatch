"""Engine contracts and the built-in engine registry."""

from .contracts import EngineDescriptor, EngineProfile, EngineSelection
from .execution import ExecutionBudget
from .registry import EngineRegistry, EngineUnavailable, registry

__all__ = [
    "EngineDescriptor",
    "ExecutionBudget",
    "EngineProfile",
    "EngineRegistry",
    "EngineSelection",
    "EngineUnavailable",
    "registry",
]
