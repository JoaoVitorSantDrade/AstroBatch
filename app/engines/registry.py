"""Explicit registry for built-in engines.

The registry deliberately does not use package entry points yet.  That keeps
V2 engine selection deterministic while allowing an external-plugin boundary
to be added later without changing pipeline APIs.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Callable
from typing import Any

from .contracts import EngineDescriptor, EngineProfile


class EngineUnavailable(LookupError):
    pass


class EngineRegistry:
    def __init__(self) -> None:
        self._entries: dict[tuple[str, str], tuple[EngineDescriptor, Callable[..., Any]]] = {}

    def register(
        self, descriptor: EngineDescriptor, implementation: Callable[..., Any]
    ) -> None:
        self._entries[(descriptor.stage, descriptor.id)] = (descriptor, implementation)

    def resolve(
        self, stage: str, engine_id: str, profile: EngineProfile | None = None
    ) -> Callable[..., Any]:
        entry = self._entries.get((stage, engine_id))
        if entry is None:
            raise EngineUnavailable(f"Engine {stage}/{engine_id} is not registered.")
        descriptor, implementation = entry
        if profile is not None and profile not in descriptor.profiles:
            raise EngineUnavailable(
                f"Engine {descriptor.label} is not available for {profile.value}."
            )
        if descriptor.optional_dependency and importlib.util.find_spec(descriptor.optional_dependency) is None:
            raise EngineUnavailable(
                f"Engine {descriptor.label} requires optional dependency "
                f"{descriptor.optional_dependency}."
            )
        return implementation

    def descriptors(self, stage: str, profile: EngineProfile | None = None) -> list[EngineDescriptor]:
        result = [item[0] for (entry_stage, _), item in self._entries.items() if entry_stage == stage]
        if profile is not None:
            result = [item for item in result if profile in item.profiles]
        return sorted(result, key=lambda item: item.label)

    def is_available(self, stage: str, engine_id: str) -> tuple[bool, str | None]:
        try:
            self.resolve(stage, engine_id)
        except EngineUnavailable as exc:
            return False, str(exc)
        return True, None


registry = EngineRegistry()
