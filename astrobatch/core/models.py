from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any


class Stage(str, Enum):
    IMPORT = "import"
    CALIBRATE = "calibrate"
    BATCH = "batch"
    FLOW = "flow"
    ALIGN = "align"
    STACK = "stack"
    REVIEW = "review"


class JobEventKind(str, Enum):
    STARTED = "started"
    PROGRESS = "progress"
    LOG = "log"
    WARNING = "warning"
    ARTIFACT = "artifact"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(slots=True)
class Artifact:
    name: str
    path: str
    kind: str = "directory"
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def filesystem_path(self) -> Path:
        return Path(self.path)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class StageResult:
    stage: Stage
    summary: str
    artifacts: list[Artifact] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class JobEvent:
    kind: JobEventKind
    stage: Stage
    message: str = ""
    progress: float | None = None
    artifact: Artifact | None = None
    details: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=utc_now)
