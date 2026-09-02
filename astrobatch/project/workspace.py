from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from astrobatch.core.models import Artifact, Stage, utc_now

PROJECT_FILE = "astrobatch.project.json"
SCHEMA_VERSION = 2


@dataclass(slots=True)
class ProjectWorkspace:
    root: Path

    @property
    def manifest_path(self) -> Path:
        return self.root / PROJECT_FILE

    @property
    def output_root(self) -> Path:
        return self.root / "outputs"

    def stage_output(self, stage: Stage) -> Path:
        return self.output_root / stage.value

    def ensure(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.output_root.mkdir(exist_ok=True)


@dataclass(slots=True)
class Project:
    name: str
    workspace: ProjectWorkspace
    source_dir: str = ""
    settings: dict[str, dict[str, Any]] = field(default_factory=dict)
    artifacts: dict[str, Artifact] = field(default_factory=dict)
    stage_status: dict[str, str] = field(default_factory=dict)
    run_history: list[dict[str, Any]] = field(default_factory=list)
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)

    def artifact_for(self, stage: Stage) -> Artifact | None:
        return self.artifacts.get(stage.value)

    def record_stage(self, stage: Stage, summary: str, artifacts: list[Artifact], metrics: dict[str, Any]) -> None:
        for artifact in artifacts:
            self.artifacts[stage.value] = artifact
        self.stage_status[stage.value] = "completed"
        self.run_history.append({"stage": stage.value, "summary": summary, "artifacts": [item.to_dict() for item in artifacts], "metrics": metrics, "completed_at": utc_now()})
        self.updated_at = utc_now()
