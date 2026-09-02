from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from astrobatch.core.models import Artifact

from .workspace import PROJECT_FILE, SCHEMA_VERSION, Project, ProjectWorkspace


class ProjectFormatError(ValueError):
    pass


class ProjectStore:
    def create(self, root: Path, name: str, source_dir: Path | None = None) -> Project:
        workspace = ProjectWorkspace(root.resolve())
        workspace.ensure()
        project = Project(name=name.strip() or root.name, workspace=workspace, source_dir=str(source_dir.resolve()) if source_dir else "")
        self.save(project)
        return project

    def save(self, project: Project) -> None:
        project.workspace.ensure()
        payload = self._serialize(project)
        destination = project.workspace.manifest_path
        handle, temporary_name = tempfile.mkstemp(prefix=f".{PROJECT_FILE}.", suffix=".tmp", dir=destination.parent)
        try:
            with os.fdopen(handle, "w", encoding="utf-8") as stream:
                json.dump(payload, stream, indent=2, ensure_ascii=False)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary_name, destination)
        except Exception:
            Path(temporary_name).unlink(missing_ok=True)
            raise

    def load(self, root_or_manifest: Path) -> Project:
        manifest = root_or_manifest if root_or_manifest.name == PROJECT_FILE else root_or_manifest / PROJECT_FILE
        try:
            payload = json.loads(manifest.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise ProjectFormatError(f"Project manifest was not found: {manifest}") from exc
        except json.JSONDecodeError as exc:
            raise ProjectFormatError(f"Project manifest is not valid JSON: {exc}") from exc
        if payload.get("schema_version") != SCHEMA_VERSION:
            raise ProjectFormatError(f"Unsupported project schema {payload.get('schema_version')!r}; expected {SCHEMA_VERSION}")
        workspace = ProjectWorkspace(manifest.parent.resolve())
        return Project(name=str(payload["name"]), workspace=workspace, source_dir=str(payload.get("source_dir", "")), settings=dict(payload.get("settings", {})), artifacts={key: Artifact(**value) for key, value in payload.get("artifacts", {}).items()}, stage_status=dict(payload.get("stage_status", {})), run_history=list(payload.get("run_history", [])), created_at=str(payload.get("created_at", "")), updated_at=str(payload.get("updated_at", "")))

    @staticmethod
    def _serialize(project: Project) -> dict[str, Any]:
        return {"schema_version": SCHEMA_VERSION, "name": project.name, "source_dir": project.source_dir, "settings": project.settings, "artifacts": {key: artifact.to_dict() for key, artifact in project.artifacts.items()}, "stage_status": project.stage_status, "run_history": project.run_history, "created_at": project.created_at, "updated_at": project.updated_at}
