"""Atomic JSON persistence shared by settings and scientific sidecars."""
import json
import os
from pathlib import Path
import tempfile


def atomic_json_write(path: Path, data) -> None:
    path = Path(path)
    payload = json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


class SettingsRepository:
    """Version marker preserves the existing section-based settings format."""
    def __init__(self, path):
        self.path = Path(path)

    def load(self):
        if not self.path.exists():
            return {}
        data = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("Settings must be a JSON object")
        if data.get("_schema_version", 1) not in (1, 2):
            raise ValueError("Unsupported settings version")
        return data

    def save(self, data):
        atomic_json_write(self.path, {**data, "_schema_version": 2})
