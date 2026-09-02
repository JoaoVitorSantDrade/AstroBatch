from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from astrobatch.core.models import Artifact, Stage
from astrobatch.project.store import ProjectFormatError, ProjectStore
from astrobatch.project.workspace import PROJECT_FILE


class V2ProjectTests(unittest.TestCase):
    def test_project_round_trip_and_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "m42"
            store = ProjectStore()
            project = store.create(root, "M42")
            project.settings["stack"] = {"method": "Median"}
            artifact = Artifact("stacked_image", str(root / "outputs" / "stack" / "result.fits"), "fits")
            project.record_stage(Stage.STACK, "Stacked 10 frames", [artifact], {"frames": 10})
            store.save(project)
            reopened = store.load(root)
            self.assertEqual(reopened.name, "M42")
            self.assertEqual(reopened.settings["stack"]["method"], "Median")
            self.assertEqual(reopened.artifact_for(Stage.STACK).path, artifact.path)
            self.assertEqual(reopened.run_history[-1]["metrics"]["frames"], 10)

    def test_rejects_unknown_schema(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / PROJECT_FILE).write_text('{"schema_version": 999, "name": "bad"}', encoding="utf-8")
            with self.assertRaises(ProjectFormatError):
                ProjectStore().load(root)
