from __future__ import annotations

import threading
import time
import unittest

from astrobatch.core.jobs import JobManager
from astrobatch.core.models import Artifact, JobEventKind, Stage, StageResult


class V2JobTests(unittest.TestCase):
    def _wait(self, manager: JobManager) -> None:
        deadline = time.monotonic() + 2
        while manager.is_running and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertFalse(manager.is_running)

    def test_events_complete_and_publish_artifact(self) -> None:
        events = []
        manager = JobManager()
        manager.add_listener(events.append)

        def runner(context):
            context.log("working")
            context.progress(1, 2, "halfway")
            return StageResult(Stage.IMPORT, "done", [Artifact("source", "C:/data")])

        manager.start(Stage.IMPORT, runner)
        self._wait(manager)
        self.assertEqual(events[0].kind, JobEventKind.STARTED)
        self.assertIn(JobEventKind.ARTIFACT, [event.kind for event in events])
        self.assertEqual(events[-1].kind, JobEventKind.COMPLETED)

    def test_rejects_parallel_jobs_and_cancels(self) -> None:
        events, started = [], threading.Event()
        manager = JobManager()
        manager.add_listener(events.append)

        def runner(context):
            started.set()
            while not context.cancelled:
                time.sleep(0.01)
            context.check_cancelled()
            raise AssertionError("unreachable")

        manager.start(Stage.STACK, runner)
        self.assertTrue(started.wait(1))
        with self.assertRaises(RuntimeError):
            manager.start(Stage.IMPORT, lambda context: StageResult(Stage.IMPORT, "x"))
        manager.cancel()
        self._wait(manager)
        self.assertEqual(events[-1].kind, JobEventKind.CANCELLED)
