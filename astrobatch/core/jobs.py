from __future__ import annotations

import threading
import traceback
from enum import Enum
from typing import Callable, Protocol

from .models import Artifact, JobEvent, JobEventKind, Stage, StageResult


class JobState(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    CANCELLING = "cancelling"


class StageRunner(Protocol):
    def __call__(self, context: "JobContext") -> StageResult: ...


class JobContext:
    def __init__(self, stage: Stage, cancel_event: threading.Event, emit: Callable[[JobEvent], None]):
        self.stage = stage
        self.cancel_event = cancel_event
        self._emit = emit

    @property
    def cancelled(self) -> bool:
        return self.cancel_event.is_set()

    def check_cancelled(self) -> None:
        if self.cancelled:
            raise JobCancelled("Processing was cancelled")

    def progress(self, current: int | float, total: int | float = 100, message: str = "") -> None:
        value = 0.0 if not total else max(0.0, min(100.0, float(current) / float(total) * 100.0))
        self._emit(JobEvent(JobEventKind.PROGRESS, self.stage, message, value))

    def log(self, message: str) -> None:
        self._emit(JobEvent(JobEventKind.LOG, self.stage, message.rstrip()))

    def warning(self, message: str) -> None:
        self._emit(JobEvent(JobEventKind.WARNING, self.stage, message.rstrip()))

    def publish(self, artifact: Artifact) -> None:
        self._emit(JobEvent(JobEventKind.ARTIFACT, self.stage, artifact=artifact))


class JobCancelled(RuntimeError):
    pass


class JobManager:
    """Runs exactly one cancellable stage and emits structured, UI-agnostic events."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._listeners: list[Callable[[JobEvent], None]] = []
        self._thread: threading.Thread | None = None
        self._cancel_event: threading.Event | None = None
        self.state = JobState.IDLE

    @property
    def is_running(self) -> bool:
        return self.state is not JobState.IDLE

    def add_listener(self, listener: Callable[[JobEvent], None]) -> None:
        self._listeners.append(listener)

    def _emit(self, event: JobEvent) -> None:
        for listener in tuple(self._listeners):
            listener(event)

    def start(self, stage: Stage, runner: StageRunner) -> None:
        with self._lock:
            if self.is_running:
                raise RuntimeError("Another pipeline job is already running")
            self.state = JobState.RUNNING
            self._cancel_event = threading.Event()
            self._thread = threading.Thread(target=self._run, args=(stage, runner, self._cancel_event), daemon=True, name=f"astrobatch-{stage.value}")
            self._thread.start()

    def cancel(self) -> None:
        with self._lock:
            if self._cancel_event is not None and self.is_running:
                self.state = JobState.CANCELLING
                self._cancel_event.set()

    def _run(self, stage: Stage, runner: StageRunner, cancel_event: threading.Event) -> None:
        self._emit(JobEvent(JobEventKind.STARTED, stage, f"{stage.value.title()} started"))
        try:
            result = runner(JobContext(stage, cancel_event, self._emit))
            if cancel_event.is_set():
                self._emit(JobEvent(JobEventKind.CANCELLED, stage, "Processing cancelled"))
            else:
                for artifact in result.artifacts:
                    self._emit(JobEvent(JobEventKind.ARTIFACT, stage, artifact=artifact))
                self._emit(JobEvent(JobEventKind.COMPLETED, stage, result.summary, 100.0, details=result.metrics))
        except JobCancelled:
            self._emit(JobEvent(JobEventKind.CANCELLED, stage, "Processing cancelled"))
        except Exception as exc:
            self._emit(JobEvent(JobEventKind.FAILED, stage, str(exc), details={"traceback": traceback.format_exc()}))
        finally:
            with self._lock:
                self.state = JobState.IDLE
                self._cancel_event = None
