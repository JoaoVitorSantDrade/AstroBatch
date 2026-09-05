"""Single-operation runner. Worker code never calls GUI APIs."""
from dataclasses import dataclass, replace
import threading
import traceback
from typing import Callable


@dataclass(frozen=True)
class ProgressEvent:
    current: int
    total: int
    phase: str = ""


@dataclass(frozen=True)
class OperationResult:
    outcome: str
    message: str


class PipelineRunner:
    """Coalesce progress and retain one completion until the UI consumes it."""

    def __init__(self, log: Callable[[str], None]):
        self.log = log
        self.cancel_event = threading.Event()
        self.thread = None
        self._lock = threading.Lock()
        self._busy = False
        self._progress = None
        self._finished = None

    @property
    def busy(self):
        with self._lock:
            return self._busy

    def progress(self, current, total, phase=""):
        with self._lock:
            self._progress = ProgressEvent(int(current), int(total), str(phase))

    def start(self, stage: str, operation: Callable) -> threading.Thread:
        with self._lock:
            if self._busy:
                raise RuntimeError("An operation is already active")
            self._busy = True
            self._progress = self._finished = None
            self.cancel_event.clear()

        def work():
            try:
                result = operation(self.log, self.progress, self.cancel_event)
                if not isinstance(result, OperationResult):
                    raise TypeError("Pipeline adapter must return OperationResult")
                if self.cancel_event.is_set() and result.outcome == "success":
                    result = replace(result, outcome="cancelled", message=f"{stage} cancelado.")
            except Exception as exc:
                self.log(traceback.format_exc())
                result = OperationResult("failed", f"{stage}: {exc}")
            with self._lock:
                self._finished = result

        self.thread = threading.Thread(target=work, name=f"pipeline-{stage}", daemon=True)
        try:
            self.thread.start()
        except Exception:
            with self._lock:
                self._busy = False
            raise
        return self.thread

    def drain(self):
        """Call from the UI thread; completion unlocks the next operation."""
        with self._lock:
            progress, finished = self._progress, self._finished
            self._progress = self._finished = None
            if finished is not None:
                self._busy = False
            return progress, finished

    def cancel(self):
        self.cancel_event.set()
