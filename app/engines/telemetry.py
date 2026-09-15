"""Low-overhead process telemetry for repeatable pipeline diagnostics.

The scientific pipeline intentionally does not depend on this module for its
results.  It is used by benchmark and profiling entry points to distinguish
CPU saturation, executor wait, and file-system activity without adding a
second implementation of the processing stages.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import threading
import time
from typing import Any


def _counter_value(counters: Any, name: str) -> int:
    value = getattr(counters, name, 0) if counters is not None else 0
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


@dataclass(frozen=True, slots=True)
class StageTelemetry:
    """Measured process deltas for one pipeline stage."""

    wall_seconds: float
    cpu_user_seconds: float
    cpu_system_seconds: float
    read_bytes: int
    write_bytes: int
    read_count: int
    write_count: int
    peak_rss_bytes: int
    peak_thread_count: int

    @property
    def cpu_seconds(self) -> float:
        return self.cpu_user_seconds + self.cpu_system_seconds

    @property
    def cpu_to_wall(self) -> float:
        return self.cpu_seconds / self.wall_seconds if self.wall_seconds > 0 else 0.0

    def as_dict(self) -> dict[str, int | float]:
        result = asdict(self)
        result["cpu_seconds"] = self.cpu_seconds
        result["cpu_to_wall"] = self.cpu_to_wall
        return result


class ProcessTelemetry:
    """Context manager sampling RSS and thread count around a stage.

    ``psutil`` is imported lazily so normal application startup has no
    diagnostic dependency.  If it is unavailable, timing still works and
    resource counters remain zero instead of breaking a scientific run.
    """

    def __init__(self, sample_interval: float = 0.01) -> None:
        self.sample_interval = max(0.001, float(sample_interval))
        self._process = None
        self._before_io = None
        self._before_cpu = None
        self._started = 0.0
        self._stop = threading.Event()
        self._sampler: threading.Thread | None = None
        self._peak_rss = 0
        self._peak_threads = 0
        self._last_thread_sample = 0.0
        self.result: StageTelemetry | None = None

    def __enter__(self) -> "ProcessTelemetry":
        self._started = time.perf_counter()
        try:
            import psutil

            self._process = psutil.Process()
            self._before_io = self._process.io_counters()
            self._before_cpu = self._process.cpu_times()
            self._sample(force_thread=True)
            self._sampler = threading.Thread(
                target=self._sample_loop,
                name="astrobatch-telemetry",
                daemon=True,
            )
            self._sampler.start()
        except Exception:
            self._process = None
        return self

    def _sample(self, force_thread: bool = False) -> None:
        if self._process is None:
            return
        try:
            self._peak_rss = max(self._peak_rss, int(self._process.memory_info().rss))
            # On Windows num_threads() asks psutil for a full process
            # snapshot and is materially more expensive than memory_info().
            # RSS still samples at the requested cadence; thread count is a
            # contention diagnostic and only needs a sparse/conservative
            # sample.  Always sample at entry/exit so short stages retain a
            # useful lower bound.
            now = time.perf_counter()
            thread_interval = max(0.1, self.sample_interval * 10.0)
            if force_thread or now - self._last_thread_sample >= thread_interval:
                self._peak_threads = max(self._peak_threads, int(self._process.num_threads()))
                self._last_thread_sample = now
        except Exception:
            pass

    def _sample_loop(self) -> None:
        while not self._stop.wait(self.sample_interval):
            self._sample()

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self._stop.set()
        if self._sampler is not None:
            self._sampler.join(timeout=max(0.1, self.sample_interval * 4))
        self._sample(force_thread=True)
        wall = max(0.0, time.perf_counter() - self._started)
        after_io = None
        after_cpu = None
        if self._process is not None:
            try:
                after_io = self._process.io_counters()
                after_cpu = self._process.cpu_times()
            except Exception:
                pass
        before_io = self._before_io
        before_cpu = self._before_cpu
        self.result = StageTelemetry(
            wall_seconds=wall,
            cpu_user_seconds=max(
                0.0,
                float(getattr(after_cpu, "user", 0.0) or 0.0)
                - float(getattr(before_cpu, "user", 0.0) or 0.0),
            ),
            cpu_system_seconds=max(
                0.0,
                float(getattr(after_cpu, "system", 0.0) or 0.0)
                - float(getattr(before_cpu, "system", 0.0) or 0.0),
            ),
            read_bytes=max(0, _counter_value(after_io, "read_bytes") - _counter_value(before_io, "read_bytes")),
            write_bytes=max(0, _counter_value(after_io, "write_bytes") - _counter_value(before_io, "write_bytes")),
            read_count=max(0, _counter_value(after_io, "read_count") - _counter_value(before_io, "read_count")),
            write_count=max(0, _counter_value(after_io, "write_count") - _counter_value(before_io, "write_count")),
            peak_rss_bytes=max(0, int(self._peak_rss)),
            # The sampler itself accounts for one thread.  Keep the raw
            # process peak; it is the conservative number needed to detect
            # oversubscription and is documented as including the observer.
            peak_thread_count=max(0, int(self._peak_threads)),
        )
