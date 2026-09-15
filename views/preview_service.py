"""Tk-independent preview workers used by Flow and reference dialogs."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from queue import Empty, Queue
from threading import Event, Lock


@dataclass(frozen=True)
class PreviewResult:
    generation: int
    key: object
    image: object = None
    error: str | None = None


class PreviewService:
    """Bounded worker pool; callers poll results from their UI thread."""

    def __init__(self, loader, max_workers: int = 2, max_pending: int = 32):
        self.loader = loader
        self.executor = ThreadPoolExecutor(max_workers=max(1, int(max_workers)),
                                           thread_name_prefix="astro-preview")
        self.max_pending = max(1, int(max_pending))
        self.results: Queue[PreviewResult] = Queue(maxsize=self.max_pending)
        self._generation = 0
        self._cancelled = Event()
        self._lock = Lock()
        self._closed = False

    def replace(self, generation: int, requests):
        generation = int(generation)
        with self._lock:
            if self._closed:
                return
            self._generation = generation
            self._cancelled.set()
            self._cancelled = Event()
            cancel = self._cancelled
        # Replacement invalidates the previous generation.  Keep the new
        # generation bounded before submitting work to the executor as well as
        # when buffering completed results.
        for key, payload in list(requests)[:self.max_pending]:
            self.executor.submit(self._run, generation, key, payload, cancel)

    def _run(self, generation, key, payload, cancel):
        if cancel.is_set():
            return
        try:
            image = self.loader(payload)
            result = PreviewResult(generation, key, image=image)
        except Exception as exc:  # worker boundary; UI receives text only
            result = PreviewResult(generation, key, error=str(exc))
        if cancel.is_set():
            return
        try:
            self.results.put_nowait(result)
        except Exception:
            pass

    def drain(self, limit: int = 64):
        values = []
        with self._lock:
            generation = self._generation
        for _ in range(max(1, int(limit))):
            try:
                result = self.results.get_nowait()
                if result.generation == generation:
                    values.append(result)
            except Empty:
                break
        return values

    def close(self):
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._cancelled.set()
        self.executor.shutdown(wait=False, cancel_futures=True)
