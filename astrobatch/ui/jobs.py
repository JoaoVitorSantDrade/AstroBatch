from __future__ import annotations

from PySide6.QtCore import QObject, Signal

from astrobatch.core.models import JobEvent


class QtJobBridge(QObject):
    event_received = Signal(object)

    def emit_event(self, event: JobEvent) -> None:
        self.event_received.emit(event)
