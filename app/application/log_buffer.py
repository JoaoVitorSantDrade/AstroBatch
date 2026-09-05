"""Bounded, nonblocking activity retention with explicit loss reporting."""
from collections import deque
import queue
import threading


class ActivityBuffer:
    def __init__(self, capacity=2000, max_message_chars=16000):
        self.capacity = capacity
        self.max_message_chars = max_message_chars
        self._messages = deque()
        self._lock = threading.Lock()
        self._dropped = 0

    def put(self, message):
        text = str(message)
        if len(text) > self.max_message_chars:
            text = text[:self.max_message_chars] + "\n[Atividade] Mensagem truncada.\n"
        with self._lock:
            if len(self._messages) >= self.capacity:
                victim = next((i for i, m in enumerate(self._messages)
                               if not any(t in m.lower() for t in ('error','erro','failed','falha','traceback'))), 0)
                del self._messages[victim]
                self._dropped += 1
            self._messages.append(text)

    def get_nowait(self):
        with self._lock:
            if self._dropped:
                count = self._dropped
                self._dropped = 0
                return f"[Atividade] {count} mensagens antigas removidas por limite de retenção.\n"
            if not self._messages:
                raise queue.Empty
            return self._messages.popleft()

    def qsize(self):
        with self._lock:
            return len(self._messages)
