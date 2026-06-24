"""Thread-safe JSONL logging for LLM and tool calls.

Every successful or failed LLM/tool invocation is appended as a JSON object on
its own line. The log file is created on the first write so multiple workers
can share the same log path without collisions.

The :class:`CallLogger` instances are designed to be set globally per run
(``set_default_logger`` / ``get_default_logger``) so the agent and the tools do
not need to be threaded through with the same logger object.
"""
from __future__ import annotations

import json
import os
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any


class CallLogger:
    """Append-only JSONL logger for arbitrary structured events."""

    def __init__(self, log_path: Path | str) -> None:
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._counter = 0

    def log(self, event: str, payload: dict[str, Any]) -> None:
        record = {
            "event": event,
            "timestamp": datetime.utcnow().isoformat(timespec="milliseconds") + "Z",
            "pid": os.getpid(),
            "thread": threading.get_ident(),
            **payload,
        }
        with self._lock:
            self._counter += 1
            record["seq"] = self._counter
            with self.log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    @property
    def event_count(self) -> int:
        with self._lock:
            return self._counter


_DEFAULT_LOGGER: CallLogger | None = None
_DEFAULT_LOCK = threading.Lock()


def set_default_logger(logger: CallLogger | None) -> None:
    """Install the global call logger that LLM / tool clients should use."""

    global _DEFAULT_LOGGER
    with _DEFAULT_LOCK:
        _DEFAULT_LOGGER = logger


def get_default_logger() -> CallLogger | None:
    """Return the currently installed global logger, if any."""

    with _DEFAULT_LOCK:
        return _DEFAULT_LOGGER


class TimedCall:
    """Context manager that captures elapsed time and emits a log event.

    Usage::

        with TimedCall("llm_call", logger, extra={"model": "gpt-5.2"}) as call:
            response = client.complete(...)
            call.set_result({"content": response})
    """

    def __init__(
        self,
        event: str,
        logger: CallLogger | None,
        *,
        extra: dict[str, Any] | None = None,
    ) -> None:
        self.event = event
        self.logger = logger
        self.extra = dict(extra or {})
        self.call_id = uuid.uuid4().hex[:12]
        self.extra.setdefault("call_id", self.call_id)
        self._result: dict[str, Any] = {}
        self._error: str | None = None
        self._start: float = 0.0

    def __enter__(self) -> "TimedCall":
        self._start = time.monotonic()
        return self

    def set_result(self, payload: dict[str, Any]) -> None:
        self._result.update(payload)

    def set_error(self, message: str) -> None:
        self._error = message

    def __exit__(self, exc_type, exc, tb) -> bool:
        elapsed_ms = int((time.monotonic() - self._start) * 1000)
        payload: dict[str, Any] = {
            "elapsed_ms": elapsed_ms,
            "status": "ok",
            **self.extra,
            **self._result,
        }
        if exc is not None or self._error is not None:
            payload["status"] = "error"
            payload["error"] = self._error or str(exc)
        if self.logger is not None:
            self.logger.log(self.event, payload)
        return False


def truncate_text(text: str, max_chars: int) -> str:
    """Truncate ``text`` to ``max_chars`` (0 keeps it intact)."""

    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[:max_chars] + f"...<truncated to {max_chars} chars>"
