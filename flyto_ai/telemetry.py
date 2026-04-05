# Copyright 2024 Flyto
# Licensed under the Apache License, Version 2.0
"""Unified telemetry infrastructure — single sink for all agent events.

Inspired by claw-code's ``TelemetrySink`` trait with ``MemoryTelemetrySink``
(testing) and ``JsonlTelemetrySink`` (production).

Replaces the scattered audit/transcript/cost recording with a single
``SessionTracer`` that fans out to one or more sinks.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

logger = logging.getLogger(__name__)


class TelemetryEventType(Enum):
    """Categories of telemetry events."""
    LLM_REQUEST = "llm_request"
    LLM_RESPONSE = "llm_response"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    PERMISSION_CHECK = "permission_check"
    HOOK_DECISION = "hook_decision"
    COST_UPDATE = "cost_update"
    SESSION_START = "session_start"
    SESSION_END = "session_end"
    COMPACTION = "compaction"
    ERROR = "error"
    CUSTOM = "custom"


@dataclass
class TelemetryEvent:
    """A single telemetry event."""
    type: TelemetryEventType
    timestamp: float = field(default_factory=time.time)
    session_id: str = ""
    sequence: int = 0
    data: Dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class TelemetrySink(Protocol):
    """Protocol for telemetry backends."""

    def emit(self, event: TelemetryEvent) -> None:
        """Record a telemetry event."""
        ...


class MemoryTelemetrySink:
    """In-memory sink for testing — collect events and assert on them.

    Usage::

        sink = MemoryTelemetrySink()
        tracer = SessionTracer("test-session", sinks=[sink])
        tracer.trace_tool_call("search_modules", {"query": "auth"}, {"ok": True}, 42)

        assert len(sink.events) == 1
        assert sink.events[0].type == TelemetryEventType.TOOL_CALL
    """

    def __init__(self) -> None:
        self.events: List[TelemetryEvent] = []

    def emit(self, event: TelemetryEvent) -> None:
        self.events.append(event)

    def events_of_type(self, event_type: TelemetryEventType) -> List[TelemetryEvent]:
        return [e for e in self.events if e.type == event_type]

    def clear(self) -> None:
        self.events.clear()


class JsonlTelemetrySink:
    """Production sink — append events as JSONL with file rotation.

    Thread-safe via a lock on each write.
    """

    def __init__(
        self,
        path: str,
        max_size_bytes: int = 256 * 1024,
        max_rotated: int = 3,
    ) -> None:
        self._path = Path(os.path.expanduser(path))
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._max_size = max_size_bytes
        self._max_rotated = max_rotated
        self._lock = threading.Lock()
        self._file = None

    def emit(self, event: TelemetryEvent) -> None:
        with self._lock:
            try:
                self._maybe_rotate()
                if self._file is None:
                    self._file = open(self._path, "a", encoding="utf-8")
                record = {
                    "ts": event.timestamp,
                    "type": event.type.value,
                    "session_id": event.session_id,
                    "seq": event.sequence,
                    "data": event.data,
                }
                self._file.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
                self._file.flush()
            except Exception as e:
                logger.debug("Telemetry write failed: %s", e)

    def _maybe_rotate(self) -> None:
        try:
            if not self._path.exists():
                return
            if self._path.stat().st_size < self._max_size:
                return
            if self._file:
                self._file.close()
                self._file = None

            stem, suffix = self._path.stem, self._path.suffix
            parent = self._path.parent

            oldest = parent / "{}.{}{}".format(stem, self._max_rotated, suffix)
            if oldest.exists():
                oldest.unlink()

            for i in range(self._max_rotated - 1, 0, -1):
                src = parent / "{}.{}{}".format(stem, i, suffix)
                dst = parent / "{}.{}{}".format(stem, i + 1, suffix)
                if src.exists():
                    src.rename(dst)

            rotated = parent / "{}.1{}".format(stem, suffix)
            self._path.rename(rotated)
        except Exception as e:
            logger.debug("Telemetry rotation failed: %s", e)

    def close(self) -> None:
        with self._lock:
            if self._file:
                self._file.close()
                self._file = None


class CompositeTelemetrySink:
    """Fan-out to multiple sinks."""

    def __init__(self, sinks: Optional[List[TelemetrySink]] = None) -> None:
        self._sinks: List[TelemetrySink] = list(sinks or [])

    def add_sink(self, sink: TelemetrySink) -> None:
        self._sinks.append(sink)

    def emit(self, event: TelemetryEvent) -> None:
        for sink in self._sinks:
            try:
                sink.emit(event)
            except Exception as e:
                logger.debug("Telemetry sink error: %s", e)


class SessionTracer:
    """Routes all agent events through telemetry sinks.

    Central tracing facade for a single session.  Replaces separate
    audit/transcript/cost recording in ``Agent``.

    Usage::

        tracer = SessionTracer("session-abc", sinks=[JsonlTelemetrySink("./events.jsonl")])
        tracer.trace_session_start(provider="openai", model="gpt-4o")
        tracer.trace_tool_call("execute_module", {"module_id": "browser.click"}, result, 150)
        tracer.trace_llm_call("gpt-4o", prompt_tokens=500, completion_tokens=200)
    """

    def __init__(
        self,
        session_id: str,
        sinks: Optional[List[TelemetrySink]] = None,
    ) -> None:
        self._session_id = session_id
        self._sink = CompositeTelemetrySink(sinks)
        self._sequence = 0

    def _emit(self, event_type: TelemetryEventType, data: Dict[str, Any]) -> None:
        self._sequence += 1
        event = TelemetryEvent(
            type=event_type,
            session_id=self._session_id,
            sequence=self._sequence,
            data=data,
        )
        self._sink.emit(event)

    def add_sink(self, sink: TelemetrySink) -> None:
        self._sink.add_sink(sink)

    # ── Trace methods ──────────────────────────────────────────────

    def trace_session_start(self, **kwargs: Any) -> None:
        self._emit(TelemetryEventType.SESSION_START, kwargs)

    def trace_session_end(self, **kwargs: Any) -> None:
        self._emit(TelemetryEventType.SESSION_END, kwargs)

    def trace_tool_call(
        self, name: str, arguments: Dict, result: Any, duration_ms: int = 0,
    ) -> None:
        self._emit(TelemetryEventType.TOOL_CALL, {
            "tool_name": name,
            "arguments": arguments,
            "duration_ms": duration_ms,
        })
        # Separate tool result event for large payloads
        result_preview = json.dumps(result, ensure_ascii=False, default=str)[:500] if result else ""
        ok = result.get("ok", True) if isinstance(result, dict) else True
        self._emit(TelemetryEventType.TOOL_RESULT, {
            "tool_name": name,
            "ok": ok,
            "result_preview": result_preview,
        })

    def trace_llm_call(
        self,
        model: str,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        cache_creation_tokens: int = 0,
        cache_read_tokens: int = 0,
        **kwargs: Any,
    ) -> None:
        self._emit(TelemetryEventType.LLM_RESPONSE, {
            "model": model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "cache_creation_tokens": cache_creation_tokens,
            "cache_read_tokens": cache_read_tokens,
            **kwargs,
        })

    def trace_permission_check(
        self, tool_name: str, level: str, allowed: bool, reason: str = "",
    ) -> None:
        self._emit(TelemetryEventType.PERMISSION_CHECK, {
            "tool_name": tool_name,
            "level": level,
            "allowed": allowed,
            "reason": reason,
        })

    def trace_hook_decision(
        self, hook_name: str, tool_name: str, allowed: bool, reason: str = "",
    ) -> None:
        self._emit(TelemetryEventType.HOOK_DECISION, {
            "hook_name": hook_name,
            "tool_name": tool_name,
            "allowed": allowed,
            "reason": reason,
        })

    def trace_compaction(self, tokens_saved: int, level: str) -> None:
        self._emit(TelemetryEventType.COMPACTION, {
            "tokens_saved": tokens_saved,
            "level": level,
        })

    def trace_error(self, error: str, context: str = "") -> None:
        self._emit(TelemetryEventType.ERROR, {
            "error": error,
            "context": context,
        })

    def trace_cost_update(
        self, model: str, cost_usd: float, total_cost_usd: float,
    ) -> None:
        self._emit(TelemetryEventType.COST_UPDATE, {
            "model": model,
            "cost_usd": cost_usd,
            "total_cost_usd": total_cost_usd,
        })
