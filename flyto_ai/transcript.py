# Copyright 2024 Flyto
# Licensed under the Apache License, Version 2.0
"""JSONL transcript — crash-safe, replayable session recording."""
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Default transcript directory
_DEFAULT_DIR = "~/.flyto/transcripts"


class TranscriptWriter:
    """Append-only JSONL writer for a single session.

    Each line is a self-contained JSON object with:
    - ts: timestamp (epoch)
    - type: "user" | "assistant" | "tool_call" | "tool_result" | "execution" | "error" | "meta"
    - data: event-specific payload

    Crash-safe: each event is flushed immediately after write.
    """

    def __init__(self, session_id: str, transcript_dir: Optional[str] = None) -> None:
        self._session_id = session_id
        base_dir = Path(os.path.expanduser(transcript_dir or _DEFAULT_DIR))
        base_dir.mkdir(parents=True, exist_ok=True)
        self._path = base_dir / "{}.jsonl".format(session_id)
        self._file = None

    @property
    def path(self) -> Path:
        """Path to the JSONL transcript file."""
        return self._path

    def _ensure_open(self):
        """Lazy-open the file."""
        if self._file is None:
            self._file = open(self._path, "a", encoding="utf-8")

    def _write_event(self, event_type: str, data: Any) -> None:
        """Write a single event line and flush."""
        try:
            self._ensure_open()
            record = {
                "ts": time.time(),
                "session_id": self._session_id,
                "type": event_type,
                "data": data,
            }
            line = json.dumps(record, ensure_ascii=False, default=str)
            self._file.write(line + "\n")
            self._file.flush()
        except Exception as e:
            logger.debug("Transcript write failed: %s", e)

    def record_user(self, message: str, metadata: Optional[Dict] = None) -> None:
        """Record a user message."""
        data = {"message": message}
        if metadata:
            data["metadata"] = metadata
        self._write_event("user", data)

    def record_assistant(self, message: str, provider: Optional[str] = None, model: Optional[str] = None) -> None:
        """Record an assistant response."""
        data = {"message": message}
        if provider:
            data["provider"] = provider
        if model:
            data["model"] = model
        self._write_event("assistant", data)

    def record_tool_call(self, name: str, arguments: Dict, round_num: int = 0) -> None:
        """Record a tool call."""
        self._write_event("tool_call", {
            "name": name,
            "arguments": arguments,
            "round": round_num,
        })

    def record_tool_result(self, name: str, result: Any, ok: bool = True) -> None:
        """Record a tool result."""
        # Truncate large results
        result_str = json.dumps(result, ensure_ascii=False, default=str) if not isinstance(result, str) else result
        if len(result_str) > 2000:
            result_str = result_str[:2000] + "...(truncated)"
        self._write_event("tool_result", {
            "name": name,
            "result": result_str,
            "ok": ok,
        })

    def record_execution(self, module_id: str, ok: bool, result_preview: str = "") -> None:
        """Record a module execution result."""
        self._write_event("execution", {
            "module_id": module_id,
            "ok": ok,
            "result_preview": result_preview[:500],
        })

    def record_error(self, error: str, context: Optional[str] = None) -> None:
        """Record an error event."""
        data = {"error": error}
        if context:
            data["context"] = context
        self._write_event("error", data)

    def record_meta(self, data: Dict) -> None:
        """Record metadata (config, usage, cost, etc.)."""
        self._write_event("meta", data)

    def close(self) -> None:
        """Flush and close the transcript file."""
        if self._file:
            try:
                self._file.flush()
                self._file.close()
            except Exception:
                pass
            self._file = None

    def __del__(self):
        self.close()


def load_transcript(path: str) -> List[Dict]:
    """Load a transcript JSONL file and return all events.

    Skips malformed lines gracefully (crash recovery).
    """
    events = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    logger.debug("Transcript: skipped malformed line %d in %s", line_num, path)
    except FileNotFoundError:
        logger.debug("Transcript file not found: %s", path)
    return events


def replay_messages(events: List[Dict]) -> List[Dict[str, str]]:
    """Extract user/assistant messages from transcript events for replay.

    Returns a list of {"role": ..., "content": ...} dicts suitable
    for feeding back into Agent.chat() as history.
    """
    messages = []
    for ev in events:
        ev_type = ev.get("type", "")
        data = ev.get("data", {})
        if ev_type == "user":
            messages.append({"role": "user", "content": data.get("message", "")})
        elif ev_type == "assistant":
            messages.append({"role": "assistant", "content": data.get("message", "")})
    return messages
