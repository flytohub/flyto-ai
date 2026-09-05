# Copyright 2024 Flyto2
# Licensed under the Apache License, Version 2.0
"""Structured audit logging for chat interactions.

Writes to both Python logger AND ~/.flyto/audit/ JSONL files
so the Console dashboard can read execution history.
"""
import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("flyto_ai.audit")

# Persistent audit store — survives across requests
_AUDIT_DIR = Path.home() / ".flyto" / "audit"
_entries: List[Dict[str, Any]] = []

# Preserve stable diagnostics without writing exception messages or task input.
_ERROR_CODES = frozenset({
    "timeout", "provider_call_failed", "no_provider_available", "no_api_key",
    "invalid_base_url", "cancelled", "goal_unverified", "browser_cleanup_failed",
})


def _operation_metadata(operation: Dict[str, Any]) -> Dict[str, Any]:
    """Project operation status without inspecting arguments or runtime objects."""
    metadata: Dict[str, Any] = {}
    for key in ("function", "module_id"):
        value = operation.get(key)
        if isinstance(value, str):
            metadata[key] = value[:128]
    for key in ("ok",):
        value = operation.get(key)
        if isinstance(value, bool):
            metadata[key] = value
    duration = operation.get("duration_ms")
    if isinstance(duration, (int, float)) and not isinstance(duration, bool):
        metadata["duration_ms"] = duration
    if operation.get("error"):
        metadata["has_error"] = True
    return metadata


def _ensure_dir():
    _AUDIT_DIR.mkdir(parents=True, exist_ok=True)


def _today_file() -> Path:
    _ensure_dir()
    return _AUDIT_DIR / "{}.jsonl".format(datetime.now().strftime("%Y-%m-%d"))


@dataclass
class ChatAuditEntry:
    """One chat interaction audit record."""
    timestamp: float = field(default_factory=time.time)
    user_message: str = ""
    provider: str = ""
    model: str = ""
    mode: str = "execute"
    tool_calls_count: int = 0
    execution_count: int = 0
    duration_ms: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    ok: bool = True
    error: Optional[str] = None
    # Extended fields for console
    tool_calls: Optional[List[Dict]] = None
    execution_results: Optional[List[Dict]] = None
    cost_usd: float = 0.0

    def emit(self) -> None:
        """Emit metadata to logger, JSONL and memory; omit task/operation content."""
        record = {
            "event": "chat_audit",
            "timestamp": self.timestamp,
            "ts": self.timestamp,
            "user_message_sha256": hashlib.sha256(
                self.user_message.encode("utf-8"),
            ).hexdigest(),
            "user_message_length": len(self.user_message),
            "provider": self.provider,
            "model": self.model,
            "mode": self.mode,
            "tool_calls_count": self.tool_calls_count,
            "execution_count": self.execution_count,
            "duration_ms": self.duration_ms,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "ok": self.ok,
            "cost_usd": self.cost_usd,
        }
        if self.error:
            record["error"] = self.error if self.error in _ERROR_CODES else "error"
        if self.tool_calls:
            record["tool_calls"] = [
                _operation_metadata(call)
                for call in self.tool_calls[:20] if isinstance(call, dict)
            ]
        if self.execution_results:
            record["execution_results"] = [
                _operation_metadata(result)
                for result in self.execution_results[:20] if isinstance(result, dict)
            ]

        # 1. Python logger
        logger.info(json.dumps(record, ensure_ascii=False))

        # 2. In-memory store (for console API — instant access)
        _entries.append(record)
        # Keep last 500 entries in memory
        if len(_entries) > 500:
            del _entries[:100]

        # 3. JSONL file (persistent across restarts)
        try:
            with open(_today_file(), "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.debug("Failed to write audit file (%s)", type(e).__name__)


def get_recent_entries(limit: int = 50) -> List[Dict[str, Any]]:
    """Get recent audit entries. Prefers in-memory, falls back to file."""
    if _entries:
        return list(reversed(_entries[-limit:]))

    # Fall back to reading files
    entries = []
    try:
        _ensure_dir()
        files = sorted(_AUDIT_DIR.glob("*.jsonl"), reverse=True)[:3]
        for f in files:
            for line in f.read_text(encoding="utf-8").strip().split("\n"):
                if line:
                    try:
                        entries.append(json.loads(line))
                    except Exception:
                        pass
    except Exception:
        pass
    entries.sort(key=lambda x: x.get("timestamp", 0), reverse=True)
    return entries[:limit]
