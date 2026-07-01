# Copyright 2024 Flyto
# Licensed under the Apache License, Version 2.0
"""Structured audit logging for chat interactions.

Writes to both Python logger AND ~/.flyto/audit/ JSONL files
so the Console dashboard can read execution history.
"""
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
        """Emit this entry to logger + JSONL file + in-memory store."""
        record = {
            "event": "chat_audit",
            "timestamp": self.timestamp,
            "ts": self.timestamp,
            "user_message": self.user_message[:200],
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
            record["error"] = self.error
        if self.tool_calls:
            record["tool_calls"] = self.tool_calls[:20]  # cap for storage
        if self.execution_results:
            record["execution_results"] = [
                {
                    "module_id": er.get("module_id", ""),
                    "ok": er.get("ok", False),
                    "error": er.get("error", ""),
                    "duration_ms": er.get("duration_ms", 0),
                }
                for er in self.execution_results[:20]
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
            logger.debug("Failed to write audit file: %s", e)


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
