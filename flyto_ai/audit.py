# Copyright 2024 Flyto
# Licensed under the Apache License, Version 2.0
"""Structured audit logging for chat interactions."""
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger("flyto_ai.audit")


@dataclass
class ChatAuditEntry:
    """One chat interaction audit record.

    Emitted as structured JSON to the ``flyto_ai.audit`` logger at INFO level.
    """
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

    def emit(self) -> None:
        """Emit this entry as a structured JSON log line."""
        record = {
            "event": "chat_audit",
            "ts": self.timestamp,
            "user_message": self.user_message[:200],
            "provider": self.provider,
            "model": self.model,
            "mode": self.mode,
            "tool_calls": self.tool_calls_count,
            "executions": self.execution_count,
            "duration_ms": self.duration_ms,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "ok": self.ok,
        }
        if self.error:
            record["error"] = self.error
        logger.info(json.dumps(record, ensure_ascii=False))
