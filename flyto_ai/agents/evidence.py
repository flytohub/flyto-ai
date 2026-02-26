# Copyright 2024 Flyto
# Licensed under the Apache License, Version 2.0
"""Audit evidence collector for Claude Code Agent sessions.

Records every significant action (tool approvals, denials, verifications)
to a JSONL file for post-hoc review.
"""
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from flyto_ai.agents.models import EvidenceRecord

logger = logging.getLogger(__name__)


class EvidenceCollector:
    """Collects and persists evidence records for a coding session."""

    def __init__(self, session_id: str, base_dir: str = "~/.flyto/evidence"):
        self._session_id = session_id
        self._records: List[EvidenceRecord] = []
        self._dir = Path(base_dir).expanduser() / session_id
        self._files_changed: set = set()

    def record(self, phase: str, action: str, data: Optional[Dict[str, Any]] = None) -> None:
        rec = EvidenceRecord(
            timestamp=time.time(),
            phase=phase,
            action=action,
            data=data or {},
        )
        self._records.append(rec)
        logger.debug("Evidence [%s] %s: %s", phase, action, data)

    def track_file_change(self, path: str) -> None:
        self._files_changed.add(path)

    @property
    def files_changed(self) -> List[str]:
        return sorted(self._files_changed)

    async def save(self) -> Optional[Path]:
        """Persist records as JSONL. Returns the file path or None on failure."""
        if not self._records:
            return None
        try:
            self._dir.mkdir(parents=True, exist_ok=True)
            out = self._dir / "evidence.jsonl"
            lines = []
            for r in self._records:
                lines.append(json.dumps({
                    "timestamp": r.timestamp,
                    "phase": r.phase,
                    "action": r.action,
                    "data": r.data,
                }, ensure_ascii=False, default=str))
            out.write_text("\n".join(lines) + "\n", encoding="utf-8")
            logger.info("Evidence saved: %s (%d records)", out, len(self._records))
            return out
        except Exception as e:
            logger.warning("Failed to save evidence: %s", e)
            return None

    def to_list(self) -> List[EvidenceRecord]:
        return list(self._records)


async def evidence_post_hook(
    collector: EvidenceCollector,
    tool_name: str,
    tool_input: Dict[str, Any],
    tool_result: Any,
    **kwargs: Any,
) -> None:
    """Claude Agent SDK PostToolUse hook — record file changes."""
    if tool_name in ("Edit", "Write"):
        path = tool_input.get("file_path", "") or tool_input.get("path", "")
        if path:
            collector.track_file_change(path)
            collector.record("coding", "file_changed", {"tool": tool_name, "path": path})
    elif tool_name == "Bash":
        collector.record("coding", "bash_executed", {
            "command": (tool_input.get("command", ""))[:200],
        })
    else:
        collector.record("coding", "tool_used", {"tool": tool_name})
