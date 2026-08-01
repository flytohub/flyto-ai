# Copyright 2024 Flyto2
# Licensed under the Apache License, Version 2.0
"""Crash-safe, secret-aware thread and trajectory persistence."""
from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

from flyto_ai.coding.contracts import CONTRACT_VERSION
from flyto_ai.redaction import is_sensitive_key


_THREAD_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
_SECRET_PATTERNS = (
    re.compile(r"\bBearer\s+[A-Za-z0-9._~+/=-]{8,}", re.IGNORECASE),
    re.compile(r"\b(?:sk|ghp|github_pat|xox[baprs])[-_A-Za-z0-9]{8,}\b"),
    re.compile(
        r"(?i)\b(password|passwd|api[_-]?key|access[_-]?token|secret)\b"
        r"\s*[:=]\s*([^\s,;]+)"
    ),
)
_SAFE_USAGE_COUNTERS = frozenset({
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "input_tokens",
    "output_tokens",
    "cache_creation_input_tokens",
    "cache_read_input_tokens",
    "prompt_eval_count",
    "eval_count",
})


def _safe_text(value: str, limit: int = 200_000) -> str:
    text = value[:limit]
    for pattern in _SECRET_PATTERNS:
        text = pattern.sub(lambda match: "{}=***".format(match.group(1)) if match.lastindex else "***", text)
    return text


def redact_evidence(value: Any) -> Any:
    """Recursively remove credential-shaped values before durable storage."""

    if isinstance(value, str):
        return _safe_text(value)
    if isinstance(value, dict):
        projected = {}
        for key, item in value.items():
            safe_key = str(key)[:128]
            normalized_key = safe_key.lower()
            if is_sensitive_key(safe_key):
                if (
                    normalized_key in _SAFE_USAGE_COUNTERS
                    and isinstance(item, int)
                    and not isinstance(item, bool)
                    and item >= 0
                ):
                    projected[safe_key] = item
                else:
                    projected[safe_key] = "***"
            else:
                projected[safe_key] = redact_evidence(item)
        return projected
    if isinstance(value, (list, tuple)):
        return [redact_evidence(item) for item in value[:1000]]
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return _safe_text(str(value), 10_000)


class ThreadStore:
    """Own per-thread metadata and an append-only JSONL event stream."""

    def __init__(self, root: Optional[str] = None) -> None:
        configured = root or os.environ.get("FLYTO_AI_CODING_STATE_DIR", "~/.flyto/coding")
        self.root = Path(configured).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        try:
            self.root.chmod(0o700)
        except OSError:
            pass

    def create(self, workspace: str, thread_id: Optional[str] = None) -> Dict[str, Any]:
        thread_id = thread_id or "thread_{}".format(uuid.uuid4().hex[:20])
        thread_dir = self._thread_dir(thread_id)
        if thread_dir.exists():
            raise FileExistsError("coding thread already exists")
        thread_dir.mkdir(mode=0o700)
        now = time.time()
        metadata = {
            "contract_version": CONTRACT_VERSION,
            "thread_id": thread_id,
            "workspace": str(Path(workspace).resolve()),
            "created_at": now,
            "updated_at": now,
            "status": "created",
            "turn_count": 0,
        }
        self._write_metadata(thread_id, metadata)
        self.append(thread_id, "thread.created", {"workspace": metadata["workspace"]})
        return metadata

    def load(self, thread_id: str, workspace: Optional[str] = None) -> Dict[str, Any]:
        path = self._thread_dir(thread_id) / "metadata.json"
        try:
            metadata = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise FileNotFoundError("coding thread does not exist") from exc
        if metadata.get("contract_version") != CONTRACT_VERSION:
            raise ValueError("coding thread contract version is unsupported")
        if workspace and metadata.get("workspace") != str(Path(workspace).resolve()):
            raise ValueError("coding thread belongs to a different workspace")
        return metadata

    def update(self, thread_id: str, **changes: Any) -> Dict[str, Any]:
        metadata = self.load(thread_id)
        allowed = {"status", "turn_count", "last_failure_code"}
        unknown = set(changes) - allowed
        if unknown:
            raise ValueError("unsupported thread metadata fields")
        metadata.update(redact_evidence(changes))
        metadata["updated_at"] = time.time()
        self._write_metadata(thread_id, metadata)
        return metadata

    def append(self, thread_id: str, event_type: str, data: Dict[str, Any]) -> Dict[str, Any]:
        thread_dir = self._thread_dir(thread_id)
        if not thread_dir.is_dir():
            raise FileNotFoundError("coding thread does not exist")
        event = {
            "contract_version": CONTRACT_VERSION,
            "event_id": uuid.uuid4().hex,
            "timestamp": time.time(),
            "type": str(event_type)[:128],
            "data": redact_evidence(data),
        }
        line = json.dumps(event, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        path = thread_dir / "events.jsonl"
        fd = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
        try:
            os.write(fd, line.encode("utf-8"))
            os.fsync(fd)
        finally:
            os.close(fd)
        return event

    def events(self, thread_id: str) -> List[Dict[str, Any]]:
        path = self._thread_dir(thread_id) / "events.jsonl"
        if not path.exists():
            return []
        events: List[Dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError("coding event must be an object")
                events.append(value)
        return events

    def replay_messages(self, thread_id: str) -> List[Dict[str, str]]:
        messages: List[Dict[str, str]] = []
        for event in self.events(thread_id):
            if event.get("type") != "conversation.message":
                continue
            data = event.get("data", {})
            role, content = data.get("role"), data.get("content")
            if role in {"user", "assistant"} and isinstance(content, str):
                messages.append({"role": role, "content": content})
        return messages[-40:]

    def evidence_path(self, thread_id: str) -> str:
        return str(self._thread_dir(thread_id) / "events.jsonl")

    def digest(self, thread_id: str) -> str:
        path = self._thread_dir(thread_id) / "events.jsonl"
        return hashlib.sha256(path.read_bytes() if path.exists() else b"").hexdigest()

    def _thread_dir(self, thread_id: str) -> Path:
        if not _THREAD_RE.fullmatch(thread_id):
            raise ValueError("invalid coding thread id")
        return self.root / thread_id

    def _write_metadata(self, thread_id: str, metadata: Dict[str, Any]) -> None:
        thread_dir = self._thread_dir(thread_id)
        payload = json.dumps(redact_evidence(metadata), ensure_ascii=False, indent=2, sort_keys=True)
        fd, raw_path = tempfile.mkstemp(prefix=".metadata-", suffix=".tmp", dir=str(thread_dir))
        try:
            os.fchmod(fd, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(raw_path, thread_dir / "metadata.json")
        except Exception:
            try:
                os.unlink(raw_path)
            except OSError:
                pass
            raise
