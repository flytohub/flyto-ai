# Copyright 2024 Flyto
# Licensed under the Apache License, Version 2.0
"""Safety layer — credential masking, circuit breaker, bounded history.

System-level protections that don't rely on LLM behavior.
"""
import logging
import re
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

# ── Credential Masking ──────────────────────────────────────────

_SENSITIVE_KEYS = frozenset({
    "password", "passwd", "pass", "pw", "secret", "token",
    "api_key", "apikey", "auth", "credential", "private_key",
})


def mask_sensitive(data: Any, depth: int = 0) -> Any:
    """Recursively mask sensitive values in dicts/lists.

    Replaces password-like values with '***' so they don't leak
    into LLM context, transcripts, or logs.
    """
    if depth > 10:
        return data

    if isinstance(data, dict):
        masked = {}
        for k, v in data.items():
            if isinstance(k, str) and k.lower() in _SENSITIVE_KEYS:
                masked[k] = "***"
            else:
                masked[k] = mask_sensitive(v, depth + 1)
        return masked

    if isinstance(data, list):
        return [mask_sensitive(item, depth + 1) for item in data]

    if isinstance(data, str):
        # Mask inline secrets: "password=xyz123" → "password=***"
        return re.sub(
            r'(password|passwd|secret|token|api_key|apikey)\s*[=:]\s*\S+',
            r'\1=***',
            data,
            flags=re.IGNORECASE,
        )

    return data


# ── Circuit Breaker ─────────────────────────────────────────────

class CircuitBreaker:
    """Stop retrying the same failing module after N consecutive failures.

    Tracks both hard failures (ok=false) and soft failures (ok=true but empty result).
    Prevents the LLM from hammering a broken selector 30+ times.
    """

    def __init__(self, max_failures: int = 3, max_empty: int = 3) -> None:
        self._max = max_failures
        self._max_empty = max_empty
        self._failures: Dict[str, int] = {}
        self._empty: Dict[str, int] = {}  # track empty results per module

    def record_result(self, module_id: str, ok: bool, result: Any = None) -> None:
        """Record a module execution result."""
        if ok:
            self._failures.pop(module_id, None)
        else:
            self._failures[module_id] = self._failures.get(module_id, 0) + 1

        # Track empty results (ok=true but data is []/{}/""/null)
        if ok and isinstance(result, dict):
            data = result.get("data", result.get("result"))
            is_empty = (
                data is None
                or data == []
                or data == {}
                or data == ""
                or (isinstance(data, list) and all(
                    not d or d == {} or d == [] or d == ""
                    for d in data
                ))
            )
            if is_empty:
                self._empty[module_id] = self._empty.get(module_id, 0) + 1
            else:
                self._empty.pop(module_id, None)

    def is_tripped(self, module_id: str) -> bool:
        """Check if a module has exceeded max consecutive failures OR empty results."""
        return (
            self._failures.get(module_id, 0) >= self._max
            or self._empty.get(module_id, 0) >= self._max_empty
        )

    def get_message(self, module_id: str) -> str:
        """Return a message explaining why the module is blocked."""
        fail_count = self._failures.get(module_id, 0)
        empty_count = self._empty.get(module_id, 0)
        if empty_count >= self._max_empty:
            return (
                "Module '{}' returned empty results {} consecutive times. "
                "The selector is probably wrong. "
                "Try browser.evaluate with JavaScript instead, or use a different selector."
            ).format(module_id, empty_count)
        return (
            "Module '{}' has failed {} consecutive times. "
            "Execution blocked to prevent infinite retry. "
            "Try a different approach or module."
        ).format(module_id, fail_count)


# ── Bounded History ─────────────────────────────────────────────

class BoundedHistory:
    """Fixed-size execution history for param fixer data flow.

    Prevents memory leak from unbounded exec_history list.
    """

    def __init__(self, max_size: int = 20) -> None:
        self._max = max_size
        self._items: List[Dict[str, Any]] = []

    def append(self, item: Dict[str, Any]) -> None:
        self._items.append(item)
        if len(self._items) > self._max:
            self._items = self._items[-self._max:]

    def items(self) -> List[Dict[str, Any]]:
        return list(self._items)

    def __len__(self) -> int:
        return len(self._items)
