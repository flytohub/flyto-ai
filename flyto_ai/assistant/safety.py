# Copyright 2024 Flyto2
# Licensed under the Apache License, Version 2.0
"""Safety layer — credential masking, circuit breaker, bounded history.

System-level protections that don't rely on LLM behavior.
"""
import logging
import hashlib
import json
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
        self._invalid: Dict[str, int] = {}

    @staticmethod
    def _parameter_key(module_id, params):
        encoded = json.dumps(params, sort_keys=True, default=lambda value: type(value).__name__)
        return module_id + ":" + hashlib.sha256(encoded.encode()).hexdigest()

    def record_result(self, module_id: str, ok: bool, result: Any = None, params=None) -> None:
        """Record a module execution result."""
        if isinstance(result, dict) and result.get("params_valid") is False:
            key = self._parameter_key(module_id, params)
            self._invalid[key] = self._invalid.get(key, 0) + 1
            return
        if ok:
            self._failures.pop(module_id, None)
        else:
            self._failures[module_id] = self._failures.get(module_id, 0) + 1

        # Track empty results (ok=true but data is []/{}/""/null)
        if ok and isinstance(result, dict) and ("data" in result or "result" in result):
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
        elif ok:
            self._empty.pop(module_id, None)

    def is_tripped(self, module_id: str, params=None) -> bool:
        """Check if a module has exceeded max consecutive failures OR empty results."""
        return (
            self._failures.get(module_id, 0) >= self._max
            or self._empty.get(module_id, 0) >= self._max_empty
            or self._invalid.get(self._parameter_key(module_id, params), 0) >= self._max
        )

    def get_message(self, module_id: str, params=None) -> str:
        """Return a message explaining why the module is blocked."""
        if self._invalid.get(self._parameter_key(module_id, params), 0) >= self._max:
            return (
                "These exact parameters repeatedly failed validation; no action was executed. "
                "Inspect the module's canonical params_schema and change the invalid arguments. "
                "A corrected call to this module remains available."
            )
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


# ── Variable Resolver ──────────────────────────────────────────

_VAR_PATTERN = re.compile(r'\$\{(?:steps\.)?(\w+)\.(\w+)\}')


def _resolve_variables(params: dict, exec_history: List[Dict[str, Any]]) -> dict:
    """Resolve ${steps.x.result} variables to actual values from exec history.

    LLM sometimes writes YAML variable syntax as literal strings.
    This resolver replaces them with actual values.
    """
    if not exec_history:
        return params

    resolved = {}
    any_resolved = False

    for key, value in params.items():
        if isinstance(value, str) and "${" in value:
            new_value = value
            for match in _VAR_PATTERN.finditer(value):
                step_ref = match.group(1)  # step name
                field_ref = match.group(2)  # field name (usually 'result')
                actual = _find_step_value(step_ref, field_ref, exec_history)
                if actual is not None:
                    if value == match.group(0):
                        # Entire value is a variable — replace completely (preserve type)
                        new_value = actual
                    else:
                        # Variable is part of a larger string — string replace
                        new_value = new_value.replace(match.group(0), str(actual))
                    any_resolved = True
                    logger.info("Var resolved: %s → %s", match.group(0), str(actual)[:50])
            resolved[key] = new_value
        else:
            resolved[key] = value

    return resolved if any_resolved else params


def _find_step_value(step_ref: str, field_ref: str, exec_history: List[Dict[str, Any]]) -> Any:
    """Find a value from execution history by step reference."""
    import json
    for r in reversed(exec_history):
        if not r.get("ok"):
            continue
        # Match by module_id suffix or step index
        mid = r.get("module_id", "")
        if step_ref in mid or step_ref.replace("_", ".") in mid:
            preview = r.get("result_preview", "")
            try:
                data = json.loads(preview) if isinstance(preview, str) else preview
                if isinstance(data, dict):
                    if field_ref in data:
                        return data[field_ref]
                    # 'result' often means the main output
                    if field_ref == "result":
                        for k in ("result", "text", "output", "content", "data", "value"):
                            if k in data and data[k]:
                                return data[k]
            except (ValueError, TypeError):
                if field_ref == "result" and isinstance(preview, str):
                    return preview
    return None
