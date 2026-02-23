# Copyright 2024 Flyto
# Licensed under the Apache License, Version 2.0
"""Sensitive key detection and log redaction."""
from typing import Any

_SENSITIVE_KEYWORDS = frozenset({
    "password", "passwd", "api_key", "apikey", "token", "secret",
    "credential", "authorization", "auth", "bearer", "cookie",
    "access_token", "refresh_token", "session_token",
})


def is_sensitive_key(key: str) -> bool:
    """Check if a key name looks sensitive (partial match)."""
    lower = key.lower()
    return any(kw in lower for kw in _SENSITIVE_KEYWORDS)


def redact_args(obj: Any) -> Any:
    """Recursively redact sensitive values in dicts and lists."""
    if isinstance(obj, dict):
        return {
            k: "***" if is_sensitive_key(k) else redact_args(v)
            for k, v in obj.items()
        }
    elif isinstance(obj, (list, tuple)):
        return [redact_args(item) for item in obj]
    return obj
