"""Structured security finding schema for test generation."""
from dataclasses import dataclass
from typing import Literal, Optional

VulnCategory = Literal[
    "sql_injection",
    "xss_reflected",
    "auth_bypass",
    # Future: "xss_stored", "ssrf", "idor", "deserialization"
]

SUPPORTED_CATEGORIES: list[str] = [
    "sql_injection",
    "xss_reflected",
    "auth_bypass",
]


@dataclass
class SecurityFinding:
    """Structured finding produced by flyto-indexer or flyto-engine."""

    category: VulnCategory
    source: str              # e.g. "request.args.get('user_id')"
    source_file: str         # e.g. "handler.py"
    source_line: int         # e.g. 42
    sink: str                # e.g. "cursor.execute(query)"
    sink_file: str           # e.g. "handler.py"
    sink_line: int           # e.g. 55
    severity: Literal["critical", "high", "medium", "low"]
    # Optional context for more accurate test generation
    param_name: Optional[str] = None        # e.g. "user_id"
    endpoint_path: Optional[str] = None     # e.g. "/api/users"
    http_method: Optional[str] = None       # e.g. "GET"
    sanitized: bool = False
    recommendation: Optional[str] = None
