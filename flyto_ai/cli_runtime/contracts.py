"""Host-owned local CLI selection and bounded inference messages."""

import json
import math
import os
import re
from dataclasses import dataclass
from typing import Literal

MAX_INPUT_BYTES = 1_000_000
MAX_OUTPUT_BYTES = 2_000_000
MAX_EVENT_BYTES = 512_000
MAX_IMAGES = 8
MAX_IMAGE_BYTES = 5_000_000
MAX_CALLS = 8


def valid_model_id(value, *, allow_empty=True):
    """Pass official IDs/aliases verbatim; never interpret them as CLI flags."""
    return isinstance(value, str) and ((allow_empty and value == "") or bool(
        re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:/@+\[\]-]{0,255}", value)
    ))


class CliRuntimeError(RuntimeError):
    """A public classification, never raw provider output or credentials."""

    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


@dataclass(frozen=True)
class CliRuntimeConfig:
    """Selected by the authenticated host, never by a model tool argument."""

    source: Literal["codex_cli", "claude_cli"]
    model: str = ""
    command: str | None = None
    timeout_seconds: float = 100.0

    def __post_init__(self):
        if self.source not in {"codex_cli", "claude_cli"}:
            raise ValueError("Unsupported local CLI source")
        if not valid_model_id(self.model):
            raise ValueError("CLI model must be a bounded identifier")
        if self.command is not None and (not isinstance(self.command, str) or not self.command or "\x00" in self.command):
            raise ValueError("CLI command must be one executable path")
        value = self.timeout_seconds
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0.1 <= value <= 300:
            raise ValueError("CLI timeout must be between 0.1 and 300 seconds")


def cli_environment() -> dict[str, str]:
    """Keep official local sign-in locations; inherit no provider/CI secrets."""
    names = ("HOME", "USER", "LOGNAME", "CODEX_HOME", "CLAUDE_CONFIG_DIR", "PATH", "TMPDIR", "LANG", "LC_ALL", "TERM", "SSL_CERT_FILE", "SSL_CERT_DIR")
    return {name: os.environ[name] for name in names if name in os.environ}


def encode_json(value, *, limit=MAX_INPUT_BYTES) -> str:
    try:
        text = json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    except (TypeError, ValueError, RecursionError) as exc:
        raise CliRuntimeError("cli_invalid_input") from exc
    if len(text.encode()) > limit:
        raise CliRuntimeError("cli_input_too_large")
    return text


def decode_json(text):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate JSON key")
            result[key] = value
        return result

    try:
        return json.loads(text, object_pairs_hook=pairs, parse_constant=lambda _: (_ for _ in ()).throw(ValueError("Nonfinite JSON")))
    except (TypeError, ValueError, RecursionError) as exc:
        raise CliRuntimeError("cli_invalid_output") from exc


INTENT_SCHEMA = {
    "type": "object", "additionalProperties": False,
    "properties": {
        "content": {"type": "string"},
        "tool_calls": {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "properties": {"name": {"type": "string"}, "arguments_json": {"type": "string"}},
            "required": ["name", "arguments_json"],
        }},
    }, "required": ["content", "tool_calls"],
}


def checked_intent(value, names: set[str]):
    if not isinstance(value, dict) or set(value) != {"content", "tool_calls"}:
        raise CliRuntimeError("cli_invalid_output")
    content, calls = value["content"], value["tool_calls"]
    if not isinstance(content, str) or len(content) > 50_000 or not isinstance(calls, list) or len(calls) > MAX_CALLS:
        raise CliRuntimeError("cli_invalid_output")
    checked = []
    for call in calls:
        if not isinstance(call, dict) or set(call) != {"name", "arguments_json"}:
            raise CliRuntimeError("cli_invalid_output")
        name, arguments = call["name"], call["arguments_json"]
        if not isinstance(name, str) or name not in names:
            raise CliRuntimeError("cli_tool_not_available")
        if not isinstance(arguments, str) or len(arguments) > 65_536:
            raise CliRuntimeError("cli_invalid_output")
        params = decode_json(arguments)
        if not isinstance(params, dict):
            raise CliRuntimeError("cli_invalid_output")
        checked.append((name, params))
    return content, checked
