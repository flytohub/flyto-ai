"""Accept only non-actuating CLI protocol events and structured completion."""

from .contracts import MAX_EVENT_BYTES, CliRuntimeError, decode_json


def failure_code(value) -> str:
    """Inspect transient error bytes in memory; retain only a fixed code."""
    text = str(value)[:8000].lower()
    for words, code in (
        (("not logged in", "unauthorized", "authentication", "login required"), "cli_auth_required"),
        (("quota", "usage limit", "credit balance", "insufficient_quota"), "cli_quota_exhausted"),
        (("rate limit", "overloaded"), "cli_capacity_unavailable"),
    ):
        if any(word in text for word in words):
            return code
    return "cli_provider_failed"


class EventReader:
    """The CLI's answer is a proposed decision, never an action receipt."""

    def __init__(self, source):
        self.source = source
        self.completed = False
        self.content = None
        self.usage = {}
        self.session_id = ""
        self.model = ""
        self._structured_calls = set()

    def read(self, raw: bytes):
        if len(raw) > MAX_EVENT_BYTES:
            raise CliRuntimeError("cli_output_too_large")
        event = decode_json(raw)
        if not isinstance(event, dict) or not isinstance(event.get("type"), str):
            raise CliRuntimeError("cli_invalid_output")
        if self.completed:
            raise CliRuntimeError("cli_invalid_output")
        if self.source == "claude_cli":
            self._claude(event)
        else:
            self._codex(event)

    def _session(self, value):
        if not isinstance(value, str) or not 1 <= len(value) <= 128:
            raise CliRuntimeError("cli_invalid_output")
        if self.session_id and self.session_id != value:
            raise CliRuntimeError("cli_session_changed")
        self.session_id = value

    def _usage(self, raw):
        if not isinstance(raw, dict):
            return
        for source, target in (("input_tokens", "prompt_tokens"), ("output_tokens", "completion_tokens")):
            value = raw.get(source)
            if isinstance(value, int) and not isinstance(value, bool) and 0 <= value <= 10**9:
                self.usage[target] = value
        self.usage["total_tokens"] = sum(self.usage.get(key, 0) for key in ("prompt_tokens", "completion_tokens"))
        for key in ('cache_creation_input_tokens', 'cache_read_input_tokens'):
            value = raw.get(key)
            if isinstance(value, int) and not isinstance(value, bool) and 0 <= value <= 10**9:
                self.usage['prompt_tokens'] = self.usage.get('prompt_tokens', 0) + value
                self.usage['total_tokens'] += value

    def _claude(self, event):
        kind = event["type"]
        if kind == "system" and event.get("subtype") == "init":
            if isinstance(event.get('model'), str) and len(event['model']) <= 128:
                self.model = event['model']
            if event.get("tools") not in (None, [], ["StructuredOutput"]):
                raise CliRuntimeError("cli_native_tools_exposed")
            if event.get("session_id"):
                self._session(event["session_id"])
        elif kind == "assistant":
            message = event.get("message") or {}
            for block in message.get("content", []):
                if isinstance(block, dict) and block.get("type") == "tool_use":
                    if block.get("name") != "StructuredOutput" or not isinstance(block.get("id"), str):
                        raise CliRuntimeError("cli_native_action_refused")
                    self._structured_calls.add(block["id"])
        elif kind == "user":
            # Claude acknowledges its non-actuating JSON-schema formatter with
            # a tool_result event. It cannot acknowledge any execution tool.
            content = (event.get("message") or {}).get("content", [])
            if not isinstance(content, list) or not content:
                raise CliRuntimeError("cli_invalid_output")
            for block in content:
                if (not isinstance(block, dict) or block.get("type") != "tool_result"
                        or block.get("tool_use_id") not in self._structured_calls):
                    raise CliRuntimeError("cli_native_action_refused")
                self._structured_calls.remove(block["tool_use_id"])
        elif kind == "result":
            if event.get("is_error") is True or event.get("subtype") != "success":
                raise CliRuntimeError(failure_code(event))
            self._session(event.get("session_id"))
            self.content = event.get("structured_output")
            if self.content is None:
                self.content = decode_json(event.get("result", ""))
            self._usage(event.get("usage"))
            self.completed = True
        elif kind not in {"system", "rate_limit_event", "stream_event", "assistant"}:
            raise CliRuntimeError("cli_invalid_output")

    def _codex(self, event):
        kind = event["type"]
        if kind == "thread.started":
            self._session(event.get("thread_id"))
        elif kind in {"item.started", "item.updated", "item.completed"}:
            item = event.get("item") or {}
            if item.get("type") not in {"agent_message", "reasoning"}:
                raise CliRuntimeError("cli_native_action_refused")
            if kind == "item.completed" and item.get("type") == "agent_message":
                self.content = decode_json(item.get("text", ""))
        elif kind == "turn.completed":
            self._usage(event.get("usage"))
            self.completed = True
        elif kind in {"error", "turn.failed"}:
            raise CliRuntimeError(failure_code(event))
        elif kind != "turn.started":
            raise CliRuntimeError("cli_invalid_output")

    def result(self):
        if not self.completed or not self.session_id or self.content is None:
            raise CliRuntimeError("cli_incomplete_output")
        return self.content, dict(self.usage)
