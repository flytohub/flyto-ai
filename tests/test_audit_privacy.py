"""Application audit logs retain metadata without copying task credentials."""
import hashlib
import json
import logging
from types import SimpleNamespace

import pytest

from flyto_ai import audit
from flyto_ai.agent import Agent
from flyto_ai.providers.base import dispatch_and_log_tool


class RuntimeHandle:
    def __str__(self):
        raise AssertionError("Audit must not inspect runtime handles")


def test_audit_metadata_is_private_in_logger_memory_and_file(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(audit, "_AUDIT_DIR", tmp_path)
    monkeypatch.setattr(audit, "_entries", [])
    goal = "Read the private calendar with password test-private-password. " * 8
    secret = "test-private-password"
    handle = RuntimeHandle()
    with caplog.at_level(logging.INFO, logger="flyto_ai.audit"):
        audit.ChatAuditEntry(
            user_message=goal, provider="openai", model="test-model",
            tool_calls_count=1, execution_count=1, total_tokens=25,
            ok=False, error="Login failed for " + secret,
            tool_calls=[{
                "function": "execute_module", "module_id": "browser.type",
                "arguments": {"sensitive_text": secret},
                "result_preview": secret, "driver": handle, "ok": False,
            }],
            execution_results=[{
                "module_id": "browser.type", "ok": False,
                "error": "Rejected " + secret, "duration_ms": 14,
                "data": {"password": secret, "driver": handle},
            }],
        ).emit()
    records = [
        json.loads(caplog.records[-1].message),
        audit.get_recent_entries()[0],
        json.loads(next(tmp_path.glob("*.jsonl")).read_text()),
    ]
    for record in records:
        serialized = json.dumps(record)
        assert secret not in serialized
        assert goal not in serialized
        assert "user_message" not in record
        assert record["user_message_length"] == len(goal)
        assert record["user_message_sha256"] == hashlib.sha256(goal.encode()).hexdigest()
        assert record["tool_calls"] == [{
            "function": "execute_module", "module_id": "browser.type", "ok": False,
        }]
        assert record["execution_results"] == [{
            "module_id": "browser.type", "ok": False, "duration_ms": 14,
            "has_error": True,
        }]
        assert record["error"] == "error"
        assert record["total_tokens"] == 25


def test_agent_audit_hashes_full_goal_instead_of_truncated_prefix(monkeypatch):
    observed = []
    monkeypatch.setattr(audit.ChatAuditEntry, "emit", lambda self: observed.append(self))
    agent = SimpleNamespace(
        _cost_tracker=None, _last_model_route=None,
        _config=SimpleNamespace(provider="openai", resolved_model="test-model"),
    )
    goal = "A" * 250 + " different ending"
    Agent._emit_audit(agent, goal, "execute", [], [], True, None, 12, {})
    assert observed[0].user_message == goal


@pytest.mark.asyncio
async def test_provider_log_omits_arguments_without_changing_dispatch_or_owner_evidence(caplog):
    args = {"module_id": "browser.type", "params": {"sensitive_text": "private-value"}}
    seen = []

    async def dispatch(name, params):
        seen.append((name, params))
        return {"status": "success", "data": {"observed": "actual-tool-value"}}

    with caplog.at_level(logging.INFO, logger="flyto_ai.providers.base"):
        result, entry, images = await dispatch_and_log_tool(
            "execute_module", args, dispatch, 2,
        )
    assert seen == [("execute_module", args)]
    assert json.loads(result)["data"]["observed"] == "actual-tool-value"
    assert entry["arguments"] is args
    assert entry["ok"] is True
    assert images == []
    assert "private-value" not in caplog.text
    assert "sensitive_text" not in caplog.text
    assert "Tool call [3]: execute_module" in caplog.text


@pytest.mark.parametrize("error", ["timeout", "provider_call_failed"])
def test_audit_preserves_known_error_codes(tmp_path, monkeypatch, error):
    monkeypatch.setattr(audit, "_AUDIT_DIR", tmp_path)
    monkeypatch.setattr(audit, "_entries", [])
    audit.ChatAuditEntry(ok=False, error=error).emit()
    assert audit.get_recent_entries()[0]["error"] == error
