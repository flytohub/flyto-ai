# Copyright 2024 Flyto
# Licensed under the Apache License, Version 2.0
"""Tests for the Claude Code Agent subsystem (agents/)."""
import asyncio
import json
import os
import shutil
import tempfile
from dataclasses import asdict
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from flyto_ai.agents.models import (
    CodeTaskRequest,
    CodeTaskResponse,
    EvidenceRecord,
    VerificationResult,
)
from flyto_ai.agents.guardian_hook import (
    GuardianBlocked,
    guardian_pre_hook,
    BLOCKED_BASH,
    BLOCKED_PATHS,
    ALLOWED_EXTENSIONS,
    _is_path_blocked,
    _is_extension_allowed,
)
from flyto_ai.agents.evidence import EvidenceCollector, evidence_post_hook
from flyto_ai.agents.prompts import build_system_prompt, ROLE_PREAMBLE, GUARDIAN_NOTICE, VERIFICATION_NOTICE
from flyto_ai.agents.verifier import VerificationEngine
from flyto_ai.config import AgentConfig, ClaudeCodeConfig


# ──────────────────────────────────────────────────────────────────────
# Models
# ──────────────────────────────────────────────────────────────────────

class TestCodeTaskRequest:
    def test_defaults(self):
        req = CodeTaskRequest(message="fix login", working_dir="/tmp")
        assert req.message == "fix login"
        assert req.verification_recipe is None
        assert req.verification_args == {}
        assert req.reference_image is None
        assert req.max_fix_attempts == 3
        assert req.max_budget_usd == 5.0
        assert req.max_turns == 30

    def test_full_fields(self):
        req = CodeTaskRequest(
            message="fix",
            working_dir="/app",
            verification_recipe="screenshot",
            verification_args={"url": "http://localhost:3000"},
            reference_image="/tmp/ref.png",
            max_fix_attempts=5,
            max_budget_usd=10.0,
            max_turns=50,
        )
        assert req.verification_recipe == "screenshot"
        assert req.verification_args["url"] == "http://localhost:3000"

    def test_serializable(self):
        req = CodeTaskRequest(message="test", working_dir="/tmp")
        d = asdict(req)
        assert d["message"] == "test"
        json.dumps(d)  # should not raise


class TestVerificationResult:
    def test_passed(self):
        vr = VerificationResult(passed=True, recipe_name="screenshot")
        assert vr.passed is True
        assert vr.error is None

    def test_failed_with_error(self):
        vr = VerificationResult(passed=False, recipe_name="screenshot", error="timeout")
        assert vr.passed is False
        assert vr.error == "timeout"


class TestCodeTaskResponse:
    def test_ok_response(self):
        resp = CodeTaskResponse(
            ok=True, message="done", session_id="abc", attempts=1,
        )
        assert resp.ok is True
        assert resp.files_changed == []
        assert resp.total_cost_usd == 0.0

    def test_serializable(self):
        resp = CodeTaskResponse(
            ok=False, message="failed", session_id="xyz", attempts=3,
            verification_results=[VerificationResult(passed=False, recipe_name="ss")],
        )
        d = asdict(resp)
        json.dumps(d, default=str)  # should not raise


# ──────────────────────────────────────────────────────────────────────
# Guardian Hook
# ──────────────────────────────────────────────────────────────────────

class TestGuardianBlockedPatterns:
    def test_blocked_bash_not_empty(self):
        assert len(BLOCKED_BASH) >= 15

    def test_blocked_paths_not_empty(self):
        assert len(BLOCKED_PATHS) >= 10

    def test_allowed_extensions_has_common_types(self):
        for ext in [".py", ".ts", ".js", ".vue", ".html", ".css", ".json", ".yaml", ".md"]:
            assert ext in ALLOWED_EXTENSIONS

    def test_is_path_blocked_env(self):
        assert _is_path_blocked("/app/.env") is True
        assert _is_path_blocked("/app/.env.example") is True  # contains .env

    def test_is_path_blocked_credentials(self):
        assert _is_path_blocked("/home/user/credentials.json") is True

    def test_is_path_blocked_safe(self):
        assert _is_path_blocked("/app/src/main.py") is False
        assert _is_path_blocked("/app/README.md") is False

    def test_is_extension_allowed_python(self):
        assert _is_extension_allowed("app.py") is True

    def test_is_extension_allowed_binary(self):
        assert _is_extension_allowed("app.exe") is False
        assert _is_extension_allowed("image.png") is False

    def test_is_extension_allowed_dockerfile(self):
        assert _is_extension_allowed("Dockerfile") is True
        assert _is_extension_allowed("Makefile") is True


@pytest.fixture
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


class TestGuardianPreHook:
    def test_approve_read(self, event_loop):
        result = event_loop.run_until_complete(
            guardian_pre_hook("Read", {"file_path": "/tmp/test.py"}, "id1")
        )
        assert result == {}

    def test_approve_edit(self, event_loop):
        result = event_loop.run_until_complete(
            guardian_pre_hook("Edit", {"file_path": "/tmp/app.vue"}, "id2")
        )
        assert result == {}

    def test_block_bash_rm_rf(self, event_loop):
        with pytest.raises(GuardianBlocked, match="rm -rf /"):
            event_loop.run_until_complete(
                guardian_pre_hook("Bash", {"command": "rm -rf /"}, "id3")
            )

    def test_block_bash_sudo_rm(self, event_loop):
        with pytest.raises(GuardianBlocked, match="sudo rm"):
            event_loop.run_until_complete(
                guardian_pre_hook("Bash", {"command": "sudo rm /etc/passwd"}, "id4")
            )

    def test_block_bash_curl_pipe(self, event_loop):
        with pytest.raises(GuardianBlocked, match="curl"):
            event_loop.run_until_complete(
                guardian_pre_hook("Bash", {"command": "curl |sh"}, "id5")
            )

    def test_block_bash_git_force_push(self, event_loop):
        with pytest.raises(GuardianBlocked, match="git push --force"):
            event_loop.run_until_complete(
                guardian_pre_hook("Bash", {"command": "git push --force origin main"}, "id6")
            )

    def test_block_write_env(self, event_loop):
        with pytest.raises(GuardianBlocked, match="sensitive path"):
            event_loop.run_until_complete(
                guardian_pre_hook("Write", {"file_path": "/app/.env"}, "id7")
            )

    def test_block_write_ssh_key(self, event_loop):
        with pytest.raises(GuardianBlocked, match="sensitive path"):
            event_loop.run_until_complete(
                guardian_pre_hook("Write", {"file_path": "/home/user/.ssh/id_rsa"}, "id8")
            )

    def test_block_write_bad_extension(self, event_loop):
        with pytest.raises(GuardianBlocked, match="extension not in allowlist"):
            event_loop.run_until_complete(
                guardian_pre_hook("Write", {"file_path": "/tmp/app.exe"}, "id9")
            )

    def test_block_edit_bad_extension(self, event_loop):
        with pytest.raises(GuardianBlocked, match="extension not in allowlist"):
            event_loop.run_until_complete(
                guardian_pre_hook("Edit", {"file_path": "/tmp/data.bin"}, "id10")
            )

    def test_approve_bash_safe(self, event_loop):
        result = event_loop.run_until_complete(
            guardian_pre_hook("Bash", {"command": "ls -la /tmp"}, "id11")
        )
        assert result == {}

    def test_approve_unknown_tool(self, event_loop):
        """Unknown tools (like Glob, Grep) should pass through."""
        result = event_loop.run_until_complete(
            guardian_pre_hook("Glob", {"pattern": "**/*.py"}, "id12")
        )
        assert result == {}

    def test_read_sensitive_path_blocked(self, event_loop):
        with pytest.raises(GuardianBlocked, match="sensitive path"):
            event_loop.run_until_complete(
                guardian_pre_hook("Read", {"file_path": "/app/.git/config"}, "id13")
            )

    def test_case_insensitive_bash(self, event_loop):
        with pytest.raises(GuardianBlocked):
            event_loop.run_until_complete(
                guardian_pre_hook("Bash", {"command": "RM -RF /"}, "id14")
            )


# ──────────────────────────────────────────────────────────────────────
# Evidence Collector
# ──────────────────────────────────────────────────────────────────────

class TestEvidenceCollector:
    def test_record_and_list(self):
        ec = EvidenceCollector("test-sess", "/tmp/flyto-test-evidence")
        ec.record("context", "indexer_query", {"length": 500})
        ec.record("coding", "tool_approved", {"tool": "Edit"})
        records = ec.to_list()
        assert len(records) == 2
        assert records[0].phase == "context"
        assert records[1].action == "tool_approved"

    def test_track_files(self):
        ec = EvidenceCollector("test-sess", "/tmp/flyto-test-evidence")
        ec.track_file_change("/tmp/b.py")
        ec.track_file_change("/tmp/a.py")
        ec.track_file_change("/tmp/b.py")  # duplicate
        assert ec.files_changed == ["/tmp/a.py", "/tmp/b.py"]  # sorted, deduped

    def test_save_creates_jsonl(self, event_loop):
        tmp = tempfile.mkdtemp()
        try:
            ec = EvidenceCollector("sess123", tmp)
            ec.record("coding", "file_changed", {"path": "/tmp/x.py"})
            path = event_loop.run_until_complete(ec.save())
            assert path is not None
            assert path.exists()
            with open(path) as f:
                lines = f.read().strip().split("\n")
            assert len(lines) == 1
            rec = json.loads(lines[0])
            assert rec["phase"] == "coding"
            assert rec["action"] == "file_changed"
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_save_empty_returns_none(self, event_loop):
        ec = EvidenceCollector("empty-sess", "/tmp/flyto-test-evidence")
        result = event_loop.run_until_complete(ec.save())
        assert result is None


class TestEvidencePostHook:
    def test_tracks_edit(self, event_loop):
        ec = EvidenceCollector("hook-test", "/tmp/flyto-test-evidence")
        event_loop.run_until_complete(
            evidence_post_hook(ec, "Edit", {"file_path": "/app/main.py"}, None)
        )
        assert "/app/main.py" in ec.files_changed
        assert ec.to_list()[-1].action == "file_changed"

    def test_tracks_write(self, event_loop):
        ec = EvidenceCollector("hook-test", "/tmp/flyto-test-evidence")
        event_loop.run_until_complete(
            evidence_post_hook(ec, "Write", {"file_path": "/app/new.py"}, None)
        )
        assert "/app/new.py" in ec.files_changed

    def test_tracks_bash(self, event_loop):
        ec = EvidenceCollector("hook-test", "/tmp/flyto-test-evidence")
        event_loop.run_until_complete(
            evidence_post_hook(ec, "Bash", {"command": "npm test"}, None)
        )
        records = ec.to_list()
        assert records[-1].action == "bash_executed"

    def test_tracks_other_tool(self, event_loop):
        ec = EvidenceCollector("hook-test", "/tmp/flyto-test-evidence")
        event_loop.run_until_complete(
            evidence_post_hook(ec, "Grep", {"pattern": "TODO"}, None)
        )
        records = ec.to_list()
        assert records[-1].action == "tool_used"


# ──────────────────────────────────────────────────────────────────────
# Prompts
# ──────────────────────────────────────────────────────────────────────

class TestBuildSystemPrompt:
    def test_basic(self):
        prompt = build_system_prompt()
        assert ROLE_PREAMBLE.strip() in prompt
        assert GUARDIAN_NOTICE.strip() in prompt
        assert "Verification" not in prompt  # no verification by default

    def test_with_verification(self):
        prompt = build_system_prompt(has_verification=True)
        assert "Verification" in prompt

    def test_with_indexer_context(self):
        prompt = build_system_prompt(indexer_context="**Projects**: flyto-ai")
        assert "flyto-ai" in prompt
        assert "Codebase Context" in prompt

    def test_all_options(self):
        prompt = build_system_prompt(
            indexer_context="health: A (95/100)",
            has_verification=True,
        )
        assert "Guardian" in prompt
        assert "Verification" in prompt
        assert "health: A (95/100)" in prompt


# ──────────────────────────────────────────────────────────────────────
# Config: ClaudeCodeConfig
# ──────────────────────────────────────────────────────────────────────

class TestClaudeCodeConfig:
    def test_defaults(self):
        cc = ClaudeCodeConfig()
        assert cc.max_budget_usd == 5.0
        assert cc.max_turns == 30
        assert cc.max_fix_attempts == 3
        assert "Read" in cc.allowed_tools
        assert "Edit" in cc.allowed_tools
        assert "Bash" in cc.allowed_tools
        assert cc.verification_timeout == 120
        assert cc.evidence_dir == "~/.flyto/evidence"

    def test_agent_config_has_claude_code(self):
        cfg = AgentConfig()
        assert isinstance(cfg.claude_code, ClaudeCodeConfig)

    def test_from_env(self):
        env = {
            "FLYTO_AI_CC_MAX_BUDGET": "10.0",
            "FLYTO_AI_CC_MAX_TURNS": "50",
            "FLYTO_AI_CC_MAX_FIX_ATTEMPTS": "5",
        }
        with patch.dict(os.environ, env, clear=False):
            cfg = AgentConfig.from_env()
        assert cfg.claude_code.max_budget_usd == 10.0
        assert cfg.claude_code.max_turns == 50
        assert cfg.claude_code.max_fix_attempts == 5

    def test_from_dict(self):
        cfg = AgentConfig.from_dict({
            "claude_code": {
                "max_budget_usd": 20.0,
                "max_turns": 100,
            }
        })
        assert cfg.claude_code.max_budget_usd == 20.0
        assert cfg.claude_code.max_turns == 100
        assert cfg.claude_code.max_fix_attempts == 3  # default

    def test_from_dict_ignores_unknown_keys(self):
        cfg = AgentConfig.from_dict({
            "claude_code": {
                "max_budget_usd": 1.0,
                "unknown_key": "ignored",
            }
        })
        assert cfg.claude_code.max_budget_usd == 1.0


# ──────────────────────────────────────────────────────────────────────
# ClaudeCodeAgent (unit tests with mocked SDK)
# ──────────────────────────────────────────────────────────────────────

class TestClaudeCodeAgentInit:
    def test_creates_with_default_config(self):
        from flyto_ai.agents.claude_code import ClaudeCodeAgent
        agent = ClaudeCodeAgent()
        assert agent._cc.max_budget_usd == 5.0

    def test_creates_with_custom_config(self):
        from flyto_ai.agents.claude_code import ClaudeCodeAgent
        cfg = AgentConfig()
        cfg.claude_code.max_budget_usd = 99.0
        agent = ClaudeCodeAgent(config=cfg)
        assert agent._cc.max_budget_usd == 99.0


class TestClaudeCodeAgentRun:
    """Test the full run() loop with mocked _run_claude_code."""

    def test_no_verification_returns_ok(self, event_loop):
        from flyto_ai.agents.claude_code import ClaudeCodeAgent
        agent = ClaudeCodeAgent()

        mock_sdk = AsyncMock(return_value={
            "session_id": "sdk-123",
            "message": "Done.",
            "cost": 0.05,
            "num_turns": 3,
            "duration_ms": 5000,
            "usage": {"input_tokens": 100, "output_tokens": 50},
        })
        agent._run_claude_code = mock_sdk

        req = CodeTaskRequest(message="fix bug", working_dir="/tmp")
        result = event_loop.run_until_complete(agent.run(req))

        assert result.ok is True
        assert result.attempts == 1
        assert result.message == "Done."
        assert result.total_cost_usd == 0.05
        mock_sdk.assert_called_once()

    def test_verification_pass_first_attempt(self, event_loop):
        from flyto_ai.agents.claude_code import ClaudeCodeAgent
        agent = ClaudeCodeAgent()

        agent._run_claude_code = AsyncMock(return_value={
            "session_id": "sdk-456",
            "message": "Fixed.",
            "cost": 0.1,
            "num_turns": 5,
            "duration_ms": 8000,
            "usage": None,
        })
        agent._verifier.verify = AsyncMock(return_value=VerificationResult(
            passed=True, recipe_name="screenshot", duration_ms=2000,
        ))

        req = CodeTaskRequest(
            message="fix login",
            working_dir="/tmp",
            verification_recipe="screenshot",
            verification_args={"url": "http://localhost:3000"},
        )
        result = event_loop.run_until_complete(agent.run(req))

        assert result.ok is True
        assert result.attempts == 1
        assert "passed" in result.message.lower()

    def test_verification_fail_then_pass(self, event_loop):
        from flyto_ai.agents.claude_code import ClaudeCodeAgent
        agent = ClaudeCodeAgent()

        agent._run_claude_code = AsyncMock(return_value={
            "session_id": "sdk-789",
            "message": "Tried.",
            "cost": 0.05,
            "num_turns": 3,
            "duration_ms": 4000,
            "usage": None,
        })

        verify_results = [
            VerificationResult(passed=False, recipe_name="ss", error="layout mismatch"),
            VerificationResult(passed=True, recipe_name="ss", duration_ms=2000),
        ]
        agent._verifier.verify = AsyncMock(side_effect=verify_results)

        req = CodeTaskRequest(
            message="fix",
            working_dir="/tmp",
            verification_recipe="ss",
            max_fix_attempts=3,
        )
        result = event_loop.run_until_complete(agent.run(req))

        assert result.ok is True
        assert result.attempts == 2
        assert agent._run_claude_code.call_count == 2

    def test_all_attempts_exhausted(self, event_loop):
        from flyto_ai.agents.claude_code import ClaudeCodeAgent
        agent = ClaudeCodeAgent()

        agent._run_claude_code = AsyncMock(return_value={
            "session_id": "sdk-fail",
            "message": "Failed.",
            "cost": 0.02,
            "num_turns": 2,
            "duration_ms": 3000,
            "usage": None,
        })
        agent._verifier.verify = AsyncMock(return_value=VerificationResult(
            passed=False, recipe_name="ss", error="still broken",
        ))

        req = CodeTaskRequest(
            message="fix",
            working_dir="/tmp",
            verification_recipe="ss",
            max_fix_attempts=2,
        )
        result = event_loop.run_until_complete(agent.run(req))

        assert result.ok is False
        assert result.attempts == 2
        assert len(result.verification_results) == 2

    def test_budget_exhausted_breaks_loop(self, event_loop):
        from flyto_ai.agents.claude_code import ClaudeCodeAgent
        cfg = AgentConfig()
        cfg.claude_code.max_budget_usd = 0.10
        agent = ClaudeCodeAgent(config=cfg)

        agent._run_claude_code = AsyncMock(return_value={
            "session_id": "sdk-budget",
            "message": "...",
            "cost": 0.10,  # exactly at budget
            "num_turns": 5,
            "duration_ms": 5000,
            "usage": None,
        })
        agent._verifier.verify = AsyncMock(return_value=VerificationResult(
            passed=False, recipe_name="ss", error="not done",
        ))

        req = CodeTaskRequest(
            message="fix",
            working_dir="/tmp",
            verification_recipe="ss",
            max_fix_attempts=5,
            max_budget_usd=0.10,
        )
        result = event_loop.run_until_complete(agent.run(req))

        assert result.ok is False
        # Should have stopped after 1 attempt due to budget
        assert agent._run_claude_code.call_count == 1

    def test_stream_events_emitted(self, event_loop):
        from flyto_ai.agents.claude_code import ClaudeCodeAgent
        agent = ClaudeCodeAgent()

        agent._run_claude_code = AsyncMock(return_value={
            "session_id": "sdk-stream",
            "message": "OK.",
            "cost": 0.01,
            "num_turns": 1,
            "duration_ms": 1000,
            "usage": None,
        })

        events = []
        def on_stream(event):
            events.append(event)

        req = CodeTaskRequest(message="test", working_dir="/tmp")
        event_loop.run_until_complete(agent.run(req, on_stream=on_stream))

        types = [e["type"] for e in events]
        assert "phase_start" in types
        assert "phase_end" in types


class TestClaudeCodeAgentBuildFeedback:
    def test_feedback_with_error(self):
        from flyto_ai.agents.claude_code import ClaudeCodeAgent
        agent = ClaudeCodeAgent()
        vr = VerificationResult(
            passed=False, recipe_name="ss",
            error="Layout mismatch",
            comparison_summary="Button misaligned",
        )
        fb = agent._build_feedback(vr)
        assert "FAILED" in fb
        assert "Layout mismatch" in fb
        assert "Button misaligned" in fb

    def test_feedback_with_extracted_text(self):
        from flyto_ai.agents.claude_code import ClaudeCodeAgent
        agent = ClaudeCodeAgent()
        vr = VerificationResult(
            passed=False, recipe_name="ss",
            extracted_data={"text": "Login Error: invalid password"},
        )
        fb = agent._build_feedback(vr)
        assert "Login Error" in fb

    def test_feedback_with_screenshot(self):
        from flyto_ai.agents.claude_code import ClaudeCodeAgent
        agent = ClaudeCodeAgent()
        vr = VerificationResult(
            passed=False, recipe_name="ss",
            screenshot_path="/tmp/screenshot.png",
        )
        fb = agent._build_feedback(vr)
        assert "/tmp/screenshot.png" in fb


# ──────────────────────────────────────────────────────────────────────
# Verifier (unit tests)
# ──────────────────────────────────────────────────────────────────────

class TestVerificationEngine:
    def test_recipe_failure_returns_error(self, event_loop):
        engine = VerificationEngine()
        # Mock _run_recipe to raise
        engine._run_recipe = AsyncMock(side_effect=RuntimeError("no flyto-core"))
        result = event_loop.run_until_complete(
            engine.verify(recipe="screenshot", args={"url": "http://localhost"})
        )
        assert result.passed is False
        assert "no flyto-core" in result.error
        assert result.duration_ms >= 0

    def test_no_reference_passes(self, event_loop):
        engine = VerificationEngine()
        engine._run_recipe = AsyncMock(return_value={
            "screenshot_path": "/tmp/shot.png",
        })
        result = event_loop.run_until_complete(
            engine.verify(recipe="screenshot", args={})
        )
        assert result.passed is True
        assert result.screenshot_path == "/tmp/shot.png"

    def test_extract_recipe_output_steps(self):
        engine = VerificationEngine()
        raw = {
            "steps": [
                {"data": {"path": "/tmp/shot.png"}},
                {"data": {"text": "Login Form"}},
            ]
        }
        out = engine._extract_recipe_output(raw)
        assert out["screenshot_path"] == "/tmp/shot.png"
        assert out["extracted_data"]["text"] == "Login Form"

    def test_extract_recipe_output_direct(self):
        engine = VerificationEngine()
        raw = {"screenshot_path": "/tmp/direct.png"}
        out = engine._extract_recipe_output(raw)
        assert out["screenshot_path"] == "/tmp/direct.png"


# ──────────────────────────────────────────────────────────────────────
# Package-level
# ──────────────────────────────────────────────────────────────────────

class TestPackageExports:
    def test_agents_init_exports(self):
        from flyto_ai.agents import (
            ClaudeCodeAgent,
            CodeTaskRequest,
            CodeTaskResponse,
            EvidenceRecord,
            VerificationResult,
        )
        assert ClaudeCodeAgent is not None
        assert CodeTaskRequest is not None

    def test_top_level_lazy_import(self):
        import flyto_ai
        cls = flyto_ai.ClaudeCodeAgent
        assert cls.__name__ == "ClaudeCodeAgent"

    def test_stream_event_types(self):
        from flyto_ai.models import StreamEventType
        assert StreamEventType.PHASE_START.value == "phase_start"
        assert StreamEventType.PHASE_END.value == "phase_end"
        assert StreamEventType.VERIFICATION_RESULT.value == "verification_result"
