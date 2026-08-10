# Copyright 2024 Flyto2
# Licensed under the Apache License, Version 2.0
"""Tests for the Claude Code Agent subsystem (agents/)."""
import asyncio
import json
import os
import shutil
import tempfile
from dataclasses import asdict
from unittest.mock import AsyncMock, patch

import pytest

from flyto_ai.agents.models import (
    CodeTaskRequest,
    CodeTaskResponse,
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
from flyto_ai.agents.prompts import build_system_prompt, ROLE_PREAMBLE, GUARDIAN_NOTICE
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
        for ext in [
            ".py", ".ts", ".js", ".mjs", ".vue", ".html", ".css", ".json",
            ".yaml", ".md", ".service",
        ]:
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

    def test_is_extension_allowed_systemd_service(self):
        assert _is_extension_allowed("turtlebot3-bringup.service") is True

    def test_is_extension_allowed_binary(self):
        assert _is_extension_allowed("app.exe") is False
        assert _is_extension_allowed("image.png") is False

    def test_is_extension_allowed_dockerfile(self):
        assert _is_extension_allowed("Dockerfile") is True
        assert _is_extension_allowed("Makefile") is True

    @pytest.mark.parametrize("name", [".gitignore", ".dockerignore", ".editorconfig"])
    def test_is_extension_allowed_closed_repository_dotfiles(self, name):
        assert _is_extension_allowed(name) is True

    def test_is_extension_allowed_arbitrary_dotfile(self):
        assert _is_extension_allowed(".bashrc") is False


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

    def test_approve_edit_allowlisted_repository_dotfile(self, event_loop):
        result = event_loop.run_until_complete(
            guardian_pre_hook("Edit", {"file_path": "/tmp/.gitignore"}, "id-dotfile")
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
        assert cc.max_turns == 100
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


# ──────────────────────────────────────────────────────────────────────
# Exact Claude SDK session continuation and pinned model
# ──────────────────────────────────────────────────────────────────────

class TestClaudeSdkSessionContinuation:
    """The SDK session id is the only identity that can resume a conversation."""

    def _agent(self, tmp_path):
        from flyto_ai.agents.claude_code import ClaudeCodeAgent
        agent = ClaudeCodeAgent()
        agent._cc.evidence_dir = str(tmp_path / "evidence")
        return agent

    def _record(self, agent):
        seen = []

        async def fake(**kwargs):
            seen.append(kwargs["session_id"])
            return {
                "session_id": agent._next_session, "message": "done", "cost": 0.0,
                "num_turns": 1, "duration_ms": 1, "usage": {"input_tokens": 2},
            }

        agent._run_claude_code = fake
        return seen

    def test_first_round_starts_without_a_session_and_adopts_the_sdk_identity(
        self, event_loop, tmp_path,
    ):
        agent = self._agent(tmp_path)
        agent._next_session = "sdk-fresh"
        seen = self._record(agent)
        result = event_loop.run_until_complete(
            agent.run(CodeTaskRequest(message="build", working_dir=str(tmp_path)))
        )
        assert seen == [None]
        assert result.ok is True
        assert result.claude_session_id == "sdk-fresh"
        # The local evidence id is a different identity entirely.
        assert result.session_id != result.claude_session_id

    def test_resumed_request_uses_the_supplied_id_on_the_first_sdk_call(
        self, event_loop, tmp_path,
    ):
        agent = self._agent(tmp_path)
        agent._next_session = "sdk-resume"
        seen = self._record(agent)
        result = event_loop.run_until_complete(agent.run(CodeTaskRequest(
            message="rework", working_dir=str(tmp_path),
            sdk_session_id="sdk-resume", service_mode=True,
        )))
        assert seen == ["sdk-resume"]
        assert result.ok is True
        assert result.claude_session_id == "sdk-resume"

    @pytest.mark.parametrize("returned", [None, "", "   ", "../escape", "a" * 200])
    def test_missing_or_unsafe_session_identity_fails_closed(
        self, event_loop, tmp_path, returned,
    ):
        agent = self._agent(tmp_path)
        agent._next_session = returned
        self._record(agent)
        result = event_loop.run_until_complete(
            agent.run(CodeTaskRequest(message="build", working_dir=str(tmp_path)))
        )
        assert result.ok is False
        assert result.claude_session_id is None

    def test_changed_session_identity_fails_closed(self, event_loop, tmp_path):
        agent = self._agent(tmp_path)
        agent._next_session = "sdk-other"
        self._record(agent)
        result = event_loop.run_until_complete(agent.run(CodeTaskRequest(
            message="rework", working_dir=str(tmp_path),
            sdk_session_id="sdk-resume", service_mode=True,
        )))
        assert result.ok is False
        assert result.claude_session_id is None

    def test_service_mode_writes_no_legacy_evidence_file(self, event_loop, tmp_path):
        agent = self._agent(tmp_path)
        agent._next_session = "sdk-quiet"
        self._record(agent)
        evidence_dir = tmp_path / "evidence"
        event_loop.run_until_complete(agent.run(CodeTaskRequest(
            message="build", working_dir=str(tmp_path),
            sdk_session_id="sdk-quiet", service_mode=True,
        )))
        assert not evidence_dir.exists() or not list(evidence_dir.glob("*"))

        event_loop.run_until_complete(
            agent.run(CodeTaskRequest(message="build", working_dir=str(tmp_path)))
        )
        assert list(evidence_dir.glob("*"))

    def test_internal_session_state_is_validated_and_optional(self):
        assert CodeTaskRequest(message="m", working_dir="/tmp").sdk_session_id is None
        assert CodeTaskRequest(message="m", working_dir="/tmp").service_mode is False
        with pytest.raises(ValueError, match="bounded opaque identifier"):
            CodeTaskRequest(message="m", working_dir="/tmp", sdk_session_id="../escape")
        with pytest.raises(ValueError, match="service_mode must be a boolean"):
            CodeTaskRequest(message="m", working_dir="/tmp", service_mode="yes")


class TestClaudeSdkOptions:
    """Option construction is inspectable without importing or calling the SDK."""

    def _options(self, tmp_path, **request_kwargs):
        from flyto_ai.agents.claude_code import ClaudeCodeAgent
        agent = ClaudeCodeAgent()
        request = CodeTaskRequest(
            message="task", working_dir=str(tmp_path), **request_kwargs,
        )
        return agent, agent._option_kwargs(
            request, session_id=request_kwargs.get("sdk_session_id"),
            system_prompt="system", max_turns=7, max_budget=1.25,
        )

    def test_model_is_pinned_and_never_auto_selected(self, tmp_path):
        from flyto_ai.agents.claude_code import DEFAULT_CLAUDE_MODEL
        agent, options = self._options(tmp_path)
        assert DEFAULT_CLAUDE_MODEL == "claude-opus-5"
        assert options["model"] == "claude-opus-5"

        agent._cc.model = "claude-sonnet-4-6"
        assert agent.resolve_model(agent._cc) == "claude-sonnet-4-6"
        for invalid in ("", "not a model", None, 5, "x" * 100):
            agent._cc.model = invalid
            assert agent.resolve_model(agent._cc) == "claude-opus-5"

    def test_legacy_mode_keeps_its_tool_catalog_and_permission_mode(self, tmp_path):
        _, options = self._options(tmp_path)
        assert "Bash" in options["allowed_tools"]
        assert options["permission_mode"] == "default"
        assert options["strict_mcp_config"] is False
        assert options["system_prompt"] == "system"
        assert "resume" not in options

    def test_service_mode_ignores_a_configured_model(self, tmp_path):
        """Configuration may vary the legacy backend; it cannot redirect service work."""
        from flyto_ai.agents.claude_code import ClaudeCodeAgent
        agent = ClaudeCodeAgent()
        agent._cc.model = "claude-sonnet-4-6"
        legacy = CodeTaskRequest(message="task", working_dir=str(tmp_path))
        service = CodeTaskRequest(
            message="task", working_dir=str(tmp_path), service_mode=True,
        )
        common = {"system_prompt": "system", "max_turns": 7, "max_budget": 1.25}
        assert agent._option_kwargs(legacy, session_id=None, **common)["model"] == (
            "claude-sonnet-4-6"
        )
        assert agent._option_kwargs(service, session_id=None, **common)["model"] == (
            "claude-opus-5"
        )
        for configured in ("claude-haiku-4-5-20251001", "", "not a model", None):
            agent._cc.model = configured
            assert agent._option_kwargs(service, session_id=None, **common)["model"] == (
                "claude-opus-5"
            )

    def test_service_mode_has_no_process_execution_tool(self, tmp_path):
        from flyto_ai.agents.claude_code import SERVICE_ALLOWED_TOOLS
        _, options = self._options(
            tmp_path, service_mode=True, sdk_session_id="sdk-1",
        )
        assert set(options["allowed_tools"]) == set(SERVICE_ALLOWED_TOOLS)
        assert "Bash" not in options["allowed_tools"]
        assert options["permission_mode"] == "acceptEdits"
        assert options["resume"] == "sdk-1"
        assert "system_prompt" not in options
        assert options["model"] == "claude-opus-5"
        assert options["strict_mcp_config"] is True

    def test_service_mode_uses_only_host_declared_mcp_servers(self, tmp_path):
        from flyto_ai.agents.claude_code import ClaudeCodeAgent

        request = CodeTaskRequest(
            message="task", working_dir=str(tmp_path), service_mode=True,
        )
        options = ClaudeCodeAgent()._option_kwargs(
            request,
            session_id=None,
            system_prompt="system",
            max_turns=7,
            max_budget=1.25,
            mcp_servers={"flyto-indexer": {"command": "indexer", "args": []}},
        )

        assert options["strict_mcp_config"] is True
        assert set(options["mcp_servers"]) == {"flyto-indexer"}


class TestClaudeCodeConfigModel:
    """The legacy direct backend has a validated, bounded model setting."""

    def test_default_is_opus_5(self):
        assert ClaudeCodeConfig().model == "claude-opus-5"
        assert AgentConfig().claude_code.model == "claude-opus-5"

    def test_from_dict_preserves_and_validates_the_model(self):
        cfg = AgentConfig.from_dict({"claude_code": {"model": "claude-sonnet-4-6"}})
        assert cfg.claude_code.model == "claude-sonnet-4-6"
        assert AgentConfig.from_dict({"claude_code": {}}).claude_code.model == "claude-opus-5"
        for invalid in ("", "not a model", "x" * 100, 5, None):
            with pytest.raises(ValueError, match="bounded model identifier"):
                AgentConfig.from_dict({"claude_code": {"model": invalid}})

    def test_from_env_reads_a_bounded_model(self, monkeypatch):
        monkeypatch.delenv("FLYTO_AI_CC_MODEL", raising=False)
        assert AgentConfig.from_env().claude_code.model == "claude-opus-5"
        monkeypatch.setenv("FLYTO_AI_CC_MODEL", "claude-haiku-4-5-20251001")
        assert AgentConfig.from_env().claude_code.model == "claude-haiku-4-5-20251001"
        monkeypatch.setenv("FLYTO_AI_CC_MODEL", "not a model")
        with pytest.raises(ValueError, match="bounded model identifier"):
            AgentConfig.from_env()

    @pytest.mark.parametrize(("field", "accepted"), [
        ("max_budget_usd", (0.01, 1000.0, 5.0, 99.0)),
        ("max_turns", (1, 100, 30)),
        ("max_fix_attempts", (1, 5, 3)),
        ("verification_timeout", (1, 3600, 120)),
    ])
    def test_numeric_bounds_accept_their_documented_range(self, field, accepted):
        for value in accepted:
            assert getattr(ClaudeCodeConfig(**{field: value}), field) == value

    @pytest.mark.parametrize(("field", "rejected"), [
        ("max_budget_usd", (True, 0, -1.0, 1000.1, float("nan"), float("inf"),
                            float("-inf"), "5.0", None)),
        ("max_turns", (True, 0, -1, 101, 3.5, "30", None)),
        ("max_fix_attempts", (True, 0, -1, 6, 3.0, "3", None)),
        ("verification_timeout", (True, 0, -1, 3601, 120.0, "120", None)),
    ])
    def test_numeric_bounds_fail_closed_without_clamping(self, field, rejected):
        for value in rejected:
            with pytest.raises(ValueError, match="claude_code " + field):
                ClaudeCodeConfig(**{field: value})

    def test_out_of_range_values_are_rejected_from_dict_and_env(self, monkeypatch):
        with pytest.raises(ValueError, match="max_fix_attempts"):
            AgentConfig.from_dict({"claude_code": {"max_fix_attempts": -1}})
        monkeypatch.setenv("FLYTO_AI_CC_MAX_FIX_ATTEMPTS", "-1")
        with pytest.raises(ValueError, match="max_fix_attempts"):
            AgentConfig.from_env()
        monkeypatch.delenv("FLYTO_AI_CC_MAX_FIX_ATTEMPTS")
        for value in ("nan", "inf", "-inf", "0", "-5", "1000.1"):
            monkeypatch.setenv("FLYTO_AI_CC_MAX_BUDGET", value)
            with pytest.raises(ValueError, match="max_budget_usd"):
                ClaudeCodeConfig.from_env()

    @pytest.mark.parametrize(("variable", "raw"), [
        ("FLYTO_AI_CC_MAX_BUDGET", "five"),
        ("FLYTO_AI_CC_MAX_BUDGET", ""),
        ("FLYTO_AI_CC_MAX_TURNS", "30.5"),
        ("FLYTO_AI_CC_MAX_TURNS", "lots"),
        ("FLYTO_AI_CC_MAX_FIX_ATTEMPTS", "0x3"),
    ])
    def test_malformed_env_conversion_names_the_setting(self, monkeypatch, variable, raw):
        monkeypatch.setenv(variable, raw)
        with pytest.raises(ValueError, match=variable):
            ClaudeCodeConfig.from_env()


class TestServiceEditAuthority:
    """Startup sandbox/approval authority reaches the SDK tool catalog."""

    def _options(self, tmp_path, *, edit_authority):
        from flyto_ai.agents.claude_code import ClaudeCodeAgent
        request = CodeTaskRequest(
            message="task", working_dir=str(tmp_path),
            service_mode=True, service_edit_authority=edit_authority,
        )
        return ClaudeCodeAgent()._option_kwargs(
            request, session_id=None, system_prompt="system",
            max_turns=7, max_budget=1.25,
        )

    def test_writable_authority_keeps_edit_tools_and_accept_edits(self, tmp_path):
        from flyto_ai.agents.claude_code import SERVICE_ALLOWED_TOOLS
        options = self._options(tmp_path, edit_authority=True)
        assert set(options["allowed_tools"]) == set(SERVICE_ALLOWED_TOOLS)
        assert options["permission_mode"] == "acceptEdits"

    def test_read_only_authority_removes_every_write_tool(self, tmp_path):
        from flyto_ai.agents.claude_code import SERVICE_READONLY_TOOLS
        options = self._options(tmp_path, edit_authority=False)
        assert set(options["allowed_tools"]) == set(SERVICE_READONLY_TOOLS)
        assert set(options["allowed_tools"]) == {"Read", "Glob", "Grep"}
        assert options["permission_mode"] == "default"
        assert options["model"] == "claude-opus-5"

    def test_service_catalog_exposes_file_scoped_search_but_not_bash(self, tmp_path):
        from flyto_ai.agents.claude_code import ClaudeCodeAgent
        for authority in (True, False):
            catalog = set(self._options(tmp_path, edit_authority=authority)["allowed_tools"])
            assert "Grep" in catalog
            assert "Bash" not in catalog
        assert set(self._options(tmp_path, edit_authority=True)["allowed_tools"]) == {
            "Read", "Edit", "Write", "Glob", "Grep",
        }
        # The legacy direct catalog is unchanged and still configurable.
        agent = ClaudeCodeAgent()
        legacy = agent._option_kwargs(
            CodeTaskRequest(message="task", working_dir=str(tmp_path)),
            session_id=None, system_prompt="s", max_turns=1, max_budget=1.0,
        )
        assert set(legacy["allowed_tools"]) >= {"Read", "Edit", "Write", "Bash", "Glob", "Grep"}

    def test_service_mode_limits_content_search_to_one_explicit_safe_file(
        self, event_loop, tmp_path,
    ):
        (tmp_path / ".env").write_text("API_TOKEN=s3cr3t\n")
        (tmp_path / "app.py").write_text("value = 1\n")
        assert event_loop.run_until_complete(guardian_pre_hook(
            "Grep", {"pattern": "value", "path": "app.py"}, "id",
            workspace=str(tmp_path), service_mode=True,
        )) == {}

        for arguments in ({"pattern": "value"}, {"pattern": "value", "path": "."}):
            with pytest.raises(GuardianBlocked) as blocked:
                event_loop.run_until_complete(guardian_pre_hook(
                    "Grep", arguments, "id",
                    workspace=str(tmp_path), service_mode=True,
                ))
            message = str(blocked.value)
            assert "requires one" in message
            for fragment in ("value", str(tmp_path)):
                assert fragment not in message

        with pytest.raises(GuardianBlocked, match="sensitive"):
            event_loop.run_until_complete(guardian_pre_hook(
                "Grep", {"pattern": "API_TOKEN", "path": ".env"}, "id",
                workspace=str(tmp_path), service_mode=True,
            ))

        # Names-only search stays available inside the workspace.
        assert event_loop.run_until_complete(guardian_pre_hook(
            "Glob", {"pattern": "**/*.py"}, "id",
            workspace=str(tmp_path), service_mode=True,
        )) == {}

    def test_legacy_content_search_remains_available(self, event_loop, tmp_path):
        assert event_loop.run_until_complete(guardian_pre_hook(
            "Grep", {"pattern": "value", "path": "."}, "id", workspace=str(tmp_path),
        )) == {}
        assert event_loop.run_until_complete(
            guardian_pre_hook("Grep", {"pattern": "value"}, "id")
        ) == {}

    @pytest.mark.parametrize("tool", ["Glob", "Grep"])
    @pytest.mark.parametrize("root", [None, 5, True, ["."], {"path": "."}])
    def test_non_string_search_paths_fail_closed(self, event_loop, tmp_path, tool, root):
        with pytest.raises(GuardianBlocked, match="search path must be a string"):
            event_loop.run_until_complete(guardian_pre_hook(
                tool, {"pattern": "*", "path": root}, "id", workspace=str(tmp_path),
            ))

    def test_flag_is_validated_and_defaults_to_the_legacy_behavior(self):
        assert CodeTaskRequest(message="m", working_dir="/tmp").service_edit_authority is True
        with pytest.raises(ValueError, match="service_edit_authority must be a boolean"):
            CodeTaskRequest(message="m", working_dir="/tmp", service_edit_authority="yes")

    @pytest.mark.parametrize("tool", ["Edit", "Write", "NotebookEdit", "MultiEdit"])
    def test_guardian_denies_mutation_without_edit_authority(
        self, event_loop, tmp_path, tool,
    ):
        with pytest.raises(GuardianBlocked, match="no workspace write authority"):
            event_loop.run_until_complete(guardian_pre_hook(
                tool, {"file_path": str(tmp_path / "app.py")}, "id",
                workspace=str(tmp_path), service_mode=True, edit_authority=False,
            ))
        assert event_loop.run_until_complete(guardian_pre_hook(
            "Read", {"file_path": str(tmp_path / "app.py")}, "id",
            workspace=str(tmp_path), service_mode=True, edit_authority=False,
        )) == {}


class TestGuardianSearchRootConfinement:
    def test_search_roots_share_the_workspace_boundary(self, event_loop, tmp_path):
        workspace = tmp_path / "ws"
        (workspace / "pkg").mkdir(parents=True)
        outside = tmp_path / "outside"
        outside.mkdir()
        (workspace / "linked").symlink_to(outside)
        (workspace / "inner").symlink_to(workspace / "pkg")

        for tool in ("Glob", "Grep"):
            for allowed in ({}, {"path": ""}, {"path": "pkg"}, {"path": str(workspace)},
                            {"path": "inner"}):
                arguments = {"pattern": "**/*.py"}
                arguments.update(allowed)
                assert event_loop.run_until_complete(guardian_pre_hook(
                    tool, arguments, "id", workspace=str(workspace),
                )) == {}
            for denied in ("..", "../outside", str(outside), "/etc", "linked"):
                with pytest.raises(GuardianBlocked) as blocked:
                    event_loop.run_until_complete(guardian_pre_hook(
                        tool, {"pattern": "*", "path": denied}, "id",
                        workspace=str(workspace),
                    ))
                message = str(blocked.value)
                assert "outside the run workspace" in message
                for fragment in (denied, str(outside), str(workspace)):
                    if fragment:
                        assert fragment not in message

    def test_search_roots_ignore_write_extension_rules(self, event_loop, tmp_path):
        (tmp_path / "data.bin").mkdir()
        assert event_loop.run_until_complete(guardian_pre_hook(
            "Glob", {"pattern": "*", "path": "data.bin"}, "id", workspace=str(tmp_path),
        )) == {}


class TestGuardianWorkspaceConfinement:
    def test_paths_must_resolve_inside_the_run_workspace(self, event_loop, tmp_path):
        workspace = tmp_path / "ws"
        workspace.mkdir()
        (workspace / "app.py").write_text("value = 1\n")
        outside = tmp_path / "outside.py"
        outside.write_text("secret = 1\n")
        (workspace / "escape.py").symlink_to(outside)

        for allowed in ("app.py", str(workspace / "app.py"), "./app.py"):
            assert event_loop.run_until_complete(guardian_pre_hook(
                "Read", {"file_path": allowed}, "id", workspace=str(workspace),
            )) == {}
        for denied in ("../outside.py", str(outside), "escape.py", "/etc/hosts"):
            with pytest.raises(GuardianBlocked, match="outside the run workspace"):
                event_loop.run_until_complete(guardian_pre_hook(
                    "Read", {"file_path": denied}, "id", workspace=str(workspace),
                ))

    def test_resolved_symlink_targets_are_rejected(self, event_loop, tmp_path):
        """A link's name says nothing about what it resolves to."""
        workspace = tmp_path / "ws"
        workspace.mkdir()
        (workspace / ".env").write_text("API_TOKEN=s3cr3t\n")
        (workspace / "real.py").write_text("value = 1\n")
        (workspace / "safe.py").symlink_to(workspace / ".env")
        (workspace / "alias.py").symlink_to(workspace / "real.py")

        for tool in ("Read", "Edit", "Write"):
            for linked in ("safe.py", "alias.py"):
                with pytest.raises(GuardianBlocked) as blocked:
                    event_loop.run_until_complete(guardian_pre_hook(
                        tool, {"file_path": linked}, "id", workspace=str(workspace),
                    ))
                message = str(blocked.value)
                assert "symlink" in message
                for fragment in ("s3cr3t", ".env", linked, str(workspace)):
                    assert fragment not in message

        # A regular in-workspace file keeps working.
        assert event_loop.run_until_complete(guardian_pre_hook(
            "Edit", {"file_path": "real.py"}, "id", workspace=str(workspace),
        )) == {}

    def test_sensitive_policy_is_reapplied_after_resolution(self, event_loop, tmp_path):
        workspace = tmp_path / "ws"
        workspace.mkdir()
        nested = workspace / "config"
        nested.mkdir()
        (nested / "credentials.json").write_text("{}\n")
        with pytest.raises(GuardianBlocked, match="sensitive"):
            event_loop.run_until_complete(guardian_pre_hook(
                "Read", {"file_path": "config/credentials.json"}, "id",
                workspace=str(workspace),
            ))

    def test_without_a_workspace_the_legacy_contract_is_unchanged(self, event_loop):
        assert event_loop.run_until_complete(
            guardian_pre_hook("Read", {"file_path": "/tmp/test.py"}, "id")
        ) == {}

    def test_errors_do_not_echo_the_path(self, event_loop):
        with pytest.raises(GuardianBlocked) as blocked:
            event_loop.run_until_complete(
                guardian_pre_hook("Write", {"file_path": "/app/deploy/.env"}, "id")
            )
        assert "/app/deploy/.env" not in str(blocked.value)


class TestGuardianServiceLandingBoundary:
    LANDING = [
        "git add -A",
        "git commit -m 'ship it'",
        "git push",
        "git   push   origin   main",
        "git push --force origin main",
        "git -C /repo push",
        "git --git-dir=/repo/.git push",
        "FLYTO=1 sudo git push",
        "/usr/bin/git push",
        "git tag v1.0.0",
        "git merge feature",
        "git rebase -i main",
        "git reset --hard HEAD~1",
        "git clean -fdx",
        "git restore --staged .",
        "git rm -r src",
        "git cherry-pick abc123",
        "git revert HEAD",
        "echo done && git push",
        "true; git commit -m x",
        "npm publish",
        "yarn publish",
        "pnpm publish",
        "twine upload dist/*",
        "poetry publish",
        "cargo publish",
        "gem push pkg.gem",
        "docker push registry/image:tag",
        "kubectl apply -f deploy.yaml",
        "kubectl rollout restart deploy/api",
        "helm upgrade api ./chart",
        "terraform apply -auto-approve",
        "pulumi up",
        "serverless deploy",
        "vercel deploy --prod",
        "netlify deploy",
        "flyctl deploy",
        "gh pr merge 12",
        "gh release create v1.0.0",
    ]

    @pytest.mark.parametrize("command", LANDING)
    def test_service_mode_denies_landing_publish_and_deploy(self, event_loop, command):
        with pytest.raises(GuardianBlocked) as blocked:
            event_loop.run_until_complete(guardian_pre_hook(
                "Bash", {"command": command}, "id", service_mode=True,
            ))
        # The denial names the matched rule, never the caller's command body.
        message = str(blocked.value)
        assert message.startswith("Bash blocked:")
        assert len(message) <= 80

    def test_denial_never_echoes_command_arguments_or_credentials(self, event_loop):
        with pytest.raises(GuardianBlocked) as blocked:
            event_loop.run_until_complete(guardian_pre_hook(
                "Bash",
                {"command": "git push https://ci-user:s3cr3t@example.invalid/repo.git main"},
                "id", service_mode=True,
            ))
        message = str(blocked.value)
        for fragment in ("s3cr3t", "ci-user", "example.invalid", "repo.git", "main"):
            assert fragment not in message

    @pytest.mark.parametrize("command", [
        "git status --short",
        "git log --oneline -5",
        "git diff --stat",
        "git show HEAD",
        "git rev-parse HEAD",
        "ls -la",
        "python -m pytest -q",
    ])
    def test_read_only_inspection_stays_available(self, event_loop, command):
        assert event_loop.run_until_complete(guardian_pre_hook(
            "Bash", {"command": command}, "id", service_mode=True,
        )) == {}
        assert event_loop.run_until_complete(
            guardian_pre_hook("Bash", {"command": command}, "id")
        ) == {}

    def test_legacy_mode_still_permits_ordinary_git_work(self, event_loop):
        assert event_loop.run_until_complete(
            guardian_pre_hook("Bash", {"command": "git commit -m x"}, "id")
        ) == {}

    def test_non_string_command_is_rejected(self, event_loop):
        with pytest.raises(GuardianBlocked):
            event_loop.run_until_complete(
                guardian_pre_hook("Bash", {"command": {"cmd": "git push"}}, "id")
            )
