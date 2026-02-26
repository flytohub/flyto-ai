# Copyright 2024 Flyto
# Licensed under the Apache License, Version 2.0
"""Claude Code Agent — orchestrates Claude Code via Agent SDK with
indexer context gathering, guardian hooks, and YAML recipe verification.

Flow:
  Phase 1: Gather codebase context from flyto-indexer
  Phase 2: Spawn Claude Code via Agent SDK to write code
  Phase 3: Run verification recipe (browser screenshot + extraction)
  Phase 4: LLM comparison (actual vs reference)
  Loop back to Phase 2 if verification fails
"""
import logging
import sys
import uuid
from typing import Any, Callable, Dict, List, Optional

from flyto_ai.agents.evidence import EvidenceCollector, evidence_post_hook
from flyto_ai.agents.guardian_hook import GuardianBlocked, guardian_pre_hook
from flyto_ai.agents.models import (
    CodeTaskRequest,
    CodeTaskResponse,
    VerificationResult,
)
from flyto_ai.agents.prompts import build_system_prompt
from flyto_ai.agents.verifier import VerificationEngine

logger = logging.getLogger(__name__)

# Type alias for streaming callback
StreamCallback = Optional[Callable[[Dict[str, Any]], None]]


class ClaudeCodeAgent:
    """High-level orchestrator: indexer → Claude Code → verify → feedback loop."""

    def __init__(self, config: Any = None):
        """
        Args:
            config: AgentConfig instance (optional). Uses config.claude_code
                    for ClaudeCodeConfig settings.
        """
        from flyto_ai.config import AgentConfig
        if config is None:
            config = AgentConfig.from_env()
        self._config = config
        self._cc = config.claude_code
        self._verifier = VerificationEngine(timeout=self._cc.verification_timeout)

    async def run(
        self,
        request: CodeTaskRequest,
        on_stream: StreamCallback = None,
    ) -> CodeTaskResponse:
        """Execute the full code → verify → fix loop.

        Args:
            request: What to build and how to verify it.
            on_stream: Optional callback for streaming phase/token events.

        Returns:
            CodeTaskResponse with pass/fail, evidence, and files changed.
        """
        session_id = uuid.uuid4().hex[:12]
        evidence = EvidenceCollector(session_id, self._cc.evidence_dir)
        verification_results: List[VerificationResult] = []

        max_attempts = min(request.max_fix_attempts, self._cc.max_fix_attempts)
        max_budget = min(request.max_budget_usd, self._cc.max_budget_usd)
        max_turns = min(request.max_turns, self._cc.max_turns)

        # ── Phase 1: Gather indexer context ──
        self._emit(on_stream, "phase_start", {"phase": "context"})
        indexer_context = ""
        try:
            from flyto_ai.agents.indexer_context import gather_context
            indexer_context = await gather_context(request.message, request.working_dir)
            if indexer_context:
                evidence.record("context", "indexer_query", {"length": len(indexer_context)})
        except Exception as e:
            logger.debug("Indexer context gathering skipped: %s", e)
        self._emit(on_stream, "phase_end", {"phase": "context"})

        # ── Phase 2-4 loop ──
        sdk_session_id: Optional[str] = None
        total_cost = 0.0
        total_turns = 0
        total_duration_ms = 0
        last_usage: Optional[Dict[str, Any]] = None

        for attempt in range(1, max_attempts + 1):
            # Phase 2: Claude Code writes code
            self._emit(on_stream, "phase_start", {"phase": "coding", "attempt": attempt})
            evidence.record("coding", "attempt_start", {"attempt": attempt})

            feedback_prefix = ""
            if attempt > 1 and verification_results:
                feedback_prefix = self._build_feedback(verification_results[-1])

            sdk_result = await self._run_claude_code(
                request=request,
                indexer_context=indexer_context,
                feedback=feedback_prefix,
                session_id=sdk_session_id,
                max_budget=max_budget - total_cost,
                max_turns=max_turns,
                evidence=evidence,
                on_stream=on_stream,
            )
            sdk_session_id = sdk_result.get("session_id")
            total_cost += sdk_result.get("cost", 0.0)
            total_turns += sdk_result.get("num_turns", 0)
            total_duration_ms += sdk_result.get("duration_ms", 0)
            last_usage = sdk_result.get("usage") or last_usage
            evidence.record("coding", "attempt_end", {
                "attempt": attempt,
                "cost": sdk_result.get("cost", 0),
                "num_turns": sdk_result.get("num_turns", 0),
                "duration_ms": sdk_result.get("duration_ms", 0),
            })
            self._emit(on_stream, "phase_end", {"phase": "coding", "attempt": attempt})

            # Phase 3: Verification
            if not request.verification_recipe:
                # No verification configured — consider it a pass
                await evidence.save()
                return CodeTaskResponse(
                    ok=True,
                    message=sdk_result.get("message", "Code changes applied."),
                    session_id=session_id,
                    attempts=attempt,
                    verification_results=verification_results,
                    evidence=evidence.to_list(),
                    files_changed=evidence.files_changed,
                    total_cost_usd=total_cost,
                    claude_session_id=sdk_session_id,
                    claude_num_turns=total_turns,
                    claude_duration_ms=total_duration_ms,
                    claude_usage=last_usage,
                )

            self._emit(on_stream, "phase_start", {"phase": "verification", "attempt": attempt})
            evidence.record("verification", "recipe_start", {"recipe": request.verification_recipe})

            vr = await self._verifier.verify(
                recipe=request.verification_recipe,
                args=request.verification_args,
                reference=request.reference_image,
            )
            verification_results.append(vr)
            evidence.record("verification", "recipe_result", {
                "passed": vr.passed,
                "duration_ms": vr.duration_ms,
                "error": vr.error,
            })

            self._emit(on_stream, "verification_result", {
                "passed": vr.passed,
                "attempt": attempt,
                "recipe": request.verification_recipe,
                "error": vr.error,
                "summary": vr.comparison_summary,
            })
            self._emit(on_stream, "phase_end", {"phase": "verification", "attempt": attempt})

            if vr.passed:
                await evidence.save()
                return CodeTaskResponse(
                    ok=True,
                    message="Verification passed on attempt {}.".format(attempt),
                    session_id=session_id,
                    attempts=attempt,
                    verification_results=verification_results,
                    evidence=evidence.to_list(),
                    files_changed=evidence.files_changed,
                    total_cost_usd=total_cost,
                    claude_session_id=sdk_session_id,
                    claude_num_turns=total_turns,
                    claude_duration_ms=total_duration_ms,
                    claude_usage=last_usage,
                )

            # Budget guard
            if total_cost >= max_budget:
                logger.warning("Budget exhausted (%.2f >= %.2f)", total_cost, max_budget)
                break

        # All attempts exhausted
        await evidence.save()
        return CodeTaskResponse(
            ok=False,
            message="Verification failed after {} attempts.".format(max_attempts),
            session_id=session_id,
            attempts=max_attempts,
            verification_results=verification_results,
            evidence=evidence.to_list(),
            files_changed=evidence.files_changed,
            total_cost_usd=total_cost,
            claude_session_id=sdk_session_id,
            claude_num_turns=total_turns,
            claude_duration_ms=total_duration_ms,
            claude_usage=last_usage,
        )

    # ── Private helpers ──

    async def _run_claude_code(
        self,
        request: CodeTaskRequest,
        indexer_context: str,
        feedback: str,
        session_id: Optional[str],
        max_budget: float,
        max_turns: int,
        evidence: EvidenceCollector,
        on_stream: StreamCallback,
    ) -> Dict[str, Any]:
        """Spawn or resume a Claude Code session via Agent SDK.

        Returns {"session_id": str, "message": str, "cost": float, ...}.
        """
        try:
            from claude_agent_sdk import (
                query,
                ClaudeAgentOptions,
                HookMatcher,
                AssistantMessage,
                ResultMessage,
                SystemMessage,
                TextBlock,
            )
        except ImportError:
            raise RuntimeError(
                "claude-agent-sdk is required for the 'code' command.\n"
                "Install with: pip install flyto-ai[agent]"
            )

        system_prompt = build_system_prompt(
            indexer_context=indexer_context,
            has_verification=bool(request.verification_recipe),
        )

        # Build prompt text
        prompt = request.message
        if feedback:
            prompt = feedback + "\n\n" + request.message

        # Build hooks
        async def _pre_hook(input_data, tool_use_id, context):
            tool_name = input_data.get("tool_name", "")
            tool_input = input_data.get("tool_input", {})
            try:
                await guardian_pre_hook(tool_name, tool_input, tool_use_id or "")
                evidence.record("coding", "tool_approved", {
                    "tool": tool_name,
                    "id": tool_use_id,
                })
                return {}
            except GuardianBlocked as e:
                evidence.record("coding", "tool_denied", {
                    "tool": tool_name,
                    "reason": str(e),
                    "id": tool_use_id,
                })
                return {
                    "hookSpecificOutput": {
                        "hookEventName": "PreToolUse",
                        "permissionDecision": "deny",
                        "permissionDecisionReason": str(e),
                    }
                }

        async def _post_hook(input_data, tool_use_id, context):
            tool_name = input_data.get("tool_name", "")
            tool_input = input_data.get("tool_input", {})
            tool_response = input_data.get("tool_response")
            await evidence_post_hook(evidence, tool_name, tool_input, tool_response)
            return {}

        # MCP servers — attach flyto-indexer if available
        mcp_servers: Dict[str, Any] = {}
        indexer_cmd = self._find_indexer_command()
        if indexer_cmd:
            mcp_servers["flyto-indexer"] = {
                "type": "stdio",
                "command": indexer_cmd[0],
                "args": indexer_cmd[1:],
            }

        # Build options
        options_kwargs: Dict[str, Any] = {
            "max_turns": max_turns,
            "max_budget_usd": max_budget,
            "cwd": request.working_dir,
            "allowed_tools": self._cc.allowed_tools,
            "permission_mode": "bypassPermissions",
            "allow_dangerously_skip_permissions": True,
            "env": {"CLAUDECODE": ""},
            "hooks": {
                "PreToolUse": [HookMatcher(hooks=[_pre_hook])],
                "PostToolUse": [HookMatcher(hooks=[_post_hook])],
            },
        }

        if mcp_servers:
            options_kwargs["mcp_servers"] = mcp_servers

        # Resume existing session or start new
        if session_id:
            options_kwargs["resume"] = session_id
        else:
            options_kwargs["system_prompt"] = system_prompt

        options = ClaudeAgentOptions(**options_kwargs)

        # Execute via query() — async iterator of messages
        result_msg = ""
        cost = 0.0
        final_session_id = session_id
        num_turns = 0
        duration_ms = 0
        usage = None

        async for message in query(prompt=prompt, options=options):
            if isinstance(message, SystemMessage):
                if getattr(message, "subtype", "") == "init":
                    final_session_id = getattr(message, "session_id", final_session_id)
            elif isinstance(message, AssistantMessage):
                for block in getattr(message, "content", []):
                    if isinstance(block, TextBlock):
                        result_msg += block.text
                        self._emit(on_stream, "token", {"content": block.text})
            elif isinstance(message, ResultMessage):
                # ResultMessage.result contains the final text
                result_text = getattr(message, "result", "")
                if result_text and not result_msg:
                    result_msg = result_text
                final_session_id = getattr(message, "session_id", final_session_id)
                cost = getattr(message, "total_cost_usd", 0.0) or 0.0
                num_turns = getattr(message, "num_turns", 0) or 0
                duration_ms = getattr(message, "duration_ms", 0) or 0
                usage = getattr(message, "usage", None)

        return {
            "session_id": final_session_id,
            "message": result_msg,
            "cost": cost,
            "num_turns": num_turns,
            "duration_ms": duration_ms,
            "usage": usage,
        }

    def _find_indexer_command(self) -> Optional[List[str]]:
        """Find the flyto-indexer MCP server command."""
        try:
            import shutil
            # Check if flyto-indexer-mcp is on PATH
            if shutil.which("flyto-indexer-mcp"):
                return ["flyto-indexer-mcp"]
            # Fallback: python -m
            return [sys.executable, "-m", "flyto_indexer.mcp_server"]
        except Exception:
            return None

    def _build_feedback(self, vr: VerificationResult) -> str:
        """Construct feedback message from a failed verification."""
        parts = ["## Verification FAILED — please fix the issues below:"]
        if vr.error:
            parts.append("**Error**: {}".format(vr.error))
        if vr.comparison_summary:
            parts.append("**Visual comparison**: {}".format(vr.comparison_summary))
        if vr.extracted_data:
            text = vr.extracted_data.get("text", "")
            if text:
                parts.append("**Extracted text from page**:\n```\n{}\n```".format(text[:2000]))
        if vr.screenshot_path:
            parts.append("**Screenshot saved at**: {}".format(vr.screenshot_path))
        return "\n\n".join(parts)

    @staticmethod
    def _emit(on_stream: StreamCallback, event_type: str, data: Dict[str, Any]) -> None:
        """Fire a stream event if callback is set."""
        if on_stream is None:
            return
        try:
            on_stream({"type": event_type, **data})
        except Exception:
            pass
