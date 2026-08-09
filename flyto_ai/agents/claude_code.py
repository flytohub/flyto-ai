# Copyright 2024 Flyto2
# Licensed under the Apache License, Version 2.0
"""Optional Claude SDK backend — orchestrates Claude Code with
indexer context gathering, guardian hooks, and YAML recipe verification.

Flow:
  Phase 1: Gather codebase context from flyto-indexer
  Phase 2: Spawn Claude Code via Agent SDK to write code
  Phase 3: Run verification recipe (browser screenshot + extraction)
  Phase 4: LLM comparison (actual vs reference)
  Loop back to Phase 2 if verification fails
"""
import logging
import os
import re
import sys
import uuid
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from flyto_ai.agents.evidence import EvidenceCollector, evidence_post_hook
from flyto_ai.agents.guardian_hook import GuardianBlocked, guardian_pre_hook
from flyto_ai.agents.models import (
    CodeTaskRequest,
    CodeTaskResponse,
    VerificationResult,
    is_safe_sdk_session_id,
)
from flyto_ai.agents.prompts import build_system_prompt
from flyto_ai.agents.verifier import VerificationEngine

logger = logging.getLogger(__name__)

# Type alias for streaming callback
StreamCallback = Optional[Callable[[Dict[str, Any]], None]]

#: The pinned Claude model. There is deliberately no fallback chain and no
#: auto-selection: an unavailable model is an operator problem, not a reason to
#: silently run a different one.
DEFAULT_CLAUDE_MODEL = "claude-opus-5"
_MODEL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
#: Service mode inspects and edits; host-owned checks run afterward. Process
#: execution is not delegated to the model.
#: Service-mode content search is limited by the guardian to one explicit,
#: regular, non-sensitive file. That gives large-file repair rounds the same
#: bytes `Read` could return without allowing a directory-wide search to cross
#: a protected result path. Process execution remains host-owned.
SERVICE_ALLOWED_TOOLS = ("Read", "Edit", "Write", "Glob", "Grep")
#: Catalog for a run whose startup authority does not permit model edits.
SERVICE_READONLY_TOOLS = ("Read", "Glob", "Grep")
#: Prefix for a provisional host thread id used when a service round fails
#: before any Claude session exists. It can never be mistaken for, or resumed
#: as, an SDK session.
HOST_THREAD_PREFIX = "host-"
#: A durable ThreadStore identifier is narrower than an opaque SDK session id.
_HOST_THREAD_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
#: Lowercase markers for provider conditions the host can name on its own.
#: The SDK reports a bounded turn limit only as exception text, so the text is
#: matched in memory and discarded; the marker list stays deliberately short so
#: an unrecognized failure keeps the conservative ``provider_failed`` code.
PROVIDER_FAILURE_MARKERS = (
    ("error_max_turns", "turn_limit_exceeded"),
    ("reached maximum number of turns", "turn_limit_exceeded"),
)
#: The recognized conditions that stop a round *after* the conversation already
#: exists. They bound how much work one round may do; they say nothing about
#: whether the work so far is valid. A round that ends this way keeps its exact
#: session so the control plane can resume it, and never reports success.
#: Anything unrecognized stays terminal.
RESUMABLE_PROVIDER_FAILURE_CODES = frozenset({"turn_limit_exceeded"})
#: Recognized stops that, by definition, consumed the whole host-configured
#: round budget. No `ResultMessage` arrives for these, so the count the host
#: itself set is the only truthful one available — and it is a host fact, not
#: something read out of the provider's error.
TURN_BUDGET_EXHAUSTED_CODES = frozenset({"turn_limit_exceeded"})
#: How much of a provider exception is ever examined, in memory, before it is
#: dropped. Nothing derived from the text itself is kept.
MAX_PROVIDER_ERROR_SCAN_CHARS = 2000


def provider_failure_code(exc: BaseException) -> str:
    """Name the bounded provider conditions the host can classify safely.

    The SDK raises a bare ``Exception`` whose text is the only signal for a
    turn limit, so the text is matched in memory against a fixed marker list
    and then discarded. It is never stored, logged, or returned, and anything
    unrecognized stays the conservative ``provider_failed``.
    """

    try:
        text = str(exc)[:MAX_PROVIDER_ERROR_SCAN_CHARS].lower()
    except Exception:  # noqa: BLE001 - a hostile __str__ is not a category
        return "provider_failed"
    for marker, code in PROVIDER_FAILURE_MARKERS:
        if marker in text:
            return code
    return "provider_failed"


class ClaudeCodeAgent:
    """Detachable compatibility backend: indexer → Claude → verify → feedback."""

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
        # The local evidence id is not the Claude SDK session id. Only the
        # latter can resume a conversation.
        session_id = uuid.uuid4().hex[:12]
        evidence = EvidenceCollector(session_id, self._cc.evidence_dir)
        service_mode = bool(getattr(request, "service_mode", False))
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
        # A resumed service round must continue the exact SDK session it was
        # given, starting with the very first call.
        sdk_session_id: Optional[str] = getattr(request, "sdk_session_id", None) or None
        requested_session = sdk_session_id
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
            returned_session = sdk_result.get("session_id")
            if not is_safe_sdk_session_id(returned_session) or (
                requested_session is not None and returned_session != requested_session
            ):
                # Fail closed: without a stable identity a later attempt would
                # silently continue some other conversation.
                await self._save_evidence(evidence, service_mode)
                return CodeTaskResponse(
                    ok=False,
                    message="Claude SDK session identity is unavailable or changed.",
                    session_id=session_id,
                    attempts=attempt,
                    verification_results=verification_results,
                    evidence=evidence.to_list(),
                    files_changed=evidence.files_changed,
                    total_cost_usd=total_cost,
                    claude_session_id=None,
                    claude_num_turns=total_turns,
                    claude_duration_ms=total_duration_ms,
                    claude_usage=last_usage,
                )
            sdk_session_id = returned_session
            requested_session = returned_session
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

            incomplete = str(sdk_result.get("incomplete_code") or "")
            if incomplete:
                # A recognized bounded stop, with the exact session preserved.
                # This is never reported as a success and never verified as one;
                # it is a real, attributable, resumable round that did not
                # finish. The message is host-composed from a fixed vocabulary.
                await self._save_evidence(evidence, service_mode)
                return CodeTaskResponse(
                    ok=False,
                    message="Claude implementation round stopped: {}.".format(incomplete),
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
                    provider_failure_code=incomplete,
                )

            # Phase 3: Verification
            if not request.verification_recipe:
                # No verification configured — consider it a pass
                await self._save_evidence(evidence, service_mode)
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
                await self._save_evidence(evidence, service_mode)
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
        await self._save_evidence(evidence, service_mode)
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
                await guardian_pre_hook(
                    tool_name, tool_input, tool_use_id or "",
                    workspace=request.working_dir,
                    service_mode=bool(getattr(request, "service_mode", False)),
                    edit_authority=bool(getattr(request, "service_edit_authority", True)),
                )
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
        options_kwargs = self._option_kwargs(
            request,
            session_id=session_id,
            system_prompt=system_prompt,
            max_turns=max_turns,
            max_budget=max_budget,
            mcp_servers=mcp_servers,
        )
        options_kwargs["hooks"] = {
            "PreToolUse": [HookMatcher(hooks=[_pre_hook])],
            "PostToolUse": [HookMatcher(hooks=[_post_hook])],
        }

        options = ClaudeAgentOptions(**options_kwargs)

        # Execute via query() — async iterator of messages
        result_msg = ""
        cost = 0.0
        final_session_id = session_id
        num_turns = 0
        duration_ms = 0
        usage = None
        incomplete = ""

        try:
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
        except Exception as exc:  # noqa: BLE001 - the SDK raises a bare Exception
            # The init message already bound this conversation's identity, and
            # the model has been editing the real workspace through host-owned
            # hooks the whole time. Losing that identity because the round hit a
            # bounded stop condition is what strands real work as unauditable.
            #
            # The exception is classified in memory and dropped right here: no
            # provider text, argument, or traceback reaches evidence, a
            # receipt, a log, or the caller. Only a recognized resumable code
            # with a safe captured session is recoverable; everything else
            # propagates and stays terminal.
            code = provider_failure_code(exc)
            if code not in RESUMABLE_PROVIDER_FAILURE_CODES or not is_safe_sdk_session_id(
                final_session_id,
            ):
                raise
            incomplete = code
            if code in TURN_BUDGET_EXHAUSTED_CODES:
                # Reaching this stop *is* the proof that the configured bound
                # was consumed, for a fresh and a resumed invocation alike.
                # Reporting 0 here is what made a real round look like it never
                # happened. The value is the host's own ceiling; nothing is
                # parsed out of the provider's message.
                num_turns = max(num_turns, max_turns)
            evidence.record("coding", "provider_incomplete", {"failure_code": code})

        return {
            "session_id": final_session_id,
            "message": result_msg,
            "cost": cost,
            "num_turns": num_turns,
            "duration_ms": duration_ms,
            "usage": usage,
            # Empty for a round that ran to completion, so the caller's ordinary
            # path is unchanged.
            "incomplete_code": incomplete,
        }

    def _option_kwargs(
        self,
        request: CodeTaskRequest,
        *,
        session_id: Optional[str],
        system_prompt: str,
        max_turns: int,
        max_budget: float,
        mcp_servers: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Build SDK options without importing or calling the SDK."""

        service_mode = bool(getattr(request, "service_mode", False))
        edits_allowed = bool(getattr(request, "service_edit_authority", True))
        if service_mode:
            service_tools = list(
                SERVICE_ALLOWED_TOOLS if edits_allowed else SERVICE_READONLY_TOOLS,
            )
        options_kwargs: Dict[str, Any] = {
            # The service route is pinned to Opus 5. Configuration can vary the
            # legacy direct backend only; it can never redirect service work.
            "model": DEFAULT_CLAUDE_MODEL if service_mode else self.resolve_model(self._cc),
            "max_turns": max_turns,
            "max_budget_usd": max_budget,
            "cwd": request.working_dir,
            "allowed_tools": service_tools if service_mode else self._cc.allowed_tools,
            # An audited service round must start from the host-declared MCP
            # set only. Without strict mode the bundled CLI also loads user
            # and project MCP settings, which can launch unrelated servers,
            # add startup latency, and make the effective route differ from
            # the control plane receipt. Legacy direct runs retain the CLI's
            # normal settings behavior.
            "strict_mcp_config": service_mode,
            # Never disable the SDK permission system implicitly.  The
            # guardian hook is an additional policy boundary, not a reason to
            # bypass the SDK's own approval prompts.  Service mode accepts
            # edits only when the startup sandbox and approval policy grant
            # write authority; it never has a process-execution tool.
            "permission_mode": (
                "acceptEdits" if service_mode and edits_allowed else "default"
            ),
            # The SDK already drops an inherited CLAUDECODE from the child
            # environment so a spawned CLI never believes it is nested inside
            # a Claude Code parent. It merges options.env *over* that
            # filtering, so passing the key at all - even empty - puts the
            # marker back and the CLI refuses to start before a session
            # exists. Leave it absent and let the SDK's filtering stand.
            "env": {},
        }
        if mcp_servers:
            options_kwargs["mcp_servers"] = mcp_servers
        # Resume existing session or start new
        if session_id:
            options_kwargs["resume"] = session_id
        else:
            options_kwargs["system_prompt"] = system_prompt
        return options_kwargs

    @staticmethod
    def resolve_model(claude_config: Any) -> str:
        """Return the validated configured model or the pinned default."""

        configured = getattr(claude_config, "model", "")
        if isinstance(configured, str) and _MODEL_RE.fullmatch(configured):
            return configured
        return DEFAULT_CLAUDE_MODEL

    @staticmethod
    async def _save_evidence(evidence: EvidenceCollector, service_mode: bool) -> None:
        """Service mode records bounded metadata in the ThreadStore instead."""

        if service_mode:
            return
        await evidence.save()

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


class ClaudeCodingAgent:
    """CodingService-compatible adapter over the optional Claude SDK backend.

    It satisfies the same callable shape as ``FlytoCodingAgent``::

        agent = ClaudeCodingAgent(store, config=config)
        result = await agent.run(CodingTaskRequest(...))

    Attribution, verification, and evidence stay host-owned: changed files come
    from independent workspace snapshots, checks come from the repository's
    ``.flyto/coding.yaml``, and the returned ``thread_id`` is the exact Claude
    SDK session so CodingService can enforce same-session rework.
    """

    MAX_MESSAGE_CHARS = 2000
    MAX_USAGE_KEYS = 16
    MAX_USAGE_VALUE = 10 ** 9

    def __init__(
        self,
        store: Any,
        *,
        config: Any = None,
        agent: Optional[ClaudeCodeAgent] = None,
    ) -> None:
        self.store = store
        self.agent = agent if agent is not None else ClaudeCodeAgent(config)

    async def run(self, request: Any) -> Any:
        """Run one bounded implementation round for the coding service."""

        from flyto_ai.coding.checks import CheckRunner, load_project_config
        from flyto_ai.coding.contracts import CodingTaskResult
        from flyto_ai.coding.workspace import WorkspaceTools, WorkspaceViolation

        thread_id = str(request.thread_id or "")
        try:
            tools = WorkspaceTools(
                request.working_dir,
                sandbox_mode=request.sandbox_mode,
                approval_policy=request.approval_policy,
                sandbox_image=request.command_sandbox_image,
            )
            checks, capabilities = load_project_config(
                request.working_dir, request.config_path,
            )
        except (ValueError, OSError, WorkspaceViolation):
            return self._failed(thread_id, "invalid_config")

        if not [check for check in checks if check.required]:
            return self._failed(thread_id, "verification_required")
        if [spec for spec in capabilities if spec.required]:
            # An optional adapter must never let a required capability pass by
            # simply not being attachable here.
            return self._failed(thread_id, "required_capability_unavailable")

        # The Claude backend gets exactly the authority the host granted at
        # startup. An impossible task fails before the model is ever called.
        writable, denial = self._edit_authority(request)
        if denial and request.require_changes:
            return self._failed(thread_id, denial)

        try:
            before = tools.snapshot()
        except WorkspaceViolation:
            return self._failed(thread_id, "snapshot_failed")

        code_request = CodeTaskRequest(
            message=request.message,
            working_dir=request.working_dir,
            max_fix_attempts=request.max_attempts,
            max_turns=request.max_rounds,
            sdk_session_id=thread_id if request.resume and thread_id else None,
            service_mode=True,
            service_edit_authority=writable,
        )
        try:
            response = await self.agent.run(code_request)
        except Exception as exc:  # noqa: BLE001 - the backend is detachable
            # A round that dies here has no Claude session, so one provisional
            # host id is derived once and used for both the durable diagnostic
            # and the returned receipt; two derivations would name two threads.
            round_thread = self.host_thread_id(thread_id)
            code = self._provider_failure_code(exc)
            self._note_provider_error(round_thread, request.working_dir, exc, code)
            return self._failed(round_thread, code)

        session = response.claude_session_id
        if (
            not is_safe_sdk_session_id(session)
            or not _HOST_THREAD_RE.fullmatch(session)
            or (
                code_request.sdk_session_id is not None
                and session != code_request.sdk_session_id
            )
        ):
            return self._failed(thread_id, "session_binding_failed")
        try:
            after = tools.snapshot()
        except WorkspaceViolation:
            return self._failed(session, "snapshot_failed")
        changed = self._attributable(tools, before, after)
        if not writable and changed:
            # A read-only run that still mutated the workspace is evidence of a
            # boundary failure, never an acceptable implementation.
            return self._failed(session, "unexpected_workspace_change")

        try:
            self._bind_thread(session, request.working_dir)
        except (ValueError, OSError):
            return self._failed(session, "thread_binding_failed")
        self.store.append(session, "coding.round", {
            "backend": "claude-sdk",
            "attempts": int(getattr(response, "attempts", 0) or 0),
            "files_changed": len(changed),
        })

        results = await CheckRunner(tools).run(tuple(checks))
        verified = CheckRunner.passed(results)
        changed_ok = bool(changed) or not request.require_changes
        ok = bool(response.ok) and verified and changed_ok
        failure_code = None
        if not response.ok:
            failure_code = self._response_failure_code(response)
        elif not verified:
            failure_code = "verification_failed"
        elif not changed_ok:
            failure_code = "no_changes"
        self.store.append(session, "coding.outcome", {
            "ok": ok, "failure_code": failure_code or "",
        })
        self.store.update(session, status="completed" if ok else "failed")

        return CodingTaskResult(
            ok=ok,
            message=self._public_message(response, request.working_dir),
            thread_id=session,
            attempts=int(getattr(response, "attempts", 0) or 0),
            status="completed" if ok else "failed",
            files_changed=list(changed),
            checks=list(results),
            capabilities=[],
            usage=self._bounded_usage(response.claude_usage),
            rounds_used=int(getattr(response, "claude_num_turns", 0) or 0),
            evidence_path="",
            failure_code=failure_code,
        )

    @staticmethod
    def _edit_authority(request: Any) -> "tuple[bool, str]":
        """Resolve write authority from the startup sandbox and approval policy.

        Returns the authority and, when it is absent, the stable reason a task
        that requires changes cannot run at all. `never` and `on-failure` keep
        the native-compatible behavior of granting writes up front.
        """

        from flyto_ai.coding.contracts import ApprovalPolicy, SandboxMode

        if SandboxMode(request.sandbox_mode) is not SandboxMode.WORKSPACE_WRITE:
            return False, "workspace_read_only"
        if ApprovalPolicy(request.approval_policy) in {
            ApprovalPolicy.ON_REQUEST, ApprovalPolicy.ALWAYS,
        }:
            # A detached service has no interactive host to pause for.
            return False, "approval_required"
        return True, ""

    def _bind_thread(self, session: str, workspace: str) -> None:
        """Create or resume the durable thread under the exact SDK session id."""

        try:
            self.store.load(session, workspace)
        except FileNotFoundError:
            self.store.create(workspace, session)

    def _attributable(
        self, tools: Any, before: Dict[str, str], after: Dict[str, str],
    ) -> List[str]:
        """Derive changed files from snapshots, never from model prose."""

        changed = tools.changed_since(before, after)
        try:
            evidence_root = Path(str(self.store.root)).resolve()
            prefix = evidence_root.relative_to(Path(tools.root).resolve()).as_posix() + "/"
        except (AttributeError, ValueError, OSError):
            return list(changed)
        return [item for item in changed if not item.startswith(prefix)]

    @classmethod
    def _public_message(cls, response: Any, workspace: str) -> str:
        """Bound the model message and keep host paths out of the receipt."""

        text = str(getattr(response, "message", "") or "")[: cls.MAX_MESSAGE_CHARS]
        for variant in cls._workspace_variants(workspace):
            text = text.replace(variant, "<workspace>")
        return "".join(
            character for character in text
            if character.isprintable() or character == "\n"
        )

    @staticmethod
    def _workspace_variants(workspace: str) -> List[str]:
        """Every canonical spelling of this run's workspace, longest first.

        A relative or symlinked request still lets the model echo the resolved
        absolute path, so redaction cannot rely on the request string alone.
        """

        if not workspace:
            return []
        variants = {str(workspace)}
        expanded = os.path.expanduser(str(workspace))
        variants.add(expanded)
        try:
            variants.add(os.path.abspath(expanded))
            variants.add(os.path.realpath(expanded))
            variants.add(str(Path(expanded).resolve()))
        except OSError:  # pragma: no cover - resolution is local and bounded
            pass
        return sorted((item for item in variants if item), key=len, reverse=True)

    @classmethod
    def _bounded_usage(cls, usage: Any) -> Dict[str, int]:
        """Convert only bounded integer counters; drop everything else."""

        if not isinstance(usage, dict):
            return {}
        projected: Dict[str, int] = {}
        for key, value in usage.items():
            if len(projected) >= cls.MAX_USAGE_KEYS:
                break
            if not isinstance(key, str) or not key.isidentifier() or len(key) > 64:
                continue
            if isinstance(value, bool) or not isinstance(value, int):
                continue
            if 0 <= value <= cls.MAX_USAGE_VALUE:
                projected[key] = value
        return projected

    @staticmethod
    def host_thread_id(supplied: Any) -> str:
        """Return a durable host thread id for any round, including failures.

        A resumed round keeps its supplied thread so its evidence stays in one
        place. A round that fails before any Claude session exists still needs
        an id the ThreadStore accepts, so it gets a provisional host id that is
        deliberately distinguishable from an SDK session.
        """

        if isinstance(supplied, str) and _HOST_THREAD_RE.fullmatch(supplied):
            return supplied
        return "{}{}".format(HOST_THREAD_PREFIX, uuid.uuid4().hex[:20])

    @staticmethod
    def _provider_failure_code(exc: BaseException) -> str:
        """Classify one provider exception; see `provider_failure_code`."""

        return provider_failure_code(exc)

    @staticmethod
    def _response_failure_code(response: Any) -> str:
        """Name a recognized bounded provider stop, else the conservative code.

        Only the fixed vocabulary above is honoured, so a backend that invents
        its own code cannot widen what the control plane treats as resumable.
        """

        code = getattr(response, "provider_failure_code", "") or ""
        if isinstance(code, str) and code in RESUMABLE_PROVIDER_FAILURE_CODES:
            return code
        return "provider_failed"

    def _note_provider_error(
        self, thread_id: str, workspace: str, exc: BaseException, code: str,
    ) -> None:
        """Durably record the sanitized category of a failed provider start.

        A start that dies before any Claude session leaves no other trace, so
        the provisional thread is created if it does not exist yet and the
        exception class plus the host's own failure code are appended under the
        exact id the caller receives. The exception message, arguments,
        traceback, and environment are never recorded: they can carry paths,
        tokens, or prompt material. A store that refuses the note still never
        fails the round.
        """

        category = type(exc).__name__
        if not category.isidentifier() or len(category) > 64:
            category = "unknown"
        try:
            self._bind_thread(thread_id, workspace)
            self.store.append(thread_id, "coding.provider_error", {
                "backend": "claude-sdk", "error_class": category,
                "failure_code": code,
            })
        except Exception:  # noqa: BLE001 - diagnostics never break the round
            return

    @classmethod
    def _failed(cls, thread_id: str, code: str) -> Any:
        """Return a stable failed result without leaking host material."""

        from flyto_ai.coding.contracts import CodingTaskResult

        return CodingTaskResult(
            ok=False,
            message="Claude coding round failed: {}".format(code),
            thread_id=cls.host_thread_id(thread_id),
            attempts=0,
            status="failed",
            files_changed=[],
            checks=[],
            capabilities=[],
            usage={},
            rounds_used=0,
            evidence_path="",
            failure_code=code,
        )
