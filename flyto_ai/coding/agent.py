# Copyright 2024 Flyto2
# Licensed under the Apache License, Version 2.0
"""Native provider-neutral Flyto2 coding agent and verification loop."""
from __future__ import annotations

import dataclasses
import logging
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from flyto_ai.coding.capabilities import CapabilityManager
from flyto_ai.coding.checks import (
    CheckRunner,
    VerificationToolUnavailable,
    round_contract,
    unlaunchable_required_checks,
)
from flyto_ai.coding.contracts import (
    VERIFICATION_CONTRACT_CHANGED,
    CapabilitySpec,
    CheckResult,
    CheckSpec,
    CodingTaskRequest,
    CodingTaskResult,
)
from flyto_ai.coding.store import ThreadStore
from flyto_ai.coding.workspace import WorkspaceTools
from flyto_ai.coding.store import mark_provider_start
from flyto_ai.providers.base import LLMProvider


logger = logging.getLogger(__name__)
StreamCallback = Optional[Callable[[Any], None]]


SYSTEM_PROMPT = """You are the native Flyto2 coding agent.
Operate only through the supplied coding and capability tools. Inspect before
editing, make the smallest complete change, preserve unrelated user work, and
never claim success from prose. The host will run source-controlled checks;
when a check fails, use its bounded evidence to repair the implementation.
Do not request, print, persist, or search for credentials. Do not weaken tests,
permissions, sandbox boundaries, or verification to make a check pass.
External capabilities are detachable and their absence grants no authority.
"""


class FlytoCodingAgent:
    """Run a model/tool loop followed by mandatory real checks and repair."""

    #: This implementer drives a real `CapabilityManager`, so it can bridge the
    #: kinds a contract may declare. Preflight reads this to decide whether a
    #: required capability is feasible before a job exists.
    attachable_capability_kinds = frozenset({"mcp-stdio", "command"})

    def __init__(self, provider: LLMProvider, *, store: Optional[ThreadStore] = None) -> None:
        self.provider = provider
        self.store = store or ThreadStore()

    async def run(
        self,
        request: CodingTaskRequest,
        *,
        on_stream: StreamCallback = None,
    ) -> CodingTaskResult:
        metadata = self.store.load(request.thread_id, request.working_dir) if request.resume else self.store.create(
            request.working_dir, request.thread_id,
        )
        thread_id = str(metadata["thread_id"])
        self.store.update(thread_id, status="preflight")
        self.store.append(thread_id, "conversation.message", {
            "role": "user", "content": request.message,
        })

        pinned = getattr(request, "pinned_contract", None)
        try:
            # One resolution per round, so the digest gated below is provably
            # the digest of the document this round derives its checks from.
            # With a host pin there is no read at all: the authorized contract
            # is carried by value, so an edit to the file on disk can neither
            # authorize itself nor strand this round.
            contract = round_contract(
                request.working_dir, request.config_path, pinned=pinned,
            )
        except Exception as exc:
            return self._fail(thread_id, "invalid_config", str(exc), attempts=0)

        authorized = str(getattr(request, "authorized_config_sha256", "") or "")
        if pinned is None and authorized and contract.digest != authorized:
            # Before checks are derived and before any provider is contacted.
            # A round that verified against a contract its job was not
            # authorized against could weaken its own required checks by
            # editing them in an earlier round.
            return self._fail(
                thread_id,
                VERIFICATION_CONTRACT_CHANGED,
                "the repository verification contract changed after this job "
                "was authorized",
                attempts=0,
            )
        configured_checks = contract.checks
        configured_capabilities = contract.capabilities
        checks = request.checks or configured_checks
        capabilities = self._merge_capabilities(configured_capabilities, request.capabilities)
        if not checks or not any(check.required for check in checks):
            return self._fail(
                thread_id, "verification_required",
                "No required real checks are configured. Add .flyto/coding.yaml or pass CheckSpec values.",
                attempts=0,
            )
        # The contract is pinned and read; now prove its required commands can
        # actually start, before a workspace, a capability handshake or a
        # provider call exists. Submit-time preflight already asked this, so
        # reaching it here means the tool went away in between - a race, not a
        # new class of problem, and the honest answer is the same one preflight
        # would have given rather than a verdict on a change nobody made yet.
        unlaunchable = unlaunchable_required_checks(checks, request.working_dir)
        if unlaunchable:
            return self._fail(
                thread_id, "verification_tool_missing",
                "a required verification tool is not installed on this host",
                attempts=0, blockers=unlaunchable,
            )

        workspace = WorkspaceTools(
            request.working_dir,
            sandbox_mode=request.sandbox_mode,
            approval_policy=request.approval_policy,
            sandbox_image=request.command_sandbox_image,
        )
        self.store.append(thread_id, "sandbox.discovered", {
            "backend": workspace.command_sandbox_backend,
            "model_commands_available": bool(workspace.command_sandbox_backend),
            "model_command_network": "denied",
            "model_command_workspace_write": "denied",
        })
        try:
            baseline = workspace.snapshot()
        except Exception as exc:
            return self._fail(
                thread_id, "snapshot_failed", str(exc), attempts=0,
                command_sandbox=workspace.command_sandbox_backend,
            )
        manager = CapabilityManager(request.working_dir)
        statuses = await manager.start(capabilities)
        self.store.append(thread_id, "capabilities.discovered", {
            "capabilities": [dataclasses.asdict(status) for status in statuses],
        })
        if not manager.required_available:
            await manager.close()
            missing = [status.name for status in statuses if status.required and not status.available]
            return self._fail(
                thread_id, "required_capability_unavailable",
                "Required capabilities unavailable: {}".format(", ".join(missing)),
                attempts=0, capabilities=statuses,
                command_sandbox=workspace.command_sandbox_backend,
            )

        messages = self.store.replay_messages(thread_id) if request.resume else []
        if not messages or messages[-1].get("content") != request.message:
            messages.append({"role": "user", "content": request.message})
        tool_definitions = workspace.definitions + manager.definitions
        check_runner = CheckRunner(workspace)
        total_usage: Dict[str, int] = {}
        total_rounds = 0
        final_message = ""
        last_checks: List[CheckResult] = []
        attempts_used = 0
        self.store.update(thread_id, status="running")

        async def dispatch(name: str, args: Dict[str, Any]) -> Dict[str, Any]:
            result = await (
                workspace.dispatch(name, args)
                if name.startswith("coding_")
                else manager.dispatch(name, args)
            )
            self.store.append(thread_id, "tool.completed", self._project_tool_event(name, args, result))
            return result

        try:
            for attempt in range(1, request.max_attempts + 1):
                attempts_used = attempt
                self.store.append(thread_id, "attempt.started", {"attempt": attempt})
                # Durable, and before the provider is entered, so a worker that
                # dies mid-call is still recorded as having started one.
                mark_provider_start(self.store)
                try:
                    final_message, tool_log, rounds, usage = await self.provider.chat(
                        messages=messages,
                        system_prompt=self._system_prompt(statuses, checks),
                        tools=tool_definitions,
                        dispatch_fn=dispatch,
                        max_rounds=request.max_rounds,
                        on_stream=on_stream,
                    )
                except Exception as exc:
                    self.store.append(thread_id, "provider.failed", {
                        "attempt": attempt, "error": str(exc)[:1000],
                    })
                    return self._fail(
                        thread_id, "provider_failed", str(exc), attempts=attempt,
                        capabilities=statuses, usage=total_usage, rounds=total_rounds,
                        command_sandbox=workspace.command_sandbox_backend,
                    )
                total_rounds += int(rounds)
                for key, value in (usage or {}).items():
                    if isinstance(value, int):
                        total_usage[key] = total_usage.get(key, 0) + value
                self.store.append(thread_id, "provider.completed", {
                    "attempt": attempt, "rounds": rounds, "usage": usage,
                    "tool_calls": [self._project_provider_log(item) for item in tool_log],
                })
                self.store.append(thread_id, "conversation.message", {
                    "role": "assistant", "content": final_message,
                })

                try:
                    last_checks = await check_runner.run(checks)
                except VerificationToolUnavailable as exc:
                    # The provider really ran, so the attempt and round counts
                    # stay honest; what is missing is any basis for a verdict.
                    # Reporting this as a failed change would blame the round
                    # for the host, and a round with no verification must never
                    # become auditable.
                    return self._fail(
                        thread_id, "verification_tool_missing",
                        "a required verification tool is not installed on this host",
                        attempts=attempt, capabilities=statuses, usage=total_usage,
                        rounds=total_rounds, blockers=exc.blockers,
                        command_sandbox=workspace.command_sandbox_backend,
                    )
                self.store.append(thread_id, "verification.completed", {
                    "attempt": attempt,
                    "checks": [dataclasses.asdict(result) for result in last_checks],
                })
                after = workspace.snapshot()
                changed = self._attributable_changes(
                    workspace, workspace.changed_since(baseline, after),
                )
                verified = CheckRunner.passed(last_checks)
                changed_ok = bool(changed) or not request.require_changes
                if verified and changed_ok:
                    self.store.update(thread_id, status="completed", turn_count=int(metadata.get("turn_count", 0)) + 1)
                    return CodingTaskResult(
                        ok=True, message=final_message or "Coding task verified.",
                        thread_id=thread_id, attempts=attempt, status="completed",
                        files_changed=changed, checks=last_checks, capabilities=statuses,
                        usage=total_usage, rounds_used=total_rounds,
                        evidence_path=self.store.evidence_path(thread_id),
                        command_sandbox=workspace.command_sandbox_backend,
                    )
                failure_summary = self._check_feedback(last_checks, changed_ok)
                if attempt < request.max_attempts:
                    messages.extend([
                        {"role": "assistant", "content": final_message},
                        {"role": "user", "content": failure_summary},
                    ])
                    self.store.append(thread_id, "conversation.message", {
                        "role": "user", "content": failure_summary,
                    })
        finally:
            await manager.close()

        after = workspace.snapshot()
        changed = self._attributable_changes(
            workspace, workspace.changed_since(baseline, after),
        )
        failure_code = "no_changes" if request.require_changes and not changed else "verification_failed"
        return self._fail(
            thread_id, failure_code,
            self._check_feedback(last_checks, bool(changed) or not request.require_changes),
            attempts=attempts_used, checks=last_checks, capabilities=statuses,
            files_changed=changed, usage=total_usage, rounds=total_rounds,
            command_sandbox=workspace.command_sandbox_backend,
        )

    def _fail(
        self,
        thread_id: str,
        code: str,
        message: str,
        *,
        attempts: int,
        checks: Optional[List[CheckResult]] = None,
        capabilities: Optional[List[Any]] = None,
        files_changed: Optional[List[str]] = None,
        usage: Optional[Dict[str, int]] = None,
        rounds: int = 0,
        command_sandbox: str = "",
        blockers: Sequence[str] = (),
    ) -> CodingTaskResult:
        self.store.append(thread_id, "task.failed", {"failure_code": code, "message": message})
        self.store.update(thread_id, status="failed", last_failure_code=code)
        return CodingTaskResult(
            ok=False, message=message[:4000], thread_id=thread_id, attempts=attempts,
            status="failed", checks=checks or [], capabilities=capabilities or [],
            files_changed=files_changed or [], usage=usage or {}, rounds_used=rounds,
            evidence_path=self.store.evidence_path(thread_id), failure_code=code,
            command_sandbox=command_sandbox, verification_blockers=tuple(blockers),
        )

    @staticmethod
    def _merge_capabilities(
        configured: Sequence[CapabilitySpec], requested: Sequence[CapabilitySpec],
    ) -> Tuple[CapabilitySpec, ...]:
        merged = {spec.name: spec for spec in configured}
        merged.update({spec.name: spec for spec in requested})
        return tuple(merged[name] for name in sorted(merged))

    def _attributable_changes(
        self, workspace: WorkspaceTools, changed: Sequence[str],
    ) -> List[str]:
        """Exclude the evidence store if an operator placed it in the repo."""
        try:
            state_prefix = self.store.root.relative_to(workspace.root).as_posix().rstrip("/")
        except ValueError:
            return list(changed)
        return [
            path for path in changed
            if path != state_prefix and not path.startswith(state_prefix + "/")
        ]

    @staticmethod
    def _system_prompt(statuses: Sequence[Any], checks: Sequence[CheckSpec]) -> str:
        capability_lines = [
            "- {}: {} (required={})".format(item.name, "available" if item.available else "unavailable", item.required)
            for item in statuses
        ] or ["- none configured"]
        check_lines = ["- {}: {}".format(item.name, " ".join(item.argv)) for item in checks]
        return "{}\nCapabilities:\n{}\nRequired verification checks:\n{}".format(
            SYSTEM_PROMPT, "\n".join(capability_lines), "\n".join(check_lines),
        )

    @staticmethod
    def _project_tool_event(name: str, args: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
        safe_args = {
            key: value for key, value in args.items()
            if key in {"path", "query", "argv", "timeout_seconds", "overwrite"}
        }
        return {
            "tool": name, "arguments": safe_args,
            "ok": bool(result.get("ok")), "exit_code": result.get("exit_code"),
            "path": result.get("path"), "sha256": result.get("sha256") or result.get("output_sha256"),
            "error": result.get("error"),
            "sandbox_backend": result.get("sandbox_backend"),
        }

    @staticmethod
    def _project_provider_log(item: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "function": item.get("function"), "ok": item.get("ok"),
            "module_id": item.get("module_id"),
        }

    @staticmethod
    def _check_feedback(results: Sequence[CheckResult], changed_ok: bool) -> str:
        parts = ["Verification failed. Fix the implementation; do not weaken the checks."]
        if not changed_ok:
            parts.append("No workspace change attributable to this run was detected.")
        for result in results:
            if result.required and not result.passed:
                parts.append("{}: {}\n{}".format(
                    result.name, result.error or "failed", result.output_preview[-2000:],
                ))
        return "\n\n".join(parts)
