# Copyright 2024 Flyto2
# Licensed under the Apache License, Version 2.0
"""Codex CLI adapter for the audited Flyto2 coding service.

The CLI is only the implementation worker.  Workspace attribution, repository
checks, continuation identity, durable evidence, and the final Codex audit stay
host-owned by :mod:`flyto_ai.coding`.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import signal
import uuid
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence


DEFAULT_TIMEOUT_SECONDS = 1800
MAX_EVENT_BYTES = 1024 * 1024
MAX_STREAM_BYTES = 8 * 1024 * 1024
MAX_MESSAGE_CHARS = 2000
MAX_ERROR_SCAN_CHARS = 4000
MAX_USAGE_KEYS = 16
MAX_USAGE_VALUE = 10 ** 9
HOST_THREAD_PREFIX = "host-"
_THREAD_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
_MODEL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")

_PROVIDER_FAILURE_MARKERS = (
    ("usage_limit_exceeded", "provider_quota_exhausted"),
    ("usage limit", "provider_quota_exhausted"),
    ("quota exceeded", "provider_quota_exhausted"),
    ("not logged in", "provider_auth_failed"),
    ("unauthorized", "provider_auth_failed"),
    ("authentication", "provider_auth_failed"),
    ("server_overloaded", "provider_capacity_unavailable"),
    ("rate limit", "provider_capacity_unavailable"),
    ("too many requests", "provider_capacity_unavailable"),
    ("cyber_policy", "provider_policy_refused"),
    ("policy refused", "provider_policy_refused"),
    ("session_budget_exceeded", "provider_job_budget_exhausted"),
    ("context_window_exceeded", "turn_limit_exceeded"),
)

_SAFE_ENV_NAMES = (
    "CODEX_HOME", "HOME", "LANG", "LC_ALL", "PATH", "SSL_CERT_DIR",
    "SSL_CERT_FILE", "TERM", "TMPDIR",
)


class CodexCliCodingAgent:
    """Run one bounded Codex CLI turn behind ``CodingService`` contracts."""

    attachable_capability_kinds = frozenset()

    def __init__(
        self,
        store: Any,
        *,
        executable: str,
        model: str,
        timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        resolved = shutil.which(str(executable or ""))
        if not resolved:
            raise ValueError("the Codex CLI executable is unavailable")
        if not isinstance(model, str) or not _MODEL_RE.fullmatch(model):
            raise ValueError("the Codex CLI backend requires a bounded --model")
        if (
            isinstance(timeout_seconds, bool)
            or not isinstance(timeout_seconds, int)
            or not 30 <= timeout_seconds <= 3600
        ):
            raise ValueError("Codex CLI timeout must be between 30 and 3600 seconds")
        self.store = store
        self.executable = resolved
        self.model = model
        self.timeout_seconds = timeout_seconds

    async def run(self, request: Any) -> Any:
        """Execute one implementation round and return only host-derived facts."""

        from flyto_ai.coding.actions import (
            ActionSandboxUnavailable,
            ProjectActionExecutor,
        )
        from flyto_ai.coding.checks import (
            CheckRunner,
            VerificationToolUnavailable,
            round_contract,
            unlaunchable_required_checks,
        )
        from flyto_ai.coding.contracts import (
            ACTION_SANDBOX_UNAVAILABLE,
            VERIFICATION_CONTRACT_CHANGED,
            CodingTaskResult,
        )
        from flyto_ai.coding.store import bind_provider_session, mark_provider_start
        from flyto_ai.coding.workspace import WorkspaceTools, WorkspaceViolation

        supplied_thread = str(request.thread_id or "")
        try:
            tools = WorkspaceTools(
                request.working_dir,
                sandbox_mode=request.sandbox_mode,
                approval_policy=request.approval_policy,
                sandbox_image=request.command_sandbox_image,
            )
            pinned = getattr(request, "pinned_contract", None)
            contract = round_contract(
                request.working_dir,
                request.config_path,
                pinned=pinned,
            )
        except (ValueError, OSError, WorkspaceViolation):
            return self._failed(supplied_thread, "invalid_config")

        authorized = str(getattr(request, "authorized_config_sha256", "") or "")
        if pinned is None and authorized and contract.digest != authorized:
            return self._failed(supplied_thread, VERIFICATION_CONTRACT_CHANGED)
        checks, capabilities = contract.checks, contract.capabilities

        if contract.actions:
            try:
                ProjectActionExecutor(
                    request.working_dir,
                    request.config_path,
                    sandbox_image=request.command_sandbox_image,
                    **({} if pinned is None else {"pinned_contract": pinned}),
                )
            except ActionSandboxUnavailable:
                return self._failed(supplied_thread, ACTION_SANDBOX_UNAVAILABLE)
            except (OSError, ValueError):
                return self._failed(supplied_thread, ACTION_SANDBOX_UNAVAILABLE)

        if not [check for check in checks if check.required]:
            return self._failed(supplied_thread, "verification_required")
        unlaunchable = unlaunchable_required_checks(checks, request.working_dir)
        if unlaunchable:
            return self._failed(
                supplied_thread,
                "verification_tool_missing",
                blockers=unlaunchable,
            )
        if [spec for spec in capabilities if spec.required]:
            return self._failed(supplied_thread, "required_capability_unavailable")

        writable, denial = self._edit_authority(request)
        if denial and request.require_changes:
            return self._failed(supplied_thread, denial)
        try:
            before = tools.snapshot()
        except WorkspaceViolation:
            return self._failed(supplied_thread, "snapshot_failed")

        prompt = self._prompt(request.message, writable=writable)
        expected_session = supplied_thread if request.resume else ""
        state: Dict[str, Any] = {
            "session": "",
            "message": "",
            "usage": {},
            "turn_completed": False,
            "invalid_output": False,
            "binding_failed": False,
            "errors": "",
        }
        stderr_text = ""
        timed_out = False
        returncode: Optional[int] = None
        process = None

        argv = self._argv(
            request.working_dir,
            writable=writable,
            resume=bool(request.resume),
            session_id=expected_session,
        )
        try:
            mark_provider_start(self.store)
            process = await asyncio.create_subprocess_exec(
                *argv,
                cwd=request.working_dir,
                env=self._environment(),
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                start_new_session=True,
                limit=MAX_EVENT_BYTES + 1,
            )
            stdout_task = asyncio.create_task(
                self._read_stdout(
                    process.stdout,
                    state,
                    expected_session=expected_session,
                    bind_session=bind_provider_session,
                ),
            )
            stderr_task = asyncio.create_task(self._read_stderr(process.stderr))
            assert process.stdin is not None
            try:
                process.stdin.write(prompt.encode("utf-8"))
                await process.stdin.drain()
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                process.stdin.close()
            try:
                returncode = await asyncio.wait_for(
                    process.wait(), timeout=self.timeout_seconds,
                )
            except asyncio.TimeoutError:
                timed_out = True
                self._terminate(process)
                await process.wait()
                returncode = process.returncode
            await stdout_task
            stderr_text = await stderr_task
        except Exception as exc:  # noqa: BLE001 - detachable process boundary
            if process is not None and process.returncode is None:
                self._terminate(process)
                try:
                    await process.wait()
                except Exception:  # noqa: BLE001 - process is already unusable
                    pass
            thread = state.get("session") or self.host_thread_id(supplied_thread)
            code = self._provider_failure_code(type(exc).__name__)
            self._note_provider_error(thread, request.working_dir, exc, code)
            return self._failed(thread, code)

        session = str(state.get("session") or "")
        if (
            state.get("binding_failed")
            or not _THREAD_RE.fullmatch(session)
            or (expected_session and session != expected_session)
        ):
            return self._failed(supplied_thread, "session_binding_failed")

        try:
            after = tools.snapshot()
        except WorkspaceViolation:
            return self._failed(session, "snapshot_failed")
        changed = self._attributable(tools, before, after)
        if not writable and changed:
            return self._failed(session, "unexpected_workspace_change")
        try:
            self._bind_thread(session, request.working_dir)
        except (ValueError, OSError):
            return self._failed(session, "thread_binding_failed")

        combined_error = "{}\n{}".format(state.get("errors", ""), stderr_text)
        provider_ok = bool(
            returncode == 0
            and state.get("turn_completed")
            and not state.get("invalid_output")
            and not timed_out
        )
        provider_failure = ""
        if not provider_ok:
            provider_failure = self._provider_failure_code(combined_error)
        self.store.append(session, "coding.round", {
            "backend": "codex-cli",
            "attempts": 1,
            "files_changed": len(changed),
        })

        try:
            results = await CheckRunner(tools).run(tuple(checks))
        except VerificationToolUnavailable as exc:
            return self._failed(
                session,
                "verification_tool_missing",
                blockers=exc.blockers,
                attempts=1,
                rounds=1,
                usage=self._safe_usage(state.get("usage")),
            )
        verified = CheckRunner.passed(results)
        changed_ok = bool(changed) or not request.require_changes
        ok = provider_ok and verified and changed_ok
        failure_code = None
        if not provider_ok:
            failure_code = provider_failure or "provider_failed"
        elif not verified:
            failure_code = "verification_failed"
        elif not changed_ok:
            failure_code = "no_changes"
        self.store.append(session, "coding.outcome", {
            "ok": ok,
            "failure_code": failure_code or "",
        })
        self.store.update(session, status="completed" if ok else "failed")

        return CodingTaskResult(
            ok=ok,
            message=self._public_message(state.get("message", ""), request.working_dir),
            thread_id=session,
            attempts=1,
            status="completed" if ok else "failed",
            files_changed=list(changed),
            checks=list(results),
            capabilities=[],
            usage=self._safe_usage(state.get("usage")),
            rounds_used=1,
            evidence_path="",
            failure_code=failure_code,
        )

    def _argv(
        self,
        workspace: str,
        *,
        writable: bool,
        resume: bool,
        session_id: str,
    ) -> list[str]:
        common = [
            self.executable,
            "exec",
            "--ignore-user-config",
            # Personal exec-policy rules are operator-specific authority and
            # must not leak into this detached implementation worker.
            "--ignore-rules",
            "--strict-config",
            "-c",
            'approval_policy="never"',
            "-c",
            'web_search="disabled"',
            "--disable",
            "plugins",
            "--disable",
            "apps",
            "--disable",
            "standalone_web_search",
            "--disable",
            "search_tool",
            "--disable",
            "browser_use",
            "--disable",
            "browser_use_external",
            "--disable",
            "computer_use",
            "--disable",
            "multi_agent",
            "--disable",
            "multi_agent_v2",
            "--disable",
            "enable_fanout",
            "--model",
            self.model,
            "--skip-git-repo-check",
            "--json",
            # These are `codex exec` options, not `exec resume` options.  They
            # must precede the resume subcommand or Codex restores the session
            # with its default read-only permission profile and an audited
            # rework round cannot modify the workspace.
            "--sandbox",
            "workspace-write" if writable else "read-only",
            "--color",
            "never",
            "--cd",
            workspace,
        ]
        if resume:
            return [*common, "resume", session_id, "-"]
        return [*common, "-"]

    @staticmethod
    def _prompt(message: str, *, writable: bool) -> str:
        authority = (
            "You may edit only this workspace."
            if writable
            else "This is a read-only round; do not modify the workspace."
        )
        return (
            "You are the implementation worker inside Flyto2's governed "
            "flyto_coding route. The host already owns planning gates, real "
            "repository checks, evidence, and the final independent audit.\n\n"
            "Rules for this bounded round:\n"
            "- Follow every AGENTS.md and repository instruction in scope.\n"
            "- {}\n"
            "- Preserve pre-existing dirty work and make the narrowest complete change.\n"
            "- Never call flyto_coding or another implementation agent recursively.\n"
            "- Do not stage, commit, push, publish, deploy, or access credentials.\n"
            "- Do not use web search, browser/computer control, plugins, or MCP tools.\n"
            "- Do not claim checks passed; the host runs source-controlled checks afterward.\n"
            "- Finish with a concise summary of the implementation only.\n\n"
            "Implementation request:\n{}"
        ).format(authority, str(message))

    @staticmethod
    def _environment() -> Dict[str, str]:
        environment = {
            name: os.environ[name]
            for name in _SAFE_ENV_NAMES
            if name in os.environ
        }
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        return environment

    async def _read_stdout(
        self,
        stream: Any,
        state: Dict[str, Any],
        *,
        expected_session: str,
        bind_session: Any,
    ) -> None:
        buffer = bytearray()
        total = 0
        discarding = False
        while True:
            chunk = await stream.read(64 * 1024)
            if not chunk:
                break
            total += len(chunk)
            if total > MAX_STREAM_BYTES:
                state["invalid_output"] = True
            for byte in chunk:
                if byte == 10:
                    if not discarding and buffer:
                        self._consume_event(
                            bytes(buffer),
                            state,
                            expected_session=expected_session,
                            bind_session=bind_session,
                        )
                    buffer.clear()
                    discarding = False
                    continue
                if discarding:
                    continue
                if len(buffer) >= MAX_EVENT_BYTES:
                    state["invalid_output"] = True
                    buffer.clear()
                    discarding = True
                    continue
                buffer.append(byte)
        if buffer and not discarding:
            self._consume_event(
                bytes(buffer),
                state,
                expected_session=expected_session,
                bind_session=bind_session,
            )

    @staticmethod
    async def _read_stderr(stream: Any) -> str:
        kept = bytearray()
        while True:
            chunk = await stream.read(64 * 1024)
            if not chunk:
                break
            remaining = MAX_ERROR_SCAN_CHARS - len(kept)
            if remaining > 0:
                kept.extend(chunk[:remaining])
        return kept.decode("utf-8", errors="replace")

    def _consume_event(
        self,
        raw: bytes,
        state: Dict[str, Any],
        *,
        expected_session: str,
        bind_session: Any,
    ) -> None:
        try:
            event = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            state["invalid_output"] = True
            return
        if not isinstance(event, Mapping) or not isinstance(event.get("type"), str):
            state["invalid_output"] = True
            return
        event_type = event["type"]
        if event_type == "thread.started":
            session = event.get("thread_id")
            if (
                not isinstance(session, str)
                or not _THREAD_RE.fullmatch(session)
                or (expected_session and session != expected_session)
                or (state.get("session") and state["session"] != session)
            ):
                state["binding_failed"] = True
                return
            if not state.get("session"):
                try:
                    bind_session(self.store, session)
                except Exception:  # noqa: BLE001 - host ownership failed closed
                    state["binding_failed"] = True
                    return
                state["session"] = session
        elif event_type == "item.completed":
            item = event.get("item")
            if isinstance(item, Mapping) and item.get("type") == "agent_message":
                text = item.get("text")
                if isinstance(text, str):
                    state["message"] = text[:MAX_MESSAGE_CHARS]
        elif event_type == "turn.completed":
            state["turn_completed"] = True
            state["usage"] = self._safe_usage(event.get("usage"))
        elif event_type in {"turn.failed", "error", "stream_error"}:
            state["errors"] = self._event_error_text(event)

    @staticmethod
    def _event_error_text(event: Mapping[str, Any]) -> str:
        candidates = [
            event.get("message"), event.get("error"), event.get("code"),
        ]
        error = event.get("error")
        if isinstance(error, Mapping):
            candidates.extend((error.get("message"), error.get("code")))
        return " ".join(
            value[:MAX_ERROR_SCAN_CHARS]
            for value in candidates
            if isinstance(value, str)
        )[:MAX_ERROR_SCAN_CHARS]

    @staticmethod
    def _terminate(process: Any) -> None:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            try:
                process.kill()
            except ProcessLookupError:
                pass

    @staticmethod
    def _edit_authority(request: Any) -> "tuple[bool, str]":
        from flyto_ai.coding.contracts import ApprovalPolicy, SandboxMode

        if SandboxMode(request.sandbox_mode) is not SandboxMode.WORKSPACE_WRITE:
            return False, "workspace_read_only"
        if ApprovalPolicy(request.approval_policy) in {
            ApprovalPolicy.ON_REQUEST,
            ApprovalPolicy.ALWAYS,
        }:
            return False, "approval_required"
        return True, ""

    def _bind_thread(self, session: str, workspace: str) -> None:
        try:
            self.store.load(session, workspace)
        except FileNotFoundError:
            self.store.create(workspace, session)

    def _attributable(
        self,
        tools: Any,
        before: Dict[str, str],
        after: Dict[str, str],
    ) -> list[str]:
        changed = tools.changed_since(before, after)
        try:
            evidence_root = Path(str(self.store.root)).resolve()
            prefix = evidence_root.relative_to(Path(tools.root).resolve()).as_posix() + "/"
        except (AttributeError, ValueError, OSError):
            return list(changed)
        return [item for item in changed if not item.startswith(prefix)]

    @classmethod
    def _public_message(cls, value: Any, workspace: str) -> str:
        text = str(value or "")[:MAX_MESSAGE_CHARS]
        variants = {str(workspace), os.path.expanduser(str(workspace))}
        try:
            variants.update({
                os.path.abspath(os.path.expanduser(str(workspace))),
                os.path.realpath(os.path.expanduser(str(workspace))),
                str(Path(workspace).expanduser().resolve()),
            })
        except OSError:
            pass
        for variant in sorted((item for item in variants if item), key=len, reverse=True):
            text = text.replace(variant, "<workspace>")
        return "".join(
            character for character in text
            if character.isprintable() or character == "\n"
        )

    @classmethod
    def _safe_usage(cls, raw: Any) -> Dict[str, int]:
        if not isinstance(raw, Mapping):
            return {}
        kept: Dict[str, int] = {}
        for key, value in sorted(raw.items()):
            if len(kept) >= MAX_USAGE_KEYS:
                break
            if not isinstance(key, str) or not key.replace("_", "").isalnum():
                continue
            if isinstance(value, bool) or not isinstance(value, int):
                continue
            if 0 <= value <= MAX_USAGE_VALUE:
                kept[key] = value
        return kept

    @staticmethod
    def _provider_failure_code(value: Any) -> str:
        try:
            text = str(value)[:MAX_ERROR_SCAN_CHARS].lower()
        except Exception:  # noqa: BLE001 - hostile text is never evidence
            return "provider_failed"
        for marker, code in _PROVIDER_FAILURE_MARKERS:
            if marker in text:
                return code
        return "provider_failed"

    @staticmethod
    def host_thread_id(supplied: Any) -> str:
        if isinstance(supplied, str) and _THREAD_RE.fullmatch(supplied):
            return supplied
        return "{}{}".format(HOST_THREAD_PREFIX, uuid.uuid4().hex[:20])

    def _note_provider_error(
        self,
        thread_id: str,
        workspace: str,
        exc: BaseException,
        code: str,
    ) -> None:
        category = type(exc).__name__
        if not category.isidentifier() or len(category) > 64:
            category = "unknown"
        try:
            self._bind_thread(thread_id, workspace)
            self.store.append(thread_id, "coding.provider_error", {
                "backend": "codex-cli",
                "error_class": category,
                "failure_code": code,
            })
        except Exception:  # noqa: BLE001 - diagnostics never change the result
            return

    @classmethod
    def _failed(
        cls,
        thread_id: str,
        code: str,
        *,
        blockers: Sequence[str] = (),
        attempts: int = 0,
        rounds: int = 0,
        usage: Optional[Dict[str, int]] = None,
    ) -> Any:
        from flyto_ai.coding.contracts import CodingTaskResult

        return CodingTaskResult(
            ok=False,
            message="Codex coding round failed: {}".format(code),
            thread_id=cls.host_thread_id(thread_id),
            attempts=attempts,
            status="failed",
            files_changed=[],
            checks=[],
            capabilities=[],
            usage=dict(usage or {}),
            rounds_used=int(rounds or 0),
            verification_blockers=tuple(blockers),
            evidence_path="",
            failure_code=code,
        )
