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
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

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
#: The SDK defaults to a 1 MiB JSON-message buffer. A strict service round can
#: legitimately receive a larger, single framed result from its host-declared
#: Indexer MCP server, so the audited route uses an explicit finite ceiling.
#: Legacy direct calls retain the SDK default and this is not a request/body or
#: tool-authority limit; it only bounds one already-authorized SDK frame.
SERVICE_MAX_BUFFER_SIZE_BYTES = 8 * 1024 * 1024
#: Service mode inspects and edits; host-owned checks run afterward. Process
#: execution is not delegated to the model.
#: Service-mode content search is limited by the guardian to one explicit,
#: regular, non-sensitive file. That gives large-file repair rounds the same
#: bytes `Read` could return without allowing a directory-wide search to cross
#: a protected result path. Process execution remains host-owned.
SERVICE_ALLOWED_TOOLS = ("Read", "Edit", "Write", "Glob", "Grep")
#: The in-process MCP server that carries the repository's declared actions.
#: One server, one tool, and the tool takes a name from a closed list -- so the
#: session gains the ability to ask for a reviewed command and gains nothing
#: resembling a shell. `Bash` is absent from every service catalog above and
#: stays absent; this is the only route to a subprocess an implementer has.
PROJECT_ACTION_SERVER = "flyto-actions"
PROJECT_ACTION_TOOL = "run_project_action"
#: Fully-qualified name the SDK exposes for the tool above.
PROJECT_ACTION_TOOL_ID = "mcp__{}__{}".format(PROJECT_ACTION_SERVER, PROJECT_ACTION_TOOL)
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
#: Ordered because the first match wins: the more specific marker for a
#: condition must precede any broader one that would also match its text.
#: The SDK's own terminal `subtype` values, which are structured data rather
#: than prose. This is the authoritative signal: the CLI emits an `is_error`
#: `ResultMessage` carrying the subtype, the session, the turn count and the
#: usage *before* it exits non-zero, so classifying the trailing exception text
#: was always reading the shadow instead of the object. Closed on purpose - an
#: unrecognized subtype is not guessed into a friendlier category.
STRUCTURED_STOP_SUBTYPES = {
    "error_max_budget_usd": "provider_job_budget_exhausted",
    "error_max_turns": "turn_limit_exceeded",
}

PROVIDER_FAILURE_MARKERS = (
    ("error_max_budget_usd", "provider_job_budget_exhausted"),
    ("reached maximum budget", "provider_job_budget_exhausted"),
    ("error_max_turns", "turn_limit_exceeded"),
    ("reached maximum number of turns", "turn_limit_exceeded"),
    # Credentials. Distinct because no amount of retrying or reworking helps:
    # somebody has to fix an API key before this host can call the provider.
    ("authentication_error", "provider_auth_failed"),
    ("invalid x-api-key", "provider_auth_failed"),
    ("could not resolve authentication method", "provider_auth_failed"),
    ("permission_error", "provider_auth_failed"),
    # Billing/quota. Also terminal, but the fix is a purchase, not a key.
    ("credit balance is too low", "provider_quota_exhausted"),
    ("billing", "provider_quota_exhausted"),
    ("quota exceeded", "provider_quota_exhausted"),
    ("exceeded your current quota", "provider_quota_exhausted"),
    # Transient capacity. The only new category a caller may sensibly retry,
    # which is exactly why it must not be lumped in with the terminal ones.
    ("rate_limit_error", "provider_capacity_unavailable"),
    ("overloaded_error", "provider_capacity_unavailable"),
    # Provider-side content policy. Reworking the prompt is the only route
    # forward, so it is neither retryable nor an implementation blocker.
    ("content policy", "provider_policy_refused"),
    ("stop_reason: refusal", "provider_policy_refused"),
    ("policy violation", "provider_policy_refused"),
)

#: Provider conditions a caller may retry unchanged. Deliberately narrow: only
#: transient capacity qualifies. Auth, quota and policy are terminal until a
#: human changes something, and an unrecognized failure is never guessed to be
#: transient. This says nothing about *where* the work resumes -- see
#: `RESUMABLE_PROVIDER_FAILURE_CODES`, which is about keeping a session.
RETRYABLE_PROVIDER_FAILURE_CODES = frozenset({"provider_capacity_unavailable"})
#: The recognized conditions that stop a round *after* the conversation already
#: exists. They bound how much work one round may do; they say nothing about
#: whether the work so far is valid. A round that ends this way keeps its exact
#: session so the control plane can resume it, and never reports success.
#: Anything unrecognized stays terminal.
RESUMABLE_PROVIDER_FAILURE_CODES = frozenset(
    {"turn_limit_exceeded", "provider_job_budget_exhausted"}
)
#: Recognized stops that, by definition, consumed the whole host-configured
#: round budget. No `ResultMessage` arrives for these, so the count the host
#: itself set is the only truthful one available — and it is a host fact, not
#: something read out of the provider's error.
TURN_BUDGET_EXHAUSTED_CODES = frozenset({"turn_limit_exceeded"})
#: How much of a provider exception is ever examined, in memory, before it is
#: dropped. Nothing derived from the text itself is kept.
MAX_PROVIDER_ERROR_SCAN_CHARS = 2000


def structured_stop_code(message: Any) -> str:
    """Classify one SDK ``ResultMessage`` that reports a terminal stop.

    Structured, bounded and closed. ``is_error`` must be exactly true - a
    truthy string or a stray object is not an assertion this host acts on - and
    the subtype must be one of the values named above. Anything else, including
    an unknown or hostile subtype, returns "" so the caller falls back to the
    conservative path rather than inventing a category.
    """

    try:
        if getattr(message, "is_error", None) is not True:
            return ""
        subtype = getattr(message, "subtype", None)
    except Exception:  # noqa: BLE001 - a hostile attribute is not a category
        return ""
    if not isinstance(subtype, str) or len(subtype) > 64:
        return ""
    return STRUCTURED_STOP_SUBTYPES.get(subtype, "")


def bounded_turns(value: Any, fallback: int = 0) -> int:
    """One non-negative bounded integer, or the host's own fallback."""

    if isinstance(value, bool) or not isinstance(value, int):
        return fallback
    if 0 <= value <= 100_000:
        return value
    return fallback


def bounded_usage(raw: Any) -> Dict[str, int]:
    """Integer counters with safe names only; never provider prose or floats."""

    if not isinstance(raw, Mapping):
        return {}
    kept: Dict[str, int] = {}
    for key, value in sorted(raw.items()):
        if len(kept) >= 16:
            break
        if not isinstance(key, str) or not key.replace("_", "").isalnum():
            continue
        if isinstance(value, bool) or not isinstance(value, int):
            continue
        if 0 <= value <= 10_000_000:
            kept[key] = value
    return kept


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


def make_project_action_handler(
    executor,
    catalog,
    authorized_digest: str,
    *,
    edit_authority: bool,
):
    """Build the async callable the action tool delegates to.

    `edit_authority` is checked again here, not only where the tool is
    attached. Every action this contract can express writes to the worktree -
    that is what actions are for - so a read-only round must not be able to run
    one even if it somehow reached a catalogued tool. Refusing at attach time
    and refusing again before the process starts are two different mistakes to
    have to make.
    """

    from flyto_ai.coding.actions import ProjectActionError

    names = sorted(item["name"] for item in catalog)

    def _refuse(code: str) -> Dict[str, Any]:
        return {
            "content": [{
                "type": "text",
                "text": "{}: declared actions are {}".format(code, ", ".join(names)),
            }],
            "is_error": True,
        }

    async def _handle(args: Dict[str, Any]) -> Dict[str, Any]:
        if not edit_authority:
            # Before any name lookup and long before any process start.
            return _refuse("project_action_requires_edit_authority")
        requested = (args or {}).get("name")
        try:
            result = executor.run(
                requested if isinstance(requested, str) else "",
                expected_config_sha256=authorized_digest,
            )
        except ProjectActionError as exc:
            # The requested name is deliberately not echoed: it came from a
            # model, and a refusal is not a place to reflect one back.
            return _refuse(getattr(exc, "code", "project_action_failed"))
        # Bounded, and explicitly not verification: the host reruns its own
        # required checks after the round regardless of what this reported.
        summary = "action {} exit={} duration_ms={}{}".format(
            result.name,
            result.exit_code,
            result.duration_ms,
            " (timed out)" if result.timed_out else "",
        )
        body = "\n".join(part for part in (summary, result.stdout, result.stderr) if part)
        return {
            "content": [{"type": "text", "text": body}],
            "is_error": not result.ok,
        }

    return _handle


class ActionSandboxMissing(RuntimeError):
    """The action surface cannot be offered because it cannot be isolated.

    Deliberately distinct from `ProjectActionBridgeUnavailable`: that one means
    the contract moved, this one means the deployment lacks the boundary. They
    need different operators doing different things, so they carry different
    stable codes.
    """

    code = "action_sandbox_unavailable"


class ProjectActionBridgeUnavailable(RuntimeError):
    """The declared action surface cannot be attached, so the round must not start.

    Raised rather than swallowed. A malformed contract or a digest that no
    longer matches the job's authority is a pre-session failure the host has to
    see: continuing without the tool would silently downgrade a round that the
    repository declared needs it, and would hide a contract edit made to obtain
    new authority.
    """

    code = "project_action_bridge_unavailable"


def build_project_action_server(
    working_dir: str,
    config_path: str = ".flyto/coding.yaml",
    *,
    edit_authority: bool = False,
    authorized_config_sha256: str = "",
    action_sandbox_image: str = "",
    pinned_contract: Optional[Any] = None,
) -> Tuple[Optional[Any], Tuple[str, ...]]:
    """Build the in-process MCP server carrying this repository's actions.

    Returns `(None, ())` - no tool at all - when the round has no edit
    authority, when the repository declares no actions, or when the SDK is
    absent. Those are the three cases where the correct catalog is the one that
    existed before this feature.

    `authorized_config_sha256` is the job's authority, established once at
    submit and carried by the request. It is *not* recomputed here: recomputing
    would let a model edit `.flyto/coding.yaml` during round one and have the
    edit authorize itself in round two. When the current contract does not hash
    to it, this raises rather than quietly attaching a different surface.

    The catalog and the digest come from one read of one document, so the pair
    can never describe two different files.
    """

    if not edit_authority:
        # Every declared action mutates the worktree. A read-only round has no
        # business holding a tool that does, so it is never built.
        return None, ()

    from flyto_ai.coding.actions import (
        ActionSandboxUnavailable,
        ProjectActionExecutor,
        action_catalog,
    )
    from flyto_ai.coding.checks import round_contract

    try:
        # A host pin is authoritative and is not re-read. Re-reading is what
        # would let this session's own edit grant it a new action surface, and
        # refusing on a digest mismatch is what made a legitimate contract
        # change unfinishable; executing the pin does neither.
        contract = round_contract(working_dir, config_path, pinned=pinned_contract)
    except (OSError, ValueError) as exc:
        # A contract this host cannot parse never grants a tool, and never
        # passes silently: preflight already accepted this repository, so an
        # unreadable contract here is a real change of state.
        raise ProjectActionBridgeUnavailable(
            "the repository action contract could not be read",
        ) from exc

    if (
        pinned_contract is None
        and authorized_config_sha256
        and contract.digest != authorized_config_sha256
    ):
        raise ProjectActionBridgeUnavailable(
            "the repository contract changed after this job was authorized",
        )
    if not contract.actions:
        return None, ()

    try:
        from claude_agent_sdk import create_sdk_mcp_server, tool
    except ImportError:  # pragma: no cover - optional SDK
        return None, ()

    try:
        # The no-pin call shape is the historical one, argument for argument.
        # `ProjectActionExecutor` is patched by tests and by callers that
        # predate pinning, so passing `pinned_contract=None` explicitly would
        # break a constructor contract that never had the parameter. The pin is
        # an addition to the call only when there really is one; the action
        # boundary itself is unchanged either way, because an executor with no
        # pin still re-reads and re-validates the contract exactly as before.
        pin_kwargs = {} if pinned_contract is None else {"pinned_contract": pinned_contract}
        executor = ProjectActionExecutor(
            working_dir, config_path, sandbox_image=action_sandbox_image,
            **pin_kwargs,
        )
    except ActionSandboxUnavailable as exc:
        # The repository declared actions and this host cannot isolate them.
        # Omitting the tool would silently downgrade a round the repository
        # said needs it; running on the host is the vulnerability. So the round
        # does not start.
        raise ActionSandboxMissing(
            "no isolation boundary is available for declared project actions",
        ) from exc
    catalog = action_catalog(contract.actions)
    # Bind to the job's authority when it has one, and to the digest just read
    # otherwise, so the executor's own re-check never passes vacuously.
    bound_digest = authorized_config_sha256 or contract.digest
    handler = make_project_action_handler(
        executor, catalog, bound_digest, edit_authority=True,
    )

    @tool(
        PROJECT_ACTION_TOOL,
        "Run one project action declared in this repository's "
        "source-controlled .flyto/coding.yaml, by name. The command, its "
        "arguments, its timeout and its working directory come from that file "
        "and cannot be changed from here. Declared actions: "
        + ", ".join(
            "{} ({})".format(item["name"], item["description"] or "no description")
            for item in catalog
        ),
        {"name": str},
    )
    async def _run_project_action(args: Dict[str, Any]) -> Dict[str, Any]:
        return await handler(args)

    server = create_sdk_mcp_server(
        name=PROJECT_ACTION_SERVER,
        version="1",
        tools=[_run_project_action],
    )
    return server, (PROJECT_ACTION_TOOL_ID,)


def signal_provider_start(agent: Any) -> None:
    """Tell the host that provider work is beginning, exactly once.

    A backend-neutral seam rather than a dependency: the host attaches
    ``on_provider_start`` before a round and this backend calls it at the one
    moment the statement is true. Nothing about the host, the store or the
    durable record is visible from here.

    Setting the marker earlier - when the adapter is entered, or before the
    action bridge is built - would record a session for every deterministic
    refusal that happens first, which is the false "started with zero attempts"
    this exists to prevent.
    """

    hook = getattr(agent, "on_provider_start", None)
    if hook is None or getattr(agent, "_provider_start_signalled", False):
        return
    try:
        setattr(agent, "_provider_start_signalled", True)
    except Exception:  # pragma: no cover - exotic backend objects stay usable
        pass
    hook()


class ProviderSessionBindingFailed(RuntimeError):
    """The provider's session identity could not be established or bound.

    Carries no provider text, no host detail and no identifier: the caller maps
    it to one stable code. Distinct from every other provider failure because it
    is not about the model's work at all - it is the host refusing to run a
    round whose identity it cannot durably own.
    """


def signal_provider_session(agent: Any, session_id: Any) -> None:
    """Bind the session the provider just established, exactly once.

    Called at SDK `System`/`init`, which is the first moment a real session
    exists and the last moment before the model can touch anything. Everything
    afterwards - a tool call, an edit, a crash, a ceiling - happens inside an
    identity the host has already written down.

    Three refusals, all fail-closed:

    * an id outside the safe SDK shape is not an identity, it is a string;
    * a *different* id arriving later means the backend silently moved
      conversations underneath a bound round, which no host can honour;
    * a hook that raises is a host that could not record ownership, and running
      unowned is precisely the state this exists to prevent.

    Repeating the identical init is a no-op, because a reconnect that lands in
    the same session has not changed anything.
    """

    hook = getattr(agent, "on_provider_session", None)
    if hook is None:
        return
    # One definition of "not a real session", shared with the host rather than
    # restated here. A `host-`/`route-` id is a placeholder the host minted for
    # itself; a backend announcing one is not naming a conversation.
    from flyto_ai.coding.continuation import is_continuable_session

    if not is_safe_sdk_session_id(session_id) or not is_continuable_session(session_id):
        raise ProviderSessionBindingFailed("provider session identity is unusable")
    bound = getattr(agent, "_provider_session_bound", "")
    if bound:
        if bound != session_id:
            raise ProviderSessionBindingFailed("provider session identity changed mid-round")
        return
    try:
        setattr(agent, "_provider_session_bound", session_id)
    except Exception:  # pragma: no cover - exotic backend objects stay usable
        pass
    try:
        hook(session_id)
    except ProviderSessionBindingFailed:
        raise
    except BaseException as exc:  # noqa: BLE001 - no host detail crosses here
        raise ProviderSessionBindingFailed(
            "provider session identity could not be bound",
        ) from exc


def sdk_system_session_id(message: Any, fallback: Any = None) -> Any:
    """Read the provider session from a Claude SDK ``SystemMessage``.

    Current Claude Agent SDK releases expose init metadata through the
    message's ``data`` mapping.  Older test doubles and SDK revisions exposed
    ``session_id`` directly.  Prefer the real mapping shape, while retaining a
    bounded compatibility fallback; validation remains the caller's job and
    therefore still fails closed for an absent or unsafe identity.
    """

    data = getattr(message, "data", None)
    if isinstance(data, Mapping) and "session_id" in data:
        return data.get("session_id")
    return getattr(message, "session_id", fallback)


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
        no_change_feedback = ""

        for attempt in range(1, max_attempts + 1):
            # Phase 2: Claude Code writes code
            self._emit(on_stream, "phase_start", {"phase": "coding", "attempt": attempt})
            evidence.record("coding", "attempt_start", {"attempt": attempt})

            feedback_parts: List[str] = []
            if no_change_feedback:
                feedback_parts.append(no_change_feedback)
            if attempt > 1 and verification_results:
                feedback_parts.append(self._build_feedback(verification_results[-1]))
            feedback_prefix = "\n\n".join(feedback_parts)

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

            # A detached service job that requires a change cannot treat a
            # prose-only answer as an implementation.  The provider-neutral
            # host will independently prove the final diff from snapshots, but
            # waiting until after this method returns used to spend every
            # required check on a known-empty tree and then strand the job as
            # ``no_changes``.  Continue the exact SDK session while its bounded
            # attempt budget remains so the implementer receives one explicit,
            # host-authored correction instead.
            if (
                service_mode
                and bool(getattr(request, "require_changes", True))
                and not self._has_mutation_evidence(evidence)
            ):
                evidence.record("coding", "no_mutation", {"attempt": attempt})
                if attempt < max_attempts:
                    no_change_feedback = (
                        "The prior attempt made no attributable workspace change. "
                        "This service job requires an implementation, not an "
                        "explanation. Inspect the requested files and use the "
                        "available Edit or Write tool to make the smallest correct "
                        "change before responding."
                    )
                    continue
                await self._save_evidence(evidence, service_mode)
                return CodeTaskResponse(
                    ok=False,
                    message=(
                        "Claude made no attributable workspace change after "
                        "{} attempts.".format(max_attempts)
                    ),
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

    @staticmethod
    def _has_mutation_evidence(evidence: EvidenceCollector) -> bool:
        """Return whether the provider used an attributable mutation path.

        File edits are recorded directly by the SDK post hook.  A repository
        project action is also a host-owned mutation path; its exact resulting
        diff is proved later by the service snapshot, so seeing that bounded
        tool is enough to avoid an unnecessary provider retry here.
        """

        if evidence.files_changed:
            return True
        return any(
            record.action == "tool_used"
            and record.data.get("tool") == PROJECT_ACTION_TOOL_ID
            for record in evidence.to_list()
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
        # The repository's declared actions, if it declared any. Built here so
        # the tool closes over this exact workspace and can never be pointed at
        # another one by anything the session says.
        action_server, action_tool_ids = build_project_action_server(
            request.working_dir,
            getattr(request, "config_path", ".flyto/coding.yaml") or ".flyto/coding.yaml",
            # Both gates, from startup authority only: a read-only round gets
            # no action tool, and the digest is the one this job was authorized
            # against at submit -- never recomputed from the current file.
            edit_authority=bool(getattr(request, "service_edit_authority", False))
            and bool(getattr(request, "service_mode", False)),
            authorized_config_sha256=str(
                getattr(request, "authorized_config_sha256", "") or "",
            ),
            action_sandbox_image=str(
                getattr(request, "action_sandbox_image", "") or "",
            ),
            pinned_contract=getattr(request, "pinned_contract", None),
        )
        if action_server is not None:
            mcp_servers[PROJECT_ACTION_SERVER] = action_server
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
            extra_tools=action_tool_ids,
        )
        options_kwargs["hooks"] = {
            "PreToolUse": [HookMatcher(hooks=[_pre_hook])],
            "PostToolUse": [HookMatcher(hooks=[_post_hook])],
        }

        options = ClaudeAgentOptions(**options_kwargs)

        # Everything above is deterministic host work: the action bridge, the
        # contract, the options. It can still refuse, and a refusal there is a
        # deployment problem with no session behind it. *This* is the provider
        # boundary, so this is where the host is told the implementer started -
        # before the first message is awaited, so a crash inside the iterator is
        # still recorded as a session that began.
        signal_provider_start(self)

        # Execute via query() — async iterator of messages
        result_msg = ""
        structured_stop = ""
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
                        established = sdk_system_session_id(message, final_session_id)
                        # Bind before accepting it locally. A session the host
                        # refuses to own is not this round's session either.
                        signal_provider_session(self, established)
                        final_session_id = established
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
                    num_turns = bounded_turns(getattr(message, "num_turns", 0))
                    duration_ms = bounded_turns(getattr(message, "duration_ms", 0))
                    usage = getattr(message, "usage", None)
                    # The SDK says why it is stopping *here*, in a field, before
                    # it exits non-zero. Reading it now is what keeps the
                    # session, the turn count and the usage attached to a round
                    # that really happened.
                    structured_stop = structured_stop_code(message) or structured_stop
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
            # The structured subtype wins over the exception text. The text
            # is a rendering of the same event and can be absent, translated or
            # reworded; the field cannot.
            if isinstance(exc, ProviderSessionBindingFailed):
                # A binding refusal is never recoverable by a bounded-stop
                # code that happened to arrive first. Identity is the
                # precondition for calling any of this a round at all.
                raise
            code = structured_stop or provider_failure_code(exc)
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
        extra_tools: Sequence[str] = (),
    ) -> Dict[str, Any]:
        """Build SDK options without importing or calling the SDK."""

        service_mode = bool(getattr(request, "service_mode", False))
        edits_allowed = bool(getattr(request, "service_edit_authority", True))
        if service_mode:
            service_tools = list(
                SERVICE_ALLOWED_TOOLS if edits_allowed else SERVICE_READONLY_TOOLS,
            )
            # Only the exact fully-qualified action tool id, only when the
            # repository declared at least one action, and only when this round
            # may edit at all. Every declared action writes to the worktree, so
            # a read-only catalog must not contain one - appending it here
            # regardless of `edits_allowed` was a straight authority bypass.
            # This widens the catalog by one named tool; it never adds `Bash`.
            if edits_allowed:
                service_tools.extend(
                    name for name in extra_tools if name == PROJECT_ACTION_TOOL_ID
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
        if service_mode:
            options_kwargs["max_buffer_size"] = SERVICE_MAX_BUFFER_SIZE_BYTES
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

    #: This adapter has no capability bridge: it refuses every *required*
    #: capability the moment it reads the contract (see `run`). Declaring the
    #: empty set is not a placeholder - it is the truthful answer, and it lets
    #: preflight refuse an infeasible contract before a job, a claim or a
    #: session exists instead of after a session has been opened.
    attachable_capability_kinds = frozenset()

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
        from flyto_ai.coding.store import bind_provider_session, mark_provider_start
        from flyto_ai.coding.contracts import (
            ACTION_SANDBOX_UNAVAILABLE,
            VERIFICATION_CONTRACT_CHANGED,
            CodingTaskResult,
        )
        from flyto_ai.coding.workspace import WorkspaceTools, WorkspaceViolation

        thread_id = str(request.thread_id or "")
        try:
            tools = WorkspaceTools(
                request.working_dir,
                sandbox_mode=request.sandbox_mode,
                approval_policy=request.approval_policy,
                sandbox_image=request.command_sandbox_image,
            )
            # One read for the whole round. Checks, capabilities and the action
            # surface all come from this document, so the digest gated below is
            # provably the digest of what this round actually uses.
            contract = round_contract(
                request.working_dir, request.config_path,
                pinned=getattr(request, "pinned_contract", None),
            )
        except (ValueError, OSError, WorkspaceViolation):
            return self._failed(thread_id, "invalid_config")

        pinned = getattr(request, "pinned_contract", None)
        authorized = str(getattr(request, "authorized_config_sha256", "") or "")
        if pinned is None and authorized and contract.digest != authorized:
            # Before checks are derived and long before the provider is called.
            # Deriving verification from a contract the job was not authorized
            # against is how a round could weaken its own required checks: edit
            # `.flyto/coding.yaml` in round one, and round two verifies against
            # the edit. Nothing about this is a provider failure.
            return self._failed(thread_id, VERIFICATION_CONTRACT_CHANGED)
        checks, capabilities = contract.checks, contract.capabilities

        if contract.actions:
            # Feasibility of the *boundary*, decided here rather than deeper in
            # the SDK layer, because "before the session starts" has to mean
            # before this adapter hands anything to a provider at all. A
            # repository that declares actions and a host that cannot isolate
            # them is a deployment fault, and no round may run: omitting the
            # tool would quietly downgrade the round, and running on the host is
            # the vulnerability itself.
            try:
                # Same reasoning as the bridge: keep the legacy call shape when
                # there is no pin, so a patched or older constructor still sees
                # exactly the arguments it has always accepted.
                ProjectActionExecutor(
                    request.working_dir,
                    request.config_path,
                    sandbox_image=request.command_sandbox_image,
                    **({} if pinned is None else {"pinned_contract": pinned}),
                )
            except ActionSandboxUnavailable:
                return self._failed(thread_id, ACTION_SANDBOX_UNAVAILABLE)
            except (OSError, ValueError):
                return self._failed(thread_id, ACTION_SANDBOX_UNAVAILABLE)

        if not [check for check in checks if check.required]:
            return self._failed(thread_id, "verification_required")
        # Same proof as the native peer, at the same point: the pinned contract
        # has been read, and nothing has been asked of a provider yet. A tool
        # that vanished between submit and here is a host defect, and saying so
        # now costs no session, no turns and no revision.
        unlaunchable = unlaunchable_required_checks(checks, request.working_dir)
        if unlaunchable:
            return self._failed(
                thread_id, "verification_tool_missing", blockers=unlaunchable,
            )
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
            require_changes=bool(request.require_changes),
            # The two host-owned facts the SDK layer cannot re-derive safely:
            # which contract this service is operating under, and which exact
            # revision of it this job was authorized against. Without them the
            # bridge re-authorizes whatever is on disk at the moment it builds,
            # which is the whole escalation being closed.
            config_path=request.config_path,
            authorized_config_sha256=authorized,
            action_sandbox_image=request.command_sandbox_image,
            # The third host-owned fact: the contract itself, by value. Without
            # it the bridge would fall back to re-reading a file this session
            # may have rewritten.
            pinned_contract=pinned,
        )
        # Armed, not asserted. The backend calls this at its own provider
        # boundary, after every deterministic precondition has passed, so an
        # action-sandbox or contract refusal below is not recorded as a session
        # that started.
        self.agent.on_provider_start = lambda: mark_provider_start(self.store)
        # Separately armed, because it answers a different question and is
        # answered later: not "did a round begin" but "which conversation is
        # this". The host binds it durably while it still owns the job.
        self.agent.on_provider_session = (
            lambda established: bind_provider_session(self.store, established)
        )
        try:
            response = await self.agent.run(code_request)
        except ProviderSessionBindingFailed:
            # The session could not be owned, so this round has no identity a
            # later reader could trust. It is deliberately the same stable code
            # a mismatched returned session produces below: both mean the host
            # will not attribute work to a conversation it cannot bind.
            return self._failed(self.host_thread_id(thread_id), "session_binding_failed")
        except ActionSandboxMissing:
            # Pre-session and non-provider: the model was never contacted, and
            # the deployment - not the round - is what needs attention.
            return self._failed(
                self.host_thread_id(thread_id), ACTION_SANDBOX_UNAVAILABLE,
            )
        except ProjectActionBridgeUnavailable:
            # Raised while assembling the session, before any provider call.
            # Reporting it as `provider_failed` blamed the model for a
            # control-plane refusal and, worse, put a contract substitution in
            # the same bucket as a transport blip. No prose from the exception
            # crosses this boundary: the stable code is the whole message.
            return self._failed(
                self.host_thread_id(thread_id), VERIFICATION_CONTRACT_CHANGED,
            )
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

        try:
            results = await CheckRunner(tools).run(tuple(checks))
        except VerificationToolUnavailable as exc:
            # The session really ran. What is missing is any basis for a
            # verdict, so the round is refused with the host's reason rather
            # than being blamed on the change - and it never becomes auditable.
            return self._failed(
                session, "verification_tool_missing",
                blockers=exc.blockers,
                # The session really ran, so every number describing it stays
                # true. A result that says "one attempt, zero rounds" invites
                # exactly the wrong conclusion about where the work went.
                attempts=int(getattr(response, "attempts", 0) or 0),
                rounds=int(getattr(response, "claude_num_turns", 0) or 0),
                usage=self._safe_usage(getattr(response, "claude_usage", None)),
            )
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
    @staticmethod
    def _safe_usage(raw: Any) -> Dict[str, int]:
        """Bounded integer counters only; nothing else from a backend payload."""

        if not isinstance(raw, Mapping):
            return {}
        kept: Dict[str, int] = {}
        for key, value in sorted(raw.items()):
            if len(kept) >= 16:
                break
            if not isinstance(key, str) or not key.replace("_", "").isalnum():
                continue
            if isinstance(value, bool) or not isinstance(value, int):
                continue
            if 0 <= value <= 10_000_000:
                kept[key] = value
        return kept

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
        """Return a stable failed result without leaking host material."""

        from flyto_ai.coding.contracts import CodingTaskResult

        return CodingTaskResult(
            ok=False,
            message="Claude coding round failed: {}".format(code),
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
