# Copyright 2024 Flyto2
# Licensed under the Apache License, Version 2.0
"""Every typed failure the coding service can raise.

Extracted from `service.py`, which was 8,227 lines. These 35 classes are a
closed vocabulary a caller branches on -- each carries a stable `code` and
nothing sensitive -- and they neither read nor touch service state, so keeping
them beside a 6,973-line class only made both harder to find. They are
re-exported from `flyto_ai.coding.service`, so every existing
`from flyto_ai.coding.service import CodingJobNotFound` keeps working; this is a
move, not a rename, and no error identity, code, or inheritance edge changed.
"""
from __future__ import annotations

from typing import Any, Dict, Sequence, Tuple

from flyto_ai.coding.contracts import (
    ACTION_RESUBMIT_AGAINST_CURRENT_CONTRACT,
    FAILURE_PHASE_VERIFICATION,
)
from flyto_ai.coding.continuation import (
    CONTINUATION_CODES,
    CONTINUATION_UNAVAILABLE,
)
from flyto_ai.coding.mission_runtime import MissionRouteError
from flyto_ai.coding.preflight import FAILURE_PHASE_PREFLIGHT


class CodingServiceError(RuntimeError):
    """Base error with a stable, non-sensitive service code.

    Three typed attributes travel with every error, not just the ones a
    subclass bothered to annotate, because a caller deciding what to do next
    needs them for *all* failures:

    `failure_phase`
        How far the job got before this refusal. `preflight` is the strongest
        statement available - it means no session was opened, no worktree claim
        was taken, and no job exists to poll.
    `retryable`
        Whether repeating the identical request could ever succeed without
        somebody changing something first. Capacity is retryable; a missing
        verification contract is not.
    `required_actions`
        Bounded tokens from a closed allowlist naming the work that would clear
        this. Empty when the caller has nothing to do.

    They are folded into `details` so they cross the MCP and HTTP facades
    through the one existing bounded projection, rather than needing each
    transport to learn about them separately.
    """

    code = "service_error"
    failure_phase = "service"
    retryable = False
    required_actions: Tuple[str, ...] = ()

    @property
    def context(self) -> Dict[str, Any]:
        """Subclass-specific bounded context; empty by default.

        Paths, prose, and credentials never appear here.
        """
        return {}

    @property
    def details(self) -> Dict[str, Any]:
        """The typed envelope plus whatever bounded context a subclass added."""

        payload: Dict[str, Any] = {
            "failure_phase": self.failure_phase,
            "retryable": bool(self.retryable),
        }
        if self.required_actions:
            payload["required_actions"] = tuple(self.required_actions)
        payload.update(self.context)
        return payload


class CodingServiceBusy(CodingServiceError):
    """The service cannot take more work right now; the request itself is fine."""

    code = "service_busy"
    failure_phase = "capacity"
    retryable = True


class CodingCapacityUnavailable(CodingServiceBusy):
    """The bounded job queue is saturated.

    Distinct from the base `service_busy` so a caller can tell "this instance is
    at its configured concurrency ceiling, back off and retry" from any other
    transient refusal. It stays a `CodingServiceBusy` subclass so existing
    handlers and the 429 mapping keep working unchanged.
    """

    code = "capacity_unavailable"


class VerificationRequired(CodingServiceError):
    """This repository has not declared how a change to it must be verified.

    Raised from preflight, so nothing was created and nothing needs cleaning
    up. Never retryable: an identical resubmission fails identically until the
    repository adds or completes its contract.
    """

    code = "verification_required"
    failure_phase = FAILURE_PHASE_PREFLIGHT
    retryable = False

    def __init__(self, message: str, required_actions: Sequence[str] = ()) -> None:
        super().__init__(message)
        self.required_actions = tuple(required_actions)


class VerificationContractInvalid(VerificationRequired):
    """A verification contract exists but cannot be honoured as written."""

    code = "verification_contract_invalid"


class VerificationContractChanged(VerificationRequired):
    """The contract moved between observing this repository and committing to it.

    Distinct from `verification_contract_invalid`: nothing is wrong with the new
    contract. It is simply not the one this submit measured the workspace
    against, and a job pinned for its whole life to a digest that no longer
    describes the file on disk is a job nobody can audit honestly.

    Refused before a job, a lease, a worktree claim, a continuation claim or a
    session exists, so the caller may simply resubmit and be admitted against
    the contract that is actually there.

    Retryable, unlike every other `VerificationRequired`. Those mean "somebody
    must fix something before this can ever work"; this one means "two things
    happened in the wrong order". Nothing is broken and there is no required
    action, so reporting it as terminal would stop a scheduler on a race that
    resolves itself on the next attempt. A continuation retry is retryable in
    exactly the same way - it just then meets the ordinary
    `continuation_contract_changed` rule, because an authority granted under
    the old contract cannot be relicensed under a new one.
    """

    code = "verification_contract_changed"
    retryable = True


class VerificationToolMissing(VerificationRequired):
    """A required check is declared correctly and its program is not installed.

    Three neighbouring refusals, three different people: `verification_required`
    is for whoever owns the repository's contract, `capability_unavailable` is
    for whoever provisions the capability bridge, and this one is for whoever
    provisions the host's tooling. Collapsing them would leave every recipient
    reading prose to find out whether the work is theirs.

    Carries the names of the checks that cannot run. They are contract
    identifiers, already validated as safe tokens, so they say *which* tool to
    install without the refusal becoming a channel for paths or argv.
    """

    code = "verification_tool_missing"

    def __init__(
        self,
        message: str,
        required_actions: Sequence[str] = (),
        blockers: Sequence[str] = (),
    ) -> None:
        super().__init__(message, required_actions)
        self.blockers = tuple(blockers)

    @property
    def context(self) -> Dict[str, Any]:
        if not self.blockers:
            return {}
        return {"verification_blockers": tuple(self.blockers)}


class CapabilityUnavailable(VerificationRequired):
    """A required capability cannot be attached on this host.

    The contract is fine; the environment is not. Kept distinct so an operator
    installing a missing tool is never sent to edit a correct contract.
    """

    code = "capability_unavailable"


class CodingAuthorityConflict(CodingServiceError):
    """This state root's active work belongs to a different startup authority.

    The coding route is startup-fixed by design: the implementer, the audit
    requirement, the contract path, the sandbox, the approval policy and the
    host lane policies are all decided before a job exists and no payload can
    reach them. A second service that would decide any of them differently is
    therefore not a peer worker on this queue - it is a different route sharing
    a directory.

    Refusing it at construction, rather than letting it start and then declining
    each item, is the difference between one bounded error and an unbounded one:
    a running incompatible service is offered every ready item forever, and each
    refusal costs a dispatch attempt and a fencing token. It is also what keeps
    such a service away from the workspace-claim sweep, which it has no standing
    to run against another authority's audit gap.

    Rotation is allowed - just not while work is live. Once every job under the
    old authority is terminal, the next service binds the root to its own.
    """

    code = "execution_authority_conflict"
    failure_phase = "startup"
    retryable = False


class CodingAuthorityUnavailable(CodingServiceError):
    """This host cannot bind a state root to one startup authority at all.

    Distinct from a conflict, because the two are fixed by different people
    doing different things: a conflict means "another authority owns this root",
    and this means "the primitive that would answer that question is missing".

    It is a refusal rather than a degradation on purpose. Without an
    inter-process lock there is no way to know whether another service is alive
    here, so continuing would advertise multi-process isolation this host cannot
    keep - two services would both start, both believe they owned the root, and
    both dispatch against it.
    """

    code = "execution_authority_unavailable"
    failure_phase = "startup"
    retryable = False


class CodingWorkspaceAuthorityConflict(CodingServiceError):
    """Another coding state root owns a configured workspace root.

    A state root brokers the jobs inside it; it cannot broker a directory tree,
    because its workspace claims live under itself. Two services on two state
    roots therefore each keep a private, self-consistent opinion about the same
    checkout - which is exactly how two sessions came to edit one tree on
    2026-08-11. The host-global registry above them refuses the second one.

    Raised before any provider, status row, reconciliation, job record, or
    workspace edit exists, so a refusal here has changed nothing.
    """

    code = "workspace_authority_conflict"
    failure_phase = "startup"
    retryable = False


class CodingWorkspaceAuthorityBusy(CodingServiceError):
    """The host-global registry was mid-transaction past the bounded deadline.

    Neither a conflict nor a fault: the registry is intact and readable, and
    another process is simply joining or reporting on it. A join holds the
    registry-wide lock for a few reads and one small write, so the same request
    normally succeeds on the next attempt.

    Retryable, and the only one of the three workspace refusals that is. It is
    kept distinct because the operator actions for the other two -- stop
    another state root, or repair the registry -- are both wrong here and both
    destructive to somebody's time.
    """

    code = "workspace_authority_busy"
    failure_phase = "startup"
    retryable = True


class CodingWorkspaceAuthorityUnavailable(CodingServiceError):
    """Workspace ownership cannot be established on this host at all.

    Distinct from a conflict for the same reason `CodingAuthorityUnavailable`
    is: a conflict names an owner, this says the mechanism that would name one
    is missing or damaged. Both fail closed.
    """

    code = "workspace_authority_unavailable"
    failure_phase = "startup"
    retryable = False


class HostReleaseValveRootUnusable(CodingServiceError):
    """The valve was pointed at something that is not an established root.

    A release is only ever a *subtraction* from a state root some other service
    already created. So the valve refuses anything it would otherwise have to
    bring into existence first: a missing directory, a missing `.service.lock`
    or `locks/` tree, a missing authority lease, a symlinked component, or a
    world-reachable directory. Creating them would mean the operator's release
    silently established a brand new root and then reported success against it,
    which is the opposite of what they asked for.

    Distinct from `CodingAuthorityUnavailable`: that says the locking primitive
    is missing, this says there is nothing here worth locking.
    """

    code = "release_valve_root_unusable"
    failure_phase = "preflight"
    retryable = False


class HostReleaseValveRefused(CodingServiceError):
    """A subtractive host release was asked to do something additive.

    The release valve exists to *remove* one piece of durable state - an
    orphaned audit's job record, or a claim nobody can evaluate. It deliberately
    starts without a startup authority of its own, so it has no standing to
    admit work, decide an audit, or run a round, and every one of those entry
    points refuses here rather than silently running under an authority the
    valve never proved.
    """

    code = "release_valve_refused"
    failure_phase = "startup"
    retryable = False


class CodingServiceReloadRequired(CodingServiceError):
    """The source tree changed after this long-lived service imported it."""

    code = "service_reload_required"


class CodingJobNotFound(CodingServiceError):
    code = "job_not_found"


class IdempotencyConflict(CodingServiceError):
    code = "idempotency_conflict"


class WorkspaceDenied(CodingServiceError):
    code = "workspace_denied"


class WorkspaceBusy(CodingServiceError):
    """Another job owns this worktree until its audit loop closes.

    Ownership spans the whole job, not one implementation round, so a second
    Codex frontend cannot edit a tree between an implementation and the exact
    revision audit that binds it. The owning job id is safe to publish: it is
    an opaque host-minted token, never a path or a prompt.
    """

    code = "workspace_busy"
    failure_phase = "workspace"

    def __init__(self, message: str, owner_job_id: str = "") -> None:
        super().__init__(message)
        self.owner_job_id = str(owner_job_id or "")

    @property
    def context(self) -> Dict[str, Any]:
        return {"owner_job_id": self.owner_job_id} if self.owner_job_id else {}


class WorkspaceClaimUnresolved(CodingServiceError):
    """A worktree carries a claim whose authority cannot be evaluated.

    This is deliberately distinct from `workspace_busy`. Busy means "a named
    live job owns this tree"; unresolved means "something claims this tree and
    the service cannot prove otherwise". Both refuse the edit, but only the
    second needs a host operator to look at the state root, because no job
    transition will ever clear it on its own.
    """

    code = "workspace_claim_unresolved"
    failure_phase = "workspace"

    def __init__(self, message: str, owner_job_id: str = "") -> None:
        super().__init__(message)
        self.owner_job_id = str(owner_job_id or "")

    @property
    def context(self) -> Dict[str, Any]:
        return {"owner_job_id": self.owner_job_id} if self.owner_job_id else {}


class AbandonStateConflict(CodingServiceError):
    """Only an audit-ready job may be abandoned, and only into a failure."""

    code = "abandon_state_conflict"


class AuditNotEnabled(CodingServiceError):
    code = "audit_not_enabled"


class AuditStateConflict(CodingServiceError):
    code = "audit_state_conflict"


class RevisionMismatch(CodingServiceError):
    code = "revision_mismatch"


class RevisionUnavailable(CodingServiceError):
    code = "revision_unavailable"


class SessionBindingFailed(CodingServiceError):
    code = "session_binding_failed"


class ContinuationRefused(CodingServiceError):
    """A resume request was not granted, and says as little as it safely can.

    Refused in `submit`, before a job, a lease, a claim or a session exists, so
    `preflight` is the truthful phase. Never retryable on its own: every code
    here needs somebody to change something first - restore the bytes, restore
    the contract, or accept that this session is finished.

    The default code is deliberately the collapsed one. A caller learns a
    specific reason only after it has already proven it owns the authority; up
    to that point every distinguishable answer would be an existence oracle for
    somebody else's session.
    """

    code = CONTINUATION_UNAVAILABLE
    failure_phase = "preflight"
    retryable = False

    def __init__(self, message: str, code: str = CONTINUATION_UNAVAILABLE) -> None:
        super().__init__(message)
        # Bounded by the module's own closed set, so a future caller cannot
        # smuggle prose into a facade through this field.
        self.code = code if code in CONTINUATION_CODES else CONTINUATION_UNAVAILABLE


#: Every way a job's durable Indexer plan authority, or the cumulative scope
#: it governs, can fail to be provable. Closed by construction: each is a stable
#: token a caller may branch on, and none of them ever carries contract content,
#: a path, a session, or provider prose.
PLAN_AUTHORITY_CODES: Tuple[str, ...] = (
    "plan_authority_unavailable",
    "plan_authority_unsealable",
    "cumulative_scope_unproven",
    "cumulative_scope_unbounded",
    "cumulative_session_unproven",
    "cumulative_workspace_unproven",
    "cumulative_claim_unproven",
    "cumulative_resume_unproven",
    "cumulative_revision_mismatch",
)


class PlanAuthorityUnprovable(CodingServiceError):
    """This job cannot prove what its root task is authorized to change.

    Lifecycle, stated precisely because the phase depends on it: this is
    *pre-implementer*, not submit-preflight. The job already exists, it already
    holds a durable worktree claim, admission and capability startup already
    happened. What has not happened is the implementer call, so no session was
    opened, no provider budget was spent and nothing was produced to audit.
    `preflight` is therefore the wrong phase - that one promises no job and no
    claim - and `verification` is the truthful one: the host could not verify
    the authority this round would have run under.

    Not retryable. Every member of the closed set means a durable fact is
    missing, stale, replayed or contradicted, so an identical rework of this
    job cannot repair it. The way forward is a fresh job planned against the
    authority that actually exists now, which is exactly what
    `resubmit_against_current_contract` names.

    The code is the whole message. The contract itself, the paths it names and
    the session it belongs to stay in the private job record.
    """

    code = "plan_authority_unavailable"
    failure_phase = FAILURE_PHASE_VERIFICATION
    retryable = False
    required_actions = (ACTION_RESUBMIT_AGAINST_CURRENT_CONTRACT,)

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code if code in PLAN_AUTHORITY_CODES else "plan_authority_unavailable"


class ReworkLimitReached(CodingServiceError):
    code = "rework_limit_reached"


class ReworkNotResumable(CodingServiceError):
    code = "rework_not_resumable"


class RouteEvidenceMissing(CodingServiceError):
    code = "route_evidence_missing"


class AuditBlockersUnresolved(CodingServiceError):
    """An auditable revision still carries host blockers, so it cannot land.

    This is deliberately not `audit_state_conflict`: the job *is* auditable and
    the auditor *is* authorized. Only `accept` is refused, and only while the
    blockers stand. `rework` remains available and is the way through.
    """

    code = "audit_blockers_unresolved"


class EmergencyAuthorityMissing(CodingServiceError):
    """An emergency-bound job cannot be served without its authority contract."""

    code = "emergency_authority_missing"


class MissionRouteRefused(CodingServiceError):
    """The mission lane refused this job, under the kernel's own stable code.

    One class, many codes. `capacity`, `conflict`, `stale fence`, `dependency`,
    `corruption`, `authority` and `unsupported host` are already a closed,
    machine-readable vocabulary in `flyto_ai.coding.mission_runtime`, so this
    adopts that code rather than inventing a parallel one - a caller branching
    on `code` sees exactly what the kernel decided, and `retryable` travels with
    it so backing off is never a guess.
    """

    failure_phase = "mission"

    def __init__(self, exc: MissionRouteError) -> None:
        super().__init__(str(exc))
        self.code = getattr(exc, "code", "mission_unavailable")
        self.retryable = bool(getattr(exc, "retryable", False))
