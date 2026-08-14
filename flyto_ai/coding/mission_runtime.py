# Copyright 2024 Flyto2
# Licensed under the Apache License, Version 2.0
"""The coding adapter between one coding job and the generic mission kernel.

`flyto_ai.orchestration.mission_control` is deliberately workload-neutral: it
knows objectives, work graphs, resources, leases and fencing tokens, and it has
no vocabulary for repositories, audits or Codex. This module is the only place
that vocabulary exists. It translates one direction - coding job to mission -
and never the other, so nothing coding-specific is ever pushed down into the
kernel.

Four separations carry this layer, and each of them exists because the obvious
shortcut is unsafe.

*Synthesis versus contract.* A caller that names no mission still gets one, and
the synthesized contract is fixed here rather than in the kernel: the objective
is the caller's own immutable request message, the desired result is an
attributable verified revision accepted by an independent Codex audit, and the
three acceptance criteria name the implementation revision, the checks pinned at
admission, and that audit. A caller that *does* name a mission has its contract
honoured exactly - and when it names an existing `mission_id`, the stored
contract is re-derived and compared against the envelope's own main axis, so a
job can never attach to a mission whose objective is not the one it was
submitted against.

*Identity versus payload.* Coordinates carry the tenant reference, the workspace
digest and the job id - three bounded tokens, all of them already digests or
opaque identifiers. The canonical workspace is claimed as a
:class:`MissionResource` by digest, never by path, so mutual exclusion over a
worktree is enforced without the store, a snapshot, or a fleet view ever holding
a filesystem path.

*Authority versus record.* Every mutating call here demands the tenant, the
mission id, the work item id, the worker identity and the live
:class:`DispatchHandle` that a dispatch produced, and all five must agree. A
handle cannot be assembled from identifiers read out of a receipt, because the
kernel authorises it against a lease this process holds; this layer refuses even
to try, and ambiguity is a refusal rather than a best guess.

*Projection versus context.* :meth:`CodingMissionRuntime.fleet` is a bounded
read of the kernel's own secret-free snapshot: identities, shape, status and
counters. It carries no objective, no criterion statement, no rationale, no
evidence value, no coordinate, no worker and no path, and it is a snapshot -
there is no operation on it. Full mission context is reachable only through
:meth:`CodingMissionRuntime.context`, which requires the owning tenant.
"""
from __future__ import annotations

import os
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, Mapping, Optional, Sequence, Tuple

from flyto_ai.coding.contracts import (
    MISSION_LANE_PRIMARY,
    MISSION_OPEN,
    MISSION_STATUS_CLOSED,
    MISSION_STATUS_READY,
    CodingMissionEnvelope,
    CodingMissionProjection,
    mission_axis_sha256,
)
from flyto_ai.orchestration.mission_control import (
    DEFAULT_QUEUE_CAPACITY,
    DISPOSITION_BLOCKED,
    DISPOSITION_DEFERRED,
    DISPOSITION_FIXED,
    LANE_REPAIR,
    MAX_SNAPSHOT_ITEMS,
    MAX_TEXT_CHARS,
    AcceptanceCriterion,
    Closure,
    DispatchHandle,
    Mission,
    MissionCapacityExceeded,
    MissionConflict,
    MissionCorrupt,
    MissionError,
    MissionResource,
    MissionStaleFence,
    MissionStore,
    MissionUnauthorized,
    MissionUnsupported,
    WorkCoordinates,
    WorkItem,
    inspect_host,
)

__all__ = [
    "CODING_QUEUE_CAPACITY",
    "CRITERION_AUDIT",
    "CRITERION_CHECKS",
    "CRITERION_REVISION",
    "MISSION_DESIRED_RESULT",
    "MISSION_RESOURCE_KIND",
    "MISSION_RESOURCE_NAMESPACE",
    "CodingMissionRuntime",
    "DispatchedWork",
    "MissionAdmission",
    "MissionAuthorityRefused",
    "MissionCapacityRefused",
    "MissionConflictRefused",
    "MissionCorruptRefused",
    "MissionDependencyRefused",
    "MissionHeartbeat",
    "MissionRepairRetryExhausted",
    "MissionRouteError",
    "MissionStaleFenceRefused",
    "MissionUnsupportedRefused",
    "synthesize_envelope",
]

#: One durable queue depth for every process that shares a state root. It is a
#: constant rather than a mirror of `max_workers`/`max_queued`, because the
#: kernel records capacity durably and refuses a second store that asks for a
#: different one: two coding services configured with different local
#: concurrency must still agree about how deep the shared queue is.
CODING_QUEUE_CAPACITY = DEFAULT_QUEUE_CAPACITY

#: The resource coordinate one coding job holds while it runs. `identity` is the
#: workspace digest the service already computes, never the worktree path, so
#: exclusion is enforced without publishing where anybody's code lives.
MISSION_RESOURCE_NAMESPACE = "flyto.coding"
MISSION_RESOURCE_KIND = "workspace"

#: The three identified criteria of a synthesized coding mission. They are
#: constants because they are read back on acceptance to build evidence: a
#: criterion whose id was generated per job could never be answered by a later
#: process holding only the mission.
CRITERION_REVISION = "implementation-revision"
CRITERION_CHECKS = "pinned-required-checks"
CRITERION_AUDIT = "codex-audit"

MISSION_DESIRED_RESULT = (
    "an attributable verified revision of this workspace, accepted by an "
    "independent Codex audit"
)
_CRITERION_STATEMENTS: Tuple[Tuple[str, str], ...] = (
    (
        CRITERION_REVISION,
        "the host hashed an attributable implementation revision over exactly the "
        "files this job's rounds changed",
    ),
    (
        CRITERION_CHECKS,
        "the required checks pinned at admission ran and passed for that revision, "
        "under the contract the job was admitted against",
    ),
    (
        CRITERION_AUDIT,
        "an independent Codex audit accepted that exact revision, with no "
        "unresolved host blocker",
    ),
)

#: Marker appended when the caller's message does not fit the kernel's bounded
#: prose field. The objective stays the caller's own words; it is never
#: paraphrased, summarised or replaced, only cut, and the cut is visible.
_TRUNCATED = " [...]"
#: Who a blocked or deferred coding work item is left with. A token, because the
#: kernel stores closure owners as identifiers rather than as prose.
MISSION_CLOSURE_OWNER = "flyto-coding-host"
#: How far ahead a terminal coding closure points its revisit: one hour. Long
#: enough that a restart storm does not make every item instantly overdue, short
#: enough that a blocked worktree is not forgotten for a day.
_REVISIT_HORIZON_SECONDS = 3600

_WORKER_MAX_CHARS = 128


# --------------------------------------------------------------------------
# refusals
# --------------------------------------------------------------------------


class MissionRouteError(RuntimeError):
    """A mission-lane refusal, carrying a stable machine code.

    Deliberately not a `CodingServiceError`. This module is imported *by* the
    service, so it cannot import the service's exception base without a cycle;
    the service translates these into its own typed refusals, which is also the
    only place that knows which of them are retryable in its own vocabulary.
    """

    code = "mission_unavailable"
    retryable = False


class MissionUnsupportedRefused(MissionRouteError):
    """This host cannot provide the primitives the mission kernel requires."""

    code = "mission_unsupported"


class MissionCapacityRefused(MissionRouteError):
    """The durable mission queue is full, or its receipts are unacknowledged."""

    code = "mission_capacity"
    retryable = True


class MissionConflictRefused(MissionRouteError):
    """Another live worker holds the authority this call needed."""

    code = "mission_conflict"
    retryable = True


class MissionStaleFenceRefused(MissionRouteError):
    """This worker's fencing era ended; a later dispatch owns the work item."""

    code = "mission_stale_fence"


class MissionDependencyRefused(MissionRouteError):
    """The mission graph refused this lineage, dependency or closure."""

    code = "mission_dependency"


class MissionRepairRetryExhausted(MissionDependencyRefused):
    """Both bounded publication children were accounted without running."""

    code = "mission_repair_retry_exhausted"


class MissionCorruptRefused(MissionRouteError):
    """Durable mission state did not validate, so nothing was trusted."""

    code = "mission_corrupt"


class MissionAuthorityRefused(MissionRouteError):
    """Tenant, mission, work item, worker and live handle did not all agree."""

    code = "mission_authority"


def _translate(exc: MissionError) -> MissionRouteError:
    """Map one kernel failure onto this layer's closed refusal vocabulary."""

    if isinstance(exc, MissionUnsupported):
        return MissionUnsupportedRefused(str(exc))
    if isinstance(exc, MissionCapacityExceeded):
        return MissionCapacityRefused(str(exc))
    if isinstance(exc, MissionStaleFence):
        return MissionStaleFenceRefused(str(exc))
    if isinstance(exc, MissionUnauthorized):
        return MissionAuthorityRefused(str(exc))
    if isinstance(exc, MissionConflict):
        return MissionConflictRefused(str(exc))
    if isinstance(exc, MissionCorrupt):
        return MissionCorruptRefused(str(exc))
    return MissionDependencyRefused(str(exc))


@contextmanager
def _translated() -> Iterator[None]:
    try:
        yield
    except MissionError as exc:
        raise _translate(exc) from exc


# --------------------------------------------------------------------------
# synthesis
# --------------------------------------------------------------------------


def _bounded_objective(message: str) -> str:
    """Project one request message into the kernel's bounded prose field.

    The kernel stores a single printable line, so a multi-line task description
    is folded on whitespace rather than rejected - and it is folded, not
    rewritten: the words are the caller's, in the caller's order. Truncation is
    marked so a reader is never shown a cut sentence as if it were the whole
    objective.
    """

    folded = " ".join(str(message or "").split())
    if not folded:
        # A mission needs an objective and this one has nothing to say. The
        # request contract already refuses an empty message, so this is the
        # unreachable-by-construction floor rather than a silent default.
        return "an unstated coding task"
    if len(folded) <= MAX_TEXT_CHARS:
        return folded
    return folded[: MAX_TEXT_CHARS - len(_TRUNCATED)] + _TRUNCATED


def coding_scope(workspace_sha256: str) -> str:
    """The fairness scope one workspace's coding work shares.

    A bounded prefix of the workspace digest, so the kernel's least-recently-
    served rotation is per worktree - and so a fleet view can group by worktree
    without ever holding one. Never the path, and never the caller's prose.
    """

    return "ws-" + str(workspace_sha256 or "")[:32]


def synthesize_envelope(message: str, *, workspace_sha256: str) -> CodingMissionEnvelope:
    """Build the mission a coding job serves when its caller named none.

    This synthesis lives here and only here. `MissionStore` stays workload
    neutral: it never learns that "revision", "checks" and "audit" are the
    things a coding job has to prove, and it would accept an entirely different
    triple from a pentest or robotics adapter without changing a line.
    """

    return CodingMissionEnvelope(
        scope=coding_scope(workspace_sha256),
        objective=_bounded_objective(message),
        desired_result=MISSION_DESIRED_RESULT,
        acceptance_criteria=tuple(
            AcceptanceCriterion(identifier, statement)
            for identifier, statement in _CRITERION_STATEMENTS
        ),
        lane=MISSION_LANE_PRIMARY,
    )


# --------------------------------------------------------------------------
# value types
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class MissionAdmission:
    """What admission placed: one mission, one work item, one projection."""

    mission_id: str
    work_item_id: str
    #: The contract this placement was made against. `None` for a repair child,
    #: which is placed from its parent's projection rather than from prose the
    #: job record deliberately does not keep.
    envelope: Optional[CodingMissionEnvelope]
    projection: CodingMissionProjection


@dataclass(frozen=True)
class DispatchedWork:
    """One store-selected work item, resolved back to its private owner.

    The handle is the authority. `tenant_ref` and `job_id` come from the work
    item's own coordinates, so the owner is read from durable kernel state
    rather than from anything the dispatching process happened to remember -
    which is what lets a second process run work a first one queued.
    """

    handle: DispatchHandle
    tenant_ref: str
    job_id: str
    workspace_sha256: str

    @property
    def work_item_id(self) -> str:
        return self.handle.work_item_id

    @property
    def mission_id(self) -> str:
        return self.handle.mission_id

    @property
    def fence(self) -> int:
        return self.handle.fence

    @property
    def is_repair(self) -> bool:
        return self.handle.lane == LANE_REPAIR


class MissionHeartbeat:
    """Periodic liveness for one dispatch, stopped before the item closes.

    Heartbeats are observability and nothing else - no lease moves, no fence is
    burned, and no other worker becomes able to take this item because one
    stopped arriving. Failures are swallowed on purpose: a heartbeat that could
    not be written must never be the reason a real implementation round dies.
    """

    __slots__ = ("_handle", "_interval", "_stop", "_thread", "_beats")

    def __init__(self, handle: DispatchHandle, *, interval: float = 5.0) -> None:
        self._handle = handle
        self._interval = max(0.05, float(interval))
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._beats = 0

    @property
    def beats(self) -> int:
        """How many heartbeats this helper has successfully recorded."""

        return self._beats

    def beat(self) -> None:
        """Record one phase-boundary heartbeat, synchronously."""

        try:
            self._handle.heartbeat()
        except (MissionError, OSError):
            return
        self._beats += 1

    def start(self) -> "MissionHeartbeat":
        thread = threading.Thread(
            target=self._run, name="flyto-mission-heartbeat", daemon=True,
        )
        self._thread = thread
        thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=self._interval * 2)
            self._thread = None

    def __enter__(self) -> "MissionHeartbeat":
        return self.start()

    def __exit__(self, *_exc: object) -> None:
        self.stop()

    def _run(self) -> None:
        while not self._stop.wait(self._interval):
            self.beat()


# --------------------------------------------------------------------------
# runtime
# --------------------------------------------------------------------------


class CodingMissionRuntime:
    """One coding service's bound view of the shared mission store.

    Constructing this creates nothing and touches no filesystem, exactly like
    the kernel it wraps, so a host that cannot support missions is discovered by
    asking rather than by damaging a directory.
    """

    def __init__(
        self,
        state_root: Path,
        *,
        worker: str,
        queue_capacity: int = CODING_QUEUE_CAPACITY,
    ) -> None:
        if not isinstance(worker, str) or not worker or len(worker) > _WORKER_MAX_CHARS:
            raise ValueError("mission worker must be a bounded identifier")
        if any(character.isspace() for character in worker) or not worker.isprintable():
            raise ValueError("mission worker must be a bounded identifier")
        self.worker = worker
        self.state_root = Path(state_root)
        self._store = MissionStore(self.state_root, queue_capacity=queue_capacity)

    # -- availability ---------------------------------------------------

    @property
    def store(self) -> MissionStore:
        """The durable store. Exposed for tests and reconciliation, not policy."""

        return self._store

    @staticmethod
    def _persisted_work_item(
        state_root: Path, work_item_id: str,
    ) -> Optional[WorkItem]:
        """Read one durable item without constructing an execution worker."""

        return MissionStore(Path(state_root)).get_work_item(work_item_id)

    @staticmethod
    def supported() -> bool:
        """Whether this host has the primitives the kernel refuses to emulate."""

        return inspect_host().supported

    def require_supported(self) -> None:
        if not self.supported():
            raise MissionUnsupportedRefused(
                "this host cannot provide the mission store's required primitives",
            )

    # -- admission ------------------------------------------------------

    def admit(
        self,
        *,
        tenant_ref: str,
        job_id: str,
        workspace_sha256: str,
        envelope: Optional[CodingMissionEnvelope],
        message: str,
        repository_sha256s: Sequence[str] = (),
    ) -> MissionAdmission:
        """Create or validate the mission, then place exactly one work item.

        Both operations carry deterministic keys derived from this job's own id,
        so a submit that was interrupted after publishing - and is retried under
        the same job - reconciles to the mission and the work item it already
        created rather than minting a second pair. Placing two work items for
        one job is the failure this makes unrepresentable: the queue would then
        contain work nobody would ever close.
        """

        self.require_supported()
        if envelope is None:
            envelope = synthesize_envelope(message, workspace_sha256=workspace_sha256)
        with _translated():
            mission = self._resolve_mission(envelope, job_id)
            item = self._store.submit_work_item(
                mission.mission_id,
                operation=self._key("place", job_id),
                coordinates=self._coordinates(tenant_ref, job_id, workspace_sha256),
                resources=tuple(
                    self.resource(value)
                    for value in (tuple(repository_sha256s) or (workspace_sha256,))
                ),
                lane=envelope.lane,
                priority=envelope.priority,
                root=envelope.is_root,
                parent_id=envelope.parent_id,
                return_to_id=envelope.return_to_id,
                depends_on_ids=envelope.depends_on_ids,
            )
        self._forget(self._key("mission", job_id))
        self._forget(self._key("place", job_id))
        return MissionAdmission(
            mission_id=mission.mission_id,
            work_item_id=item.work_item_id,
            envelope=envelope,
            projection=envelope.projection(
                mission_id=mission.mission_id,
                work_item_id=item.work_item_id,
                status=MISSION_STATUS_READY,
                mission_status=mission.status,
            ),
        )

    def _resolve_mission(self, envelope: CodingMissionEnvelope, job_id: str) -> Mission:
        """Honour an existing mission's contract, or record a new immutable one.

        A named mission is *validated*, never amended. The stored contract is
        re-addressed through the same domain-separated digest the envelope uses,
        so attaching to a mission whose objective, desired result, scope or
        criteria differ by a single character is a refusal - which is the only
        way lineage into an existing graph can be safe.
        """

        if envelope.mission_id is not None:
            mission = self._store.get_mission(envelope.mission_id)
            stored = mission_axis_sha256(
                mission.scope,
                mission.objective,
                mission.desired_result,
                mission.acceptance_criteria,
            )
            if stored != envelope.main_axis_sha256:
                raise MissionDependencyRefused(
                    "the named mission states a different immutable contract",
                )
            if mission.status != MISSION_OPEN:
                raise MissionDependencyRefused("the named mission is already completed")
            return mission
        return self._store.create_mission(
            operation=self._key("mission", job_id),
            scope=envelope.scope,
            objective=envelope.objective,
            desired_result=envelope.desired_result,
            acceptance_criteria=envelope.acceptance_criteria,
        )

    def submit_repair(
        self,
        *,
        tenant_ref: str,
        job_id: str,
        workspace_sha256: str,
        projection: CodingMissionProjection,
        round_index: int,
        repository_sha256s: Sequence[str] = (),
    ) -> MissionAdmission:
        """Place one repair child under the same mission, in the repair lane.

        The key binds this job and this repair round, so a rework that is
        retried - by the same worker or by a second one that read the same
        record - reconciles to the child that already exists. Duplicate repair
        items are what would let one audit produce two implementation rounds in
        the same session.

        The child depends on the parent it repairs, and returns to the mission's
        root: the route home always points at the main axis.

        The parent is named by the job's *stored* projection rather than by an
        envelope, because the mission's prose is not kept in a job record and
        must not be reconstructed from anything a later round wrote. Scope,
        priority, criteria and the main-axis digest all come from the projection
        the admission produced, so the child can never describe a different
        contract from its parent.
        """

        self.require_supported()
        mission_id = projection.mission_id
        parent_id = projection.work_item_id
        key = self._key("repair-{}".format(int(round_index)), job_id)
        with _translated():
            parent = self._store.get_work_item(parent_id)
            if parent.mission_id != mission_id:
                raise MissionAuthorityRefused(
                    "the repaired work item belongs to a different mission",
                )
            self._require_owner(parent, tenant_ref, job_id)
            root_id = self._root_id(parent)
            item = self._store.submit_work_item(
                mission_id,
                operation=key,
                coordinates=self._coordinates(tenant_ref, job_id, workspace_sha256),
                resources=tuple(
                    self.resource(value)
                    for value in (tuple(repository_sha256s) or (workspace_sha256,))
                ),
                lane=LANE_REPAIR,
                priority=projection.priority,
                root=False,
                parent_id=parent_id,
                return_to_id=root_id,
                depends_on_ids=(parent_id,),
            )
        self._forget(key)
        return MissionAdmission(
            mission_id=mission_id,
            work_item_id=item.work_item_id,
            envelope=None,
            projection=CodingMissionProjection(
                mission_id=mission_id,
                scope=projection.scope,
                work_item_id=item.work_item_id,
                main_axis_sha256=projection.main_axis_sha256,
                criteria_ids=projection.criteria_ids,
                lane=LANE_REPAIR,
                priority=projection.priority,
                status=MISSION_STATUS_READY,
                mission_status=MISSION_OPEN,
                parent_id=parent_id,
                return_to_id=root_id,
            ),
        )

    def submit_repair_retry(
        self,
        *,
        tenant_ref: str,
        job_id: str,
        workspace_sha256: str,
        projection: CodingMissionProjection,
        round_index: int,
        retry_index: int,
        repository_sha256s: Sequence[str] = (),
    ) -> MissionAdmission:
        """Place one bounded retry after a repair route failed before provider.

        The failed repair child remains immutable and accounted as blocked.
        The retry records it as lineage, but depends on the fixed main-axis
        item: depending on a blocked item would correctly make the retry
        unrunnable in the mission kernel.
        """

        self.require_supported()
        mission_id = projection.mission_id
        blocked_id = projection.work_item_id
        key = self._key(
            "repair-route-retry-{}-{}".format(int(round_index), int(retry_index)),
            job_id,
        )
        with _translated():
            blocked = self._store.get_work_item(blocked_id)
            if blocked.mission_id != mission_id or blocked.lane != LANE_REPAIR:
                raise MissionAuthorityRefused(
                    "the blocked repair belongs to a different mission lane",
                )
            self._require_owner(blocked, tenant_ref, job_id)
            if (
                blocked.status != MISSION_STATUS_CLOSED
                or blocked.disposition != DISPOSITION_BLOCKED
            ):
                raise MissionDependencyRefused(
                    "the repair route retry requires one accounted blocked child",
                )
            root_id = self._root_id(blocked)
            root = self._store.get_work_item(root_id)
            self._require_owner(root, tenant_ref, job_id)
            if (
                not root.is_root
                or root.status != MISSION_STATUS_CLOSED
                or root.disposition != DISPOSITION_FIXED
            ):
                raise MissionDependencyRefused(
                    "the repair route retry requires the fixed audited main axis",
                )
            coordinates = self._coordinates(tenant_ref, job_id, workspace_sha256)
            resources = tuple(
                self.resource(value)
                for value in (tuple(repository_sha256s) or (workspace_sha256,))
            )
            item = self._store.submit_work_item(
                mission_id,
                operation=key,
                coordinates=coordinates,
                resources=resources,
                lane=LANE_REPAIR,
                priority=projection.priority,
                root=False,
                parent_id=blocked_id,
                return_to_id=root_id,
                depends_on_ids=(root_id,),
            )
            if not self._repair_retry_shape(
                item,
                mission_id=mission_id,
                parent_id=blocked_id,
                return_to_id=root_id,
                coordinates=coordinates,
                resources=resources,
                priority=projection.priority,
            ):
                raise MissionDependencyRefused(
                    "the persisted repair route retry changed identity",
                )
            if item.status == MISSION_STATUS_CLOSED:
                if not self._retry_was_host_deferred(item, job_id):
                    raise MissionDependencyRefused(
                        "the persisted repair route retry is no longer runnable",
                    )
                orphaned_retry_id = item.work_item_id
                reconcile_key = self._key(
                    "repair-route-retry-reconcile-{}-{}".format(
                        int(round_index), int(retry_index),
                    ),
                    job_id,
                )
                item = self._store.submit_work_item(
                    mission_id,
                    operation=reconcile_key,
                    coordinates=coordinates,
                    resources=resources,
                    lane=LANE_REPAIR,
                    priority=projection.priority,
                    root=False,
                    parent_id=orphaned_retry_id,
                    return_to_id=root_id,
                    depends_on_ids=(root_id,),
                )
                if not self._repair_retry_shape(
                    item,
                    mission_id=mission_id,
                    parent_id=orphaned_retry_id,
                    return_to_id=root_id,
                    coordinates=coordinates,
                    resources=resources,
                    priority=projection.priority,
                ):
                    raise MissionDependencyRefused(
                        "the reconciled repair route retry changed identity",
                    )
            if (
                item.status == MISSION_STATUS_CLOSED
                and self._retry_was_host_deferred(item, job_id)
            ):
                raise MissionRepairRetryExhausted(
                    "the bounded repair route publication recovery was exhausted",
                )
            if item.status != MISSION_STATUS_READY or item.attempts != 0:
                raise MissionDependencyRefused(
                    "the repair route retry is not durably ready",
                )
        return MissionAdmission(
            mission_id=mission_id,
            work_item_id=item.work_item_id,
            envelope=None,
            projection=CodingMissionProjection(
                mission_id=mission_id,
                scope=projection.scope,
                work_item_id=item.work_item_id,
                main_axis_sha256=projection.main_axis_sha256,
                criteria_ids=projection.criteria_ids,
                lane=LANE_REPAIR,
                priority=projection.priority,
                status=MISSION_STATUS_READY,
                mission_status=MISSION_OPEN,
                parent_id=item.parent_id,
                return_to_id=root_id,
            ),
        )

    @staticmethod
    def _repair_retry_shape(
        item: WorkItem,
        *,
        mission_id: str,
        parent_id: str,
        return_to_id: str,
        coordinates: WorkCoordinates,
        resources: Tuple[MissionResource, ...],
        priority: int,
    ) -> bool:
        """Bind a recalled mission operation to the retry we requested."""

        return (
            item.mission_id == mission_id
            and item.lane == LANE_REPAIR
            and item.priority == priority
            and not item.is_root
            and item.parent_id == parent_id
            and item.return_to_id == return_to_id
            and item.coordinates == coordinates
            and item.resources == resources
            and item.depends_on_ids == (return_to_id,)
        )

    @staticmethod
    def _retry_was_host_deferred(item: WorkItem, job_id: str) -> bool:
        """Recognize only the orphan accounting produced by this host."""

        closure = item.closure
        return bool(
            item.attempts == 1
            and closure is not None
            and closure.disposition == DISPOSITION_DEFERRED
            and closure.rationale
            == "the host could not run this work item: job_not_runnable"
            and closure.risk
            == (
                "the mission's objective is not reached and this workspace's "
                "next round has to be submitted again"
            )
            and closure.evidence_refs
            == ("reason-job_not_runnable", "job-{}".format(job_id))
            and closure.owner == MISSION_CLOSURE_OWNER
        )

    def acknowledge_repair_retry(
        self,
        *,
        job_id: str,
        round_index: int,
        retry_index: int,
    ) -> None:
        """Forget retry receipts only after the owner record names the child."""

        for kind in (
            "repair-route-retry-{}-{}".format(round_index, retry_index),
            "repair-route-retry-reconcile-{}-{}".format(
                round_index, retry_index,
            ),
        ):
            self._forget(self._key(kind, job_id))

    def _root_id(self, item: WorkItem) -> str:
        """Walk lineage to the main axis, bounded by the kernel's own depth."""

        current = item
        for _ in range(64):
            if current.is_root:
                return current.work_item_id
            parent_id = current.parent_id
            if not parent_id:
                break
            current = self._store.get_work_item(parent_id)
        raise MissionDependencyRefused("this work item has no route to a main axis")

    # -- dispatch -------------------------------------------------------

    @contextmanager
    def dispatch(self) -> Iterator[Optional[DispatchedWork]]:
        """Take the next work item the *store* chose, or yield `None`.

        Order is the store's, not this process's. Nothing here filters
        candidates by which job this instance happens to have submitted, which
        is exactly what keeps two services sharing a state root on one queue
        instead of two - and what keeps a per-submit executor from deciding the
        order by accident of thread scheduling.
        """

        self.require_supported()
        key = "cmd-" + uuid.uuid4().hex
        try:
            with _translated():
                dispatcher = self._store.dispatch(operation=key, worker=self.worker)
                with dispatcher as handle:
                    if handle is None:
                        yield None
                    else:
                        yield self._resolve(handle)
        finally:
            self._forget(key)

    def _resolve(self, handle: DispatchHandle) -> DispatchedWork:
        coordinates = handle.coordinates
        return DispatchedWork(
            handle=handle,
            tenant_ref=coordinates.project,
            job_id=coordinates.location,
            workspace_sha256=coordinates.repository,
        )

    def queue_state(self) -> Tuple[int, int]:
        """How many work items are waiting, and how many are being run.

        Read from the store rather than from anything this process remembers, so
        a second service sharing the state root sees the same two numbers. The
        kernel's ``queue_depth`` is already exactly the count of items in
        ``ready`` - items a dispatch could take - and ``dispatched`` is the count
        some worker, in any process, currently holds a lease on. Subtracting one
        from the other would report an idle queue whenever a single ready item
        sat behind a single running one, which is precisely the handoff a
        two-instance queue depends on.
        """

        if not self.supported():
            return (0, 0)
        try:
            metrics = self._store.metrics()
        except MissionError:
            return (0, 0)
        return (max(0, metrics.queue_depth), max(0, metrics.dispatched))

    def ready_work(self) -> int:
        """How many work items a dispatch could take right now."""

        return self.queue_state()[0]

    # -- closure --------------------------------------------------------

    def close_fixed(
        self,
        work: DispatchedWork,
        *,
        tenant_ref: str,
        job_id: str,
        mission_id: str,
        work_item_id: str,
    ) -> WorkItem:
        """Close one work item as delivered, on this live handle's authority."""

        self._authorize(work, tenant_ref, job_id, mission_id, work_item_id)
        with _translated():
            item = work.handle.close(
                Closure(disposition=DISPOSITION_FIXED),
                operation=self._key("close", job_id, work_item_id),
            )
        self._forget(self._key("close", job_id, work_item_id))
        return item

    def close_accounted(
        self,
        work: DispatchedWork,
        *,
        tenant_ref: str,
        job_id: str,
        mission_id: str,
        work_item_id: str,
        disposition: str,
        rationale: str,
        risk: str,
        evidence_refs: Sequence[str],
        owner: str = MISSION_CLOSURE_OWNER,
        revisit_in_seconds: int = _REVISIT_HORIZON_SECONDS,
    ) -> WorkItem:
        """Close one work item short of delivery, with the whole accounting.

        `blocked` and `deferred` are legitimate outcomes and are exactly the
        ones a silent skip would hide behind, so the kernel demands a rationale,
        a risk, evidence refs, a named owner and a future revisit - and this
        layer supplies all five rather than letting a terminal coding failure
        leave the queue quietly.
        """

        if disposition not in (DISPOSITION_BLOCKED, DISPOSITION_DEFERRED):
            raise MissionAuthorityRefused(
                "an accounted closure is blocked or deferred, never fixed",
            )
        self._authorize(work, tenant_ref, job_id, mission_id, work_item_id)
        refs = tuple(dict.fromkeys(str(ref) for ref in evidence_refs if ref))
        with _translated():
            item = work.handle.close(
                Closure(
                    disposition=disposition,
                    rationale=rationale,
                    risk=risk,
                    evidence_refs=refs or ("no-evidence-recorded",),
                    owner=owner,
                    revisit_at=int(time.time()) + int(revisit_in_seconds),
                ),
                operation=self._key("close", job_id, work_item_id),
            )
        self._forget(self._key("close", job_id, work_item_id))
        return item

    # -- completion -----------------------------------------------------

    def complete(
        self,
        *,
        tenant_ref: str,
        job_id: str,
        mission_id: str,
        work_item_id: str,
        evidence: Mapping[str, str],
    ) -> Optional[Mission]:
        """Complete a mission whose root has been accepted, or decline to.

        Three refusals live here rather than in the caller. A side item's accept
        never completes anything: the branch is not the axis. A mission with any
        open work item is left open, because a sibling that is still running has
        not been accounted for and the kernel would - correctly - refuse. And
        evidence must cover exactly the criteria the mission actually declared,
        which is why it is built from the stored contract rather than from
        whatever the accepting caller believed the criteria were.
        """

        self.require_supported()
        with _translated():
            item = self._store.get_work_item(work_item_id)
            self._require_owner(item, tenant_ref, job_id)
            if item.mission_id != mission_id:
                raise MissionAuthorityRefused(
                    "the accepted work item belongs to a different mission",
                )
            if not item.is_root:
                # A side item hands control back to its recorded ancestor. It
                # closes, and it says nothing at all about the main axis.
                return None
            snapshot = self._store.snapshot(
                mission_id=mission_id, limit=MAX_SNAPSHOT_ITEMS,
            )
            if snapshot.truncated or any(
                summary.status != MISSION_STATUS_CLOSED
                for summary in snapshot.work_items
            ):
                # Truncated means "this view did not see every item", which is
                # never a basis for declaring the whole graph finished.
                return None
            mission = self._store.get_mission(mission_id)
            if mission.status != MISSION_OPEN:
                return mission
            supplied = self._evidence(mission, evidence)
            completed = self._store.complete_mission(
                mission_id, supplied, operation=self._key("complete", mission_id),
            )
        self._forget(self._key("complete", mission_id))
        return completed

    @staticmethod
    def _evidence(mission: Mission, offered: Mapping[str, str]) -> Dict[str, str]:
        """Answer exactly the mission's own criteria, and nothing else.

        A criterion the caller has no specific evidence for still gets the
        accepted revision, because that is what an acceptance actually proves.
        A key the mission never declared is dropped rather than passed through:
        the kernel would reject the whole completion for it, and an unknown
        criterion is a caller bug, not a reason to strand a finished mission.
        """

        fallback = str(offered.get(CRITERION_REVISION) or "")
        supplied: Dict[str, str] = {}
        for criterion in mission.criteria_ids:
            value = str(offered.get(criterion) or "") or fallback
            if not value:
                raise MissionDependencyRefused(
                    "acceptance evidence is missing for a declared criterion",
                )
            supplied[criterion] = value
        return supplied

    # -- reconciliation -------------------------------------------------

    def reclaim(self, work_item_id: str) -> bool:
        """Requeue a dispatched item whose execution lease is *provably* free.

        Evidence, never age. The kernel reclaims exactly when it can take the
        lease itself, which is impossible while any live process holds it, so a
        worker that is slow, paused or simply quiet is never stolen from. A live
        holder raises `MissionConflictRefused`, and the caller leaves it alone.
        """

        self.require_supported()
        key = self._key("reclaim", work_item_id, uuid.uuid4().hex[:16])
        with _translated():
            reclaimed = self._store.reclaim(work_item_id, operation=key)
        self._forget(key)
        return reclaimed

    def work_item(self, work_item_id: str) -> Optional[WorkItem]:
        """Read one work item, or `None` when the store has no such row."""

        if not self.supported():
            return None
        try:
            return self._store.get_work_item(work_item_id)
        except MissionError:
            return None

    def is_workspace_claimed(self, workspace_sha256: str) -> bool:
        """Whether some running work item currently holds this worktree."""

        if not self.supported():
            return False
        try:
            return self._store.is_claimed(self.resource(workspace_sha256))
        except MissionError:
            return False

    # -- observability --------------------------------------------------

    def fleet(self, *, limit: int = 50) -> Dict[str, Any]:
        """A bounded, snapshot-only, secret-free view of every mission here.

        Observable and never actionable: there is no identifier in this payload
        that any operation on this runtime will accept as authority, and nothing
        in it is ever fed back to a model as conversational context. It carries
        no objective, no criterion statement, no rationale, no evidence value,
        no coordinate, no worker and no path, because the kernel's own snapshot
        carries none of those either.
        """

        if not self.supported():
            return {"available": False, "missions": [], "work_items": [], "truncated": False}
        try:
            snapshot = self._store.snapshot(limit=limit)
        except MissionError as exc:
            return {
                "available": False,
                "error_code": _translate(exc).code,
                "missions": [],
                "work_items": [],
                "truncated": False,
            }
        return {
            "available": True,
            "truncated": bool(snapshot.truncated),
            "missions": [
                {
                    "mission_id": summary.mission_id,
                    "scope": summary.scope,
                    "status": summary.status,
                    "criteria_ids": list(summary.criteria_ids),
                    "work_items": summary.work_items,
                    "closed_work_items": summary.closed_work_items,
                }
                for summary in snapshot.missions
            ],
            "work_items": [
                {
                    "work_item_id": item.work_item_id,
                    "mission_id": item.mission_id,
                    "scope": item.scope,
                    "lane": item.lane,
                    "priority": item.priority,
                    "is_root": item.is_root,
                    "parent_id": item.parent_id or "",
                    "return_to_id": item.return_to_id or "",
                    "status": item.status,
                    "disposition": item.disposition or "",
                    "fence": item.fence,
                    "attempts": item.attempts,
                    "heartbeats": item.heartbeats,
                    "resource_count": item.resource_count,
                    "dependency_count": item.dependency_count,
                }
                for item in snapshot.work_items
            ],
            "metrics": {
                "queue_capacity": snapshot.metrics.queue_capacity,
                "queue_depth": snapshot.metrics.queue_depth,
                "dispatched": snapshot.metrics.dispatched,
                "closed_fixed": snapshot.metrics.closed_fixed,
                "closed_deferred": snapshot.metrics.closed_deferred,
                "closed_blocked": snapshot.metrics.closed_blocked,
                "missions_open": snapshot.metrics.missions_open,
                "missions_completed": snapshot.metrics.missions_completed,
                "conflicts": snapshot.metrics.conflicts,
                "stale_fence_rejects": snapshot.metrics.stale_fence_rejects,
                "capacity_rejects": snapshot.metrics.capacity_rejects,
            },
        }

    def scheduler_order(self, *, limit: int = 50) -> Tuple[str, ...]:
        """Current ready-work preference, read-only and non-authoritative."""

        if not self.supported():
            return ()
        try:
            return self._store.scheduler_order(limit=limit)
        except MissionError:
            return ()

    def context(
        self, *, tenant_ref: str, job_id: str, work_item_id: str,
    ) -> Dict[str, Any]:
        """Full mission context for one work item, for its owning tenant only.

        This is the surface that *does* carry the contract's prose, so it is
        bound to the tenant reference and the job id recorded in the work item's
        own coordinates. A caller holding only identifiers from a fleet view
        cannot reach it, which is the whole point of keeping them separate.
        """

        self.require_supported()
        with _translated():
            item = self._store.get_work_item(work_item_id)
            self._require_owner(item, tenant_ref, job_id)
            mission = self._store.get_mission(item.mission_id)
        return {
            "mission_id": mission.mission_id,
            "scope": mission.scope,
            "objective": mission.objective,
            "desired_result": mission.desired_result,
            "acceptance_criteria": [
                {"id": criterion.id, "statement": criterion.statement}
                for criterion in mission.acceptance_criteria
            ],
            "status": mission.status,
            "work_item_id": item.work_item_id,
            "work_item_status": item.status,
            "lane": item.lane,
            "fence": item.fence,
            "attempts": item.attempts,
            "heartbeats": item.heartbeats,
            "disposition": item.disposition or "",
        }

    # -- projection -----------------------------------------------------

    @staticmethod
    def project(
        envelope: CodingMissionEnvelope,
        *,
        mission_id: str,
        work_item_id: str,
        status: str,
        disposition: str = "",
        mission_status: str = MISSION_OPEN,
        parent_id: str = "",
        return_to_id: str = "",
        returned_to_main_axis: bool = False,
    ) -> CodingMissionProjection:
        """Build the receipt-side projection of one placed work item."""

        return CodingMissionProjection(
            mission_id=mission_id,
            scope=envelope.scope,
            work_item_id=work_item_id,
            main_axis_sha256=envelope.main_axis_sha256,
            criteria_ids=envelope.criteria_ids,
            lane=envelope.lane if not parent_id else LANE_REPAIR,
            priority=envelope.priority,
            status=status,
            disposition=disposition,
            mission_status=mission_status,
            parent_id=parent_id,
            return_to_id=return_to_id,
            returned_to_main_axis=returned_to_main_axis,
        )

    @staticmethod
    def advance(
        projection: Mapping[str, Any],
        *,
        status: str,
        disposition: str = "",
        mission_status: Optional[str] = None,
        returned_to_main_axis: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Move one stored projection to a new lifecycle state, revalidating it.

        The stored mapping goes back through the closed decoder before anything
        is changed, so a record edited in place cannot smuggle a field - and the
        result is re-encoded through the same decoder, so it can never publish
        one either.
        """

        current = CodingMissionProjection.from_mapping(projection)
        updated = CodingMissionProjection(
            mission_id=current.mission_id,
            scope=current.scope,
            work_item_id=current.work_item_id,
            main_axis_sha256=current.main_axis_sha256,
            criteria_ids=current.criteria_ids,
            lane=current.lane,
            priority=current.priority,
            status=status,
            disposition=disposition,
            mission_status=(
                current.mission_status if mission_status is None else mission_status
            ),
            parent_id=current.parent_id,
            return_to_id=current.return_to_id,
            returned_to_main_axis=(
                current.returned_to_main_axis
                if returned_to_main_axis is None
                else returned_to_main_axis
            ),
        )
        return updated.to_mapping()

    # -- internals ------------------------------------------------------

    @staticmethod
    def resource(workspace_sha256: str) -> MissionResource:
        """The canonical worktree claim: a digest, never a path."""

        return MissionResource(
            namespace=MISSION_RESOURCE_NAMESPACE,
            kind=MISSION_RESOURCE_KIND,
            identity=str(workspace_sha256),
        )

    @staticmethod
    def _coordinates(
        tenant_ref: str, job_id: str, workspace_sha256: str,
    ) -> WorkCoordinates:
        """Private tenant and job identity, bounded, as three opaque tokens.

        The kernel never branches on what these spell, and a snapshot never
        carries them. They exist so a dispatch in *any* process can find the
        private record that owns the work - which is the only way store order
        can be authoritative across processes.
        """

        return WorkCoordinates(
            project=str(tenant_ref),
            repository=str(workspace_sha256),
            location=str(job_id),
        )

    @staticmethod
    def _require_owner(item: WorkItem, tenant_ref: str, job_id: str) -> None:
        if (
            item.coordinates.project != str(tenant_ref)
            or item.coordinates.location != str(job_id)
        ):
            raise MissionAuthorityRefused(
                "this work item is not owned by the named tenant and job",
            )

    def _authorize(
        self,
        work: DispatchedWork,
        tenant_ref: str,
        job_id: str,
        mission_id: str,
        work_item_id: str,
    ) -> None:
        """Demand tenant, mission, work item, worker and live handle at once.

        Every one of these is checkable and every one of them is checked. A
        handle constructed anywhere but a real dispatch has no lease to be
        authorised against, so it cannot reach this at all; what this adds is
        that a *real* handle cannot be used to close somebody else's work.
        """

        if not isinstance(work, DispatchedWork) or not isinstance(
            work.handle, DispatchHandle,
        ):
            raise MissionAuthorityRefused("a live dispatch handle is required")
        if work.handle.closed:
            raise MissionAuthorityRefused("this work item has already been closed")
        if (
            work.tenant_ref != str(tenant_ref)
            or work.job_id != str(job_id)
            or work.mission_id != str(mission_id)
            or work.work_item_id != str(work_item_id)
        ):
            raise MissionAuthorityRefused(
                "tenant, job, mission and work item must all name this dispatch",
            )
        if work.handle.worker != self.worker:
            raise MissionAuthorityRefused("this dispatch belongs to another worker")

    @staticmethod
    def _key(kind: str, *parts: str) -> str:
        """A deterministic operation key: same logical operation, same name."""

        return "-".join(("cm", kind) + tuple(str(part) for part in parts))

    def _forget(self, key: str) -> None:
        """Release one receipt now that its outcome is durably recorded here.

        Retention is bounded and is released only by a caller saying "I have
        this outcome". Every call above records what it did in the job record
        before this runs, so the receipt has stopped being the only evidence
        that the operation took effect.
        """

        try:
            self._store.acknowledge_operation(key)
        except (MissionError, OSError):
            return


def worker_identity(instance_id: str) -> str:
    """One bounded worker token per process, distinct across processes."""

    return "w-{}-{}".format(str(instance_id)[:24] or "anon", os.getpid())
