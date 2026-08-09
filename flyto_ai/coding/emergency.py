# Copyright 2024 Flyto2
# Licensed under the Apache License, Version 2.0
"""Host-owned emergency overflow authority for a broken route infrastructure.

The strict route in `flyto_ai.coding.route` is the normal path and stays
mandatory. This module exists for one narrow situation: the route
*infrastructure itself* is unavailable, so every job fails before the
implementer is ever invoked and the control plane would otherwise strand all
coding indefinitely.

The overflow lane is deliberately hard to reach:

- only a positively classified infrastructure failure opens it
  (`ROUTE_INFRASTRUCTURE_FAILURE_CODES`), and only in a lane that runs before
  the implementer (`EMERGENCY_TRIGGER_LANES`);
- a domain refusal, blocked gate, stale index, malformed evidence, failed
  check, failed implementation, Core validation failure, Indexer post failure,
  audit rejection, or rework exhaustion never opens it;
- it is startup authority. There is no per-job field, no model-reachable
  switch, and no environment override;
- it does not skip the audit. An emergency round still ends at
  `awaiting_codex_audit` bound to an exact revision, and it never commits,
  pushes, or publishes;
- it never claims the strict lanes passed. `CodingRouteReceipt(strict=True)`
  is untouched; acceptance of an emergency round requires this separate
  digest-validated `EmergencyAuthorityReceipt`.

The breaker is monotonic inside one process: it opens once and never closes
again, so it cannot oscillate. Recovery is a new service instance — a repaired
build starts with a closed circuit and publishes a new build id, while an old
instance stays visibly old in the runtime status index.
"""
from __future__ import annotations

import hashlib
import json
import re
import threading
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

from flyto_ai.coding.route import RouteLane


EMERGENCY_CONTRACT_VERSION = "flyto.coding-emergency.v1"
_EMERGENCY_DOMAIN = b"flyto.coding-emergency.v1\n"
_ACTION_RE = re.compile(r"^[a-z][a-z0-9_.:-]{1,63}$")
_ID_RE = re.compile(r"^[a-z0-9]{8,64}$")
_BACKEND_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
_JOB_ID_RE = re.compile(r"^job_[a-f0-9]{24}$")
_SHA256_RE = re.compile(r"^[a-f0-9]{64}$")
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x1f\x7f-\x9f]")
MAX_EMERGENCY_SESSION_ID_CHARS = 128

#: The only route failures that count as broken infrastructure. Each one means
#: the capability process could not be reached or would not answer inside its
#: bound — never that it answered and refused.
ROUTE_INFRASTRUCTURE_FAILURE_CODES = frozenset({
    "capability_unavailable",
    "capability_timeout",
})
#: Lanes that run strictly before the implementer. A failure anywhere else
#: means an edit may already exist, so it can never open the overflow lane.
EMERGENCY_TRIGGER_LANES = frozenset({
    RouteLane.INDEXER_PRE.value,
    RouteLane.BLUEPRINT.value,
})
#: `emergency` opened the lane for this job; `emergency_rework` is a later
#: round of the same job continuing on the same authority. Neither is a
#: strict-route outcome and neither may be presented as one.
EMERGENCY_MODES = ("emergency", "emergency_rework")
CIRCUIT_STATES = ("closed", "open")
MAX_EMERGENCY_FAILURE_THRESHOLD = 10


class EmergencyAuthorityError(ValueError):
    """The emergency authority contract does not hold. Fail closed."""


@dataclass(frozen=True)
class EmergencyOverflowPolicy:
    """Startup-only overflow authority. No job payload can reach these fields.

    `backend` names the implementer this authority was granted for. The
    service refuses to use the lane when its configured implementer is a
    different one, so enabling overflow can never silently redirect work to a
    backend the operator did not choose.
    """

    enabled: bool = False
    backend: str = ""
    #: One positively classified pre-edit infrastructure failure is enough by
    #: default. Each Codex conversation gets its own stdio process and many see
    #: exactly one job, so a per-process count above 1 would still strand every
    #: one of them. Counters are per-instance and are never shared between
    #: processes or builds.
    failure_threshold: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("emergency enabled must be a boolean")
        if not isinstance(self.backend, str):
            raise ValueError("emergency backend must be a string")
        if isinstance(self.failure_threshold, bool) or not isinstance(
            self.failure_threshold, int,
        ):
            raise ValueError("emergency failure_threshold must be an integer")
        if not 1 <= self.failure_threshold <= MAX_EMERGENCY_FAILURE_THRESHOLD:
            raise ValueError(
                "emergency failure_threshold must be between 1 and {}".format(
                    MAX_EMERGENCY_FAILURE_THRESHOLD,
                ),
            )
        if self.enabled and not _BACKEND_RE.fullmatch(self.backend):
            raise ValueError("an enabled emergency policy requires a safe backend id")
        if not self.enabled and self.backend:
            raise ValueError("a disabled emergency policy cannot name a backend")

    def applies_to(self, backend: str) -> bool:
        """Whether this authority covers the implementer the service selected."""
        return self.enabled and bool(backend) and backend == self.backend


@dataclass(frozen=True)
class EmergencyTrigger:
    """The positively classified infrastructure failure that opened the lane."""

    lane: str
    action: str
    code: str

    def __post_init__(self) -> None:
        if self.lane not in EMERGENCY_TRIGGER_LANES:
            raise EmergencyAuthorityError("emergency trigger lane is not pre-implementer")
        if self.code not in ROUTE_INFRASTRUCTURE_FAILURE_CODES:
            raise EmergencyAuthorityError("emergency trigger code is not infrastructure")
        if self.action and not _ACTION_RE.fullmatch(self.action):
            raise EmergencyAuthorityError("emergency trigger action is not a safe action")


def classify_overflow_trigger(
    lane: str, action: str, code: str,
) -> Optional[EmergencyTrigger]:
    """Return a trigger only for a positively classified infrastructure failure.

    Every other failure category returns `None`, which keeps the round on the
    ordinary fail-closed path. This is an allowlist: an unrecognised code is
    never treated as infrastructure.
    """
    if code not in ROUTE_INFRASTRUCTURE_FAILURE_CODES:
        return None
    if lane not in EMERGENCY_TRIGGER_LANES:
        return None
    safe_action = action if isinstance(action, str) and _ACTION_RE.fullmatch(action) else ""
    return EmergencyTrigger(lane=lane, action=safe_action, code=code)


@dataclass(frozen=True)
class EmergencyAuthorityReceipt:
    """Digest-validated proof that one round ran on the emergency lane.

    This is the *only* evidence that can make an emergency round auditable. It
    is separate from `CodingRouteReceipt` on purpose: a strict route receipt
    must never be able to describe a round whose strict lanes did not run.
    """

    mode: str
    circuit_state: str
    trigger_lane: str
    trigger_action: str
    trigger_code: str
    implementer_backend: str
    instance_id: str
    build_id: str
    contract_version: str = EMERGENCY_CONTRACT_VERSION
    #: Host-owned binding facts. They enter the digest, so an intact receipt
    #: copied into another job's record no longer validates: the job id,
    #: request digest, implementation session, and exact revision must all
    #: still match the record that carries it.
    job_id: str = ""
    request_sha256: str = ""
    session_id: str = ""
    revision_sha256: str = ""
    implementer_started: bool = True
    checks_enforced: bool = False
    audit_required: bool = True
    digest: str = ""

    def __post_init__(self) -> None:
        if self.contract_version != EMERGENCY_CONTRACT_VERSION:
            raise EmergencyAuthorityError("unsupported emergency contract version")
        if self.mode not in EMERGENCY_MODES:
            raise EmergencyAuthorityError("emergency mode is unknown")
        if self.circuit_state != "open":
            # A closed breaker never authorized anything. Recording one here
            # would be a claim the runtime cannot support.
            raise EmergencyAuthorityError("emergency authority requires an open circuit")
        if self.trigger_lane not in EMERGENCY_TRIGGER_LANES:
            raise EmergencyAuthorityError("emergency trigger lane is not pre-implementer")
        if self.trigger_code not in ROUTE_INFRASTRUCTURE_FAILURE_CODES:
            raise EmergencyAuthorityError("emergency trigger code is not infrastructure")
        if self.trigger_action and not _ACTION_RE.fullmatch(self.trigger_action):
            raise EmergencyAuthorityError("emergency trigger action is not a safe action")
        if not _BACKEND_RE.fullmatch(self.implementer_backend):
            raise EmergencyAuthorityError("emergency implementer_backend must be a safe id")
        for name in ("instance_id", "build_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not _ID_RE.fullmatch(value):
                raise EmergencyAuthorityError(
                    "emergency {} must be an opaque lowercase token".format(name),
                )
        if self.job_id and not _JOB_ID_RE.fullmatch(self.job_id):
            raise EmergencyAuthorityError("emergency job_id must be a service job id")
        for name in ("request_sha256", "revision_sha256"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, str):
                raise EmergencyAuthorityError("emergency {} must be a string".format(name))
            if value and not _SHA256_RE.fullmatch(value):
                raise EmergencyAuthorityError(
                    "emergency {} must be a lowercase sha256 digest".format(name),
                )
        if isinstance(self.session_id, bool) or not isinstance(self.session_id, str):
            raise EmergencyAuthorityError("emergency session_id must be a string")
        if (
            len(self.session_id) > MAX_EMERGENCY_SESSION_ID_CHARS
            or _CONTROL_CHARS_RE.search(self.session_id)
        ):
            raise EmergencyAuthorityError("emergency session_id must be a bounded token")
        if self.revision_sha256 and not self.session_id:
            # A bound revision without the session that produced it cannot
            # prove which implementation round it belongs to.
            raise EmergencyAuthorityError("emergency revision requires its session id")
        for name in ("implementer_started", "checks_enforced", "audit_required"):
            if not isinstance(getattr(self, name), bool):
                raise EmergencyAuthorityError("emergency {} must be a boolean".format(name))
        if not self.implementer_started:
            # The receipt exists because the emergency implementer ran. There
            # is no such thing as emergency authority without an implementer.
            raise EmergencyAuthorityError("emergency authority requires a started implementer")
        if not self.audit_required:
            raise EmergencyAuthorityError("emergency rounds always require an audit")
        expected = self.compute_digest(self.to_payload())
        if not self.digest:
            object.__setattr__(self, "digest", expected)
        elif self.digest != expected:
            raise EmergencyAuthorityError("emergency digest does not match its facts")

    def to_payload(self) -> Dict[str, Any]:
        """Return the canonical digest input: every field except the digest."""
        return {
            "contract_version": self.contract_version,
            "mode": self.mode,
            "circuit_state": self.circuit_state,
            "trigger_lane": self.trigger_lane,
            "trigger_action": self.trigger_action,
            "trigger_code": self.trigger_code,
            "implementer_backend": self.implementer_backend,
            "instance_id": self.instance_id,
            "build_id": self.build_id,
            "job_id": self.job_id,
            "request_sha256": self.request_sha256,
            "session_id": self.session_id,
            "revision_sha256": self.revision_sha256,
            "implementer_started": self.implementer_started,
            "checks_enforced": self.checks_enforced,
            "audit_required": self.audit_required,
        }

    @property
    def sealed(self) -> bool:
        """Whether this authority is bound to one exact audited round.

        An unsealed authority proves the emergency lane ran; only a sealed one
        can authorize an audit, because only it names the job, request,
        session, and revision the verdict would apply to.
        """
        return bool(
            self.job_id and self.request_sha256
            and self.session_id and self.revision_sha256,
        )

    def seal(
        self, *, job_id: str, request_sha256: str, session_id: str, revision_sha256: str,
    ) -> "EmergencyAuthorityReceipt":
        """Return the same authority bound to one exact audited round."""
        import dataclasses

        return dataclasses.replace(
            self,
            job_id=job_id,
            request_sha256=request_sha256,
            session_id=session_id,
            revision_sha256=revision_sha256,
            digest="",
        )

    @staticmethod
    def compute_digest(payload: Mapping[str, Any]) -> str:
        encoded = json.dumps(
            dict(payload), ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        )
        digest = hashlib.sha256()
        digest.update(_EMERGENCY_DOMAIN)
        digest.update(encoded.encode("utf-8"))
        return digest.hexdigest()

    def to_mapping(self) -> Dict[str, Any]:
        payload = self.to_payload()
        payload["digest"] = self.digest
        return payload

    def trigger(self) -> EmergencyTrigger:
        """Return the recorded trigger so a later round keeps the same cause."""
        return EmergencyTrigger(
            lane=self.trigger_lane, action=self.trigger_action, code=self.trigger_code,
        )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "EmergencyAuthorityReceipt":
        """Revalidate a persisted authority; a tampered record fails closed."""
        if not isinstance(value, Mapping):
            raise EmergencyAuthorityError("emergency authority must be an object")
        unknown = set(value) - {
            "contract_version", "mode", "circuit_state", "trigger_lane",
            "trigger_action", "trigger_code", "implementer_backend", "instance_id",
            "build_id", "job_id", "request_sha256", "session_id", "revision_sha256",
            "implementer_started", "checks_enforced", "audit_required", "digest",
        }
        if unknown:
            raise EmergencyAuthorityError("unsupported emergency authority fields")

        def text(name: str) -> str:
            item = value.get(name, "")
            if isinstance(item, bool) or not isinstance(item, str):
                raise EmergencyAuthorityError("emergency {} must be a string".format(name))
            return item

        def flag(name: str, default: bool) -> bool:
            item = value.get(name, default)
            if not isinstance(item, bool):
                raise EmergencyAuthorityError("emergency {} must be a boolean".format(name))
            return item

        return cls(
            contract_version=text("contract_version") or EMERGENCY_CONTRACT_VERSION,
            mode=text("mode"),
            circuit_state=text("circuit_state"),
            trigger_lane=text("trigger_lane"),
            trigger_action=text("trigger_action"),
            trigger_code=text("trigger_code"),
            implementer_backend=text("implementer_backend"),
            instance_id=text("instance_id"),
            build_id=text("build_id"),
            job_id=text("job_id"),
            request_sha256=text("request_sha256"),
            session_id=text("session_id"),
            revision_sha256=text("revision_sha256"),
            implementer_started=flag("implementer_started", True),
            checks_enforced=flag("checks_enforced", False),
            audit_required=flag("audit_required", True),
            digest=text("digest"),
        )


class EmergencyCircuitBreaker:
    """Count infrastructure failures for one process and open exactly once.

    Monotonic on purpose. A breaker that could close again would oscillate
    between the strict route and the overflow lane while the infrastructure
    flapped, and each flap would be a differently authorized round. Recovery is
    a new process: a repaired build starts closed with a fresh instance id.
    """

    def __init__(self, policy: EmergencyOverflowPolicy) -> None:
        if not isinstance(policy, EmergencyOverflowPolicy):
            raise ValueError("emergency policy must be an EmergencyOverflowPolicy")
        self.policy = policy
        self._lock = threading.Lock()
        self._failures = 0
        self._activations = 0
        self._state = "closed"

    @property
    def state(self) -> str:
        with self._lock:
            return self._state

    @property
    def activations(self) -> int:
        with self._lock:
            return self._activations

    @property
    def failures(self) -> int:
        with self._lock:
            return self._failures

    def record_infrastructure_failure(self) -> bool:
        """Count one classified failure; return True when the lane may be used.

        A disabled policy still returns False after counting nothing, so an
        operator who never granted the authority sees no behavioral change.
        """
        with self._lock:
            if not self.policy.enabled:
                return False
            self._failures += 1
            if self._failures >= self.policy.failure_threshold:
                self._state = "open"
            return self._state == "open"

    def note_activation(self) -> str:
        """Record that one emergency round ran and return the circuit state."""
        with self._lock:
            if self._state != "open":
                raise EmergencyAuthorityError("a closed circuit cannot authorize a round")
            self._activations += 1
            return self._state

    def force_open(self) -> str:
        """Reopen the breaker for a job already bound to emergency authority.

        A rework round of an emergency job must stay on the same path even in a
        process whose own counter has not tripped yet. It never runs unless the
        persisted authority already proved a real infrastructure trigger.
        """
        with self._lock:
            if not self.policy.enabled:
                raise EmergencyAuthorityError("emergency authority is not enabled")
            self._state = "open"
            return self._state
