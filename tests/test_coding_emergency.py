"""Emergency overflow authority, runtime status, and their service integration.

The overflow lane exists for one situation: the route infrastructure itself is
unreachable, so every job would otherwise fail before the implementer runs.
Most of this module is therefore negative control — proof that everything
*except* a classified pre-edit infrastructure failure stays fail-closed.
"""
from __future__ import annotations

import json
import os
import stat
import sys
import time
from pathlib import Path

import pytest
from flyto_ai.coding.contracts import (
    CheckResult,
    CodingAuditFinding,
    CodingAuditVerdict,
    CodingJobReceipt,
    CodingJobState,
    CodingTaskRequest,
    CodingTaskResult,
    TERMINAL_CODING_JOB_STATES,
)
from flyto_ai.coding.emergency import (
    EMERGENCY_CONTRACT_VERSION,
    EMERGENCY_TRIGGER_LANES,
    ROUTE_INFRASTRUCTURE_FAILURE_CODES,
    EmergencyAuthorityError,
    EmergencyAuthorityReceipt,
    EmergencyCircuitBreaker,
    EmergencyOverflowPolicy,
    classify_overflow_trigger,
)
from flyto_ai.coding.route import CodingRoutePolicy, RouteLane
from flyto_ai.coding.store import mark_provider_start
from flyto_ai.coding.route_status import (
    MAX_STATUS_INSTANCES,
    ROUTE_STATUS_CONTRACT_VERSION,
    STATUS_INSTANCE_TTL_SECONDS,
    CodingRouteStatus,
    RouteStatusPublisher,
    project_index_row,
    route_mode,
    service_build_id,
)
from flyto_ai.coding.service import (
    CodingService,
    EmergencyAuthorityMissing,
    RouteEvidenceMissing,
)

from tests.test_coding_route import (
    BLUEPRINT_FIXTURE,
    INDEXER_FIXTURE,
    _blueprint_spec,
    _indexer_spec,
)


# ── emergency contract ────────────────────────────────────────────────


#: Preflight refuses a repository that never declared how it wants to be
#: verified, before a job, a worktree claim or an implementer session exists.
#: A fixture that means to exercise anything past preflight therefore has to
#: declare a contract, exactly as a real repository does.
_TEST_VERIFICATION_CONTRACT = """version: flyto.coding-config.v1
checks:
  - name: declared
    argv: [python, --version]
    timeout_seconds: 30
    required: true
"""


def _declare_verification(workspace) -> None:
    config = workspace / ".flyto" / "coding.yaml"
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text(_TEST_VERIFICATION_CONTRACT, encoding="utf-8")


def _authority(**overrides) -> EmergencyAuthorityReceipt:
    values = dict(
        mode="emergency", circuit_state="open", trigger_lane="indexer_pre",
        trigger_action="search", trigger_code="capability_timeout",
        implementer_backend="claude", instance_id="a" * 24, build_id="b" * 24,
        checks_enforced=True,
    )
    values.update(overrides)
    return EmergencyAuthorityReceipt(**values)


def test_the_policy_is_startup_authority_bound_to_one_named_implementer():
    assert EmergencyOverflowPolicy().enabled is False
    assert EmergencyOverflowPolicy().applies_to("claude") is False
    # One classified failure is enough: each Codex conversation is its own
    # process and many of them only ever see a single job.
    assert EmergencyOverflowPolicy().failure_threshold == 1

    granted = EmergencyOverflowPolicy(enabled=True, backend="claude")
    assert granted.applies_to("claude") is True
    assert granted.applies_to("native") is False
    assert granted.applies_to("") is False

    with pytest.raises(ValueError, match="requires a safe backend"):
        EmergencyOverflowPolicy(enabled=True, backend="")
    with pytest.raises(ValueError, match="cannot name a backend"):
        EmergencyOverflowPolicy(enabled=False, backend="claude")
    with pytest.raises(ValueError, match="failure_threshold must be between"):
        EmergencyOverflowPolicy(enabled=True, backend="claude", failure_threshold=0)
    with pytest.raises(ValueError, match="failure_threshold must be an integer"):
        EmergencyOverflowPolicy(enabled=True, backend="claude", failure_threshold=True)


@pytest.mark.parametrize("code", sorted(ROUTE_INFRASTRUCTURE_FAILURE_CODES))
def test_classified_infrastructure_failures_open_the_lane(code):
    for lane in sorted(EMERGENCY_TRIGGER_LANES):
        trigger = classify_overflow_trigger(lane, "search", code)
        assert trigger is not None
        assert (trigger.lane, trigger.code) == (lane, code)


@pytest.mark.parametrize("code", [
    # Every one of these is the infrastructure answering and refusing, or a
    # failure that happens at or after implementation.
    "domain_failure", "gate_not_satisfied", "gate_not_remediable",
    "gate_needs_external_authority", "index_stale", "malformed_evidence",
    "plan_contract_missing", "plan_step_not_allowlisted", "plan_bound_exceeded",
    "call_bound_exceeded", "response_bound_exceeded", "catalog_missing",
    "required_action_missing", "required_checks_failed", "required_checks_missing",
    "implementation_not_successful", "validation_failed", "strict_verify_failed",
    "core_proof_unavailable", "core_validation_failed", "action_not_allowlisted",
    "capability_close_failed", "service_restarted", "rework_limit_reached",
])
def test_every_other_failure_category_stays_fail_closed(code):
    assert classify_overflow_trigger("indexer_pre", "search", code) is None
    assert classify_overflow_trigger("blueprint", "list_blueprints", code) is None


@pytest.mark.parametrize("lane", ["core", "indexer_post", "", "unknown_lane"])
def test_an_infrastructure_code_outside_a_pre_lane_never_opens_the_lane(lane):
    for code in sorted(ROUTE_INFRASTRUCTURE_FAILURE_CODES):
        assert classify_overflow_trigger(lane, "task.validate", code) is None


def test_a_hostile_action_name_is_dropped_rather_than_persisted():
    trigger = classify_overflow_trigger(
        "indexer_pre", "search; rm -rf /", "capability_timeout",
    )
    assert trigger is not None and trigger.action == ""


def test_the_authority_receipt_digests_every_fact_it_carries():
    authority = _authority()
    assert authority.contract_version == EMERGENCY_CONTRACT_VERSION
    assert len(authority.digest) == 64
    assert authority.sealed is False
    restored = EmergencyAuthorityReceipt.from_mapping(authority.to_mapping())
    assert restored.digest == authority.digest

    sealed = authority.seal(
        job_id="job_" + "a" * 24, request_sha256="c" * 64,
        session_id="sdk-session-1", revision_sha256="d" * 64,
    )
    assert sealed.sealed is True
    assert sealed.digest != authority.digest
    assert EmergencyAuthorityReceipt.from_mapping(sealed.to_mapping()).sealed


@pytest.mark.parametrize(("changes", "match"), [
    ({"circuit_state": "closed"}, "requires an open circuit"),
    ({"mode": "strict"}, "mode is unknown"),
    ({"trigger_lane": "indexer_post"}, "not pre-implementer"),
    ({"trigger_code": "domain_failure"}, "not infrastructure"),
    ({"implementer_backend": "not a name"}, "safe id"),
    ({"instance_id": "SHORT"}, "opaque lowercase token"),
    ({"implementer_started": False}, "requires a started implementer"),
    ({"audit_required": False}, "always require an audit"),
    ({"job_id": "nope"}, "must be a service job id"),
    ({"request_sha256": "zz"}, "lowercase sha256"),
    ({"revision_sha256": "d" * 64}, "revision requires its session id"),
    ({"session_id": "x" * 129}, "bounded token"),
])
def test_the_authority_receipt_refuses_incoherent_facts(changes, match):
    with pytest.raises(EmergencyAuthorityError, match=match):
        _authority(**changes)


def test_a_tampered_authority_fails_closed():
    payload = _authority().to_mapping()
    payload["trigger_code"] = "capability_unavailable"
    with pytest.raises(EmergencyAuthorityError, match="digest does not match"):
        EmergencyAuthorityReceipt.from_mapping(payload)

    payload = _authority().to_mapping()
    payload["unexpected"] = 1
    with pytest.raises(EmergencyAuthorityError, match="unsupported emergency"):
        EmergencyAuthorityReceipt.from_mapping(payload)

    payload = _authority().to_mapping()
    payload["checks_enforced"] = "yes"
    with pytest.raises(EmergencyAuthorityError, match="must be a boolean"):
        EmergencyAuthorityReceipt.from_mapping(payload)


def test_the_breaker_opens_on_the_first_failure_and_never_closes_again():
    breaker = EmergencyCircuitBreaker(
        EmergencyOverflowPolicy(enabled=True, backend="claude"),
    )
    assert breaker.state == "closed"
    assert breaker.record_infrastructure_failure() is True
    assert breaker.state == "open"
    assert breaker.note_activation() == "open"
    assert breaker.activations == 1
    # Monotonic: more failures never toggle it back, so it cannot oscillate.
    assert breaker.record_infrastructure_failure() is True
    assert breaker.state == "open"


def test_a_higher_threshold_holds_the_lane_shut_until_it_is_reached():
    breaker = EmergencyCircuitBreaker(
        EmergencyOverflowPolicy(enabled=True, backend="claude", failure_threshold=3),
    )
    assert breaker.record_infrastructure_failure() is False
    assert breaker.record_infrastructure_failure() is False
    assert breaker.state == "closed"
    with pytest.raises(EmergencyAuthorityError, match="closed circuit"):
        breaker.note_activation()
    assert breaker.record_infrastructure_failure() is True
    assert breaker.state == "open"


def test_a_disabled_policy_never_opens_or_authorizes_anything():
    breaker = EmergencyCircuitBreaker(EmergencyOverflowPolicy())
    for _ in range(5):
        assert breaker.record_infrastructure_failure() is False
    assert breaker.state == "closed" and breaker.failures == 0
    with pytest.raises(EmergencyAuthorityError):
        breaker.force_open()


def test_breaker_counters_are_per_instance_and_never_shared():
    policy = EmergencyOverflowPolicy(enabled=True, backend="claude", failure_threshold=2)
    first, second = EmergencyCircuitBreaker(policy), EmergencyCircuitBreaker(policy)
    assert first.record_infrastructure_failure() is False
    # A second process starts its own count; it does not inherit the first's.
    assert second.record_infrastructure_failure() is False
    assert second.state == "closed"


def test_a_public_receipt_rejects_inconsistent_or_mixed_authority():
    sealed = _authority().seal(
        job_id="job_" + "a" * 24, request_sha256="c" * 64,
        session_id="sdk-session-1", revision_sha256="d" * 64,
    )
    base = dict(
        job_id="job_" + "a" * 24, state=CodingJobState.AWAITING_CODEX_AUDIT,
        submitted_at=1.0, updated_at=2.0,
        implementation_backend="claude",
        implementation_session_id="sdk-session-1",
        implementation_revision_sha256="d" * 64,
        implementer_started=True,
        emergency_authority=sealed.to_mapping(),
    )
    receipt = CodingJobReceipt(**base)
    assert receipt.emergency_authority["mode"] == "emergency"

    # The sealed binding must describe this very receipt.
    with pytest.raises(ValueError, match="does not bind this receipt's job"):
        CodingJobReceipt(**{**base, "job_id": "job_" + "b" * 24})
    with pytest.raises(ValueError, match="implementation session"):
        CodingJobReceipt(**{**base, "implementation_session_id": "other-session"})
    with pytest.raises(ValueError, match="implementation revision"):
        CodingJobReceipt(**{**base, "implementation_revision_sha256": "e" * 64})
    with pytest.raises(ValueError, match="different implementation backend"):
        CodingJobReceipt(**{**base, "implementation_backend": "native"})

    from flyto_ai.coding.route import CodingRouteReceipt

    passed = CodingRouteReceipt(strict=False, ok=True, lanes=())
    with pytest.raises(ValueError, match="both route and emergency authority"):
        CodingJobReceipt(**{**base, "route_receipt": passed.to_mapping()})


def test_an_unsealed_or_uncheck_authority_can_never_be_landable():
    accepted = dict(
        job_id="job_" + "a" * 24, state=CodingJobState.CODEX_ACCEPTED,
        submitted_at=1.0, updated_at=2.0,
        implementation_backend="claude",
        implementation_session_id="sdk-session-1",
        implementation_revision_sha256="d" * 64,
        audit_count=1, rework_count=0, audit_findings_sha256="f" * 64,
        landable=True, implementer_started=True,
    )
    with pytest.raises(ValueError, match="must bind its exact round"):
        CodingJobReceipt(**accepted, emergency_authority=_authority().to_mapping())
    unchecked = _authority(checks_enforced=False).seal(
        job_id="job_" + "a" * 24, request_sha256="c" * 64,
        session_id="sdk-session-1", revision_sha256="d" * 64,
    )
    with pytest.raises(ValueError, match="requires passed required checks"):
        CodingJobReceipt(**accepted, emergency_authority=unchecked.to_mapping())


# ── runtime status ────────────────────────────────────────────────────


def _status(**overrides) -> CodingRouteStatus:
    values = dict(
        instance_id="a" * 24, build_id="b" * 24, service_version="9.9.9",
        process_id=os.getpid(), started_at=1000.0, updated_at=1001.0,
        implementation_backend="claude", job_id="job_" + "a" * 24,
        state="failed", mode="strict", lane="indexer_pre", action="search",
        failure_code="route_capability_timeout",
    )
    values.update(overrides)
    return CodingRouteStatus(**values)


def _publisher(root: Path, instance_id: str, **overrides) -> RouteStatusPublisher:
    values = dict(build_id="b" * 24, version="9.9.9", started_at=1000.0)
    values.update(overrides)
    return RouteStatusPublisher(root, instance_id=instance_id, **values)


def test_the_status_schema_is_closed_and_narrows_untrusted_values():
    status = _status(
        job_id="not-a-job", lane="Indexer Pre", action="search; rm -rf /",
        failure_code="NOT A CODE", implementation_revision_sha256="zz",
        implementation_session_id="x" * 200,
    )
    assert status.job_id == "" and status.lane == "" and status.action == ""
    assert status.failure_code == "" and status.implementation_revision_sha256 == ""
    assert status.implementation_session_id == ""

    with pytest.raises(ValueError, match="mode is unknown"):
        _status(mode="degraded-ish")
    with pytest.raises(ValueError, match="lifecycle is unknown"):
        _status(lifecycle="zombie")
    with pytest.raises(ValueError, match="state is unknown"):
        _status(state="inventing_a_state")
    with pytest.raises(ValueError, match="circuit_state is unknown"):
        _status(circuit_state="half-open")
    with pytest.raises(ValueError, match="must be an opaque token"):
        _status(instance_id="TOO-SHORT")
    with pytest.raises(ValueError, match="last_publish_failure_code is unknown"):
        _status(last_publish_failure_code="PermissionError")
    with pytest.raises(ValueError, match="must record a started implementer"):
        _status(landable=True, implementer_started=False)
    with pytest.raises(ValueError, match="unsupported route status fields"):
        CodingRouteStatus.from_mapping({**_status().to_mapping(), "prompt": "secret"})


def test_status_carries_no_prompt_path_or_error_prose(tmp_path):
    publisher = _publisher(tmp_path, "a" * 24)
    publisher.publish(_status())
    raw = publisher.instance_path().read_text(encoding="utf-8")
    payload = json.loads(raw)
    assert payload["contract_version"] == ROUTE_STATUS_CONTRACT_VERSION
    forbidden = {
        "message", "working_dir", "error", "files", "implementation_files",
        "result", "argv", "environment", "command", "prompt",
    }
    assert forbidden.isdisjoint(payload)
    assert str(tmp_path) not in raw


def test_status_files_are_written_atomically_at_mode_0600(tmp_path):
    publisher = _publisher(tmp_path, "a" * 24)
    publisher.publish(_status())
    for path in (publisher.instance_path(), publisher.index_path):
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
    # No temporary file is left behind by a completed publish.
    assert not list(publisher.root.glob(".status-*"))


def test_two_live_instances_coexist_and_identify_their_builds(tmp_path):
    old = _publisher(tmp_path, "1" * 24, build_id="0" * 24, version="1.0.0")
    new = _publisher(tmp_path, "2" * 24, build_id="9" * 24, version="2.0.0")
    old.publish(_status(instance_id="1" * 24, build_id="0" * 24, service_version="1.0.0"))
    new.publish(_status(instance_id="2" * 24, build_id="9" * 24, service_version="2.0.0"))

    rows = {row["instance_id"]: row for row in new.inspect()}
    assert set(rows) == {"1" * 24, "2" * 24}
    assert rows["1" * 24]["build_id"] == "0" * 24
    assert rows["2" * 24]["build_id"] == "9" * 24
    assert rows["2" * 24]["current"] is True and rows["1" * 24]["current"] is False
    assert rows["1" * 24]["build_stale"] is True
    assert rows["1" * 24]["reload_required"] is True
    assert rows["2" * 24]["build_stale"] is False
    # An older instance's own file is untouched by the newer publisher.
    assert json.loads(old.instance_path().read_text())["service_version"] == "1.0.0"


def test_an_instance_only_ever_writes_its_own_file(tmp_path):
    publisher = _publisher(tmp_path, "1" * 24)
    with pytest.raises(ValueError, match="belongs to another instance"):
        publisher.publish(_status(instance_id="2" * 24))


def test_the_index_read_is_bounded_and_projected_through_a_closed_schema(tmp_path):
    publisher = _publisher(tmp_path, "1" * 24)
    publisher.publish(_status(instance_id="1" * 24))

    forged = {
        "contract_version": ROUTE_STATUS_CONTRACT_VERSION,
        "updated_at": 5.0,
        "instances": [
            {"instance_id": "1" * 24, "secret": "sk-nope", "state": "invented"},
            {"instance_id": "1" * 24, "state": "failed"},
            {"instance_id": "BAD"},
            {"no_instance_id": True},
            "not-an-object",
        ],
    }
    publisher.index_path.write_text(json.dumps(forged), encoding="utf-8")
    rows = publisher.read_index()["instances"]
    # Duplicates collapse, unusable rows are dropped, unknown fields never
    # survive, and an invented state degrades instead of propagating.
    assert len(rows) == 1
    assert "secret" not in rows[0]
    assert rows[0]["state"] == ""

    publisher.index_path.write_text("x" * (300 * 1024), encoding="utf-8")
    assert publisher.read_index()["instances"] == []
    publisher.index_path.write_text("{not json", encoding="utf-8")
    assert publisher.read_index()["instances"] == []
    publisher.index_path.write_text(json.dumps({"instances": []}), encoding="utf-8")
    assert publisher.read_index()["instances"] == []


def test_a_malformed_index_is_never_republished(tmp_path):
    publisher = _publisher(tmp_path, "1" * 24)
    publisher.index_path.parent.mkdir(parents=True, exist_ok=True)
    publisher.index_path.write_text(
        json.dumps({
            "contract_version": ROUTE_STATUS_CONTRACT_VERSION,
            "instances": [{"instance_id": "2" * 24, "poison": "value"}],
        }),
        encoding="utf-8",
    )
    publisher.publish(_status(instance_id="1" * 24))
    written = json.loads(publisher.index_path.read_text())
    assert all("poison" not in row for row in written["instances"])


def test_stale_and_excess_instances_are_collected_deterministically(tmp_path):
    now = 1_000_000.0
    publisher = _publisher(tmp_path, "0" * 24, started_at=now)
    stale_id = "5" * 24
    stale = _publisher(tmp_path, stale_id, started_at=1.0)
    stale.publish(_status(
        instance_id=stale_id, updated_at=now - STATUS_INSTANCE_TTL_SECONDS - 10,
    ))
    assert stale.instance_path().exists()

    publisher.publish(_status(instance_id="0" * 24, updated_at=now))
    rows = {row["instance_id"] for row in publisher.read_index()["instances"]}
    assert rows == {"0" * 24}
    assert not stale.instance_path().exists()


def test_the_index_never_grows_past_its_bound(tmp_path):
    now = 2_000_000.0
    for index in range(MAX_STATUS_INSTANCES + 5):
        instance_id = "{:024x}".format(index)
        publisher = _publisher(tmp_path, instance_id)
        publisher.publish(_status(instance_id=instance_id, updated_at=now + index))
    final = _publisher(tmp_path, "f" * 24)
    final.publish(_status(instance_id="f" * 24, updated_at=now + 1000))
    rows = final.read_index()["instances"]
    assert len(rows) <= MAX_STATUS_INSTANCES
    files = list(final.root.glob("instance-*.json"))
    assert len(files) <= MAX_STATUS_INSTANCES


def test_liveness_and_staleness_are_annotated_for_local_inspection(tmp_path):
    now = time.time()
    publisher = _publisher(tmp_path, "1" * 24)
    publisher.publish(_status(instance_id="1" * 24, updated_at=now))
    live = publisher.inspect()[0]
    assert live["stale"] is False
    assert live["age_stale"] is False and live["build_stale"] is False
    assert live["reload_required"] is False
    assert live["alive"] is True
    assert live["lane"] == "indexer_pre" and live["action"] == "search"

    dead = _publisher(tmp_path, "2" * 24)
    dead.publish(_status(
        instance_id="2" * 24, process_id=2 ** 22, updated_at=now,
    ))
    rows = {row["instance_id"]: row for row in publisher.inspect()}
    assert rows["2" * 24]["alive"] in (False, None)


def test_project_index_row_rejects_what_it_cannot_trust():
    assert project_index_row("string") is None
    assert project_index_row({"instance_id": "SHORT"}) is None
    row = project_index_row({
        "instance_id": "1" * 24, "process_id": True, "updated_at": "soon",
        "lifecycle": "invented", "mode": "invented", "circuit_state": "invented",
    })
    assert row["process_id"] == 0 and row["updated_at"] == 0.0
    assert row["lifecycle"] == "active" and row["mode"] == "strict"
    assert row["circuit_state"] == "closed"


def test_route_mode_reads_the_durable_execution_mode_first():
    assert route_mode({}) == "strict"
    assert route_mode({"execution_mode": "emergency"}) == "emergency"
    # Even before an authority exists, a persisted trigger says overflow.
    assert route_mode({"emergency_trigger": {"lane": "indexer_pre"}}) == "emergency"
    assert route_mode({"emergency_authority": {"mode": "emergency"}}) == "emergency"


def test_the_build_id_is_stable_within_one_process():
    assert service_build_id() == service_build_id()
    assert len(service_build_id()) == 32


def test_a_changed_source_build_blocks_new_jobs_before_mutation(
    tmp_path, monkeypatch,
):
    import flyto_ai.coding.service as service_module

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    _declare_verification(workspace)
    box: dict = {}
    service = _emergency_service(tmp_path, workspace, box, enabled=False)
    try:
        monkeypatch.setattr(
            service_module,
            "current_service_build_id",
            lambda: "f" * 32 if service.build_id != "f" * 32 else "e" * 32,
        )
        with pytest.raises(
            service_module.CodingServiceReloadRequired,
            match="reload the MCP worker",
        ) as caught:
            service.submit("t", "source-drift", _request(workspace))
        assert caught.value.code == "service_reload_required"
        assert not list((service.state_root / "tenants").rglob("job_*.json"))
    finally:
        service.close()


def test_a_changed_source_build_still_returns_an_idempotent_existing_job(
    tmp_path, monkeypatch,
):
    import flyto_ai.coding.service as service_module

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    _declare_verification(workspace)
    box: dict = {}
    service = _emergency_service(tmp_path, workspace, box, enabled=False)
    try:
        first = service.submit("t", "existing-job", _request(workspace))
        monkeypatch.setattr(
            service_module,
            "current_service_build_id",
            lambda: "f" * 32 if service.build_id != "f" * 32 else "e" * 32,
        )
        repeated = service.submit("t", "existing-job", _request(workspace))
        assert repeated.job_id == first.job_id
    finally:
        service.close()


# ── service integration ───────────────────────────────────────────────


class EmergencyImplementer:
    """Writes a real file and passes a required check, like a real backend."""

    def __init__(self, store, session="sdk-emergency-1", fail=False, boom=False):
        self.store = store
        self.session = session
        self.rounds = 0
        self.requests: list = []
        self.fail = fail
        self.boom = boom

    async def run(self, request):
        self.rounds += 1
        self.requests.append(request.message)
        # A real adapter signals the host at its provider boundary, and only
        # there. This fixture stands in for a backend that reached that
        # boundary, so it signals before it can succeed *or* explode - which is
        # what makes "the durable start survives an exception" a real claim
        # rather than a property of being called at all.
        mark_provider_start(self.store)
        if self.boom:
            raise RuntimeError("implementer exploded")
        (Path(request.working_dir) / "notes.txt").write_text(
            "round {}\n".format(self.rounds), encoding="utf-8",
        )
        try:
            self.store.load(self.session, request.working_dir)
        except FileNotFoundError:
            self.store.create(request.working_dir, self.session)
        self.store.append(self.session, "coding.round", {"round": self.rounds})
        checks = [CheckResult(
            name="unit", passed=not self.fail, required=True,
            exit_code=0 if not self.fail else 1, duration_ms=1,
            output_sha256="0" * 64,
        )]
        return CodingTaskResult(
            ok=not self.fail, message="applied", thread_id=self.session,
            attempts=1, status="completed" if not self.fail else "failed",
            files_changed=["notes.txt"], checks=checks,
        )


def _emergency_service(
    tmp_path: Path,
    workspace: Path,
    box: dict,
    *,
    enabled: bool = True,
    emergency_backend: str = "claude",
    backend: str = "claude",
    indexer_argv=None,
    state_dir: str = "emergency-state",
    threshold: int = 1,
    implementer_kwargs=None,
) -> CodingService:
    """Build a strict audited service whose Indexer cannot be launched."""

    fixture = tmp_path / "blueprint_fixture.py"
    if not fixture.exists():
        fixture.write_text(BLUEPRINT_FIXTURE)
    policy = CodingRoutePolicy(
        strict=True,
        indexer=_indexer_spec(
            # An argv that does not exist is a real capability launch failure.
            argv=indexer_argv or (sys.executable, str(tmp_path / "absent.py")),
        ),
        blueprint=_blueprint_spec(argv=(sys.executable, str(fixture))),
        core_enabled=True,
    )

    def factory(store):
        if box.get("agent") is None:
            box["agent"] = EmergencyImplementer(store, **(implementer_kwargs or {}))
        else:
            box["agent"].store = store
        return box["agent"]

    return CodingService(
        factory,
        state_root=str(tmp_path / state_dir),
        workspace_roots=(str(workspace),),
        max_workers=1, max_queued=4,
        require_codex_audit=True,
        implementation_backend=backend,
        route_policy=policy,
        emergency_policy=(
            EmergencyOverflowPolicy(
                enabled=True, backend=emergency_backend, failure_threshold=threshold,
            ) if enabled else EmergencyOverflowPolicy()
        ),
    )


def _request(workspace: Path, message: str = "improve the workspace"):
    return CodingTaskRequest(message=message, working_dir=str(workspace))


def _settle(service, tenant, job_id, timeout=60):
    settled = TERMINAL_CODING_JOB_STATES | {CodingJobState.AWAITING_CODEX_AUDIT}
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        receipt = service.get(tenant, job_id)
        if receipt.state in settled:
            return receipt
        time.sleep(0.02)
    raise AssertionError("emergency job did not settle")


def _job_record(service, tenant, job_id):
    path = service._tenant_dir(service._tenant_ref(tenant)) / "jobs" / (job_id + ".json")
    return service._read_json(path)


def test_a_classified_launch_failure_overflows_once_and_reaches_audit(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    _declare_verification(workspace)
    box: dict = {}
    service = _emergency_service(tmp_path, workspace, box)
    try:
        queued = service.submit("t", "emergency-1", _request(workspace))
        awaiting = _settle(service, "t", queued.job_id)
        assert awaiting.state is CodingJobState.AWAITING_CODEX_AUDIT
        assert box["agent"].rounds == 1, "the overflow lane runs exactly one round"
        assert awaiting.implementer_started is True
        assert awaiting.implementation_backend == "claude"
        assert awaiting.landable is False
        # It never claims the strict lanes passed.
        assert awaiting.route_receipt is None
        authority = awaiting.emergency_authority
        assert authority["mode"] == "emergency"
        assert authority["circuit_state"] == "open"
        assert authority["trigger_lane"] == RouteLane.INDEXER_PRE.value
        assert authority["trigger_code"] == "capability_unavailable"
        assert authority["checks_enforced"] is True
        assert authority["audit_required"] is True
        assert authority["job_id"] == queued.job_id
        assert authority["revision_sha256"] == awaiting.implementation_revision_sha256

        accepted = service.audit(
            "t", queued.job_id, awaiting.implementation_revision_sha256,
            CodingAuditVerdict.ACCEPT, (),
        )
        assert accepted.state is CodingJobState.CODEX_ACCEPTED
        assert accepted.landable is True
    finally:
        service.close()


def test_emergency_rework_stays_in_the_same_session_and_authority(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    _declare_verification(workspace)
    box: dict = {}
    service = _emergency_service(tmp_path, workspace, box)
    try:
        queued = service.submit("t", "emergency-rework", _request(workspace))
        first = _settle(service, "t", queued.job_id)
        assert first.state is CodingJobState.AWAITING_CODEX_AUDIT

        service.audit(
            "t", queued.job_id, first.implementation_revision_sha256,
            CodingAuditVerdict.REWORK,
            (CodingAuditFinding("needs_test", "blocker", "add coverage"),),
        )
        second = _settle(service, "t", queued.job_id)
        assert second.state is CodingJobState.AWAITING_CODEX_AUDIT
        assert box["agent"].rounds == 2
        # Same session, new revision, still on the emergency authority path.
        assert second.implementation_session_id == first.implementation_session_id
        assert second.implementation_revision_sha256 != first.implementation_revision_sha256
        assert second.route_receipt is None
        assert second.emergency_authority["mode"] == "emergency_rework"
        assert second.emergency_authority["session_id"] == second.implementation_session_id

        accepted = service.audit(
            "t", queued.job_id, second.implementation_revision_sha256,
            CodingAuditVerdict.ACCEPT, (),
        )
        assert accepted.state is CodingJobState.CODEX_ACCEPTED
        assert accepted.landable is True
    finally:
        service.close()


def test_a_service_without_the_startup_flag_never_overflows(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    _declare_verification(workspace)
    box: dict = {}
    service = _emergency_service(tmp_path, workspace, box, enabled=False)
    try:
        queued = service.submit("t", "no-authority", _request(workspace))
        failed = _settle(service, "t", queued.job_id)
        assert failed.state is CodingJobState.FAILED
        assert failed.failure_code == "route_capability_unavailable"
        assert failed.implementer_started is False
        assert failed.emergency_authority is None
        assert box.get("agent") is None or box["agent"].rounds == 0
        assert not (workspace / "notes.txt").exists()
    finally:
        service.close()


def test_authority_granted_to_another_backend_never_overflows(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    _declare_verification(workspace)
    box: dict = {}
    with pytest.raises(ValueError, match="must match the selected implementation"):
        _emergency_service(
            tmp_path, workspace, box, backend="native", emergency_backend="claude",
        )


def test_a_transplanted_authority_starts_no_implementation(tmp_path):
    """A sealed receipt copied into another job must not reach the model."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    _declare_verification(workspace)
    donor_box: dict = {}
    donor = _emergency_service(tmp_path, workspace, donor_box, state_dir="donor")
    try:
        queued = donor.submit("t", "donor-job", _request(workspace))
        awaiting = _settle(donor, "t", queued.job_id)
        stolen = dict(awaiting.emergency_authority)
    finally:
        donor.close()

    victim_box: dict = {}
    victim = _emergency_service(tmp_path, workspace, victim_box, state_dir="victim")
    try:
        target = victim.submit("t", "victim-job", _request(workspace))
        path = (
            victim._tenant_dir(victim._tenant_ref("t"))
            / "jobs" / (target.job_id + ".json")
        )
        _settle(victim, "t", target.job_id)
        rounds_before = victim_box["agent"].rounds
        record = victim._read_json(path)
        record["emergency_authority"] = stolen
        record["state"] = CodingJobState.QUEUED.value
        record["implementer_started"] = False
        victim._write_json(path, record)

        # Re-running the round with a transplanted authority must not call the
        # implementer even once, whether it is offered as rework or not.
        progress = victim._read_json(path)
        assert progress["emergency_authority"]["job_id"] == queued.job_id
        with pytest.raises(EmergencyAuthorityMissing, match="another job"):
            victim._bound_emergency_authority(progress, target.job_id, rework=True)
        with pytest.raises(EmergencyAuthorityMissing, match="initial round"):
            victim._bound_emergency_authority(progress, target.job_id, rework=False)
        assert victim_box["agent"].rounds == rounds_before

        # It is not auditable either.
        with pytest.raises(EmergencyAuthorityMissing):
            victim._require_execution_authority(progress)
    finally:
        victim.close()


def test_an_initial_round_carrying_authority_fails_closed(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    _declare_verification(workspace)
    box: dict = {}
    service = _emergency_service(tmp_path, workspace, box)
    try:
        queued = service.submit("t", "replayed", _request(workspace))
        awaiting = _settle(service, "t", queued.job_id)
        record = _job_record(service, "t", queued.job_id)
        assert record["emergency_authority"]["job_id"] == queued.job_id
        # Its own valid authority is still illegal for a fresh initial round.
        with pytest.raises(EmergencyAuthorityMissing, match="initial round"):
            service._bound_emergency_authority(record, queued.job_id, rework=False)
        assert awaiting.state is CodingJobState.AWAITING_CODEX_AUDIT
    finally:
        service.close()


@pytest.mark.parametrize(("mutate", "cause"), [
    # Flipping a digested fact without redigesting is caught by the digest.
    (lambda a: {**a, "checks_enforced": False}, "digest does not match"),
    (lambda a: {**a, "implementer_backend": "native"}, "digest does not match"),
    # An emptied record cannot even be parsed back into an authority.
    (lambda a: {}, "mode is unknown"),
    (lambda a: {**a, "digest": ""}, "requires an open circuit"),
])
def test_tampered_or_empty_authority_is_never_auditable(tmp_path, mutate, cause):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    _declare_verification(workspace)
    box: dict = {}
    service = _emergency_service(tmp_path, workspace, box)
    try:
        queued = service.submit("t", "tamper", _request(workspace))
        awaiting = _settle(service, "t", queued.job_id)
        record = _job_record(service, "t", queued.job_id)
        forged = mutate(dict(awaiting.emergency_authority))
        if forged.get("digest") == "":
            # A cleared digest cannot be recomputed into authority either: the
            # remaining facts must still satisfy every coherence rule.
            forged["circuit_state"] = "closed"
        record["emergency_authority"] = forged
        with pytest.raises(EmergencyAuthorityMissing, match="is invalid") as caught:
            service._require_execution_authority(record)
        assert cause in str(caught.value.__cause__)
    finally:
        service.close()


def test_a_record_claiming_both_authorities_is_refused(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    _declare_verification(workspace)
    box: dict = {}
    service = _emergency_service(tmp_path, workspace, box)
    try:
        queued = service.submit("t", "mixed", _request(workspace))
        awaiting = _settle(service, "t", queued.job_id)
        record = _job_record(service, "t", queued.job_id)
        from flyto_ai.coding.route import CodingRouteReceipt

        record["route_receipt"] = CodingRouteReceipt(
            strict=False, ok=True, lanes=(),
        ).to_mapping()
        assert isinstance(awaiting.emergency_authority, dict)
        with pytest.raises(RouteEvidenceMissing, match="both route and emergency"):
            service._require_execution_authority(record)
    finally:
        service.close()


def test_a_domain_failure_never_opens_the_lane(tmp_path):
    """A reachable Indexer that refuses is not broken infrastructure."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    _declare_verification(workspace)
    fixture = tmp_path / "refusing_indexer.py"
    fixture.write_text(INDEXER_FIXTURE.replace(
        'return {"results": [{"symbol_id": "p:app.py:function:main"}]}, False',
        'return {"error": "refused"}, True',
    ))
    box: dict = {}
    service = _emergency_service(
        tmp_path, workspace, box, indexer_argv=(sys.executable, str(fixture)),
    )
    try:
        queued = service.submit("t", "domain-failure", _request(workspace))
        failed = _settle(service, "t", queued.job_id)
        assert failed.state is CodingJobState.FAILED
        assert failed.failure_code == "route_domain_failure"
        assert failed.implementer_started is False
        assert failed.emergency_authority is None
        assert box.get("agent") is None or box["agent"].rounds == 0
    finally:
        service.close()


def test_a_post_implementation_failure_never_overflows_but_keeps_its_proof(tmp_path):
    """Indexer post-work refusing is not a reason to re-run the model."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    _declare_verification(workspace)
    fixture = tmp_path / "post_failing_indexer.py"
    fixture.write_text(INDEXER_FIXTURE.replace(
        'if action == "validate":\n        return {"pass": True, "checks": []}, False',
        'if action == "validate":\n        return {"pass": False, "checks": []}, False',
    ))
    box: dict = {}
    service = _emergency_service(
        tmp_path, workspace, box, indexer_argv=(sys.executable, str(fixture)),
    )
    try:
        queued = service.submit("t", "post-failure", _request(workspace))
        failed = _settle(service, "t", queued.job_id)
        assert failed.state is CodingJobState.FAILED
        assert failed.failure_code == "route_validation_failed"
        # The implementer really ran exactly once and is not run again.
        assert box["agent"].rounds == 1
        assert failed.implementer_started is True
        assert failed.emergency_authority is None
        assert failed.landable is False
        # Proof that implementation happened survives the failure.
        assert failed.implementation_session_id == "sdk-emergency-1"
        assert len(failed.implementation_revision_sha256) == 64
        # It is still not auditable.
        with pytest.raises(RouteEvidenceMissing):
            service._require_execution_authority(
                _job_record(service, "t", queued.job_id),
            )
    finally:
        service.close()


def test_a_failed_check_on_the_overflow_lane_is_terminal_and_non_landable(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    _declare_verification(workspace)
    box: dict = {}
    service = _emergency_service(
        tmp_path, workspace, box, implementer_kwargs={"fail": True},
    )
    try:
        queued = service.submit("t", "failing-checks", _request(workspace))
        failed = _settle(service, "t", queued.job_id)
        assert failed.state is CodingJobState.FAILED
        assert failed.landable is False
        assert failed.implementer_started is True
        # It still says it ran through the overflow lane.
        record = _job_record(service, "t", queued.job_id)
        assert route_mode(record) == "emergency"
        assert record["emergency_trigger"]["code"] == "capability_unavailable"
        assert box["agent"].rounds == 1
    finally:
        service.close()


def test_the_durable_start_and_trigger_survive_an_implementer_exception(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    _declare_verification(workspace)
    box: dict = {}
    service = _emergency_service(
        tmp_path, workspace, box, implementer_kwargs={"boom": True},
    )
    try:
        queued = service.submit("t", "exploding", _request(workspace))
        failed = _settle(service, "t", queued.job_id)
        assert failed.state is CodingJobState.FAILED
        assert failed.failure_code == "service_execution_failed"
        assert failed.implementer_started is True
        record = _job_record(service, "t", queued.job_id)
        # The overflow attempt is durable even though the model never returned.
        assert record["execution_mode"] == "emergency"
        assert record["emergency_trigger"]["lane"] == "indexer_pre"
        assert record["emergency_authority"] is None

        status = json.loads(service._status.instance_path().read_text())
        assert status["mode"] == "emergency"
        assert status["implementer_started"] is True
        assert status["lane"] == "indexer_pre"
    finally:
        service.close()


def test_a_strict_success_carries_no_emergency_fields(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    _declare_verification(workspace)
    fixture = tmp_path / "working_indexer.py"
    fixture.write_text(INDEXER_FIXTURE)
    box: dict = {}
    service = _emergency_service(
        tmp_path, workspace, box, indexer_argv=(sys.executable, str(fixture)),
    )
    try:
        queued = service.submit("t", "strict-ok", _request(workspace))
        awaiting = _settle(service, "t", queued.job_id)
        assert awaiting.state is CodingJobState.AWAITING_CODEX_AUDIT
        assert awaiting.route_receipt["ok"] is True
        assert awaiting.emergency_authority is None
        assert awaiting.implementer_started is True
        record = _job_record(service, "t", queued.job_id)
        assert route_mode(record) == "strict"
        assert record["emergency_trigger"] is None
        assert service._breaker.state == "closed"
        accepted = service.audit(
            "t", queued.job_id, awaiting.implementation_revision_sha256,
            CodingAuditVerdict.ACCEPT, (),
        )
        assert accepted.landable is True
    finally:
        service.close()


def test_status_tracks_a_job_and_survives_a_graceful_close(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    _declare_verification(workspace)
    box: dict = {}
    service = _emergency_service(tmp_path, workspace, box, enabled=False)
    instance_path = service._status.instance_path()
    try:
        queued = service.submit("t", "status-1", _request(workspace))
        assert json.loads(instance_path.read_text())["job_id"] == queued.job_id
        failed = _settle(service, "t", queued.job_id)
        assert failed.state is CodingJobState.FAILED
        live = json.loads(instance_path.read_text())
        assert live["state"] == "failed"
        assert live["lifecycle"] == "active"
        assert live["failure_code"] == "route_capability_unavailable"
        assert live["lane"] == "indexer_pre"
        assert live["implementer_started"] is False
        assert service.status_health()["failures"] == 0
    finally:
        service.close()

    closed = json.loads(instance_path.read_text())
    # Closing changes lifecycle and time only; every diagnostic fact remains.
    assert closed["lifecycle"] == "closed"
    assert closed["job_id"] == live["job_id"]
    assert closed["state"] == "failed"
    assert closed["failure_code"] == "route_capability_unavailable"
    assert closed["lane"] == "indexer_pre"
    assert closed["updated_at"] >= live["updated_at"]

    # A later instance sees the old one as a distinct, closed row.
    reader = _publisher(tmp_path / "emergency-state", "9" * 24)
    rows = {row["instance_id"]: row for row in reader.inspect()}
    assert closed["instance_id"] in rows
    assert rows[closed["instance_id"]]["lifecycle"] == "closed"


def test_a_broken_recorder_is_counted_with_a_stable_code_and_recovers(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    _declare_verification(workspace)
    box: dict = {}
    service = _emergency_service(tmp_path, workspace, box, enabled=False)
    try:
        original = service._status.publish

        def broken(status):
            raise OSError("state root is unwritable")

        service._status.publish = broken
        service._publish_status({"job_id": "job_" + "a" * 24, "state": "queued"})
        health = service.status_health()
        assert health["failures"] == 1
        assert health["last_failure_code"] == "status_write_failed"

        service._status.publish = original
        service._publish_status({"job_id": "job_" + "a" * 24, "state": "queued"})
        published = json.loads(service._status.instance_path().read_text())
        # The recovered status reports the earlier failure rather than hiding it.
        assert published["publish_failures"] == 1
        assert published["last_publish_failure_code"] == "status_write_failed"
        assert service.status_health()["published"] >= 1
    finally:
        service.close()


def test_two_service_processes_share_one_state_root_without_clobbering(tmp_path):
    """Each instance owns its own file; the shared index lists them both."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    _declare_verification(workspace)
    first_box: dict = {}
    second_box: dict = {}
    first = _emergency_service(
        tmp_path, workspace, first_box, enabled=False, state_dir="shared",
    )
    second = _emergency_service(
        tmp_path, workspace, second_box, enabled=False, state_dir="shared",
    )
    try:
        assert first.instance_id != second.instance_id
        assert first.build_id == second.build_id
        one = first.submit("t", "shared-1", _request(workspace))
        _settle(first, "t", one.job_id)
        two = second.submit("t", "shared-2", _request(workspace, "second task"))
        _settle(second, "t", two.job_id)

        # Neither instance overwrote the other's status file.
        first_status = json.loads(first._status.instance_path().read_text())
        second_status = json.loads(second._status.instance_path().read_text())
        assert first_status["job_id"] == one.job_id
        assert second_status["job_id"] == two.job_id
        rows = {row["instance_id"]: row for row in second._status.inspect()}
        assert {first.instance_id, second.instance_id} <= set(rows)
        assert rows[first.instance_id]["job_id"] == one.job_id
        # Both jobs stay independently readable from either process.
        assert first.get("t", two.job_id).job_id == two.job_id
        assert second.get("t", one.job_id).job_id == one.job_id
    finally:
        first.close()
        second.close()


def test_an_old_build_row_stays_distinguishable_from_the_repaired_build(tmp_path):
    """A pre-upgrade instance keeps its build id; a new one publishes its own."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    _declare_verification(workspace)
    box: dict = {}
    service = _emergency_service(tmp_path, workspace, box, enabled=False)
    try:
        legacy = _publisher(
            service.state_root, "7" * 24, build_id="0" * 24, version="1.0.0",
        )
        legacy.publish(_status(
            instance_id="7" * 24, build_id="0" * 24, service_version="1.0.0",
            updated_at=time.time(),
        ))
        queued = service.submit("t", "build-compare", _request(workspace))
        _settle(service, "t", queued.job_id)

        rows = {row["instance_id"]: row for row in service._status.inspect()}
        assert rows["7" * 24]["build_id"] == "0" * 24
        assert rows["7" * 24]["service_version"] == "1.0.0"
        assert rows[service.instance_id]["build_id"] == service.build_id
        assert rows[service.instance_id]["build_id"] != "0" * 24
        assert rows[service.instance_id]["current"] is True
    finally:
        service.close()
