"""The host release valve against a state root it did not configure.

The release valve's whole purpose is to retire a job whose worker is gone. In
production that worker is a *strict* route — a Claude implementer, the four
mandatory lanes, an emergency overflow grant — and the operator running the
valve is not. Building an ordinary service to do the release therefore built an
ordinary service's *startup authority* too, and the state root refused it: the
recorded authority did not match, so rotation was attempted, and rotation
requires every job terminal — including the one open `awaiting_codex_audit` job
the command existed to retire. The release was impossible exactly when it was
needed.

These tests plant a genuinely non-default authority (Claude, strict route,
emergency overflow — computed from a real service configured that way, never
hand-invented) and then drive the real CLI as an operator would.
"""
from __future__ import annotations

import json
import os
import threading
from pathlib import Path

import pytest
from flyto_ai.coding.continuation import (
    STATE_SETTLED,
    ContinuationStore,
    is_continuable_session,
    session_ref,
)
from flyto_ai.coding.contracts import CodingAuditVerdict, CodingJobState
from flyto_ai.coding.emergency import EmergencyOverflowPolicy
from flyto_ai.coding.mcp_server import CodingMCPServer
from flyto_ai.coding.service import (
    AUTHORITY_LOCK_NAME,
    AUTHORITY_MARKER_NAME,
    AUTHORITY_MARKER_VERSION,
    SERVICE_LOCK_NAME,
    CodingService,
    CodingServiceBusy,
    CodingServiceError,
    HostReleaseValveRefused,
    HostReleaseValveRootUnusable,
)

from tests.test_cli_code_release import _run
from tests.test_coding_route import _policy
from tests.test_coding_service import (
    ReworkingProvider,
    _audited_service,
    _awaiting,
    _blocker,
    _request,
)

TENANT = "tenant-audit"


def _never_implements(store):
    raise AssertionError("the authority probe must never run a coding round")


def _claude_strict_authority(tmp_path: Path) -> dict:
    """The exact execution authority a Claude strict-route host records.

    Computed by constructing a service with that configuration rather than by
    writing a dict, so the marker these tests plant is the same shape and the
    same digests the real strict route would leave behind. The probe lives on
    its own throwaway state root and never submits anything.
    """

    probe_workspace = tmp_path / "authority-probe-workspace"
    probe_workspace.mkdir()
    probe = CodingService(
        _never_implements,
        state_root=str(tmp_path / "authority-probe-state"),
        workspace_roots=(str(probe_workspace),),
        max_workers=1,
        max_queued=4,
        require_codex_audit=True,
        implementation_backend="claude",
        route_policy=_policy(),
        emergency_policy=EmergencyOverflowPolicy(
            enabled=True, backend="claude", failure_threshold=2,
        ),
    )
    try:
        authority = probe._execution_authority()
    finally:
        probe.close()
    return authority


def _record_paths(state_dir: Path) -> list:
    return sorted((state_dir / "tenants").glob("*/jobs/job_*.json"))


def _plant_authority(state_dir: Path, authority: dict) -> None:
    """Make this state root read as one a strict Claude route bound and left.

    Both halves matter. The marker is what an ordinary construction compares
    itself against, and the per-record fingerprint is what `_bind_startup_authority`
    walks — a test that planted only one of them would prove less than the bug.
    """

    (state_dir / AUTHORITY_MARKER_NAME).write_text(
        json.dumps(
            {"marker_version": AUTHORITY_MARKER_VERSION, "authority": authority},
            sort_keys=True,
            indent=2,
        ),
        encoding="utf-8",
    )
    for path in _record_paths(state_dir):
        record = json.loads(path.read_text(encoding="utf-8"))
        record["execution_authority"] = dict(authority)
        path.write_text(json.dumps(record, sort_keys=True), encoding="utf-8")


def _orphaned_pair(tmp_path: Path):
    """Two audit-ready jobs on two worktrees, sharing one state root."""

    state_dir = tmp_path / "service-state"
    left = tmp_path / "workspace-left"
    left.mkdir()
    right = tmp_path / "workspace-right"
    right.mkdir()

    # Both services declare the same tree set: they share one state root, and
    # the configured set is part of startup authority precisely so two
    # differently-configured workers cannot certify each other's jobs.
    service = _audited_service(
        tmp_path, left, provider=ReworkingProvider(), extra_roots=(right,),
    )
    try:
        first = _awaiting(service, TENANT, "strict-001", left)
        left_claim = Path(service._workspace_claim_path(str(left)))
    finally:
        service.close()

    service = _audited_service(
        tmp_path, right, provider=ReworkingProvider(), extra_roots=(left,),
    )
    try:
        second = _awaiting(service, TENANT, "strict-002", right)
        right_claim = Path(service._workspace_claim_path(str(right)))
    finally:
        service.close()

    return {
        "state_dir": state_dir,
        "left": left,
        "right": right,
        "first": first,
        "second": second,
        "left_claim": left_claim,
        "right_claim": right_claim,
    }


def _tenant_ref() -> str:
    return CodingService._tenant_ref(TENANT)


def _resume_path(state_dir: Path, job_id: str) -> Path:
    """The durable resume envelope — the artifact rework is rebuilt from.

    An `awaiting_codex_audit` job always has one: it is what lets a rework round
    reach a worker that never held the implementation session in memory. It is
    therefore the concrete thing a release has to take away, and the concrete
    thing a release of a *different* job must leave alone.
    """

    return state_dir / "tenants" / _tenant_ref() / "resume" / (job_id + ".json")


def _resumable(state_dir: Path, job_id: str) -> bool:
    return _resume_path(state_dir, job_id).exists()


def _continuation_state(state_dir: Path, record: dict) -> str:
    """What a later worker could still re-enter for this job.

    `unavailable` means no authority a resume could ever find: either the record
    names no continuable session, or the session names no stored authority.
    Anything else is the authority's own state, so a live `open` or `claimed`
    continuation surviving a release shows up here as itself rather than being
    quietly folded into a pass.
    """

    session = str(record.get("continuation_session_id") or "")
    if not is_continuable_session(session):
        return "unavailable"
    authority = ContinuationStore(state_dir).load(_tenant_ref(), session)
    if authority is None:
        return "unavailable"
    return authority.state


def _continuation_bytes(state_dir: Path, *, without: dict) -> dict:
    """Every stored continuation except the one belonging to `without`.

    Excluding the released job's own entry is deliberate: settling its authority
    is a *write*, so comparing the whole directory would forbid the very
    cleanup this release promises. What must not move is everybody else's.
    """

    directory = state_dir / "tenants" / _tenant_ref() / "continuation"
    if not directory.is_dir():
        return {}
    session = str(without.get("continuation_session_id") or "")
    excluded = session_ref(session) if is_continuable_session(session) else ""
    return {
        path.name: path.read_bytes()
        for path in sorted(directory.iterdir())
        if path.is_file() and not (excluded and path.name.startswith(excluded))
    }


def _record_for(state_dir: Path, job_id: str) -> Path:
    for path in _record_paths(state_dir):
        if path.stem == job_id:
            return path
    raise AssertionError("no durable record for {}".format(job_id))


def _cli_args(fixture) -> tuple:
    return (
        "--tenant", TENANT,
        "--workspace-root", str(fixture["left"]),
        "--workspace-root", str(fixture["right"]),
        "--state-dir", str(fixture["state_dir"]),
    )


def test_an_ordinary_service_still_cannot_bind_the_planted_root(
    tmp_path: Path,
) -> None:
    """The regression's premise, asserted rather than assumed.

    If a default service could simply start here, the valve would be solving
    nothing. This pins the failure the release path has to route around.
    """

    fixture = _orphaned_pair(tmp_path)
    _plant_authority(fixture["state_dir"], _claude_strict_authority(tmp_path))
    from flyto_ai.coding.service import CodingAuthorityConflict

    with pytest.raises(CodingAuthorityConflict):
        _audited_service(tmp_path, fixture["left"], provider=ReworkingProvider())


def test_release_retires_one_foreign_authority_job_and_leaves_the_rest(
    tmp_path: Path, monkeypatch, capsys,
) -> None:
    """The whole contract of the fix, in one operator command."""

    fixture = _orphaned_pair(tmp_path)
    _plant_authority(fixture["state_dir"], _claude_strict_authority(tmp_path))

    state_dir = fixture["state_dir"]
    target_id = fixture["first"].job_id
    survivor_id = fixture["second"].job_id

    marker = state_dir / AUTHORITY_MARKER_NAME
    marker_before = marker.read_bytes()
    survivor = _record_for(state_dir, survivor_id)
    survivor_before = survivor.read_bytes()
    assert fixture["left_claim"].exists() and fixture["right_claim"].exists()

    # The release's cleanup contract is about artifacts that exist. Both jobs
    # are audit-ready, so both must currently be reworkable from disk; a test
    # that asserted removal without first proving presence would pass on an
    # empty directory.
    assert _resumable(state_dir, target_id)
    assert _resumable(state_dir, survivor_id)
    survivor_resume_before = _resume_path(state_dir, survivor_id).read_bytes()
    target_before = json.loads(_record_for(state_dir, target_id).read_text())
    continuation_before = _continuation_bytes(state_dir, without=target_before)
    survivor_continuation_before = _continuation_state(
        state_dir, json.loads(survivor.read_text()),
    )

    code, out, err = _run(
        monkeypatch, capsys, *_cli_args(fixture),
        "--abandon-job", target_id, "--json",
    )
    assert code == 0, err
    assert json.loads(out) == {
        "operation": "abandon_job",
        "job_id": target_id,
        "state": CodingJobState.FAILED.value,
        "failure_code": "job_abandoned",
        "landable": False,
    }

    # The recorded authority is not rotated, weakened, or reproduced.
    assert marker.read_bytes() == marker_before

    # Exactly one job moved, and it moved only to failed/job_abandoned. Its own
    # fingerprint is left as the strict route wrote it: the valve subtracted a
    # job, it did not re-attribute one.
    retired_record = _record_for(state_dir, target_id)
    retired = json.loads(retired_record.read_text())
    assert retired["state"] == CodingJobState.FAILED.value
    assert retired["failure_code"] == "job_abandoned"
    assert retired["landable"] is False
    assert retired["execution_authority"] == json.loads(marker_before)["authority"]

    # The release is not just a state change. The retired job's durable resume
    # envelope is gone, so no worker can rebuild its rework round from disk...
    assert not _resumable(state_dir, target_id)
    # ...and nothing resumable is left behind for it either: a settled
    # authority, or none at all. An `open` or `claimed` authority surviving here
    # would mean an abandoned job could still be re-entered.
    assert _continuation_state(state_dir, retired) in {STATE_SETTLED, "unavailable"}

    # The other open job under the same recorded authority is untouched, in its
    # record, its resume envelope, and its continuation authority alike.
    assert survivor.read_bytes() == survivor_before
    assert _resumable(state_dir, survivor_id)
    assert _resume_path(state_dir, survivor_id).read_bytes() == survivor_resume_before
    assert _continuation_state(
        state_dir, json.loads(survivor.read_text()),
    ) == survivor_continuation_before
    assert _continuation_bytes(state_dir, without=retired) == continuation_before

    # Only the target worktree is released.
    assert not fixture["left_claim"].exists()
    assert fixture["right_claim"].exists()


def test_abandon_retires_only_the_target_while_a_live_peer_holds_the_lease(
    tmp_path: Path, monkeypatch, capsys,
) -> None:
    """Unrelated live peers no longer pin a kernel-accounted orphan forever."""

    fixture = _orphaned_pair(tmp_path)
    marker = fixture["state_dir"] / AUTHORITY_MARKER_NAME
    survivor = _record_for(fixture["state_dir"], fixture["second"].job_id)
    live = _audited_service(
        tmp_path,
        fixture["left"],
        provider=ReworkingProvider(),
        extra_roots=(fixture["right"],),
    )
    try:
        before = marker.read_bytes()
        survivor_before = survivor.read_bytes()
        code, out, err = _run(
            monkeypatch, capsys, *_cli_args(fixture),
            "--abandon-job", fixture["first"].job_id, "--json",
        )
        assert code == 0, err
        assert "Traceback" not in err and "Traceback" not in out
        assert marker.read_bytes() == before
        assert live.get(TENANT, fixture["first"].job_id).state is CodingJobState.FAILED
        assert survivor.read_bytes() == survivor_before
        assert live.get(TENANT, fixture["second"].job_id).state is (
            CodingJobState.AWAITING_CODEX_AUDIT
        )
    finally:
        live.close()


def test_repair_workspace_stays_fail_closed_under_a_foreign_authority(
    tmp_path: Path, monkeypatch, capsys,
) -> None:
    """Reaching the evaluation is the fix; refusing a live owner is unchanged."""

    fixture = _orphaned_pair(tmp_path)
    _plant_authority(fixture["state_dir"], _claude_strict_authority(tmp_path))

    code, _out, err = _run(
        monkeypatch, capsys, *_cli_args(fixture),
        "--repair-workspace", str(fixture["left"]), "--json",
    )
    assert code == 2
    assert "workspace_busy" in err
    assert fixture["left_claim"].exists()


def _valve(fixture) -> CodingService:
    return CodingService.open_host_release_valve(
        state_root=str(fixture["state_dir"]),
        workspace_roots=(str(fixture["left"]), str(fixture["right"])),
    )


def _online_abandon_valve(fixture) -> CodingService:
    return CodingService.open_host_abandon_valve(
        state_root=str(fixture["state_dir"]),
        workspace_roots=(str(fixture["left"]), str(fixture["right"])),
    )


def test_online_abandon_valve_refuses_claim_repair_and_additive_work(
    tmp_path: Path,
) -> None:
    """Sharing the authority lease never turns the valve into a worker."""

    fixture = _orphaned_pair(tmp_path)
    live = _audited_service(
        tmp_path,
        fixture["left"],
        provider=ReworkingProvider(),
        extra_roots=(fixture["right"],),
    )
    valve = _online_abandon_valve(fixture)
    try:
        assert valve._authority_fd != -1
        assert valve._release_valve_can_repair_workspace is False
        with pytest.raises(HostReleaseValveRefused):
            valve.repair_workspace_claim(str(fixture["left"]))
        with pytest.raises(HostReleaseValveRefused):
            valve.submit(TENANT, "online-valve-001", _request(fixture["left"]))
        with pytest.raises(HostReleaseValveRefused):
            valve.audit(
                TENANT,
                fixture["first"].job_id,
                "c3" * 32,
                CodingAuditVerdict.ACCEPT,
                (),
            )
        with pytest.raises(HostReleaseValveRefused):
            valve._pump_dispatch()
    finally:
        valve.close()
        live.close()


def test_online_abandon_refuses_a_target_whose_job_lease_is_live(
    tmp_path: Path, monkeypatch, capsys,
) -> None:
    """The shared root lease cannot override the exact execution lease."""

    fixture = _orphaned_pair(tmp_path)
    target = _record_for(fixture["state_dir"], fixture["first"].job_id)
    marker = fixture["state_dir"] / AUTHORITY_MARKER_NAME
    lease_path = (
        fixture["state_dir"] / "locks" / "jobs"
        / (fixture["first"].job_id + ".lock")
    )
    import fcntl

    lease_fd = os.open(lease_path, os.O_CREAT | os.O_RDWR, 0o600)
    fcntl.flock(lease_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    before_record = target.read_bytes()
    before_marker = marker.read_bytes()
    try:
        code, out, err = _run(
            monkeypatch, capsys, *_cli_args(fixture),
            "--abandon-job", fixture["first"].job_id, "--json",
        )
        assert code == 2
        assert "service_busy" in err
        assert "Traceback" not in err and "Traceback" not in out
        assert target.read_bytes() == before_record
        assert marker.read_bytes() == before_marker
    finally:
        fcntl.flock(lease_fd, fcntl.LOCK_UN)
        os.close(lease_fd)


def test_online_abandon_valve_close_releases_both_descriptors(tmp_path: Path) -> None:
    """The live-safe constructor has the same bounded descriptor lifetime."""

    fixture = _orphaned_pair(tmp_path)
    valve = _online_abandon_valve(fixture)
    valve.close()

    assert valve._authority_fd == -1
    assert valve._lock_fd == -1
    exclusive = _valve(fixture)
    exclusive.close()


def test_online_abandon_and_audit_have_one_serialized_winner(tmp_path: Path) -> None:
    """Concurrent audit and abandon cannot both authorize the same job."""

    fixture = _orphaned_pair(tmp_path)
    live = _audited_service(
        tmp_path,
        fixture["left"],
        provider=ReworkingProvider(),
        extra_roots=(fixture["right"],),
    )
    valve = _online_abandon_valve(fixture)
    barrier = threading.Barrier(2)
    outcomes = {}

    def audit() -> None:
        barrier.wait()
        try:
            live.audit(
                TENANT,
                fixture["first"].job_id,
                fixture["first"].implementation_revision_sha256,
                CodingAuditVerdict.ACCEPT,
                (),
            )
        except CodingServiceError as exc:
            outcomes["audit"] = exc.code
        else:
            outcomes["audit"] = "accepted"

    def abandon() -> None:
        barrier.wait()
        try:
            valve.abandon(TENANT, fixture["first"].job_id)
        except CodingServiceError as exc:
            outcomes["abandon"] = exc.code
        else:
            outcomes["abandon"] = "abandoned"

    threads = [threading.Thread(target=audit), threading.Thread(target=abandon)]
    try:
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=5)
        assert not any(thread.is_alive() for thread in threads)
        assert sorted(outcomes.values()).count("accepted") + sorted(
            outcomes.values(),
        ).count("abandoned") == 1
        final = live.get(TENANT, fixture["first"].job_id)
        assert final.state in {CodingJobState.CODEX_ACCEPTED, CodingJobState.FAILED}
        assert not fixture["left_claim"].exists()
    finally:
        valve.close()
        live.close()


def test_the_valve_refuses_every_additive_operation(tmp_path: Path) -> None:
    """Subtractive by construction, not merely by the CLI's choice of calls."""

    fixture = _orphaned_pair(tmp_path)
    _plant_authority(fixture["state_dir"], _claude_strict_authority(tmp_path))
    valve = _valve(fixture)
    try:
        # No implementer exists, and no status row is published for a valve.
        with pytest.raises(HostReleaseValveRefused):
            valve.agent_factory(None)
        assert valve._status is None

        with pytest.raises(HostReleaseValveRefused):
            valve.submit(TENANT, "valve-001", _request(fixture["left"]))
        with pytest.raises(HostReleaseValveRefused):
            valve.audit(
                TENANT, fixture["first"].job_id, "c3" * 32,
                CodingAuditVerdict.ACCEPT, (),
            )
        with pytest.raises(HostReleaseValveRefused):
            valve.audit(
                TENANT, fixture["first"].job_id, "c3" * 32,
                CodingAuditVerdict.REWORK, (_blocker(),),
            )
        with pytest.raises(HostReleaseValveRefused):
            valve._pump_dispatch()
    finally:
        valve.close()


def test_the_valve_publishes_no_fourth_mcp_tool(tmp_path: Path) -> None:
    """The audited inventory is exactly submit/get/audit, valve or not."""

    fixture = _orphaned_pair(tmp_path)
    _plant_authority(fixture["state_dir"], _claude_strict_authority(tmp_path))
    valve = _valve(fixture)
    try:
        listed = CodingMCPServer(valve, TENANT).handle({
            "jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {},
        })
        names = [tool["name"] for tool in listed["result"]["tools"]]
        assert names == [
            "flyto_coding_submit", "flyto_coding_get", "flyto_coding_audit",
        ]
        assert not any(
            token in name
            for name in names
            for token in ("abandon", "release", "repair")
        )
    finally:
        valve.close()


def test_the_valve_leaves_the_marker_untouched_when_it_refuses(
    tmp_path: Path, monkeypatch, capsys,
) -> None:
    """A refused release is a no-op, including on an unknown job id."""

    fixture = _orphaned_pair(tmp_path)
    _plant_authority(fixture["state_dir"], _claude_strict_authority(tmp_path))
    marker = fixture["state_dir"] / AUTHORITY_MARKER_NAME
    before = marker.read_bytes()
    records = {path: path.read_bytes() for path in _record_paths(fixture["state_dir"])}

    code, out, err = _run(
        monkeypatch, capsys, *_cli_args(fixture),
        "--abandon-job", "job_" + "9" * 24, "--json",
    )
    assert code == 2
    assert "Traceback" not in err and "Traceback" not in out
    assert marker.read_bytes() == before
    assert {path: path.read_bytes() for path in _record_paths(fixture["state_dir"])} == records


# --- construction boundary -------------------------------------------------
#
# The valve must prove exclusivity *before* it creates anything. These
# regressions are behavioural on purpose: they inspect the filesystem and the
# constructed object rather than asserting that some flag was set.


def _tree(root: Path) -> set:
    """Every path under `root`, relative, or empty when the root is absent."""

    if not root.exists():
        return set()
    return {str(item.relative_to(root)) for item in root.rglob("*")}


def test_a_nonexistent_root_is_refused_and_not_created(tmp_path: Path) -> None:
    """The valve never brings a state root into existence."""

    missing = tmp_path / "no-such-root"
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    with pytest.raises(HostReleaseValveRootUnusable) as excinfo:
        CodingService.open_host_release_valve(
            state_root=str(missing), workspace_roots=(str(workspace),),
        )

    assert excinfo.value.code == "release_valve_root_unusable"
    assert not missing.exists()


def test_a_partial_root_is_refused_and_left_exactly_as_found(tmp_path: Path) -> None:
    """A directory no service established is not completed into one.

    This is the case that made the old path unsafe: the ordinary constructor
    would have created `.service.lock`, `locks/jobs`, `locks/workspaces` and
    the authority lease here, and only then discovered whether it was allowed.
    """

    partial = tmp_path / "partial-root"
    partial.mkdir(mode=0o700)
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    with pytest.raises(HostReleaseValveRootUnusable):
        CodingService.open_host_release_valve(
            state_root=str(partial), workspace_roots=(str(workspace),),
        )

    assert _tree(partial) == set()
    assert not (partial / SERVICE_LOCK_NAME).exists()
    assert not (partial / AUTHORITY_LOCK_NAME).exists()
    assert not (partial / "locks").exists()


def test_a_root_missing_only_its_lock_directories_is_refused(tmp_path: Path) -> None:
    """Every piece of the established furniture is required, not just one."""

    root = tmp_path / "half-root"
    root.mkdir(mode=0o700)
    (root / SERVICE_LOCK_NAME).write_bytes(b"")
    (root / AUTHORITY_LOCK_NAME).write_bytes(b"")
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    with pytest.raises(HostReleaseValveRootUnusable):
        CodingService.open_host_release_valve(
            state_root=str(root), workspace_roots=(str(workspace),),
        )

    assert not (root / "locks").exists()


def test_a_symlinked_state_root_is_refused_without_writing_through_it(
    tmp_path: Path,
) -> None:
    """A link in the path has no lookup to intercept, so nothing lands beyond it."""

    fixture = _orphaned_pair(tmp_path)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    link = tmp_path / "linked-root"
    link.symlink_to(elsewhere, target_is_directory=True)

    with pytest.raises(HostReleaseValveRootUnusable):
        CodingService.open_host_release_valve(
            state_root=str(link), workspace_roots=(str(fixture["left"]),),
        )

    assert _tree(elsewhere) == set()


def test_a_live_service_blocks_the_valve_with_zero_mutation(tmp_path: Path) -> None:
    """`service_busy`, and every durable byte is unchanged afterwards."""

    fixture = _orphaned_pair(tmp_path)
    marker = fixture["state_dir"] / AUTHORITY_MARKER_NAME
    live = _audited_service(
        tmp_path,
        fixture["left"],
        provider=ReworkingProvider(),
        extra_roots=(fixture["right"],),
    )
    try:
        before_marker = marker.read_bytes()
        before_records = {
            path: path.read_bytes() for path in _record_paths(fixture["state_dir"])
        }

        with pytest.raises(CodingServiceBusy) as excinfo:
            _valve(fixture)

        assert excinfo.value.code == "service_busy"
        assert marker.read_bytes() == before_marker
        assert {
            path: path.read_bytes() for path in _record_paths(fixture["state_dir"])
        } == before_records
    finally:
        live.close()


def test_valve_construction_builds_no_runtime_machinery(tmp_path: Path) -> None:
    """No executor, mission runtime, status publisher or background thread.

    Asserted on the constructed object, not on a flag: these are `None`
    because nothing ever built them, so no later path can start one.
    """

    fixture = _orphaned_pair(tmp_path)
    _plant_authority(fixture["state_dir"], _claude_strict_authority(tmp_path))
    before_threads = threading.active_count()
    valve = _valve(fixture)
    try:
        assert valve._executor is None
        assert valve._mission is None
        assert valve._status is None
        assert valve._release_valve is True
        assert valve._authority_fd != -1  # exclusivity really was proven
        assert threading.active_count() == before_threads
        assert not [
            thread for thread in threading.enumerate()
            if thread.name.startswith("flyto-coding")
        ]
    finally:
        valve.close()


def test_valve_close_releases_every_descriptor(tmp_path: Path) -> None:
    """After close, the exclusive lease is available again."""

    fixture = _orphaned_pair(tmp_path)
    valve = _valve(fixture)
    valve.close()

    assert valve._authority_fd == -1
    assert valve._lock_fd == -1
    # The strongest available proof the lease really was released: a second
    # valve can take it exclusively, which a still-held lease would refuse.
    again = _valve(fixture)
    try:
        assert again._authority_fd != -1
    finally:
        again.close()


def test_a_refused_constructor_leaves_no_descriptor_behind(tmp_path: Path) -> None:
    """A refusal after the first descriptor still closes it.

    Proven by refusing many times: one leaked descriptor per attempt would
    exhaust this process's file table long before the loop ends, and the
    successful valve afterwards would fail to open anything.
    """

    fixture = _orphaned_pair(tmp_path)
    partial = tmp_path / "partial-root"
    partial.mkdir(mode=0o700)
    (partial / SERVICE_LOCK_NAME).write_bytes(b"")

    for _ in range(400):
        with pytest.raises(HostReleaseValveRootUnusable):
            CodingService.open_host_release_valve(
                state_root=str(partial), workspace_roots=(str(fixture["left"]),),
            )

    valve = _valve(fixture)
    try:
        assert valve._authority_fd != -1
    finally:
        valve.close()
