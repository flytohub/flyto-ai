# Copyright 2024 Flyto2
# Licensed under the Apache License, Version 2.0
"""Demand-scoped host-global tree ownership.

Ownership of a configured workspace tree is held only while this state root has
durable non-terminal work, never for the lifetime of an idle service. An idle
Codex worker that kept its trees for its whole task lifetime blocked every
other state root indefinitely, even after its one job settled. These tests pin
the acquire-on-first-job / release-on-last-terminal invariant, the two-state-
root behaviour, the same-state-root peer gap, and the audit-versus-release race.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from flyto_ai.coding.contracts import CodingAuditVerdict, CodingJobState
from flyto_ai.coding.service import (
    AbandonStateConflict,
    CodingService,
    CodingWorkspaceAuthorityConflict,
)
from flyto_ai.coding.workspace_authority import (
    describe_workspace_root,
    state_root_has_open_work,
)

from tests.test_coding_service import (
    ReworkingProvider,
    _audited_service,
    _awaiting,
    _blocker,
    _request,
    _wait,
)
from tests.test_coding_workspace_ownership import _declare_verification, _workspace

pytest.importorskip("fcntl")


def _owns(service: CodingService) -> bool:
    return service._workspace_root_authority is not None


def _accept(service: CodingService, tenant: str, receipt):
    return service.audit(
        tenant, receipt.job_id, receipt.implementation_revision_sha256,
        CodingAuditVerdict.ACCEPT, (),
    )


def test_an_idle_audited_service_owns_no_configured_tree(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path, "workspace")
    _declare_verification(workspace)
    service = _audited_service(tmp_path, workspace, provider=ReworkingProvider())
    try:
        assert _owns(service) is False
        assert describe_workspace_root(None, workspace)["status"] == "unregistered"
    finally:
        service.close()


def test_the_first_accepted_job_establishes_ownership(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path, "workspace")
    _declare_verification(workspace)
    service = _audited_service(tmp_path, workspace, provider=ReworkingProvider())
    try:
        assert _owns(service) is False
        _awaiting(service, "tenant-audit", "own-001", workspace)
        assert _owns(service) is True
        assert describe_workspace_root(None, workspace)["status"] == "live"
    finally:
        service.close()


def test_rework_and_awaiting_audit_retain_ownership(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path, "workspace")
    _declare_verification(workspace)
    service = _audited_service(tmp_path, workspace, provider=ReworkingProvider())
    try:
        owner = _awaiting(service, "tenant-audit", "own-001", workspace)
        assert _owns(service) is True
        service.audit(
            "tenant-audit", owner.job_id, owner.implementation_revision_sha256,
            CodingAuditVerdict.REWORK, (_blocker(),),
        )
        assert _owns(service) is True
        reworked = _wait(service, "tenant-audit", owner.job_id)
        assert reworked.state is CodingJobState.AWAITING_CODEX_AUDIT
        assert _owns(service) is True
    finally:
        service.close()


def test_accept_releases_ownership_after_the_last_open_job(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path, "workspace")
    _declare_verification(workspace)
    service = _audited_service(tmp_path, workspace, provider=ReworkingProvider())
    try:
        owner = _awaiting(service, "tenant-audit", "own-001", workspace)
        assert _owns(service) is True
        accepted = _accept(service, "tenant-audit", owner)
        assert accepted.state is CodingJobState.CODEX_ACCEPTED
        assert _owns(service) is False
        # Released, not never-claimed: the durable record survives the last
        # terminal job with nothing unresolved, so the next start adopts it.
        # `unregistered` would mean nobody had ever claimed this tree.
        assert describe_workspace_root(None, workspace)["status"] == "adoptable"
    finally:
        service.close()


def test_one_terminal_job_does_not_release_while_another_is_open(
    tmp_path: Path,
) -> None:
    alpha = _workspace(tmp_path, "alpha")
    beta = _workspace(tmp_path, "beta")
    _declare_verification(alpha)
    _declare_verification(beta)
    service = _audited_service(
        tmp_path, alpha, provider=ReworkingProvider(), extra_roots=(beta,),
    )
    try:
        first = _awaiting(service, "tenant-audit", "a-001", alpha)
        second = _awaiting(service, "tenant-audit", "b-001", beta)
        assert _owns(service) is True

        _accept(service, "tenant-audit", first)
        assert state_root_has_open_work(service.state_root) is True
        assert _owns(service) is True

        _accept(service, "tenant-audit", second)
        assert _owns(service) is False
    finally:
        service.close()


def test_an_idle_state_root_does_not_block_a_second_one(tmp_path: Path) -> None:
    workspace = _workspace(tmp_path, "workspace")
    _declare_verification(workspace)
    idle = _audited_service(tmp_path / "A", workspace, provider=ReworkingProvider())
    try:
        assert _owns(idle) is False
        active = _audited_service(
            tmp_path / "B", workspace, provider=ReworkingProvider(),
        )
        try:
            active_owner = _awaiting(active, "tenant-audit", "b-001", workspace)
            assert active_owner.state is CodingJobState.AWAITING_CODEX_AUDIT
            assert _owns(active) is True
        finally:
            active.close()
    finally:
        idle.close()


def test_an_active_state_root_blocks_a_second_one_until_it_settles(
    tmp_path: Path,
) -> None:
    workspace = _workspace(tmp_path, "workspace")
    _declare_verification(workspace)
    first = _audited_service(tmp_path / "A", workspace, provider=ReworkingProvider())
    try:
        owner = _awaiting(first, "tenant-audit", "a-001", workspace)
        assert _owns(first) is True

        second = _audited_service(
            tmp_path / "B", workspace, provider=ReworkingProvider(),
        )
        try:
            with pytest.raises(CodingWorkspaceAuthorityConflict) as excinfo:
                second.submit("tenant-audit", "b-001", _request(workspace))
            assert excinfo.value.code == "workspace_authority_conflict"
            tenants = Path(second.state_root) / "tenants"
            assert not tenants.exists() or not list(tenants.rglob("jobs/*.json"))

            _accept(first, "tenant-audit", owner)
            assert _owns(first) is False

            admitted = _awaiting(second, "tenant-audit", "b-002", workspace)
            assert admitted.state is CodingJobState.AWAITING_CODEX_AUDIT
            assert _owns(second) is True
        finally:
            second.close()
    finally:
        first.close()


def test_same_state_root_peers_never_open_a_release_gap(tmp_path: Path) -> None:
    alpha = _workspace(tmp_path, "alpha")
    beta = _workspace(tmp_path, "beta")
    _declare_verification(alpha)
    _declare_verification(beta)
    first = _audited_service(
        tmp_path, alpha, provider=ReworkingProvider(), extra_roots=(beta,),
    )
    second = _audited_service(
        tmp_path, alpha, provider=ReworkingProvider(), extra_roots=(beta,),
    )
    try:
        held = _awaiting(first, "tenant-audit", "a-001", alpha)
        transient = _awaiting(second, "tenant-audit", "b-001", beta)
        assert _owns(first) is True and _owns(second) is True

        # The second peer settles its own job. alpha is still open under the
        # same state root, so no peer may release the tree out from under the
        # first. This peer drops its own hold; the first keeps the shared one.
        _accept(second, "tenant-audit", transient)
        assert state_root_has_open_work(first.state_root) is True
        assert _owns(first) is True

        # A third state root is still blocked while alpha is open.
        intruder = _audited_service(
            tmp_path / "C", alpha, provider=ReworkingProvider(), extra_roots=(beta,),
        )
        try:
            with pytest.raises(CodingWorkspaceAuthorityConflict):
                intruder.submit("tenant-audit", "c-001", _request(alpha))
        finally:
            intruder.close()

        _accept(first, "tenant-audit", held)
        assert _owns(first) is False
    finally:
        first.close()
        second.close()


def test_abandon_of_rework_running_is_refused_and_keeps_ownership(
    tmp_path: Path,
) -> None:
    """The audit-versus-release race: rework in flight is not abandonable."""

    workspace = _workspace(tmp_path, "workspace")
    _declare_verification(workspace)
    service = _audited_service(tmp_path, workspace, provider=ReworkingProvider())
    try:
        owner = _awaiting(service, "tenant-audit", "race-001", workspace)
        service.audit(
            "tenant-audit", owner.job_id, owner.implementation_revision_sha256,
            CodingAuditVerdict.REWORK, (_blocker(),),
        )
        with pytest.raises(AbandonStateConflict) as excinfo:
            service.abandon("tenant-audit", owner.job_id)
        assert excinfo.value.code == "abandon_state_conflict"
        assert state_root_has_open_work(service.state_root) is True
        assert _owns(service) is True
    finally:
        service.close()
