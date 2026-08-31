import hashlib
import json
import time
from pathlib import Path

import pytest

import flyto_ai.coding.fast_get as fast_get
from flyto_ai.coding.fast_get import (
    DurableGetFallback,
    DurableGetUnavailable,
    DurableJobReceiptReader,
)
from flyto_ai.coding.route import (
    CodingRouteReceipt,
    RouteCallRecord,
    RouteLaneReceipt,
    RouteLaneStatus,
)
from flyto_ai.coding.service import CodingJobNotFound, RouteEvidenceMissing

TENANT = "local-codex"
JOB_ID = "job_" + "a" * 24


def _tenant_ref(tenant: str) -> str:
    return hashlib.sha256(tenant.encode("utf-8")).hexdigest()


def _jobs(state: Path, tenant: str = TENANT) -> Path:
    path = state / "tenants" / _tenant_ref(tenant) / "jobs"
    path.mkdir(parents=True, mode=0o700, exist_ok=True)
    path.chmod(0o700)
    return path


def _write_record(
    state: Path,
    record: dict,
    *,
    tenant: str = TENANT,
    job_id: str = JOB_ID,
) -> Path:
    path = _jobs(state, tenant) / (job_id + ".json")
    path.write_text(
        json.dumps(record, ensure_ascii=False, separators=(",", ":")),
        encoding="utf-8",
    )
    path.chmod(0o600)
    return path


def _record(job_id: str = JOB_ID, state: str = "running") -> dict:
    return {
        "job_id": job_id,
        "state": state,
        "submitted_at": 1_700_000_000.0,
        "updated_at": 1_700_000_001.0,
        "working_dir": "/private/workspace",
        "message": "must never cross the public receipt",
    }


def _lane(name: str, actions: tuple[str, ...], gates: tuple[str, ...] = ()):
    return RouteLaneReceipt(
        lane=name,
        required=True,
        status=RouteLaneStatus.APPLIED,
        reason_code="completed",
        calls=tuple(
            RouteCallRecord(name, action, True, "completed") for action in actions
        ),
        gates_passed=gates,
    )


def _strict_route() -> CodingRouteReceipt:
    return CodingRouteReceipt(
        strict=True,
        ok=True,
        lanes=(
            _lane(
                "indexer_pre",
                ("structure", "search", "task.plan", "task.gate.assess"),
                ("task.gate.assess",),
            ),
            RouteLaneReceipt(
                lane="blueprint",
                required=True,
                status=RouteLaneStatus.NOT_APPLICABLE,
                reason_code="no_relevant_blueprint",
            ),
            RouteLaneReceipt(
                lane="core",
                required=True,
                status=RouteLaneStatus.NOT_APPLICABLE,
                reason_code="no_core_surface_changed",
            ),
            _lane(
                "indexer_post",
                ("task.validate", "task.gate.verify", "verify.strict"),
                ("task.gate.verify", "verify.strict"),
            ),
        ),
    )


def _audit_ready_record(job_id: str = JOB_ID) -> dict:
    record = _record(job_id, "awaiting_codex_audit")
    record.update({
        "implementation_backend": "codex",
        "implementation_session_id": "session-01234567",
        "implementation_revision_sha256": "b" * 64,
        "implementer_started": True,
        "route_receipt": _strict_route().to_mapping(),
    })
    return record


def _reader(state: Path, tenant: str = TENANT) -> DurableJobReceiptReader:
    return DurableJobReceiptReader(
        str(state),
        tenant,
        implementation_backend="codex",
    )


def test_exact_reader_is_tenant_bound_and_redacts_private_record_fields(tmp_path) -> None:
    state = tmp_path / "state"
    path = _write_record(state, _record(state="running"))
    _write_record(
        state,
        _record(state="failed"),
        tenant="another-tenant",
    )
    before = (path.read_bytes(), path.stat().st_ino, path.stat().st_mtime_ns)
    names_before = tuple(sorted(item.name for item in path.parent.iterdir()))

    public = _reader(state).read(JOB_ID)

    assert public["job_id"] == JOB_ID
    assert public["state"] == "running"
    assert public["job_terminal"] is False
    assert "working_dir" not in public
    assert "message" not in public
    assert (path.read_bytes(), path.stat().st_ino, path.stat().st_mtime_ns) == before
    assert tuple(sorted(item.name for item in path.parent.iterdir())) == names_before
    assert _reader(state, "another-tenant").read(JOB_ID)["state"] == "failed"
    with pytest.raises(CodingJobNotFound):
        _reader(state, "missing-tenant").read(JOB_ID)


def test_exact_reader_revalidates_route_digest_and_public_receipt_schema(tmp_path) -> None:
    state = tmp_path / "state"
    valid = _audit_ready_record()
    _write_record(state, valid)
    assert _reader(state).read(JOB_ID)["route_receipt"]["digest"] == (
        valid["route_receipt"]["digest"]
    )

    tampered = _audit_ready_record()
    tampered["route_receipt"]["lanes"][0]["reason_code"] = "forged"
    _write_record(state, tampered)
    with pytest.raises(RouteEvidenceMissing):
        _reader(state).read(JOB_ID)

    mismatched = _record("job_" + "b" * 24)
    _write_record(state, mismatched)
    with pytest.raises(ValueError, match="does not match"):
        _reader(state).read(JOB_ID)


def test_exact_reader_refuses_symlinks_and_non_private_records(tmp_path) -> None:
    state = tmp_path / "state"
    outside = tmp_path / "outside.json"
    outside.write_text(json.dumps(_record()), encoding="utf-8")
    outside.chmod(0o600)
    target = _jobs(state) / (JOB_ID + ".json")
    target.symlink_to(outside)
    with pytest.raises(DurableGetUnavailable):
        _reader(state).read(JOB_ID)

    target.unlink()
    _write_record(state, _record()).chmod(0o644)
    with pytest.raises(DurableGetUnavailable):
        _reader(state).read(JOB_ID)


@pytest.mark.parametrize(
    "raw",
    [
        f'{{"job_id":"{JOB_ID}","job_id":"{JOB_ID}"}}',
        f'{{"job_id":"{JOB_ID}","state":"running","value":NaN}}',
        "[" * 1500 + "0" + "]" * 1500,
    ],
)
def test_exact_reader_rejects_ambiguous_json(tmp_path, raw) -> None:
    state = tmp_path / "state"
    path = _jobs(state) / (JOB_ID + ".json")
    path.write_text(raw, encoding="utf-8")
    path.chmod(0o600)
    with pytest.raises(ValueError, match="invalid"):
        _reader(state).read(JOB_ID)


def test_exact_reader_bounds_one_record_without_enumerating_siblings(tmp_path) -> None:
    state = tmp_path / "state"
    path = _jobs(state) / (JOB_ID + ".json")
    path.write_bytes(b"{" + b" " * (1024 * 1024) + b"}")
    path.chmod(0o600)
    with pytest.raises(DurableGetUnavailable, match="bound"):
        _reader(state).read(JOB_ID)


def test_terminal_ready_mission_falls_back_for_canonical_reconciliation(tmp_path) -> None:
    state = tmp_path / "state"
    record = _record(state="failed")
    record["mission"] = {"status": "dispatched"}
    _write_record(state, record)
    with pytest.raises(DurableGetFallback):
        _reader(state).read(JOB_ID)


def test_exact_read_latency_is_independent_of_large_durable_history(tmp_path) -> None:
    state = tmp_path / "state"
    jobs = _jobs(state)
    payload = json.dumps(_record(), separators=(",", ":"))
    for index in range(5000):
        job_id = f"job_{index:024x}"
        path = jobs / (job_id + ".json")
        path.write_text(payload.replace(JOB_ID, job_id), encoding="utf-8")
        path.chmod(0o600)
    target = f"job_{4999:024x}"

    started = time.perf_counter()
    public = _reader(state).read(target)
    elapsed = time.perf_counter() - started

    assert public["job_id"] == target
    # Includes strict parsing and redaction, but no directory enumeration.  The
    # generous bound catches an accidental O(history) scan without making the
    # test a microbenchmark of the host filesystem.
    assert elapsed < 1.0


def test_invalid_job_id_is_refused_without_a_filesystem_lookup(tmp_path, monkeypatch) -> None:
    reader = _reader(tmp_path / "state")
    monkeypatch.setattr(reader, "_open_jobs_directory", lambda: pytest.fail("opened state"))
    with pytest.raises(CodingJobNotFound):
        reader.read("../foreign-job")


def test_unsupported_platform_disables_exact_reader(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(fast_get, "_O_NOFOLLOW", 0)
    assert fast_get.durable_job_receipt_reader(
        str(tmp_path), TENANT, implementation_backend="codex",
    ) is None
