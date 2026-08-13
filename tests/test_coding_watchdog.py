"""Deterministic watchdog, transition history, launchd, and GitHub heartbeat."""
from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

from flyto_ai.coding.route_status import MAX_STATUS_INDEX_BYTES
from flyto_ai.coding.watchdog import (
    DEFAULT_GITHUB_VARIABLE,
    MAX_LATEST_BYTES,
    WATCHDOG_HEARTBEAT_SCHEMA,
    WATCHDOG_SCHEMA,
    WatchdogError,
    WatchdogRecorder,
    _validate_status_index,
    evaluate_watchdog,
    github_heartbeat_payload,
    launch_agent_definition,
    launch_agent_label,
    publish_github_heartbeat,
    run_watchdog_once,
)


NOW = 2_000_000_000.0
BUILD = "a" * 32


def _window(*tasks):
    return {"available": True, "tasks": list(tasks)}


def _task(state="running", *, age=10, job_id="job_" + "b" * 24):
    return {"state": state, "updated_at": NOW - age, "job_id": job_id}


def _instance(*, job_id="job_" + "b" * 24, build=BUILD, **values):
    row = {
        "lifecycle": "active",
        "alive": True,
        "job_id": job_id,
        "build_id": build,
        "mode": "strict",
        "circuit_state": "closed",
        "publish_failures": 0,
    }
    row.update(values)
    return row


def test_live_current_execution_is_healthy_and_secret_free():
    report = evaluate_watchdog(
        [_instance()], _window(_task()), reader_build_id=BUILD, observed_at=NOW,
    )

    assert report["health"] == "healthy"
    assert report["reason_codes"] == []
    assert report["counts"]["live_current_build"] == 1
    serialized = json.dumps(report)
    assert "job_" not in serialized
    assert "/Users/" not in serialized


def test_orphaned_execution_is_critical_but_recent_queue_has_grace():
    orphaned = evaluate_watchdog(
        [], _window(_task(age=181)), reader_build_id=BUILD, observed_at=NOW,
        orphan_grace_seconds=180,
    )
    recent = evaluate_watchdog(
        [], _window(_task(state="queued", age=179)), reader_build_id=BUILD,
        observed_at=NOW, orphan_grace_seconds=180,
    )

    assert orphaned["health"] == "critical"
    assert orphaned["counts"]["orphaned_tasks"] == 1
    assert "execution_liveness" in orphaned["reason_codes"]
    assert recent["health"] == "healthy"


def test_emergency_stale_build_and_overdue_audit_are_degraded():
    report = evaluate_watchdog(
        [_instance(build="c" * 32, mode="emergency", circuit_state="open")],
        _window(_task(), _task(state="awaiting_codex_audit", age=3601,
                              job_id="job_" + "d" * 24)),
        reader_build_id=BUILD,
        observed_at=NOW,
    )

    assert report["health"] == "degraded"
    assert set(report["reason_codes"]) == {
        "codex_audit_backlog", "emergency_spillway", "rolling_build_reload",
    }


def test_idle_without_model_process_does_not_require_always_on_ai():
    report = evaluate_watchdog(
        [_instance(job_id="", alive=False, lifecycle="closed")],
        _window(),
        reader_build_id=BUILD,
        observed_at=NOW,
    )
    assert report["health"] == "healthy"


def test_recorder_updates_latest_and_logs_only_transitions(tmp_path):
    health = tmp_path / "health"
    first = evaluate_watchdog(
        [_instance()], _window(_task()), reader_build_id=BUILD, observed_at=NOW,
    )
    second = dict(first, observed_at=int(NOW + 60))

    with WatchdogRecorder(health) as recorder:
        stored_first, changed_first = recorder.record(first)
    with WatchdogRecorder(health) as recorder:
        stored_second, changed_second = recorder.record(second)

    assert changed_first is True
    assert changed_second is False
    assert stored_first["transition"] is True
    assert stored_second["transition"] is False
    assert len((health / "history.jsonl").read_text().splitlines()) == 1
    assert json.loads((health / "latest.json").read_text())["observed_at"] == int(NOW + 60)
    assert stat.S_IMODE((health / "latest.json").stat().st_mode) == 0o600
    assert stat.S_IMODE(health.stat().st_mode) == 0o700


def test_recorder_lock_prevents_overlapping_watchdogs(tmp_path):
    with WatchdogRecorder(tmp_path) as _first:
        with pytest.raises(WatchdogError, match="watchdog_already_running"):
            with WatchdogRecorder(tmp_path):
                pass


def test_github_payload_is_small_and_contains_no_local_identifiers():
    report = evaluate_watchdog(
        [], _window(_task(age=181)), reader_build_id=BUILD, observed_at=NOW,
    )
    payload = github_heartbeat_payload(report)

    assert payload["schema"] == WATCHDOG_HEARTBEAT_SCHEMA
    assert payload["health"] == "critical"
    serialized = json.dumps(payload)
    assert "job_" not in serialized
    assert "session" not in serialized
    assert len(serialized) < 2048


def test_github_heartbeat_upserts_with_one_bounded_call(monkeypatch):
    calls = []

    def fake_run(argv, **kwargs):
        calls.append((argv, kwargs))
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    monkeypatch.setattr("flyto_ai.coding.watchdog.subprocess.run", fake_run)
    report = evaluate_watchdog([], _window(), reader_build_id=BUILD, observed_at=NOW)
    publish_github_heartbeat(
        "flytohub/flyto-ai", DEFAULT_GITHUB_VARIABLE, report, gh_command="/usr/bin/gh",
    )

    # One upsert, not PATCH-then-parse-the-error-then-POST: a missing and an
    # existing variable now take the identical path.
    assert len(calls) == 1
    argv = calls[0][0]
    assert argv[:5] == [
        "/usr/bin/gh", "variable", "set", DEFAULT_GITHUB_VARIABLE, "--repo",
    ]
    assert argv[5] == "flytohub/flyto-ai"
    assert argv[6] == "--body"
    assert json.loads(argv[7])["schema"] == WATCHDOG_HEARTBEAT_SCHEMA
    assert not any("job_" in item for item in argv)
    assert calls[0][1]["timeout"] == 20
    assert calls[0][1]["check"] is False


def test_github_heartbeat_failure_is_not_retried_as_a_create(monkeypatch):
    calls = []

    def fake_run(argv, **_kwargs):
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 1, stdout="", stderr="HTTP 404: Not Found")

    monkeypatch.setattr("flyto_ai.coding.watchdog.subprocess.run", fake_run)
    report = evaluate_watchdog([], _window(), reader_build_id=BUILD, observed_at=NOW)

    with pytest.raises(WatchdogError, match="github_heartbeat_failed"):
        publish_github_heartbeat(
            "flytohub/flyto-ai", DEFAULT_GITHUB_VARIABLE, report, gh_command="gh",
        )
    assert len(calls) == 1


def test_oversized_heartbeat_fails_locally_instead_of_being_truncated(monkeypatch):
    monkeypatch.setattr(
        "flyto_ai.coding.watchdog.subprocess.run",
        lambda *_a, **_k: pytest.fail("an oversized payload must not reach gh"),
    )
    monkeypatch.setattr("flyto_ai.coding.watchdog.MAX_GITHUB_VARIABLE_BYTES", 32)
    report = evaluate_watchdog([], _window(), reader_build_id=BUILD, observed_at=NOW)

    with pytest.raises(WatchdogError, match="github_heartbeat_payload_too_large"):
        publish_github_heartbeat(
            "flytohub/flyto-ai", DEFAULT_GITHUB_VARIABLE, report, gh_command="gh",
        )


def test_hung_gh_becomes_a_stable_code_instead_of_escaping(monkeypatch):
    def hang(argv, **kwargs):
        raise subprocess.TimeoutExpired(argv, kwargs.get("timeout", 20))

    monkeypatch.setattr("flyto_ai.coding.watchdog.subprocess.run", hang)
    report = evaluate_watchdog([], _window(), reader_build_id=BUILD, observed_at=NOW)

    with pytest.raises(WatchdogError, match="github_heartbeat_failed"):
        publish_github_heartbeat(
            "flytohub/flyto-ai", DEFAULT_GITHUB_VARIABLE, report, gh_command="gh",
        )


def test_launch_agent_rejects_an_interval_the_run_path_would_reject(tmp_path):
    with pytest.raises(WatchdogError, match="github_heartbeat_interval_invalid"):
        launch_agent_definition(
            state_root=tmp_path / "state",
            health_root=tmp_path / "health",
            github_repository="flytohub/flyto-ai",
            github_heartbeat_interval=30,
        )


def test_launch_agent_path_is_derived_from_the_real_home(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", classmethod(lambda _cls: Path("/net/homes/op")))
    definition = launch_agent_definition(
        state_root=tmp_path / "state", health_root=tmp_path / "health",
    )

    assert definition["EnvironmentVariables"]["PATH"].startswith(
        "/net/homes/op/.local/bin:",
    )


def test_launch_agent_is_bounded_and_contains_no_token(tmp_path):
    definition = launch_agent_definition(
        state_root=tmp_path / "state",
        health_root=tmp_path / "health",
        github_repository="flytohub/flyto-ai",
        notify=True,
        executable="/example/python",
    )

    assert definition["ProgramArguments"][:4] == [
        "/example/python", "-m", "flyto_ai", "code-watchdog",
    ]
    assert definition["StartInterval"] == 60
    serialized = json.dumps(definition)
    assert "TOKEN" not in serialized
    assert "ANTHROPIC" not in serialized
    assert "OPENAI" not in serialized
    assert "--notify" in definition["ProgramArguments"]


def test_invalid_remote_config_fails_before_subprocess(monkeypatch):
    monkeypatch.setattr(
        "flyto_ai.coding.watchdog.subprocess.run",
        lambda *_args, **_kwargs: pytest.fail("invalid config must not execute gh"),
    )
    with pytest.raises(WatchdogError, match="github_heartbeat_config_invalid"):
        publish_github_heartbeat(
            "https://github.com/flytohub/flyto-ai",
            DEFAULT_GITHUB_VARIABLE,
            {},
            gh_command="gh",
        )


@pytest.mark.parametrize(
    "options, code",
    [
        ({"stuck_seconds": 30}, "watchdog_stuck_seconds_invalid"),
        ({"stuck_seconds": True}, "watchdog_stuck_seconds_invalid"),
        ({"orphan_grace_seconds": -1}, "watchdog_orphan_grace_invalid"),
        ({"github_heartbeat_interval": 30}, "github_heartbeat_interval_invalid"),
        ({"interval_seconds": 5}, "watchdog_interval_invalid"),
        ({"github_variable": "lower-case"}, "github_heartbeat_config_invalid"),
        ({"github_repository": "https://github.com/a/b"}, "github_heartbeat_config_invalid"),
    ],
)
def test_install_rejects_every_value_the_run_path_would_reject(tmp_path, options, code):
    # An accepted-at-install, rejected-at-wake value is the worst outcome: the
    # LaunchAgent looks installed and the watchdog is silent forever.
    arguments = {
        "state_root": tmp_path / "state",
        "health_root": tmp_path / "health",
        "github_repository": "flytohub/flyto-ai",
    }
    arguments.update(options)

    with pytest.raises(WatchdogError, match=code):
        launch_agent_definition(**arguments)


def test_run_path_rejects_the_same_install_values(tmp_path):
    with pytest.raises(WatchdogError, match="watchdog_stuck_seconds_invalid"):
        run_watchdog_once(
            state_root=tmp_path / "state",
            health_root=tmp_path / "health",
            stuck_seconds=30,
        )


def test_a_variable_name_is_validated_even_without_a_repository(tmp_path):
    with pytest.raises(WatchdogError, match="github_heartbeat_config_invalid"):
        launch_agent_definition(
            state_root=tmp_path / "state",
            health_root=tmp_path / "health",
            github_variable="9_LEADING_DIGIT",
        )


@pytest.mark.parametrize(
    "state, health",
    [
        ("root", "root"),
        ("root", "root/health"),
        ("root/state", "root"),
    ],
)
def test_state_and_health_roots_may_not_overlap(tmp_path, state, health):
    # The observer must never write into the tree it observes: a health record
    # inside the state root would make the watchdog trigger on its own writes
    # and mutate durable coding-service state it has no authority over.
    for builder in (launch_agent_definition, run_watchdog_once):
        with pytest.raises(WatchdogError, match="watchdog_paths_overlap"):
            builder(state_root=tmp_path / state, health_root=tmp_path / health)


def test_disjoint_roots_are_accepted(tmp_path):
    definition = launch_agent_definition(
        state_root=tmp_path / "state", health_root=tmp_path / "health",
    )
    assert os.path.realpath(tmp_path / "state") in definition["ProgramArguments"]


def test_a_symlinked_health_dir_cannot_smuggle_itself_into_the_state_root(tmp_path):
    # A lexical `abspath` comparison is not a containment check. Without
    # resolving both roots, `--health-dir <link>` names the coding-service tree
    # through a symlink, the overlap guard sees two unrelated strings, and the
    # observer starts writing into the durable state it is forbidden to touch.
    state = tmp_path / "state"
    (state / "health").mkdir(parents=True)
    link = tmp_path / "link"
    link.symlink_to(state / "health", target_is_directory=True)

    for builder in (launch_agent_definition, run_watchdog_once):
        with pytest.raises(WatchdogError, match="watchdog_paths_overlap"):
            builder(state_root=state, health_root=link)


def test_install_and_uninstall_agree_on_the_label_through_a_symlink(tmp_path):
    # The label identifies a directory, not a spelling. If install resolves the
    # state root and uninstall does not, `--uninstall` removes nothing and
    # reports success while the agent keeps waking forever.
    real = tmp_path / "state"
    real.mkdir()
    link = tmp_path / "alias"
    link.symlink_to(real, target_is_directory=True)

    assert launch_agent_label(link) == launch_agent_label(real)
    definition = launch_agent_definition(
        state_root=link, health_root=tmp_path / "health",
    )
    assert definition["Label"] == launch_agent_label(real)


def _tree(root: Path):
    """Every path under `root` with its bytes, for an exact no-mutation check."""

    snapshot = {}
    for path in sorted(root.rglob("*")):
        key = str(path.relative_to(root))
        snapshot[key] = path.read_bytes() if path.is_file() else None
    return snapshot


def test_one_full_turn_observes_the_state_root_and_writes_only_health(tmp_path):
    """The whole point of the tool, end to end, against a real state root.

    Every other test here drives one function. This one proves the composed
    path an unattended LaunchAgent actually runs: read the published index and
    the task window, evaluate, and record — while leaving the coding-service
    tree byte-for-byte identical. A watchdog that mutated what it observes
    would trigger on its own writes and corrupt durable state it has no
    authority over, and no single-function test can catch that.
    """

    from flyto_ai.coding.route_status import (
        ROUTE_STATUS_CONTRACT_VERSION,
        ROUTE_STATUS_DIRNAME,
        ROUTE_STATUS_INDEX_FILENAME,
    )

    state = tmp_path / "coding-service"
    health = tmp_path / "health"
    status = state / ROUTE_STATUS_DIRNAME
    status.mkdir(parents=True)
    (status / ROUTE_STATUS_INDEX_FILENAME).write_text(
        json.dumps({
            "contract_version": ROUTE_STATUS_CONTRACT_VERSION,
            "instances": [],
        }),
        encoding="utf-8",
    )
    before = _tree(state)

    report = run_watchdog_once(state_root=state, health_root=health)

    assert report["schema"] == WATCHDOG_SCHEMA
    assert report["health"] in {"healthy", "degraded"}
    assert report["github"] == "disabled"
    assert report["transition"] is True
    assert report["previous_health"] == ""
    # A readable, well-formed, publisher-owned index must never be reported as
    # an unreadable state root.
    statuses = {item["code"]: item["status"] for item in report["checks"]}
    assert statuses["state_readable"] == "pass"

    assert _tree(state) == before, "the observer mutated the tree it observes"
    assert json.loads((health / "latest.json").read_text())["fingerprint"] == (
        report["fingerprint"]
    )
    assert len((health / "history.jsonl").read_text().splitlines()) == 1

    # Nothing local, identifying or secret reaches the record.
    serialized = json.dumps(report)
    assert "job_" not in serialized
    assert str(tmp_path) not in serialized

    # A second identical turn is not a transition, so history stays quiet.
    again = run_watchdog_once(state_root=state, health_root=health)
    assert again["transition"] is False
    assert again["previous_health"] == report["health"]
    assert len((health / "history.jsonl").read_text().splitlines()) == 1
    assert _tree(state) == before


def test_a_symlinked_record_is_never_read_through(tmp_path):
    """`--health-dir` may legitimately sit under a world-writable parent.

    A planted symlink at `latest.json` must not let another local user choose
    what this watchdog reads back as its own previous state — `previous_health`
    drives the desktop notification, and `fingerprint` decides whether a real
    transition is recorded at all.
    """

    health = tmp_path / "health"
    health.mkdir(mode=0o700)
    elsewhere = tmp_path / "elsewhere.json"
    elsewhere.write_text(
        json.dumps({"health": "healthy", "fingerprint": "planted"}), encoding="utf-8",
    )
    (health / "latest.json").symlink_to(elsewhere)

    with WatchdogRecorder(health) as recorder:
        assert recorder.previous() == {}


def test_a_symlinked_history_cannot_redirect_the_append(tmp_path):
    """History is appended by name, so the name must not be a link.

    `_atomic_write` is safe by construction because `os.replace` overwrites the
    link itself, but an `O_APPEND` open follows one. Without `O_NOFOLLOW` this
    is a write primitive into any file the watchdog's own user can write.
    """

    health = tmp_path / "health"
    health.mkdir(mode=0o700)
    target = tmp_path / "victim.txt"
    target.write_text("untouched", encoding="utf-8")
    (health / "history.jsonl").symlink_to(target)
    report = evaluate_watchdog(
        [_instance()], _window(_task()), reader_build_id=BUILD, observed_at=NOW,
    )

    with WatchdogRecorder(health) as recorder:
        with pytest.raises(WatchdogError, match="watchdog_history_unwritable"):
            recorder.record(report)

    assert target.read_text(encoding="utf-8") == "untouched"
    # The turn's actual contract still completed: latest is durable, and it is
    # a real file rather than the link that was refused.
    latest = health / "latest.json"
    assert not latest.is_symlink()
    assert json.loads(latest.read_text())["fingerprint"] == report["fingerprint"]


def test_a_symlinked_lock_fails_closed_instead_of_locking_elsewhere(tmp_path):
    health = tmp_path / "health"
    health.mkdir(mode=0o700)
    (health / "watchdog.lock").symlink_to(tmp_path / "not-the-lock")

    with pytest.raises(WatchdogError, match="watchdog_lock_unavailable"):
        with WatchdogRecorder(health):
            pass

    # Following the link would have created the target as a side effect.
    assert not (tmp_path / "not-the-lock").exists()


def test_an_unrecordable_send_marker_still_records_health(tmp_path, monkeypatch):
    """The heartbeat is secondary; the local record is the turn's contract.

    If `mark_github_sent` fails after the heartbeat was published, abandoning
    the turn would leave the remote switch reading `healthy` while the local
    record that a human inspects was never written — silence in exactly the
    state the watchdog exists to make loud.
    """

    import flyto_ai.coding.watchdog as watchdog

    state = tmp_path / "coding-service"
    state.mkdir()
    health = tmp_path / "health"
    monkeypatch.setattr(watchdog, "publish_github_heartbeat", lambda *a, **k: None)

    def unwritable(self, report):
        raise OSError("github.json is a directory")

    monkeypatch.setattr(watchdog.WatchdogRecorder, "mark_github_sent", unwritable)

    report = run_watchdog_once(
        state_root=state, health_root=health, github_repository="flyto2/flyto-ai",
    )

    assert report["github"] == "github_state_unrecordable"
    assert "github_heartbeat" in report["reason_codes"]
    assert json.loads((health / "latest.json").read_text())["fingerprint"] == (
        report["fingerprint"]
    )


def test_status_index_uses_the_publisher_byte_limit_not_the_record_limit(tmp_path):
    from flyto_ai.coding.route_status import (
        ROUTE_STATUS_CONTRACT_VERSION,
        ROUTE_STATUS_DIRNAME,
        ROUTE_STATUS_INDEX_FILENAME,
    )

    status = tmp_path / ROUTE_STATUS_DIRNAME
    status.mkdir(parents=True)
    index = status / ROUTE_STATUS_INDEX_FILENAME
    payload = {
        "contract_version": ROUTE_STATUS_CONTRACT_VERSION,
        "instances": [],
        "padding": "p" * (MAX_LATEST_BYTES + 1024),
    }
    index.write_text(json.dumps(payload), encoding="utf-8")
    size = index.stat().st_size
    assert MAX_LATEST_BYTES < size <= MAX_STATUS_INDEX_BYTES

    # Large but valid, and the publisher — not the watchdog — owns this bound.
    # Judging it by the watchdog's own record limit invents an incident.
    assert _validate_status_index(tmp_path) == "pass"

    index.write_text(
        json.dumps(dict(payload, padding="p" * (MAX_STATUS_INDEX_BYTES + 1024))),
        encoding="utf-8",
    )
    assert _validate_status_index(tmp_path) == "fail"


WORKFLOW = Path(__file__).parents[1] / ".github" / "workflows" / "coding-watchdog.yml"
WORKFLOW_ENV = {
    "MAX_AGE_SECONDS": "2700",
    "MAX_CLOCK_SKEW_SECONDS": "300",
    "MAX_HEARTBEAT_BYTES": "65536",
}


def _workflow_validator() -> str:
    """Extract the heredoc validator the workflow actually runs."""

    text = WORKFLOW.read_text(encoding="utf-8")
    start = text.index("python3 - <<'PY'\n") + len("python3 - <<'PY'\n")
    end = text.index("\n          PY\n", start)
    return textwrap.dedent(text[start:end])


def _run_validator(tmp_path, heartbeat, **overrides):
    output = tmp_path / "output"
    summary = tmp_path / "summary"
    # Truncate: the validator appends, and pytest reuses one tmp_path across
    # every call inside a single test.
    output.write_text("", encoding="utf-8")
    summary.write_text("", encoding="utf-8")
    environment = dict(os.environ)
    environment.update(WORKFLOW_ENV)
    environment.update({
        "HEARTBEAT": heartbeat,
        "GITHUB_OUTPUT": str(output),
        "GITHUB_STEP_SUMMARY": str(summary),
        "RUNNER_TEMP": str(tmp_path),
    })
    environment.update(overrides)
    completed = subprocess.run(
        [sys.executable, "-c", _workflow_validator()],
        env=environment, capture_output=True, text=True, timeout=60, check=False,
    )
    assert completed.returncode == 0, completed.stderr
    parsed = dict(
        line.split("=", 1) for line in output.read_text().splitlines() if "=" in line
    )
    return parsed, (tmp_path / "coding-watchdog.md").read_text(encoding="utf-8")


def _heartbeat(**values) -> str:
    payload = {
        "schema": "flyto.coding-watchdog-heartbeat.v1",
        "observed_at": int(time.time()),
        "health": "healthy",
        "fingerprint": "f" * 64,
        "reader_build_id": BUILD,
        "reason_codes": [],
    }
    payload.update(values)
    return json.dumps(payload)


def test_workflow_accepts_a_current_healthy_heartbeat(tmp_path):
    outputs, body = _run_validator(tmp_path, _heartbeat())

    assert outputs["healthy"] == "true"
    assert outputs["reason"] == "healthy"
    assert "Result: `healthy`" in body


@pytest.mark.parametrize(
    "heartbeat, reason",
    [
        ("", "heartbeat_missing"),
        ("   ", "heartbeat_missing"),
        ("not json", "heartbeat_invalid"),
        ("[1, 2, 3]", "heartbeat_invalid"),
        ('"a string"', "heartbeat_invalid"),
        ("null", "heartbeat_invalid"),
    ],
)
def test_workflow_rejects_unparseable_heartbeats(tmp_path, heartbeat, reason):
    outputs, _body = _run_validator(tmp_path, heartbeat)

    assert outputs["healthy"] == "false"
    assert outputs["reason"] == reason
    assert outputs["observed_at"] == "0"


@pytest.mark.parametrize(
    "values, reason",
    [
        ({"schema": "other.v1"}, "heartbeat_schema_invalid"),
        ({"observed_at": "1700000000"}, "heartbeat_timestamp_invalid"),
        ({"observed_at": True}, "heartbeat_timestamp_invalid"),
        ({"observed_at": 0}, "heartbeat_timestamp_invalid"),
        ({"observed_at": -1}, "heartbeat_timestamp_invalid"),
        ({"observed_at": 10 ** 18}, "heartbeat_timestamp_invalid"),
        ({"health": "fine"}, "heartbeat_health_invalid"),
        ({"health": None}, "heartbeat_health_invalid"),
        ({"health": "degraded", "reason_codes": "not-a-list"}, "heartbeat_invalid"),
    ],
)
def test_workflow_rejects_malformed_heartbeat_fields(tmp_path, values, reason):
    outputs, _body = _run_validator(tmp_path, _heartbeat(**values))

    assert outputs["healthy"] == "false"
    assert outputs["reason"] == reason


def test_workflow_survives_deeply_nested_json(tmp_path):
    # RecursionError is not a ValueError. Letting it escape would abort the
    # step before it can open an incident, so a malformed variable would buy
    # silence rather than an alert.
    outputs, _body = _run_validator(tmp_path, "[" * 2_000 + "]" * 2_000)

    assert outputs["healthy"] == "false"
    assert outputs["reason"] == "heartbeat_invalid"


def test_workflow_rejects_an_oversized_heartbeat_before_parsing(tmp_path):
    outputs, _body = _run_validator(
        tmp_path, _heartbeat(fingerprint="f" * 70_000),
    )

    assert outputs["reason"] == "heartbeat_oversized"


def test_workflow_flags_stale_and_future_dated_heartbeats(tmp_path):
    stale, _stale_body = _run_validator(
        tmp_path, _heartbeat(observed_at=int(time.time()) - 10_000),
    )
    future, _future_body = _run_validator(
        tmp_path, _heartbeat(observed_at=int(time.time()) + 10_000),
    )

    assert stale["reason"] == "heartbeat_stale"
    assert future["reason"] == "heartbeat_clock_invalid"


def test_workflow_reports_local_reason_codes_without_trusting_them(tmp_path):
    outputs, body = _run_validator(
        tmp_path,
        _heartbeat(
            health="critical",
            reason_codes=[
                "execution_liveness",
                "Not A Code",
                "shell$(whoami)",
                42,
                "state_readable",
            ],
        ),
    )

    assert outputs["healthy"] == "false"
    assert outputs["reason"] == "local_critical:execution_liveness,state_readable"
    assert "whoami" not in body


def test_workflow_heartbeat_cannot_forge_a_healthy_step_output(tmp_path):
    # A newline in any rendered field would let the untrusted variable append
    # its own `healthy=true` line to GITHUB_OUTPUT and silence the switch.
    outputs, _body = _run_validator(
        tmp_path,
        _heartbeat(health="degraded", reason_codes=["ok\nhealthy=true"]),
    )

    assert outputs["healthy"] == "false"
    assert outputs["reason"] == "local_degraded:unknown"


def test_workflow_declares_every_bound_the_validator_reads():
    text = WORKFLOW.read_text(encoding="utf-8")

    for key, value in WORKFLOW_ENV.items():
        assert '{}: "{}"'.format(key, value) in text


def test_repository_workflow_is_deterministic_and_not_agentic():
    workflow = Path(__file__).parents[1] / ".github" / "workflows" / "coding-watchdog.yml"
    text = workflow.read_text(encoding="utf-8")

    assert "vars.FLYTO_CODING_HEARTBEAT" in text
    assert "issues: write" in text
    assert "coding-watchdog" in text
    assert "claude" not in text.lower()
    assert "codex" not in text.lower()
    assert "gh-aw" not in text.lower()
