# Copyright 2024 Flyto2
# Licensed under the Apache License, Version 2.0
"""Two production incidents, and the arithmetic that should have prevented both.

*The route spent work to discover it had no budget.*  A legal plan of the
largest size the route accepts could not physically fit inside the per-lane call
bound, because the bound was checked after each dispatch rather than before it.
The live incident issued thirty-three calls and only then refused, before the
implementer had begun.  What is asserted here is therefore not the receipt - a
bounded receipt was exactly what the incident produced - but the number of times
the real dispatcher was invoked.

*A declared check could be impossible to launch, and preflight said yes.*
Proving a required ``CheckSpec`` exists is not proving its program is installed,
so a missing tool arrived as a failed verification: a statement about the change,
when nothing about the change was at fault, and only after a job, a claim and a
session existed.

Nothing here touches provider vocabulary, a repository layout or a workload
domain; the subject is call arithmetic and executable resolution.
"""
import asyncio
import os
import shutil
import stat
from pathlib import Path

import pytest

from flyto_ai.coding import preflight as preflight_module
from flyto_ai.coding.checks import (
    CheckRunner,
    VerificationToolUnavailable,
    unlaunchable_required_checks,
)
from flyto_ai.coding.contracts import CheckSpec, CodingTaskRequest
from flyto_ai.coding.preflight import (
    ACTION_INSTALL_VERIFICATION_TOOL,
    CODE_VERIFICATION_TOOL_MISSING,
    MAX_PREFLIGHT_BLOCKERS,
    PREFLIGHT_ACTIONS,
    PREFLIGHT_CODES,
    preflight_repository,
)
from flyto_ai.coding.route import (
    CodingRouteError,
    CodingRouteOrchestrator,
    CodingRoutePolicy,
    RouteLane,
    RouteLimits,
)
from flyto_ai.coding.contracts import CapabilitySpec
from flyto_ai.coding.workspace import resolve_executable

# --------------------------------------------------------------------------
# incident A: the route may never issue call N+1 when the bound is N
# --------------------------------------------------------------------------


class _CountingDispatcher:
    """A capability dispatcher that records every invocation it really receives.

    The count is the evidence.  A route can refuse with a perfectly bounded
    receipt while having already spent the calls, which is precisely what the
    incident did, so the receipt is not what these tests believe.
    """

    def __init__(self, plan_steps=(), gate_passes=True):
        self.calls = []
        self._plan_steps = list(plan_steps)
        self._gate_passes = gate_passes

    async def __call__(self, tool, arguments):
        self.calls.append((tool, dict(arguments or {}).get("action") or tool))
        action = dict(arguments or {}).get("action")
        if tool == "task" and action == "plan":
            return {"ok": True, "task_profile": {}, "execution_plan": self._plan_steps}
        if tool == "task" and action == "gate":
            return {"ok": True, "pass": self._gate_passes, "required_state": {}}
        if tool == "task" and action == "validate":
            return {"ok": True, "pass": True}
        return {"ok": True, "result": {}}

    @property
    def count(self):
        return len(self.calls)


def _analysis_steps(count):
    """`count` host-owned analysis steps, none of them a gate."""

    return [
        {
            "id": "step_{:02d}_scope".format(index),
            "tool": "impact",
            "args": {"target": "proj:app.py:function:main"},
            "purpose": "scope_callers",
            "required": True,
            "depends_on": [],
        }
        for index in range(count)
    ]


def _gate_step(step_id, phase):
    return {
        "id": step_id,
        "tool": "task",
        "args": {"action": "gate", "next_phase": phase},
        "purpose": "gate",
        "required": True,
        "depends_on": [],
    }


def _indexer_spec():
    return CapabilitySpec(
        name="flyto-indexer",
        argv=("python3", "-m", "flyto_ai.mcp_server"),
        required=True,
        required_tools=("task", "verify", "structure", "search", "impact"),
        allowed_tools=("task", "verify", "structure", "search", "impact"),
        tool_permissions=(
            ("task", "read_only"), ("verify", "read_only"),
            ("structure", "read_only"), ("search", "read_only"),
            ("impact", "read_only"),
        ),
    )


def _route(dispatcher, limits=None):
    policy = CodingRoutePolicy(
        strict=False, indexer=_indexer_spec(), limits=limits or RouteLimits(),
    )
    return CodingRouteOrchestrator(policy, capability_dispatch=dispatcher)


def _request(tmp_path):
    return CodingTaskRequest(message="bounded change", working_dir=str(tmp_path))


def test_a_maximum_size_plan_fits_the_default_budget(tmp_path):
    """The incident, at the default limits, with the largest legal plan.

    Three host discovery calls, thirty-two mandatory plan steps and two
    canonical gates is thirty-seven; the previous default of thirty-two made
    that shape unrunnable by arithmetic alone.
    """

    limits = RouteLimits()
    dispatcher = _CountingDispatcher(_analysis_steps(limits.max_plan_steps))
    route = _route(dispatcher, limits)

    receipt, context = asyncio.run(route._indexer_pre(_request(tmp_path)))

    assert receipt.status.value == "applied"
    assert receipt.reason_code == "completed"
    # Every plan step ran; nothing was truncated to fit.
    assert dispatcher.count == 3 + limits.max_plan_steps + 2
    assert dispatcher.count <= limits.max_calls_per_lane
    assert set(receipt.gates_passed) == {"task.gate.assess", "task.gate.implement"}
    assert context["task_contract"]


@pytest.mark.parametrize("bound", [1, 2, 3, 4, 7])
def test_the_indexer_pre_lane_never_dispatches_past_its_bound(tmp_path, bound):
    """Hostile bounds: the dispatcher itself must never be entered N+1 times."""

    dispatcher = _CountingDispatcher(_analysis_steps(8))
    route = _route(dispatcher, RouteLimits(max_calls_per_lane=bound, max_plan_steps=8))

    with pytest.raises(CodingRouteError) as excinfo:
        asyncio.run(route._indexer_pre(_request(tmp_path)))

    assert excinfo.value.lane is RouteLane.INDEXER_PRE
    assert dispatcher.count <= bound, (dispatcher.count, bound)


def test_an_unrunnable_plan_is_refused_before_its_first_step(tmp_path):
    """Discovering this halfway through has already spent the work.

    The plan is legal and every step of it is mandatory, so the honest answer
    when it cannot fit is one stable refusal *before* the first step - not a
    partial execution that stops when the budget runs out.
    """

    steps = _analysis_steps(6)
    dispatcher = _CountingDispatcher(steps)
    route = _route(dispatcher, RouteLimits(max_calls_per_lane=6, max_plan_steps=8))

    with pytest.raises(CodingRouteError) as excinfo:
        asyncio.run(route._indexer_pre(_request(tmp_path)))

    assert excinfo.value.code == "plan_call_budget_exceeded"
    # Exactly the three discovery calls: not one plan step was dispatched.
    assert dispatcher.count == 3
    assert [tool for tool, _ in dispatcher.calls] == ["structure", "search", "task"]


def test_a_plan_that_exactly_fits_is_not_refused(tmp_path):
    """The feasibility test is arithmetic, not a safety margin that rejects work."""

    steps = _analysis_steps(4)
    dispatcher = _CountingDispatcher(steps)
    # 3 discovery + 4 steps + 2 host-run canonical gates.
    route = _route(dispatcher, RouteLimits(max_calls_per_lane=9, max_plan_steps=8))

    receipt, _ = asyncio.run(route._indexer_pre(_request(tmp_path)))

    assert receipt.reason_code == "completed"
    assert dispatcher.count == 9


def test_a_plan_scheduling_its_own_gates_is_costed_correctly(tmp_path):
    """A gate the plan schedules is not also run again by the host."""

    steps = [
        _gate_step("g1", "assess"),
        _analysis_steps(1)[0],
        _gate_step("g2", "implement"),
    ]
    dispatcher = _CountingDispatcher(steps)
    route = _route(dispatcher, RouteLimits(max_calls_per_lane=6, max_plan_steps=8))

    receipt, _ = asyncio.run(route._indexer_pre(_request(tmp_path)))

    assert dispatcher.count == 6  # 3 discovery + 3 steps, no extra host gates
    assert list(receipt.gates_passed) == ["task.gate.assess", "task.gate.implement"]


@pytest.mark.parametrize("phase", ["verify", "audit", "assess_extra", "IMPLEMENT"])
def test_an_unknown_gate_phase_fails_closed(tmp_path, phase):
    steps = [_gate_step("g1", phase)]
    dispatcher = _CountingDispatcher(steps)
    route = _route(dispatcher)

    with pytest.raises(CodingRouteError) as excinfo:
        asyncio.run(route._indexer_pre(_request(tmp_path)))

    assert excinfo.value.code == "plan_gate_phase_unknown"
    # Refused before any gate was dispatched.
    assert dispatcher.count == 3


def test_an_unspecified_gate_phase_keeps_the_canonical_default(tmp_path):
    """An omitted phase is the long-standing "assess" default, not an unknown one."""

    step = _gate_step("g1", "assess")
    step["args"].pop("next_phase")
    dispatcher = _CountingDispatcher([step])
    route = _route(dispatcher)

    receipt, _ = asyncio.run(route._indexer_pre(_request(tmp_path)))

    assert "task.gate.assess" in receipt.gates_passed


def test_a_repeated_gate_phase_fails_closed(tmp_path):
    """Gate expansion is bounded: a phase twice is a plan this host will not run."""

    steps = [_gate_step("g1", "assess"), _gate_step("g2", "assess")]
    dispatcher = _CountingDispatcher(steps)
    route = _route(dispatcher)

    with pytest.raises(CodingRouteError) as excinfo:
        asyncio.run(route._indexer_pre(_request(tmp_path)))

    assert excinfo.value.code == "plan_gate_phase_repeated"
    assert dispatcher.count == 3


def test_exhausting_the_budget_fabricates_no_successful_call(tmp_path):
    """A budget refusal dispatches nothing and invents no completed record."""

    dispatcher = _CountingDispatcher(_analysis_steps(2))
    route = _route(dispatcher, RouteLimits(max_calls_per_lane=2, max_plan_steps=8))

    with pytest.raises(CodingRouteError):
        asyncio.run(route._indexer_pre(_request(tmp_path)))

    trace = route._trace
    assert trace is not None
    assert trace.dispatches <= 2
    assert dispatcher.count <= 2
    # No record claims success for a call that never went out.
    assert len(trace.calls) <= dispatcher.count


@pytest.mark.parametrize("bound", [1, 2, 3])
def test_every_lane_respects_the_bound_before_dispatching(tmp_path, bound):
    """Indexer post, Blueprint and Core are charged by the same gate."""

    for lane_name in ("_blueprint", "_core", "_indexer_post"):
        dispatcher = _CountingDispatcher()
        route = _route(dispatcher, RouteLimits(max_calls_per_lane=bound))
        lane = getattr(route, lane_name)
        request = _request(tmp_path)
        try:
            if lane_name == "_indexer_post":
                asyncio.run(lane(request, {"project": "p", "task_contract": {"a": 1}}, None))
            else:
                asyncio.run(lane(request))
        except (CodingRouteError, Exception):
            pass
        assert dispatcher.count <= bound, (lane_name, dispatcher.count, bound)


# --------------------------------------------------------------------------
# incident B: a declared check that cannot be launched is an environment defect
# --------------------------------------------------------------------------


def _contract(tmp_path, argv, required=True, name="unit"):
    (tmp_path / ".flyto").mkdir(parents=True, exist_ok=True)
    rendered = ", ".join('"{}"'.format(part) for part in argv)
    (tmp_path / ".flyto" / "coding.yaml").write_text(
        "version: flyto.coding-config.v1\n"
        "checks:\n"
        "  - name: {}\n"
        "    argv: [{}]\n"
        "    required: {}\n".format(name, rendered, "true" if required else "false"),
        encoding="utf-8",
    )
    return str(tmp_path)


def test_the_resolver_is_the_one_the_runner_uses():
    """Not a copy of the rule: literally the same function, so it cannot drift."""

    import flyto_ai.coding.workspace as workspace_module

    source = Path(workspace_module.__file__).read_text(encoding="utf-8")
    body = source.split("async def _run_process", 1)[1][:600]
    assert "resolve_executable(argv[0])" in body
    assert "shutil.which(argv[0]) if not" not in source


def test_resolution_matches_every_argv_spelling(tmp_path):
    installed = shutil.which("python3") or shutil.which("python")
    assert installed is not None

    # Plain name on PATH.
    assert resolve_executable(Path(installed).name) is not None
    # Absolute path.
    assert resolve_executable(installed) == installed
    # Absolute path that is not there.
    assert resolve_executable(str(tmp_path / "definitely-absent")) is None
    # Relative path, checked as a path.
    script = tmp_path / "tool.sh"
    script.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        assert resolve_executable("./tool.sh") is not None
    finally:
        os.chdir(cwd)
    # Present but not executable is not launchable.
    plain = tmp_path / "not-exec.sh"
    plain.write_text("#!/bin/sh\n", encoding="utf-8")
    assert resolve_executable(str(plain)) is None


def test_a_missing_required_check_executable_is_refused(tmp_path):
    workspace = _contract(tmp_path, ["definitely-not-installed-xyz", "--version"])

    outcome = preflight_repository(workspace)

    assert outcome.ok is False
    assert outcome.code == CODE_VERIFICATION_TOOL_MISSING
    assert outcome.required_actions == (ACTION_INSTALL_VERIFICATION_TOOL,)
    assert outcome.blockers == ("unit",)
    # A distinct answer from the two neighbouring refusals.
    assert outcome.code in PREFLIGHT_CODES
    assert ACTION_INSTALL_VERIFICATION_TOOL in PREFLIGHT_ACTIONS


def test_an_optional_missing_check_does_not_block(tmp_path):
    workspace = _contract(tmp_path, ["definitely-not-installed-xyz"], required=False)
    # An optional-only contract has no required check at all, so the refusal is
    # the older, narrower one - not the tool refusal.
    outcome = preflight_repository(workspace)
    assert outcome.code != CODE_VERIFICATION_TOOL_MISSING

    (tmp_path / ".flyto" / "coding.yaml").write_text(
        "version: flyto.coding-config.v1\n"
        "checks:\n"
        "  - name: required_one\n"
        "    argv: [\"python3\", \"-c\", \"pass\"]\n"
        "    required: true\n"
        "  - name: optional_one\n"
        "    argv: [\"definitely-not-installed-xyz\"]\n"
        "    required: false\n",
        encoding="utf-8",
    )
    outcome = preflight_repository(str(tmp_path))
    assert outcome.ok is True, outcome.code


def test_an_installed_check_is_never_executed(tmp_path, monkeypatch):
    """A non-zero baseline may be the defect the task exists to fix."""

    marker = tmp_path / "ran"
    script = tmp_path / "tool.sh"
    script.write_text(
        "#!/bin/sh\ntouch '{}'\nexit 1\n".format(marker), encoding="utf-8",
    )
    script.chmod(script.stat().st_mode | stat.S_IXUSR)
    workspace = _contract(tmp_path, [str(script)])

    outcome = preflight_repository(workspace)

    assert outcome.ok is True
    assert not marker.exists()


def test_blockers_are_bounded_and_name_only_checks(tmp_path):
    lines = ["version: flyto.coding-config.v1", "checks:"]
    for index in range(MAX_PREFLIGHT_BLOCKERS + 4):
        lines.append("  - name: check_{}".format(index))
        lines.append("    argv: [\"absent-tool-{}\"]".format(index))
        lines.append("    required: true")
    (tmp_path / ".flyto").mkdir(parents=True, exist_ok=True)
    (tmp_path / ".flyto" / "coding.yaml").write_text("\n".join(lines), encoding="utf-8")

    outcome = preflight_repository(str(tmp_path))

    assert outcome.code == CODE_VERIFICATION_TOOL_MISSING
    assert len(outcome.blockers) <= MAX_PREFLIGHT_BLOCKERS
    for blocker in outcome.blockers:
        assert blocker.startswith("check_")
        assert "/" not in blocker and " " not in blocker


def test_the_runner_refuses_an_impossible_verifier_typed(tmp_path):
    """The invariant-bug path: typed, named, and not a verification verdict."""

    checks = (
        CheckSpec(name="unit", argv=("definitely-not-installed-xyz",), required=True),
    )
    assert unlaunchable_required_checks(checks) == ("unit",)

    class _Tools:
        pass

    with pytest.raises(VerificationToolUnavailable) as excinfo:
        asyncio.run(CheckRunner(_Tools()).run(checks))

    assert excinfo.value.blockers == ("unit",)
    assert "definitely-not-installed" not in str(excinfo.value)


def test_preflight_and_runner_share_one_answer(tmp_path):
    """Whatever the runner would refuse, preflight refuses first, by name."""

    checks = (
        CheckSpec(name="alpha", argv=("definitely-not-installed-xyz",), required=True),
        CheckSpec(name="beta", argv=("also-absent-abc",), required=True),
    )
    assert unlaunchable_required_checks(checks) == ("alpha", "beta")
    assert preflight_module.unlaunchable_required_checks(checks) == ("alpha", "beta")


# --------------------------------------------------------------------------
# incident B: the typed refusal, and the work it must not have done first
# --------------------------------------------------------------------------


def _service(tmp_path, workspace):
    from flyto_ai.coding.service import CodingService

    def _never_built(store):  # pragma: no cover - preflight refuses first
        raise AssertionError("no implementer may be constructed for a refusal")

    return CodingService(
        _never_built,
        state_root=str(tmp_path / "service-state"),
        workspace_roots=(workspace,),
        max_workers=1,
        max_queued=2,
    )


def test_the_service_refuses_with_a_typed_terminal_code(tmp_path):
    """A distinct code, phase, retryability and action - not a variable message."""

    from flyto_ai.coding.service import VerificationToolMissing

    workspace = _contract(tmp_path, ["definitely-not-installed-xyz"])
    service = _service(tmp_path, workspace)

    with pytest.raises(VerificationToolMissing) as excinfo:
        service._require_verifiable_repository(workspace)

    error = excinfo.value
    assert error.code == "verification_tool_missing"
    assert error.failure_phase == "preflight"
    assert error.retryable is False
    assert error.required_actions == (ACTION_INSTALL_VERIFICATION_TOOL,)
    assert error.blockers == ("unit",)


def test_the_refusal_projects_bounded_blockers_and_no_paths(tmp_path):
    """What crosses MCP and HTTP is identifiers, never paths, argv or prose."""

    from flyto_ai.coding.service import VerificationToolMissing, error_details

    error = VerificationToolMissing(
        "a required verification tool is not installed on this host",
        (ACTION_INSTALL_VERIFICATION_TOOL,),
        ("unit", "lint"),
    )
    projected = error_details(error)

    assert projected["failure_phase"] == "preflight"
    assert projected["retryable"] is False
    assert projected["required_actions"] == [ACTION_INSTALL_VERIFICATION_TOOL]
    assert projected["verification_blockers"] == ["lint", "unit"]
    rendered = repr(projected)
    for forbidden in ("/", "\\", "argv", "PATH", "not installed", "Traceback"):
        assert forbidden not in rendered


def test_projection_still_refuses_anything_that_is_not_an_identifier():
    """The one identifier list is narrow, not an open string channel."""

    from flyto_ai.coding.service import error_details

    class _Hostile(Exception):
        details = {
            "verification_blockers": [
                "unit",
                "/usr/local/bin/pytest",
                "rm -rf /",
                "a" * 400,
                {"nested": 1},
            ],
            "other_list": ["unit", "not_an_action"],
        }

    projected = error_details(_Hostile())

    assert projected["verification_blockers"] == ["unit"]
    assert "other_list" not in projected


def test_the_refusal_creates_no_job_no_claim_and_no_session(tmp_path):
    """Refused before a job id exists, so there is nothing to poll or release."""

    workspace = _contract(tmp_path, ["definitely-not-installed-xyz"])
    service = _service(tmp_path, workspace)

    before_jobs = dict(getattr(service, "_jobs", {}) or {})
    with pytest.raises(Exception) as excinfo:
        service._require_verifiable_repository(workspace)
    assert excinfo.value.code == "verification_tool_missing"

    assert dict(getattr(service, "_jobs", {}) or {}) == before_jobs
    assert not getattr(service, "_resume", {})
    assert not getattr(service, "_pending", set())


def test_an_installed_required_check_passes_preflight_in_the_service(tmp_path):
    workspace = _contract(tmp_path, ["python3", "--version"])
    service = _service(tmp_path, workspace)

    digest = service._require_verifiable_repository(workspace)

    assert isinstance(digest, str) and len(digest) == 64


# --------------------------------------------------------------------------
# incident C: a compound plan's sub-tasks each own their canonical gates
# --------------------------------------------------------------------------
#
# Job job_2bd89fbe119147a38b9b5ee0 failed with route_plan_gate_phase_repeated
# before Claude was ever called.  The Indexer had returned two independently
# compiled sub-tasks, each ending with its own `assess` and `implement` - which
# is what "independently compiled" means - and the host judged gate uniqueness
# globally, so the second sub-task's `assess` looked like a duplicate.  The
# effect was that every multi-sub-task task was refused before implementation.


class _CompoundDispatcher(_CountingDispatcher):
    """A dispatcher whose gate honours the Indexer's compound state machine.

    It refuses a sub-task's gate until the host has told it, through
    `completed_subtasks`, that every earlier sub-task really finished. That is
    what turns "the four gates were called" into "the four gates were called in
    an order the Indexer would accept".
    """

    def __init__(self, plan, expected_order):
        super().__init__()
        self._plan = plan
        self._expected = list(expected_order)
        self.gate_calls = []

    async def __call__(self, tool, arguments):
        args = dict(arguments or {})
        action = args.get("action")
        if tool == "task" and action == "plan":
            self.calls.append((tool, "plan"))
            return dict(self._plan, ok=True)
        if tool == "task" and action == "gate":
            state = args.get("current_state") or {}
            done = tuple(state.get("completed_subtasks") or ())
            self.gate_calls.append((args.get("next_phase"), done))
            self.calls.append((tool, "gate"))
            index = len(self.gate_calls) - 1
            if index < len(self._expected):
                phase, required = self._expected[index]
                if args.get("next_phase") != phase or done != required:
                    return {
                        "ok": True, "pass": False,
                        "required_state": {"completed_subtasks": list(required)},
                    }
            return {"ok": True, "pass": True, "required_state": {}}
        return await super().__call__(tool, arguments)


def _subtask(index, *, gates=("assess", "implement"), analysis=1, ids=None):
    """One compiled sub-task: some analysis, then its own canonical gates."""

    local = ids or ["step_01_scope"]
    steps = [
        {
            "id": local[position % len(local)],
            "tool": "impact",
            "args": {"target": "proj:app.py:function:main"},
            "purpose": "scope_callers",
            "required": True,
            "depends_on": [],
        }
        for position in range(analysis)
    ]
    for position, phase in enumerate(gates):
        steps.append({
            "id": "step_gate_{:02d}_{}".format(position, phase),
            "tool": "task",
            "args": {"action": "gate", "next_phase": phase},
            "purpose": "gate",
            "required": True,
            "depends_on": [],
        })
    return {"execution_plan": steps}


def test_the_live_two_subtask_compound_plan_reaches_implementation(tmp_path):
    """The exact shape from the incident, through the production pre lane."""

    plan = {
        "task_profile": {},
        "sub_tasks": [_subtask(1), _subtask(2)],
    }
    expected = [
        ("assess", ()),
        ("implement", ()),
        ("assess", ("subtask_1",)),
        ("implement", ("subtask_1",)),
    ]
    dispatcher = _CompoundDispatcher(plan, expected)
    route = _route(dispatcher)

    receipt, context = asyncio.run(route._indexer_pre(_request(tmp_path)))

    assert receipt.status.value == "applied"
    assert receipt.reason_code == "completed"
    # Four gates, in the declared order, each seeing the sub-tasks the host had
    # actually watched finish - and not one of them re-run or coalesced.
    assert dispatcher.gate_calls == expected
    # No extra global fallback gate: every canonical phase was scheduled.
    assert dispatcher.count == 3 + 2 * 3
    assert list(receipt.gates_passed) == [
        "task.gate.assess:subtask_1",
        "task.gate.implement:subtask_1",
        "task.gate.assess:subtask_2",
        "task.gate.implement:subtask_2",
    ]
    # The contract handed on is byte-for-byte what the Indexer returned - no
    # host scope metadata and no completion bookkeeping was inserted into it.
    assert context["task_contract"] == dict(plan, ok=True)
    assert "completed_subtasks" not in context["task_contract"]
    for sub_task in context["task_contract"]["sub_tasks"]:
        for step in sub_task["execution_plan"]:
            assert not str(step["id"]).startswith("subtask_")
            assert "completed_subtasks" not in (step.get("args") or {})
    for _, done in dispatcher.gate_calls:
        assert isinstance(done, tuple)


def test_three_subtasks_reusing_local_ids_are_accepted_and_scoped(tmp_path):
    """Local ids repeat across compiled plans; the receipt still says which is which."""

    plan = {
        "task_profile": {},
        "sub_tasks": [_subtask(index) for index in (1, 2, 3)],
    }
    expected = [
        ("assess", ()), ("implement", ()),
        ("assess", ("subtask_1",)), ("implement", ("subtask_1",)),
        ("assess", ("subtask_1", "subtask_2")),
        ("implement", ("subtask_1", "subtask_2")),
    ]
    dispatcher = _CompoundDispatcher(plan, expected)

    receipt, _ = asyncio.run(_route(dispatcher)._indexer_pre(_request(tmp_path)))

    assert dispatcher.gate_calls == expected
    assert len(set(receipt.gates_passed)) == 6
    assert dispatcher.count == 3 + 3 * 3


def test_a_duplicate_phase_inside_one_subtask_is_still_refused(tmp_path):
    """Per-scope, not permissive: one compiled plan may still not gate twice."""

    plan = {
        "task_profile": {},
        "sub_tasks": [_subtask(1, gates=("assess", "assess"))],
    }
    dispatcher = _CompoundDispatcher(plan, [])

    with pytest.raises(CodingRouteError) as excinfo:
        asyncio.run(_route(dispatcher)._indexer_pre(_request(tmp_path)))

    assert excinfo.value.code == "plan_gate_phase_repeated"
    assert dispatcher.count == 3  # refused before the first plan step


def test_a_duplicate_phase_in_the_root_plan_is_still_refused(tmp_path):
    """The root execution plan is a scope too, and it has the same rule."""

    plan = {
        "task_profile": {},
        "execution_plan": [_gate_step("g1", "assess"), _gate_step("g2", "assess")],
    }
    dispatcher = _CompoundDispatcher(plan, [])

    with pytest.raises(CodingRouteError) as excinfo:
        asyncio.run(_route(dispatcher)._indexer_pre(_request(tmp_path)))

    assert excinfo.value.code == "plan_gate_phase_repeated"
    assert dispatcher.count == 3


def test_an_unknown_phase_in_a_subtask_is_refused_before_dispatch(tmp_path):
    plan = {"task_profile": {}, "sub_tasks": [_subtask(1, gates=("assess", "verify"))]}
    dispatcher = _CompoundDispatcher(plan, [])

    with pytest.raises(CodingRouteError) as excinfo:
        asyncio.run(_route(dispatcher)._indexer_pre(_request(tmp_path)))

    assert excinfo.value.code == "plan_gate_phase_unknown"
    assert dispatcher.count == 3


def test_an_interrupted_subtask_is_never_announced_as_complete(tmp_path):
    """A sub-task whose last step failed must not unlock the next one's gate."""

    plan = {
        "task_profile": {},
        "sub_tasks": [
            {"execution_plan": [
                {"id": "step_01_scope", "tool": "impact",
                 "args": {"target": "x"}, "required": True, "depends_on": []},
                {"id": "step_02_bad", "tool": "not_an_allowlisted_tool",
                 "args": {}, "required": True, "depends_on": []},
            ]},
            _subtask(2),
        ],
    }
    dispatcher = _CompoundDispatcher(plan, [])

    with pytest.raises(CodingRouteError) as excinfo:
        asyncio.run(_route(dispatcher)._indexer_pre(_request(tmp_path)))

    assert excinfo.value.code == "plan_step_not_allowlisted"
    # The later sub-task's gate was never reached, so nothing was told that the
    # interrupted sub-task had finished.
    assert dispatcher.gate_calls == []


def test_a_partly_gated_compound_plan_gets_exactly_the_missing_fallback(tmp_path):
    """Fallback is per missing *phase*, not per sub-task, and only when absent."""

    plan = {
        "task_profile": {},
        "sub_tasks": [_subtask(1, gates=("assess",)), _subtask(2, gates=("assess",))],
    }
    dispatcher = _CompoundDispatcher(plan, [
        ("assess", ()), ("assess", ("subtask_1",)), ("implement", ("subtask_1", "subtask_2")),
    ])

    receipt, _ = asyncio.run(_route(dispatcher)._indexer_pre(_request(tmp_path)))

    # Two scheduled assess gates, and exactly one host-run implement.
    assert [phase for phase, _ in dispatcher.gate_calls] == ["assess", "assess", "implement"]
    assert list(receipt.gates_passed) == [
        "task.gate.assess:subtask_1",
        "task.gate.assess:subtask_2",
        "task.gate.implement",
    ]
    assert dispatcher.count == 3 + 2 * 2 + 1


def test_a_gate_heavy_compound_plan_fits_the_default_budget(tmp_path):
    """The largest legal gate-heavy compound plan still runs, and stays bounded."""

    limits = RouteLimits()
    count = limits.max_plan_steps // 2  # each sub-task is one analysis + one gate
    plan = {
        "task_profile": {},
        "sub_tasks": [
            _subtask(index, gates=("assess",), analysis=1)
            for index in range(1, count + 1)
        ],
    }
    dispatcher = _CompoundDispatcher(plan, [])
    receipt, _ = asyncio.run(_route(dispatcher, limits)._indexer_pre(_request(tmp_path)))

    assert receipt.reason_code == "completed"
    # 3 discovery + every mandatory step + exactly one missing fallback gate.
    assert dispatcher.count == 3 + limits.max_plan_steps + 1
    assert dispatcher.count <= limits.max_calls_per_lane


def test_a_compound_plan_whose_gates_cannot_fit_is_refused_first(tmp_path):
    """Arithmetic still comes before dispatch, compound or not."""

    plan = {"task_profile": {}, "sub_tasks": [_subtask(index) for index in (1, 2)]}
    dispatcher = _CompoundDispatcher(plan, [])
    route = _route(dispatcher, RouteLimits(max_calls_per_lane=8, max_plan_steps=16))

    with pytest.raises(CodingRouteError) as excinfo:
        asyncio.run(route._indexer_pre(_request(tmp_path)))

    assert excinfo.value.code == "plan_call_budget_exceeded"
    assert dispatcher.count == 3


def test_a_missing_fallback_gate_is_counted_before_the_first_step(tmp_path):
    """The host-run gate is part of the demand, not an afterthought.

    Two sub-tasks of two steps each schedule `assess` only, so the host must
    still run one `implement` itself: four steps plus one fallback is five, and
    a lane with four calls left cannot do it. Leaving that fallback out of the
    arithmetic would let the plan start and run out partway through - which is
    the failure mode the pre-flight exists to remove.
    """

    plan = {
        "task_profile": {},
        "sub_tasks": [_subtask(1, gates=("assess",)), _subtask(2, gates=("assess",))],
    }
    dispatcher = _CompoundDispatcher(plan, [])
    route = _route(dispatcher, RouteLimits(max_calls_per_lane=7, max_plan_steps=16))

    with pytest.raises(CodingRouteError) as excinfo:
        asyncio.run(route._indexer_pre(_request(tmp_path)))

    assert excinfo.value.code == "plan_call_budget_exceeded"
    assert dispatcher.count == 3


# --------------------------------------------------------------------------
# incident D: scope is an authority decision, and scopes run whole
# --------------------------------------------------------------------------


def test_a_root_plan_cannot_buy_a_second_gate_by_naming_itself_a_subtask(tmp_path):
    """Provenance must be assigned by the host, never read off a step id.

    A step id is data the planner supplies. When the scope was parsed back out
    of it, an ordinary root plan could label two of its own gates
    ``subtask_1:g1`` and ``subtask_2:g2`` and be treated as two independent
    plans - so two ``assess`` gates in one plan stopped looking like the
    duplicate they are, and the refusal arrived much later as a remediation
    failure instead.
    """

    plan = {
        "task_profile": {},
        "execution_plan": [
            _gate_step("subtask_1:g1", "assess"),
            _gate_step("subtask_2:g2", "assess"),
        ],
    }
    dispatcher = _CompoundDispatcher(plan, [])

    with pytest.raises(CodingRouteError) as excinfo:
        asyncio.run(_route(dispatcher)._indexer_pre(_request(tmp_path)))

    assert excinfo.value.code == "plan_gate_phase_repeated"
    # Exactly the three discovery calls: refused before any gate dispatched.
    assert dispatcher.count == 3
    assert dispatcher.gate_calls == []


def test_ordinary_root_ids_resembling_the_reserved_prefix_still_run(tmp_path):
    """The repair must not start rejecting legal ids that merely look reserved."""

    plan = {
        "task_profile": {},
        "execution_plan": [
            {"id": "subtask_1:analysis", "tool": "impact",
             "args": {"target": "x"}, "required": True, "depends_on": []},
            _gate_step("subtask_2:assess", "assess"),
            _gate_step("subtask_9:implement", "implement"),
        ],
    }
    dispatcher = _CompoundDispatcher(plan, [("assess", ()), ("implement", ())])

    receipt, _ = asyncio.run(_route(dispatcher)._indexer_pre(_request(tmp_path)))

    # One root plan: legacy markers, unscoped, exactly as before.
    assert list(receipt.gates_passed) == ["task.gate.assess", "task.gate.implement"]
    assert [done for _, done in dispatcher.gate_calls] == [(), ()]


def test_a_locally_unordered_subtask_never_interleaves_with_the_next(tmp_path):
    """Each compiled plan is sorted on its own; declared scope order is absolute.

    Flattening every plan and sorting once let a sub-task whose first listed
    step depends on a later local step be deferred - the sorter walked on into
    ``subtask_2``, ran its gates first, and announced it complete before
    ``subtask_1`` had finished anything.
    """

    subtask_1 = {"execution_plan": [
        # Listed first, but depends on a step listed after it.
        {"id": "step_02_gate", "tool": "task",
         "args": {"action": "gate", "next_phase": "assess"},
         "required": True, "depends_on": ["step_01_scope"]},
        {"id": "step_01_scope", "tool": "impact",
         "args": {"target": "x"}, "required": True, "depends_on": []},
        {"id": "step_03_gate", "tool": "task",
         "args": {"action": "gate", "next_phase": "implement"},
         "required": True, "depends_on": ["step_02_gate"]},
    ]}
    plan = {"task_profile": {}, "sub_tasks": [subtask_1, _subtask(2)]}
    expected = [
        ("assess", ()),
        ("implement", ()),
        ("assess", ("subtask_1",)),
        ("implement", ("subtask_1",)),
    ]
    dispatcher = _CompoundDispatcher(plan, expected)

    receipt, _ = asyncio.run(_route(dispatcher)._indexer_pre(_request(tmp_path)))

    # Sub-task one entirely, then sub-task two - and completion announced 1
    # before 2, never the other way round.
    assert dispatcher.gate_calls == expected
    assert list(receipt.gates_passed) == [
        "task.gate.assess:subtask_1",
        "task.gate.implement:subtask_1",
        "task.gate.assess:subtask_2",
        "task.gate.implement:subtask_2",
    ]


def test_a_cross_scope_dependency_is_not_expressible(tmp_path):
    """A name from another compiled plan is simply unknown, and unknown refuses."""

    plan = {
        "task_profile": {},
        "sub_tasks": [
            _subtask(1),
            {"execution_plan": [
                {"id": "step_01_scope", "tool": "impact", "args": {"target": "x"},
                 "required": True, "depends_on": ["step_gate_00_assess"]},
            ]},
        ],
    }
    dispatcher = _CompoundDispatcher(plan, [])

    with pytest.raises(CodingRouteError) as excinfo:
        asyncio.run(_route(dispatcher)._indexer_pre(_request(tmp_path)))

    assert excinfo.value.code == "plan_dependency_unknown"
    assert dispatcher.count == 3


def test_a_dependency_cycle_inside_one_subtask_is_refused(tmp_path):
    plan = {
        "task_profile": {},
        "sub_tasks": [{"execution_plan": [
            {"id": "a", "tool": "impact", "args": {"target": "x"},
             "required": True, "depends_on": ["b"]},
            {"id": "b", "tool": "impact", "args": {"target": "y"},
             "required": True, "depends_on": ["a"]},
        ]}],
    }
    dispatcher = _CompoundDispatcher(plan, [])

    with pytest.raises(CodingRouteError):
        asyncio.run(_route(dispatcher)._indexer_pre(_request(tmp_path)))
    assert dispatcher.count == 3


def test_local_ids_may_repeat_across_subtasks_but_not_within_one(tmp_path):
    """Independently compiled plans restart their ids; one plan may not."""

    shared = {"execution_plan": [
        {"id": "step_01_scope", "tool": "impact", "args": {"target": "x"},
         "required": True, "depends_on": []},
        _gate_step("step_02_gate", "assess"),
    ]}
    plan = {"task_profile": {}, "sub_tasks": [dict(shared), dict(shared)]}
    dispatcher = _CompoundDispatcher(plan, [
        ("assess", ()), ("assess", ("subtask_1",)), ("implement", ("subtask_1", "subtask_2")),
    ])
    receipt, _ = asyncio.run(_route(dispatcher)._indexer_pre(_request(tmp_path)))
    assert list(receipt.gates_passed)[:2] == [
        "task.gate.assess:subtask_1", "task.gate.assess:subtask_2",
    ]

    duplicated = {"task_profile": {}, "sub_tasks": [{"execution_plan": [
        {"id": "same", "tool": "impact", "args": {"target": "x"},
         "required": True, "depends_on": []},
        {"id": "same", "tool": "impact", "args": {"target": "y"},
         "required": True, "depends_on": []},
    ]}]}
    hostile = _CompoundDispatcher(duplicated, [])
    with pytest.raises(CodingRouteError) as excinfo:
        asyncio.run(_route(hostile)._indexer_pre(_request(tmp_path)))
    assert excinfo.value.code == "plan_step_id_duplicated"


def test_a_mixed_root_and_compound_plan_runs_root_first(tmp_path):
    """Root plan, then each sub-task, in declared order."""

    plan = {
        "task_profile": {},
        "execution_plan": [_gate_step("root_assess", "assess")],
        "sub_tasks": [_subtask(1, gates=("implement",))],
    }
    dispatcher = _CompoundDispatcher(plan, [("assess", ()), ("implement", ())])

    receipt, _ = asyncio.run(_route(dispatcher)._indexer_pre(_request(tmp_path)))

    assert list(receipt.gates_passed) == [
        "task.gate.assess",
        "task.gate.implement:subtask_1",
    ]
