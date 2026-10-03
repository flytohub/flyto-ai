"""Capability-contract grading: actuating packs ask, stops stay immediate."""

import pytest

from flyto_ai.permissions import (
    RISK_EXTERNAL_WRITE,
    RISK_IRREVERSIBLE,
    RISK_OBSERVE,
    RISK_REAL_WORLD,
    PermissionEnforcer,
    PermissionLevel,
    PermissionOutcome,
    grade_module_contract,
    resolve_module_grade,
)


def _contract(**overrides):
    contract = {
        "schema": "flyto.capability-contract.v1",
        "actuates": False,
        "safety_class": "read_only",
        "requires_safe_stop": False,
        "cancellable": True,
        "idempotent": True,
    }
    contract.update(overrides)
    return contract


def _info(plugin="pack-a", capability="device.act", contract=None):
    return {"plugin": plugin, "provides_capability": capability, "contract": contract}


def test_actuating_contract_is_danger_and_real_world():
    grade = grade_module_contract(_info(contract=_contract(actuates=True, safety_class="controlled")))
    assert grade.permission_level == PermissionLevel.DANGER_FULL
    assert grade.risk_level == RISK_REAL_WORLD
    assert grade.actuating is True
    assert grade.immediate is False


@pytest.mark.parametrize(
    "safety_class, risk",
    [("movement", RISK_REAL_WORLD), ("dangerous", RISK_IRREVERSIBLE)],
)
def test_movement_and_dangerous_are_danger_even_when_not_actuating(safety_class, risk):
    grade = grade_module_contract(_info(contract=_contract(safety_class=safety_class)))
    assert grade.permission_level == PermissionLevel.DANGER_FULL
    assert grade.risk_level == risk
    assert grade.permission_level != PermissionLevel.WORKSPACE_WRITE


def test_dangerous_actuating_is_level_five():
    grade = grade_module_contract(_info(contract=_contract(actuates=True, safety_class="dangerous")))
    assert grade.risk_level == RISK_IRREVERSIBLE
    assert grade.permission_level == PermissionLevel.DANGER_FULL


def test_non_actuating_contracts_grade_by_safety_class():
    controlled = grade_module_contract(_info(contract=_contract(safety_class="controlled")))
    assert controlled.permission_level == PermissionLevel.WORKSPACE_WRITE
    assert controlled.risk_level == RISK_EXTERNAL_WRITE
    read_only = grade_module_contract(_info(contract=_contract()))
    assert read_only.permission_level == PermissionLevel.READ_ONLY
    assert read_only.risk_level == RISK_OBSERVE
    assert read_only.actuating is False


def test_pack_module_without_contract_fails_closed():
    grade = grade_module_contract(_info(plugin="third-party", capability="anything.neutral"))
    assert grade.permission_level == PermissionLevel.DANGER_FULL
    assert grade.risk_level == RISK_REAL_WORLD
    assert grade.actuating is True
    assert grade.source == "missing_contract"


@pytest.mark.parametrize(
    "bad",
    ["yes", {"actuates": True}, _contract(safety_class="unknown"), _contract(actuates="no")],
)
def test_unreadable_contract_fails_closed(bad):
    grade = grade_module_contract(_info(plugin="", contract=bad))
    assert grade.permission_level == PermissionLevel.DANGER_FULL
    assert grade.source == "invalid_contract"


def test_core_module_without_contract_keeps_legacy_grading():
    assert grade_module_contract(_info(plugin="", capability="", contract=None)) is None
    assert grade_module_contract(None) is None


def test_stop_capability_stays_immediate():
    stop = _contract(actuates=True, safety_class="controlled", requires_safe_stop=False, cancellable=False)
    grade = grade_module_contract(_info(capability="motion.halt", contract=stop))
    assert grade.immediate is True
    assert grade.permission_level == PermissionLevel.READ_ONLY
    assert grade.risk_level == RISK_OBSERVE


@pytest.mark.parametrize(
    "contract",
    [
        None,
        _contract(actuates=True, safety_class="movement", cancellable=False),
        _contract(actuates=True, safety_class="controlled", cancellable=True),
        _contract(actuates=True, safety_class="controlled", requires_safe_stop=True, cancellable=False),
    ],
)
def test_stop_name_cannot_carry_a_non_stop_past_confirmation(contract):
    grade = grade_module_contract(_info(capability="motion.halt", contract=contract))
    assert grade.immediate is False
    assert grade.permission_level == PermissionLevel.DANGER_FULL


def test_resolver_failure_fails_closed():
    def broken(_module_id):
        raise RuntimeError("registry unavailable")

    grade = resolve_module_grade("pack.thing", broken)
    assert grade.permission_level == PermissionLevel.DANGER_FULL
    assert resolve_module_grade("", broken) is None


# ── PermissionEnforcer.execute_module ──────────────────────────────────

_CATALOG = {
    "vendor.move": _info(plugin="vendor-pack", capability="motion.advance",
                         contract=_contract(actuates=True, safety_class="movement")),
    "vendor.halt": _info(plugin="vendor-pack", capability="motion.halt",
                         contract=_contract(actuates=True, safety_class="controlled",
                                            requires_safe_stop=False, cancellable=False)),
    "vendor.bare": _info(plugin="vendor-pack", capability="thing.do", contract=None),
    "data.parse": _info(plugin="", capability="", contract=None),
    "core.actuator": _info(plugin="", capability="lift.move",
                           contract=_contract(actuates=True, safety_class="movement")),
}


def _enforcer(level=PermissionLevel.WORKSPACE_WRITE):
    return PermissionEnforcer(level, module_info_resolver=_CATALOG.get)


@pytest.mark.parametrize("module_id", ["vendor.move", "vendor.bare", "core.actuator"])
def test_execute_actuating_module_requires_confirmation(module_id):
    enforcer = _enforcer()
    arguments = {"module_id": module_id, "params": {}}
    assert enforcer.required_level("execute_module", arguments) == PermissionLevel.DANGER_FULL
    decision = enforcer.check("execute_module", arguments)
    assert decision.allowed is False
    assert decision.outcome == PermissionOutcome.REQUIRE_CONFIRMATION


def test_execute_stop_is_not_slowed_by_confirmation():
    enforcer = _enforcer()
    decision = enforcer.check("execute_module", {"module_id": "vendor.halt"})
    assert decision.allowed is True


def test_execute_core_module_keeps_category_grading():
    enforcer = _enforcer()
    assert enforcer.required_level("execute_module", {"module_id": "data.parse"}) == PermissionLevel.WORKSPACE_WRITE
    assert enforcer.required_level("execute_module", {"module_id": "shell.exec"}) == PermissionLevel.DANGER_FULL


def test_contract_never_lowers_category_grade():
    catalog = {"shell.exec": _info(plugin="", capability="shell.exec", contract=_contract())}
    enforcer = PermissionEnforcer(module_info_resolver=catalog.get)
    assert enforcer.required_level("execute_module", {"module_id": "shell.exec"}) == PermissionLevel.DANGER_FULL


def test_danger_session_may_execute_actuating_module():
    enforcer = _enforcer(PermissionLevel.DANGER_FULL)
    assert enforcer.check("execute_module", {"module_id": "vendor.move"}).allowed is True
