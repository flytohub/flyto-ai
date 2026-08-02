# Copyright 2024 Flyto2
# Licensed under the Apache License, Version 2.0
"""Closed-loop tests for execution resource, sandbox, secret, and approval policy."""
from __future__ import annotations

import asyncio

import pytest

from flyto_ai.coding.execution_policy import (
    ApprovalDecision,
    ExecutionLimits,
    ExecutionPolicy,
    ExecutionPolicyController,
)
from flyto_ai.permissions import PermissionLevel


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("max_calls", True, "must be an integer"),
        ("max_calls", 0, "outside"),
        ("max_failures", 0, "outside"),
        ("max_concurrency", 0, "outside"),
        ("max_argument_depth", 129, "outside"),
        ("max_result_nodes", 0, "outside"),
        ("max_result_bytes", 16 * 1024 * 1024 + 1, "outside"),
    ],
)
def test_execution_limits_reject_invalid_scalars_and_ranges(field, value, message):
    values = {field: value}
    with pytest.raises(ValueError, match=message):
        ExecutionLimits(**values)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"limits": {}}, "limits must"),
        ({"approval_level": "danger_full"}, "approval_level"),
        ({"reject_sensitive_arguments": 1}, "must be a boolean"),
        ({"allow_outside_workspace_paths": 1}, "must be a boolean"),
        ({"allowed_sensitive_keys": ("",)}, "non-empty"),
        ({"allowed_sensitive_keys": ("api_key", "api_key")}, "duplicates"),
        ({"workspace_path_keys": (1,)}, "non-empty"),
        ({"workspace_path_keys": ("Path", "path")}, "duplicates"),
    ],
)
def test_execution_policy_contract_rejects_ambiguous_values(kwargs, message):
    with pytest.raises(ValueError, match=message):
        ExecutionPolicy(**kwargs)


def test_approval_decision_and_controller_dependencies_are_strict(tmp_path):
    with pytest.raises(ValueError, match="boolean"):
        ApprovalDecision(1)
    with pytest.raises(ValueError, match="approver_ref"):
        ApprovalDecision(True, approver_ref="x" * 257)
    file_workspace = tmp_path / "file"
    file_workspace.write_text("not a directory")
    with pytest.raises(ValueError, match="must be a directory"):
        ExecutionPolicyController(str(file_workspace))
    with pytest.raises(ValueError, match="ExecutionPolicy"):
        ExecutionPolicyController(str(tmp_path), {})
    with pytest.raises(ValueError, match="must be callable"):
        ExecutionPolicyController(str(tmp_path), approval_resolver=1)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("provider", "remote", "level", "arguments", "code"),
    [
        ("", "observe", PermissionLevel.READ_ONLY, {}, "invalid_tool"),
        ("cap_test", "", PermissionLevel.READ_ONLY, {}, "invalid_tool"),
        ("cap_test", "observe", "read_only", {}, "invalid_permission"),
        ("cap_test", "observe", PermissionLevel.READ_ONLY, [], "invalid_arguments"),
        ("cap_test", "observe", PermissionLevel.READ_ONLY, {1: "value"}, "invalid_arguments"),
        (
            "cap_test", "observe", PermissionLevel.READ_ONLY,
            {"value": object()}, "invalid_arguments",
        ),
    ],
)
async def test_policy_admission_rejects_invalid_boundary_shapes(
    tmp_path, provider, remote, level, arguments, code,
):
    controller = ExecutionPolicyController(str(tmp_path))
    denied = await controller.admit(provider, remote, level, arguments)
    assert denied.policy_code == code
    completion = await denied.finish(False, {})
    assert completion.allowed is False


@pytest.mark.asyncio
async def test_policy_rejects_non_json_oversize_secret_and_workspace_escape(tmp_path):
    controller = ExecutionPolicyController(
        str(tmp_path),
        ExecutionPolicy(limits=ExecutionLimits(max_argument_bytes=64)),
    )
    invalid = await controller.admit(
        "cap_test_observe", "observe", PermissionLevel.READ_ONLY, {"value": float("nan")},
    )
    assert invalid.policy_code == "invalid_arguments"
    oversized = await controller.admit(
        "cap_test_observe", "observe", PermissionLevel.READ_ONLY, {"value": "x" * 100},
    )
    assert oversized.policy_code == "argument_budget"

    ordinary = ExecutionPolicyController(str(tmp_path))
    secret = await ordinary.admit(
        "cap_test_observe", "observe", PermissionLevel.READ_ONLY,
        {"nested": {"api_key": "never-persist"}},
    )
    assert secret.policy_code == "secret_argument"
    assert "never-persist" not in str(secret.denial_payload())
    escape = await ordinary.admit(
        "cap_test_observe", "observe", PermissionLevel.READ_ONLY,
        {"output_path": "../escape.txt"},
    )
    assert escape.policy_code == "sandbox_path"


@pytest.mark.asyncio
async def test_policy_bounds_argument_and_result_structure_without_recursion(tmp_path):
    controller = ExecutionPolicyController(
        str(tmp_path),
        ExecutionPolicy(limits=ExecutionLimits(
            max_argument_depth=2,
            max_argument_nodes=5,
            max_result_depth=2,
            max_result_nodes=5,
        )),
    )
    deep = await controller.admit(
        "cap_test_observe", "observe", PermissionLevel.READ_ONLY,
        {"a": {"b": {"c": 1}}},
    )
    assert deep.policy_code == "invalid_arguments"
    cyclic = {}
    cyclic["self"] = cyclic
    rejected = await controller.admit(
        "cap_test_observe", "observe", PermissionLevel.READ_ONLY, cyclic,
    )
    assert rejected.policy_code == "invalid_arguments"

    admission = await controller.admit(
        "cap_test_observe", "observe", PermissionLevel.READ_ONLY, {},
    )
    completion = await admission.finish(True, {"a": {"b": {"c": 1}}})
    assert completion.allowed is False
    assert "depth budget" in completion.error


@pytest.mark.asyncio
async def test_policy_allows_safe_usage_counters_and_workspace_paths(tmp_path):
    controller = ExecutionPolicyController(str(tmp_path))
    admission = await controller.admit(
        "cap_test_observe", "observe", PermissionLevel.READ_ONLY,
        {"token_budget": 100, "output_path": "artifacts/result.json"},
    )
    assert admission.allowed is True
    completion = await admission.finish(True, {"ok": True})
    assert completion.allowed is True

    domain_neutral = await ExecutionPolicyController(str(tmp_path)).admit(
        "cap_robot_plan", "plan", PermissionLevel.READ_ONLY,
        {"path": "/world/frame/route"},
    )
    assert domain_neutral.allowed is True
    await domain_neutral.finish(True, {"ok": True})

    configured = ExecutionPolicyController(
        str(tmp_path), ExecutionPolicy(workspace_path_keys=("artifact",)),
    )
    escaped = await configured.admit(
        "cap_test_observe", "observe", PermissionLevel.READ_ONLY,
        {"artifact": "../outside.json"},
    )
    assert escaped.policy_code == "sandbox_path"

    flexible = ExecutionPolicyController(
        str(tmp_path),
        ExecutionPolicy(
            allowed_sensitive_keys=("api_key",),
            allow_outside_workspace_paths=True,
        ),
    )
    admitted = await flexible.admit(
        "cap_test", "observe", PermissionLevel.READ_ONLY,
        {"api_key": "host-injected-fixture", "output_path": "/tmp/result"},
    )
    assert admitted.allowed is True
    invalid_completion = await admitted.finish("yes", {"opaque": object()})
    assert invalid_completion.allowed is False
    assert "completion status" in invalid_completion.error


@pytest.mark.asyncio
async def test_dangerous_call_requires_real_approval_when_configured(tmp_path):
    requests = []

    async def approve(request):
        requests.append(request)
        return ApprovalDecision(True, approver_ref="operator:fixture")

    policy = ExecutionPolicy(
        approval_level=PermissionLevel.DANGER_FULL,
        reject_sensitive_arguments=False,
    )
    unavailable = ExecutionPolicyController(str(tmp_path), policy)
    denied = await unavailable.admit(
        "cap_lab_probe", "probe", PermissionLevel.DANGER_FULL, {},
    )
    assert denied.policy_code == "approval_denied"

    controller = ExecutionPolicyController(
        str(tmp_path), policy, approval_resolver=approve,
    )
    admission = await controller.admit(
        "cap_lab_probe", "probe", PermissionLevel.DANGER_FULL,
        {"authorization": "sensitive-value", "scope": "fixture-only"},
    )
    assert admission.allowed is True
    assert admission.approval.approver_ref == "operator:fixture"
    assert requests[0].arguments["authorization"] == "***"
    await admission.finish(True, {"ok": True})


@pytest.mark.asyncio
async def test_invalid_or_failed_approval_resolver_fails_closed_without_error_leak(tmp_path):
    policy = ExecutionPolicy(approval_level=PermissionLevel.READ_ONLY)

    async def invalid(_request):
        return True

    invalid_controller = ExecutionPolicyController(
        str(tmp_path), policy, approval_resolver=invalid,
    )
    denied = await invalid_controller.admit(
        "cap_test_observe", "observe", PermissionLevel.READ_ONLY, {},
    )
    assert denied.allowed is False
    assert "invalid decision" in denied.error

    async def failed(_request):
        raise RuntimeError("internal-secret-detail")

    failed_controller = ExecutionPolicyController(
        str(tmp_path), policy, approval_resolver=failed,
    )
    denied = await failed_controller.admit(
        "cap_test_observe", "observe", PermissionLevel.READ_ONLY, {},
    )
    assert denied.error == "approval resolver failed"

    async def slow(_request):
        await asyncio.sleep(2)
        return ApprovalDecision(True)

    timeout_policy = ExecutionPolicy(
        limits=ExecutionLimits(approval_timeout_seconds=1),
        approval_level=PermissionLevel.READ_ONLY,
    )
    timeout_controller = ExecutionPolicyController(
        str(tmp_path), timeout_policy, approval_resolver=slow,
    )
    denied = await timeout_controller.admit(
        "cap_test_observe", "observe", PermissionLevel.READ_ONLY, {},
    )
    assert denied.error == "approval resolver timed out"


@pytest.mark.asyncio
async def test_call_failure_result_and_concurrency_budgets_are_exact(tmp_path):
    call_controller = ExecutionPolicyController(
        str(tmp_path), ExecutionPolicy(limits=ExecutionLimits(max_calls=1)),
    )
    first = await call_controller.admit(
        "cap_test_run", "run", PermissionLevel.WORKSPACE_WRITE, {},
    )
    await first.finish(True, {"ok": True})
    exhausted = await call_controller.admit(
        "cap_test_run", "run", PermissionLevel.WORKSPACE_WRITE, {},
    )
    assert exhausted.policy_code == "call_budget"

    failure_controller = ExecutionPolicyController(
        str(tmp_path), ExecutionPolicy(limits=ExecutionLimits(max_failures=1)),
    )
    failed = await failure_controller.admit(
        "cap_test_run", "run", PermissionLevel.WORKSPACE_WRITE, {},
    )
    await failed.finish(False, {"ok": False})
    exhausted = await failure_controller.admit(
        "cap_test_run", "run", PermissionLevel.WORKSPACE_WRITE, {},
    )
    assert exhausted.policy_code == "failure_budget"

    result_controller = ExecutionPolicyController(
        str(tmp_path), ExecutionPolicy(limits=ExecutionLimits(max_result_bytes=16)),
    )
    admitted = await result_controller.admit(
        "cap_test_run", "run", PermissionLevel.WORKSPACE_WRITE, {},
    )
    completion = await admitted.finish(True, {"value": "x" * 100})
    assert completion.allowed is False
    assert "result exceeds" in completion.error
    with pytest.raises(RuntimeError, match="already finished"):
        await admitted.finish(True, {})

    elapsed_controller = ExecutionPolicyController(
        str(tmp_path),
        ExecutionPolicy(limits=ExecutionLimits(max_elapsed_seconds=1)),
    )
    elapsed_controller._started_at -= 2
    exhausted = await elapsed_controller.admit(
        "cap_test_run", "run", PermissionLevel.WORKSPACE_WRITE, {},
    )
    assert exhausted.policy_code == "elapsed_budget"


@pytest.mark.asyncio
async def test_concurrency_lease_blocks_until_previous_call_finishes(tmp_path):
    controller = ExecutionPolicyController(
        str(tmp_path), ExecutionPolicy(limits=ExecutionLimits(max_concurrency=1)),
    )
    first = await controller.admit(
        "cap_test_run", "run", PermissionLevel.WORKSPACE_WRITE, {},
    )
    second_task = asyncio.create_task(controller.admit(
        "cap_test_run", "run", PermissionLevel.WORKSPACE_WRITE, {},
    ))
    await asyncio.sleep(0.02)
    assert second_task.done() is False
    await first.finish(True, {"ok": True})
    second = await second_task
    assert second.allowed is True
    await second.finish(True, {"ok": True})
    snapshot = await controller.snapshot()
    assert snapshot["calls"] == 2
    assert snapshot["active"] == 0
