# Copyright 2024 Flyto2
# Licensed under the Apache License, Version 2.0
"""Build bounded Blueprint execution-verification receipts.

The SHA-256 digest is tamper evidence for the nested canonical evidence only.
It is not a signature, an identity claim, a sensor reading, or physical
attestation.  Callers remain responsible for observing and validating the
execution before constructing a receipt.
"""
from __future__ import annotations

import hashlib
import json
import re
from typing import Any, Dict, Mapping, Optional, Sequence


EXECUTION_VERIFICATION_RECEIPT_VERSION = "flyto.execution-verification-receipt.v1"
_EVIDENCE_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/-]{0,191}\Z")
_SAFE_IDENTIFIER_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/-]{0,191}\Z")
_EXECUTION_ID_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_DIGEST_RE = re.compile(r"sha256:[0-9a-f]{64}\Z")
_MAX_EVIDENCE_BYTES = 32_768
_MAX_INTEGER = (1 << 53) - 1
_MAX_COLLECTION_ITEMS = 256
_TOP_LEVEL_FIELDS = frozenset({
    "modules", "steps", "checks", "counts", "structural_digest",
    "outcome_success",
})
_STEP_IDENTIFIER_FIELDS = frozenset({
    "step_id", "module_id", "source_module_id", "source_step_id",
})
_STEP_RELATIONSHIP_FIELDS = frozenset({"depends_on"})
_STEP_BOOLEAN_FIELDS = frozenset({"validation_ok", "assertions_ok", "executed"})
_STEP_COUNT_FIELDS = frozenset({"assertion_count", "attempt_count"})
_STEP_FIELDS = (
    _STEP_IDENTIFIER_FIELDS | _STEP_RELATIONSHIP_FIELDS
    | _STEP_BOOLEAN_FIELDS | _STEP_COUNT_FIELDS
)
_RAW_ALIASES = frozenset({
    "api_key", "arguments", "cookie", "error", "exception", "message",
    "params", "password", "payload", "prompt", "raw", "raw_params",
    "raw_result", "result", "selector", "secret", "text", "token", "url",
    "user_text",
})


def _canonical_evidence(evidence: Mapping[str, Any]) -> tuple[Dict[str, Any], bytes]:
    if type(evidence) is not dict:
        raise ValueError("verification evidence must be an exact JSON object")
    unknown = set(evidence) - _TOP_LEVEL_FIELDS
    if unknown or any(type(key) is not str for key in evidence):
        raise ValueError("verification evidence fields do not match v1 grammar")
    digest = evidence.get("structural_digest")
    if type(digest) is not str or not _DIGEST_RE.fullmatch(digest):
        raise ValueError("verification structural digest is malformed")
    _validate_identifiers(evidence.get("modules", []), "modules")
    _validate_steps(evidence.get("steps", []))
    _validate_scalar_map(evidence.get("checks", {}), "checks", bool)
    _validate_scalar_map(evidence.get("counts", {}), "counts", int)
    if "outcome_success" in evidence and type(evidence["outcome_success"]) is not bool:
        raise ValueError("verification outcome_success must be an exact boolean")
    encoded = json.dumps(
        evidence, ensure_ascii=False, allow_nan=False,
        sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    if len(encoded) > _MAX_EVIDENCE_BYTES:
        raise ValueError("verification evidence exceeds v1 byte limit")
    return json.loads(encoded), encoded


def _safe_identifier(value: Any) -> bool:
    return (
        type(value) is str and _SAFE_IDENTIFIER_RE.fullmatch(value) is not None
        and ".." not in value and "//" not in value
    )


def _safe_structural_key(value: Any) -> bool:
    return _safe_identifier(value) and value.casefold() not in _RAW_ALIASES


def _bounded_collection(value: Any, label: str, expected_type: type) -> None:
    if type(value) is not expected_type or len(value) > _MAX_COLLECTION_ITEMS:
        raise ValueError("verification {} must be a bounded {}".format(
            label, expected_type.__name__,
        ))


def _validate_identifiers(value: Any, label: str) -> None:
    _bounded_collection(value, label, list)
    if any(not _safe_identifier(item) for item in value) or len(set(value)) != len(value):
        raise ValueError("verification {} contains malformed or duplicate identifiers".format(label))


def _validate_scalar_map(value: Any, label: str, value_type: type) -> None:
    _bounded_collection(value, label, dict)
    for key, item in value.items():
        if not _safe_structural_key(key):
            raise ValueError("verification {} key is malformed".format(label))
        if value_type is bool and type(item) is not bool:
            raise ValueError("verification checks require exact booleans")
        if value_type is int and (
            type(item) is not int or not 0 <= item <= _MAX_INTEGER
        ):
            raise ValueError("verification counts require non-negative exact integers")


def _validate_steps(value: Any) -> None:
    _bounded_collection(value, "steps", list)
    step_ids = []
    relationships = {}
    for step in value:
        if type(step) is not dict or set(step) - _STEP_FIELDS:
            raise ValueError("verification step fields do not match v1 grammar")
        if "step_id" not in step or "module_id" not in step:
            raise ValueError("verification steps require step_id and module_id")
        for key in _STEP_IDENTIFIER_FIELDS & set(step):
            if not _safe_identifier(step[key]):
                raise ValueError("verification step identifier is malformed")
        for key in _STEP_BOOLEAN_FIELDS & set(step):
            if type(step[key]) is not bool:
                raise ValueError("verification step booleans must be exact")
        for key in _STEP_COUNT_FIELDS & set(step):
            if type(step[key]) is not int or not 0 <= step[key] <= _MAX_INTEGER:
                raise ValueError("verification step counts must be non-negative integers")
        for key in _STEP_RELATIONSHIP_FIELDS & set(step):
            _validate_identifiers(step[key], "step relationship")
        step_ids.append(step["step_id"])
        relationships[step["step_id"]] = step.get("depends_on", [])
    if len(set(step_ids)) != len(step_ids):
        raise ValueError("verification steps contain duplicate identifiers")
    declared = set(step_ids)
    for step_id, dependencies in relationships.items():
        if step_id in dependencies:
            raise ValueError("verification step cannot depend on itself")
        if any(dependency not in declared for dependency in dependencies):
            raise ValueError("verification step relationship is missing a declared step")
    _reject_relationship_cycles(relationships)


def _reject_relationship_cycles(relationships: Mapping[str, Sequence[str]]) -> None:
    visiting = set()
    visited = set()

    def visit(step_id: str) -> None:
        if step_id in visiting:
            raise ValueError("verification step relationships contain a cycle")
        if step_id in visited:
            return
        visiting.add(step_id)
        for dependency in relationships[step_id]:
            visit(dependency)
        visiting.remove(step_id)
        visited.add(step_id)

    for step_id in relationships:
        visit(step_id)


def learn_verified_blueprint(
    engine: Any,
    workflow: Mapping[str, Any],
    receipt: Mapping[str, Any],
    plan_ir_version: str,
) -> tuple[Any, Optional[Dict[str, Any]]]:
    """Call Blueprint and accept only an ok response with a safe identity."""
    learned = engine.learn_from_execution(
        workflow,
        name=workflow.get("name") or "Verified MCP workflow",
        tags=["mcp", "verified", plan_ir_version],
        verification=receipt,
    )
    if type(learned) is not dict or learned.get("ok") is not True:
        return learned, None
    data = learned.get("data")
    blueprint_id = data.get("id") if type(data) is dict else learned.get("blueprint_id")
    if not _safe_identifier(blueprint_id):
        return learned, None
    identity = {"blueprint_id": blueprint_id}
    if type(data) is dict and type(data.get("score")) is int:
        identity["score"] = data["score"]
    return learned, identity


def attempt_verified_blueprint_learning(
    engine: Any,
    workflow: Mapping[str, Any],
    execution_evidence: Mapping[str, Any],
    checks: Mapping[str, bool],
    evidence_count: int,
    plan_ir_version: str,
) -> tuple[Any, Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """Build the receipt first, then call Blueprint only across a valid boundary."""
    result = execution_evidence.get("result", {})
    runtime = result.get("evidence", {})
    receipt = try_build_closed_loop_verification_receipt(
        result.get("execution_id"),
        result.get("executions", []), checks, evidence_count,
        str(runtime.get("workflow_hash") or ""),
    )
    if receipt is None:
        return None, None, None
    learned, identity = learn_verified_blueprint(
        engine, workflow, receipt, plan_ir_version,
    )
    return learned, identity, receipt


def verified_learning_distillation(
    engine: Any,
    workflow: Mapping[str, Any],
    execution_evidence: Mapping[str, Any],
    checks: Mapping[str, bool],
    evidence_count: int,
    plan_ir_version: str,
) -> tuple[bool, Dict[str, Any]]:
    """Return the complete fail-closed distillation state for one learning call."""
    learned, identity, receipt = attempt_verified_blueprint_learning(
        engine, workflow, execution_evidence, checks, evidence_count,
        plan_ir_version,
    )
    if receipt is None:
        return False, {
            "eligible": False,
            "reason": "execution verification receipt rejected",
        }
    state = {"result": learned, "verification_receipt": receipt}
    if identity is None:
        state.update(
            eligible=False,
            reason="Blueprint rejected verified learning",
        )
        return False, state
    state.update(identity)
    return True, state


def build_execution_verification_receipt(
    evidence_id: str,
    evidence: Mapping[str, Any],
    *,
    outcome_success: Optional[bool] = None,
) -> Dict[str, Any]:
    """Return the exact six-field v1 receipt for observed safe evidence.

    ``outcome_success`` must be the exact observed result when this receipt is
    used with ``report_outcome``; it is inserted before canonicalization and
    therefore covered by the evidence digest.
    """
    if (
        type(evidence_id) is not str
        or not _EVIDENCE_ID_RE.fullmatch(evidence_id)
        or ".." in evidence_id
        or "//" in evidence_id
    ):
        raise ValueError("verification evidence_id is malformed")
    bounded = dict(evidence)
    if outcome_success is not None:
        if type(outcome_success) is not bool:
            raise TypeError("outcome_success must be an exact boolean")
        if "outcome_success" in bounded and bounded["outcome_success"] is not outcome_success:
            raise ValueError("outcome_success does not match evidence")
        bounded["outcome_success"] = outcome_success
    canonical, encoded = _canonical_evidence(bounded)
    return {
        "receipt_version": EXECUTION_VERIFICATION_RECEIPT_VERSION,
        "success": True,
        "status": "verified",
        "evidence_id": evidence_id,
        "evidence_sha256": hashlib.sha256(encoded).hexdigest(),
        "evidence": canonical,
    }


def build_closed_loop_verification_receipt(
    execution_id: str,
    executions: Sequence[Mapping[str, Any]],
    checks: Mapping[str, bool],
    evidence_count: int,
    structural_digest: str,
) -> Dict[str, Any]:
    """Project full ClosedLoop records into safe structural receipt evidence."""
    if type(execution_id) is not str or not _EXECUTION_ID_RE.fullmatch(execution_id):
        raise ValueError("closed-loop execution_id is malformed")
    steps = []
    for index, item in enumerate(executions, start=1):
        validation = item.get("validation")
        assertions = item.get("assertions") or []
        steps.append({
            "step_id": str(item.get("step_id") or "step_{}".format(index)),
            "module_id": str(item.get("module_id") or ""),
            "validation_ok": bool(isinstance(validation, dict) and
                                  (validation.get("valid") is True or validation.get("ok") is True)),
            "assertions_ok": all(type(assertion) is dict and assertion.get("ok") is True
                                 for assertion in assertions),
            "executed": item.get("executed") is True,
            "assertion_count": sum(1 for assertion in assertions
                                   if isinstance(assertion, dict) and assertion.get("ok") is True),
        })
    return build_execution_verification_receipt(
        "closed-loop:{}".format(execution_id),
        {
            "checks": dict(checks),
            "counts": {"executed_steps": len(executions), "evidence_items": evidence_count},
            "steps": steps,
            "structural_digest": structural_digest,
        }, outcome_success=checks.get("outcome_success"),
    )


def try_build_closed_loop_verification_receipt(
    execution_id: str,
    executions: Sequence[Mapping[str, Any]],
    checks: Mapping[str, bool],
    evidence_count: int,
    structural_digest: str,
) -> Optional[Dict[str, Any]]:
    """Return a receipt or ``None`` so callers can fail closed at the boundary."""
    try:
        return build_closed_loop_verification_receipt(
            execution_id, executions, checks, evidence_count, structural_digest,
        )
    except (TypeError, ValueError):
        return None


def build_benchmark_verification_receipt(
    evidence_id: str,
    structural_digest: str,
    step_count: int,
    *,
    modules: Optional[Sequence[str]] = None,
    outcome_success: Optional[bool] = None,
) -> Dict[str, Any]:
    """Build the shared verified-fixture or warm-outcome receipt projection."""
    evidence: Dict[str, Any] = {
        "checks": {"fixture_valid": True},
        "counts": {"step_count": step_count},
        "structural_digest": structural_digest,
    }
    if modules is not None:
        evidence["modules"] = list(modules)
    return build_execution_verification_receipt(
        evidence_id, evidence, outcome_success=outcome_success,
    )


def build_benchmark_workflow_receipt(
    evidence_id: str,
    workflow: Mapping[str, Any],
    *,
    outcome_success: Optional[bool] = None,
) -> Dict[str, Any]:
    """Build a fixture receipt directly from one structural workflow projection."""
    encoded = json.dumps(
        workflow, ensure_ascii=False, allow_nan=False,
        sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    digest = "sha256:" + hashlib.sha256(encoded).hexdigest()
    modules = None if outcome_success is not None else [
        step["module"] for step in workflow["steps"]
    ]
    return build_benchmark_verification_receipt(
        evidence_id, digest, len(workflow["steps"]), modules=modules,
        outcome_success=outcome_success,
    )
