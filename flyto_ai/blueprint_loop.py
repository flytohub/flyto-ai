# Copyright 2024 Flyto2
# Licensed under the Apache License, Version 2.0
"""Deterministic, permission-checked execution for expanded blueprints."""
from __future__ import annotations

import asyncio
import copy
import hashlib
import json
import re
import time
import uuid
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

from flyto_ai.closed_loop_v3 import (
    CHECKPOINT_VERSION,
    CheckpointStore,
    PlanIR,
    RepairDecision,
    RepairFn,
    checkpoint_key,
    repair_from_result,
)
from flyto_ai.execution_verification import try_build_closed_loop_verification_receipt
from flyto_ai.redaction import redact_args
from flyto_ai.tools.blueprint_tools import _CLOSED_LOOP_EVIDENCE_CAPABILITY
DispatchFn = Callable[[str, Dict[str, Any]], Awaitable[Dict[str, Any]]]
PreflightFn = Callable[[Dict[str, Any]], Awaitable[Dict[str, Any]]]
_PREVIEW_LIMIT = 500
_EXECUTOR_VERSION = "blueprint-loop.v3"
_MAX_RETRIES = 3
_MAX_RETRY_DELAY_MS = 5000
_REF_RE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_-]*)(?=[.\[])")
_EXPR_RE = re.compile(r"\$\{([^{}]+)\}")
_PATH_TOKEN_RE = re.compile(r"([^. \[\]]+)|\[(\d+)\]")
_ASSERTION_OPS = {
    "contains",
    "equals",
    "exists",
    "falsy",
    "not_equals",
    "truthy",
}
_MISSING = object()


def _is_ok(result: Any) -> bool:
    """Normalize flyto-core's supported success result shapes."""
    if not isinstance(result, dict):
        return False
    if "ok" in result:
        return bool(result["ok"])
    return result.get("status") == "success"


def _preview(value: Any) -> str:
    """Build a bounded, redacted preview suitable for audit logs."""
    safe = redact_args(value)
    return json.dumps(safe, ensure_ascii=False, default=str)[:_PREVIEW_LIMIT]


def _validation_ok(result: Any) -> bool:
    """Normalize validate_params results without treating missing ``ok`` as failure."""
    if not isinstance(result, dict):
        return False
    if result.get("valid") is False or result.get("ok") is False:
        return False
    return result.get("valid") is True or result.get("ok") is True


async def _call(
    dispatch: DispatchFn,
    name: str,
    arguments: Dict[str, Any],
) -> Dict[str, Any]:
    """Convert dispatcher exceptions into structured failures."""
    try:
        result = await dispatch(name, arguments)
    except Exception as exc:
        return {
            "ok": False,
            "error": str(exc) or type(exc).__name__,
            "exception_type": type(exc).__name__,
        }
    if isinstance(result, dict):
        return result
    return {"ok": False, "error": "Dispatcher returned a non-object result"}


def _workflow_hash(steps: List[Dict[str, Any]]) -> str:
    """Return a stable content hash without exposing workflow parameters."""
    canonical = json.dumps(
        steps,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return "sha256:{}".format(hashlib.sha256(canonical.encode("utf-8")).hexdigest())


def _collect_references(value: Any) -> List[str]:
    """Collect Core-style ``${step_id.field}`` references recursively."""
    references: List[str] = []
    if isinstance(value, str):
        references.extend(match.group(1) for match in _REF_RE.finditer(value))
    elif isinstance(value, dict):
        for nested in value.values():
            references.extend(_collect_references(nested))
    elif isinstance(value, list):
        for nested in value:
            references.extend(_collect_references(nested))
    return references


def _parse_path(path: str) -> List[Any]:
    """Parse dotted keys and numeric indexes into lookup tokens."""
    tokens: List[Any] = []
    for key, index in _PATH_TOKEN_RE.findall(path):
        tokens.append(int(index) if index else key)
    return tokens


def _get_path(value: Any, path: str) -> Tuple[bool, Any]:
    """Read a nested value without conflating missing keys with ``None``."""
    current = value
    if not path:
        return True, current
    for token in _parse_path(path):
        if isinstance(token, int):
            if not isinstance(current, list) or token >= len(current):
                return False, None
            current = current[token]
        elif isinstance(current, dict) and token in current:
            current = current[token]
        else:
            return False, None
    return True, current


def _lookup_step_value(context: Dict[str, Any], expression: str) -> Any:
    """Resolve one Core-compatible step expression for the fallback resolver."""
    tokens = _parse_path(expression)
    if not tokens:
        return _MISSING
    step_id = tokens.pop(0)
    if not isinstance(step_id, str) or step_id not in context:
        return _MISSING

    output = context[step_id]
    path = ".".join(str(token) for token in tokens)
    found, value = _get_path(output, path)
    if found:
        return value

    # Core's resolver treats ``${step.result}`` as shorthand for
    # ``${step.data.result}``.
    if isinstance(output, dict) and "data" in output:
        found, value = _get_path(output["data"], path)
        if found:
            return value
    return _MISSING


def _fallback_resolve(value: Any, context: Dict[str, Any]) -> Any:
    """Resolve Core-style expressions when flyto-core is not importable."""
    if isinstance(value, dict):
        return {key: _fallback_resolve(item, context) for key, item in value.items()}
    if isinstance(value, list):
        return [_fallback_resolve(item, context) for item in value]
    if not isinstance(value, str):
        return value

    exact = _EXPR_RE.fullmatch(value)
    if exact:
        resolved = _lookup_step_value(context, exact.group(1))
        return value if resolved is _MISSING else resolved

    def replace(match: re.Match) -> str:
        resolved = _lookup_step_value(context, match.group(1))
        return match.group(0) if resolved is _MISSING else str(resolved)

    return _EXPR_RE.sub(replace, value)


def _resolve_params(params: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    """Resolve prior-step outputs with flyto-core's canonical resolver."""
    if not _collect_references(params):
        return copy.deepcopy(params)
    try:
        from core.engine.variable_resolver import VariableResolver

        resolved = VariableResolver({}, context).resolve(copy.deepcopy(params))
    except ImportError:
        resolved = _fallback_resolve(copy.deepcopy(params), context)

    unresolved = _collect_references(resolved)
    if unresolved:
        raise ValueError(
            "Unresolved step output reference(s): {}".format(
                ", ".join(sorted(set(unresolved))),
            ),
        )
    if not isinstance(resolved, dict):
        raise ValueError("Resolved step params must be an object")
    return resolved


def _normalize_retry(raw: Any) -> Dict[str, Any]:
    """Validate Core-compatible retry configuration with safe upper bounds."""
    if raw in (None, False):
        return {"count": 0, "delay_ms": 0, "backoff": "constant"}
    if isinstance(raw, bool):
        raise ValueError("retry must be an integer or object")
    if isinstance(raw, int):
        raw = {"count": raw}
    if not isinstance(raw, dict):
        raise ValueError("retry must be an integer or object")

    count = raw.get("count", 0)
    delay_ms = raw.get("delay_ms", 0)
    backoff = raw.get("backoff", "constant")
    if not isinstance(count, int) or not 0 <= count <= _MAX_RETRIES:
        raise ValueError("retry.count must be between 0 and {}".format(_MAX_RETRIES))
    if not isinstance(delay_ms, int) or not 0 <= delay_ms <= _MAX_RETRY_DELAY_MS:
        raise ValueError(
            "retry.delay_ms must be between 0 and {}".format(_MAX_RETRY_DELAY_MS),
        )
    if backoff not in {"constant", "linear", "exponential"}:
        raise ValueError("retry.backoff must be constant, linear, or exponential")
    return {"count": count, "delay_ms": delay_ms, "backoff": backoff}


def _retry_wait_seconds(retry: Dict[str, Any], retry_index: int) -> float:
    """Calculate bounded retry delay using flyto-core-compatible strategies."""
    delay_ms = retry["delay_ms"]
    if retry["backoff"] == "linear":
        delay_ms *= retry_index + 1
    elif retry["backoff"] == "exponential":
        delay_ms *= 2 ** retry_index
    return min(delay_ms, _MAX_RETRY_DELAY_MS) / 1000


def _normalize_assertions(step: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Validate declarative output assertions before execution."""
    raw = step.get("assertions", step.get("assert"))
    if raw is None:
        return []
    if isinstance(raw, dict):
        raw = [raw]
    if not isinstance(raw, list) or not all(isinstance(item, dict) for item in raw):
        raise ValueError("assertions must be an object or list of objects")

    assertions = []
    for item in raw:
        operation = item.get("op", "equals")
        path = item.get("path", "")
        if operation not in _ASSERTION_OPS:
            raise ValueError("Unsupported assertion operation: {}".format(operation))
        if not isinstance(path, str):
            raise ValueError("assertion.path must be a string")
        assertions.append({
            "path": path,
            "op": operation,
            "expected": item.get("value"),
        })
    return assertions


def _evaluate_assertions(
    result: Dict[str, Any],
    assertions: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Evaluate output assertions and return redacted evidence."""
    evidence = []
    for assertion in assertions:
        found, actual = _get_path(result, assertion["path"])
        operation = assertion["op"]
        expected = assertion["expected"]
        if operation == "exists":
            ok = found
        elif operation == "truthy":
            ok = found and bool(actual)
        elif operation == "falsy":
            ok = found and not bool(actual)
        elif operation == "equals":
            ok = found and actual == expected
        elif operation == "not_equals":
            ok = found and actual != expected
        else:  # contains
            try:
                ok = found and expected in actual
            except TypeError:
                ok = False
        evidence.append({
            "path": assertion["path"],
            "op": operation,
            "ok": ok,
            "expected": redact_args(expected),
            "actual_preview": _preview(actual) if found else "<missing>",
        })
    return evidence


def _prepare_steps(
    steps: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Perform side-effect-free structural preflight for the whole workflow."""
    prepared = []
    errors = []
    seen_ids = set()

    if not isinstance(steps, list) or not steps:
        return [], [{"step_id": None, "error": "Blueprint has no executable steps"}]

    for index, raw_step in enumerate(steps, start=1):
        if not isinstance(raw_step, dict):
            errors.append({
                "step_id": "step_{}".format(index),
                "error": "Blueprint step must be an object",
            })
            continue

        step = copy.deepcopy(raw_step)
        step_id = str(step.get("id") or "step_{}".format(index))
        module_id = str(step.get("module") or step.get("module_id") or "")
        params = step.get("params") or {}
        step_errors = []
        if step_id in seen_ids:
            step_errors.append("Duplicate step id: {}".format(step_id))
        if not module_id:
            step_errors.append("Step module is required")
        if not isinstance(params, dict):
            step_errors.append("Step params must be an object")
            params = {}

        references = _collect_references(params)
        invalid_references = sorted({ref for ref in references if ref not in seen_ids})
        if invalid_references:
            step_errors.append(
                "Missing or forward step reference(s): {}".format(
                    ", ".join(invalid_references),
                ),
            )

        try:
            retry = _normalize_retry(step.get("retry"))
        except ValueError as exc:
            retry = {"count": 0, "delay_ms": 0, "backoff": "constant"}
            step_errors.append(str(exc))
        try:
            assertions = _normalize_assertions(step)
        except ValueError as exc:
            assertions = []
            step_errors.append(str(exc))

        prepared.append({
            "index": index,
            "step_id": step_id,
            "module_id": module_id,
            "params": params,
            "references": references,
            "retry": retry,
            "assertions": assertions,
            "validation": None,
        })
        errors.extend({"step_id": step_id, "error": error} for error in step_errors)
        seen_ids.add(step_id)

    return prepared, errors


async def execute_blueprint_loop(
    blueprint_id: str,
    steps: List[Dict[str, Any]],
    dispatch: DispatchFn,
    preflight: Optional[PreflightFn] = None,
    *,
    checkpoint_store: Optional[CheckpointStore] = None,
    repair: Optional[RepairFn] = None,
    max_repairs: int = 1,
    selection_mode: str = "model_selected",
) -> Dict[str, Any]:
    """Validate and execute blueprint steps, then report one idempotent outcome.

    ``dispatch`` must be the agent's safe dispatcher. This keeps nested module
    calls behind the same permission, policy, extension-hook, and middleware
    boundaries as ordinary LLM-requested tool calls.

    ``preflight`` is an optional side-effect-free Agent access check. When
    provided, every module must pass permission and policy checks before the
    first module executes.
    """
    started = time.monotonic()
    selection_mode = (
        "deterministic" if selection_mode == "deterministic" else "model_selected"
    )
    execution_id = "bp_{}".format(uuid.uuid4().hex)
    plan_ir = PlanIR.compile(blueprint_id, steps)
    steps = plan_ir.to_steps()
    workflow_hash = plan_ir.workflow_hash
    executions: List[Dict[str, Any]] = []
    results: List[Dict[str, Any]] = []
    prepared, prepare_errors = _prepare_steps(steps)
    preflight_errors = plan_ir.gate()
    for error in prepare_errors:
        if error not in preflight_errors:
            preflight_errors.append(error)
    deferred_validation_steps = [
        item["step_id"] for item in prepared if item["references"]
    ]
    validation_passed = True
    failed_module = ""
    failed_step_id = ""
    failed_phase = ""
    completed_step_ids: List[str] = []
    context: Dict[str, Any] = {}
    total_attempts = 0
    assertion_passed = True
    max_repairs = max(0, min(int(max_repairs), 3))
    repair_trace: List[Dict[str, Any]] = []
    checkpoint_id = checkpoint_key(blueprint_id, workflow_hash)
    checkpoint_loaded = False
    checkpoint_cleared = False
    checkpoint_write_count = 0
    checkpoint_source_execution_id = ""
    resumed_step_ids: List[str] = []
    checkpoint_status = ""
    checkpoint_error = ""

    if not preflight_errors and checkpoint_store is not None:
        try:
            checkpoint = checkpoint_store.load(checkpoint_id)
        except Exception as exc:
            checkpoint = None
            preflight_errors.append({
                "step_id": None,
                "module_id": "<checkpoint>",
                "error": "Checkpoint load failed: {}".format(exc),
            })
        if isinstance(checkpoint, dict):
            valid_checkpoint = (
                checkpoint.get("version") == CHECKPOINT_VERSION
                and checkpoint.get("blueprint_id") == blueprint_id
                and checkpoint.get("workflow_hash") == workflow_hash
                and checkpoint.get("status") in {
                    "partial",
                    "failed",
                    "awaiting_outcome",
                }
                and isinstance(checkpoint.get("context"), dict)
                and isinstance(checkpoint.get("completed_step_ids"), list)
            )
            if not valid_checkpoint:
                preflight_errors.append({
                    "step_id": None,
                    "module_id": "<checkpoint>",
                    "error": "Checkpoint contract mismatch",
                })
            else:
                prepared_ids = [item["step_id"] for item in prepared]
                candidate_ids = checkpoint["completed_step_ids"]
                prefix = prepared_ids[:len(candidate_ids)]
                if candidate_ids != prefix:
                    preflight_errors.append({
                        "step_id": None,
                        "module_id": "<checkpoint>",
                        "error": "Checkpoint completed steps are not a plan prefix",
                    })
                else:
                    checkpoint_loaded = True
                    checkpoint_status = str(checkpoint.get("status") or "")
                    checkpoint_source_execution_id = str(
                        checkpoint.get("execution_id") or "",
                    )
                    resumed_step_ids = list(candidate_ids)
                    completed_step_ids = list(candidate_ids)
                    context = copy.deepcopy(checkpoint["context"])
                    if (
                        checkpoint_status == "awaiting_outcome"
                        and checkpoint_source_execution_id
                    ):
                        execution_id = checkpoint_source_execution_id
    def save_checkpoint(status: str) -> Optional[str]:
        nonlocal checkpoint_write_count
        if checkpoint_store is None:
            return None
        payload = {
            "version": CHECKPOINT_VERSION,
            "status": status,
            "blueprint_id": blueprint_id,
            "workflow_hash": workflow_hash,
            "execution_id": execution_id,
            "completed_step_ids": completed_step_ids,
            "context": context,
            "failed_step_id": failed_step_id or None,
            "failed_phase": failed_phase or None,
            "updated_at": time.time(),
        }
        try:
            checkpoint_store.save(checkpoint_id, payload)
            checkpoint_write_count += 1
        except Exception as exc:
            return str(exc) or type(exc).__name__
        return None
    # Agent permission/policy preflight happens for the complete workflow
    # before parameter validation or any side effect.
    if not preflight_errors and preflight is not None:
        for item in prepared:
            arguments = {
                "module_id": item["module_id"],
                "params": item["params"],
            }
            try:
                access = await preflight(arguments)
            except Exception as exc:
                access = {
                    "ok": False,
                    "error": str(exc) or type(exc).__name__,
                }
            if not isinstance(access, dict) or not access.get("ok"):
                preflight_errors.append({
                    "step_id": item["step_id"],
                    "module_id": item["module_id"],
                    "error": (
                        access.get("error", "Module access denied")
                        if isinstance(access, dict)
                        else "Module access preflight returned a non-object result"
                    ),
                })
    # Validate every static step before the first execute call. Dynamic steps
    # are validated immediately after their references resolve.
    if not preflight_errors:
        for item in prepared:
            if item["references"]:
                continue
            arguments = {
                "module_id": item["module_id"],
                "params": item["params"],
            }
            validation = await _call(dispatch, "validate_params", arguments)
            item["validation"] = validation
            if not _validation_ok(validation):
                validation_passed = False
                preflight_errors.append({
                    "step_id": item["step_id"],
                    "module_id": item["module_id"],
                    "error": "Blueprint step validation failed",
                    "validation": validation,
                })
    if preflight_errors:
        first_error = preflight_errors[0]
        failed_step_id = str(first_error.get("step_id") or "")
        failed_module = str(first_error.get("module_id") or "<preflight>")
        failed_phase = "preflight"
        failure = {
            "ok": False,
            "error": first_error.get("error", "Blueprint preflight failed"),
            "preflight_errors": preflight_errors,
        }
        executions.append({
            "function": "execute_module",
            "arguments": {},
            "module_id": failed_module,
            "step_id": failed_step_id,
            "ok": False,
            "executed": False,
            "phase": "preflight",
            "result_preview": _preview(failure),
            "validation": redact_args(first_error.get("validation", {})),
            "blueprint_step": next(
                (
                    item["index"] for item in prepared
                    if item["step_id"] == failed_step_id
                ),
                0,
            ),
        })
        results.append({
            "module_id": failed_module,
            "step_id": failed_step_id,
            "ok": False,
            "executed": False,
            "result": failure,
        })
    else:
        resumed_set = set(resumed_step_ids)
        for item in prepared:
            module_id = item["module_id"]
            step_id = item["step_id"]
            if step_id in resumed_set:
                resumed_result = context.get(step_id, {})
                executions.append({
                    "function": "execute_module",
                    "arguments": {},
                    "module_id": module_id,
                    "step_id": step_id,
                    "ok": True,
                    "executed": False,
                    "phase": "resume",
                    "attempt_count": 0,
                    "attempts": [],
                    "assertions": [],
                    "result_preview": _preview(resumed_result),
                    "validation": {},
                    "blueprint_step": item["index"],
                })
                results.append({
                    "module_id": module_id,
                    "step_id": step_id,
                    "ok": True,
                    "executed": False,
                    "resumed": True,
                    "result": resumed_result,
                })
                continue
            try:
                params = _resolve_params(item["params"], context)
            except ValueError as exc:
                validation_passed = False
                failed_module = module_id
                failed_step_id = step_id
                failed_phase = "resolution"
                failure = {"ok": False, "error": str(exc)}
                executions.append({
                    "function": "execute_module",
                    "arguments": {},
                    "module_id": module_id,
                    "step_id": step_id,
                    "ok": False,
                    "executed": False,
                    "phase": "resolution",
                    "result_preview": _preview(failure),
                    "validation": {},
                    "blueprint_step": item["index"],
                })
                results.append({
                    "module_id": module_id,
                    "step_id": step_id,
                    "ok": False,
                    "executed": False,
                    "result": failure,
                })
                break
            arguments = {"module_id": module_id, "params": params}
            validation = item["validation"]
            if validation is None:
                validation = await _call(dispatch, "validate_params", arguments)
            if not _validation_ok(validation):
                validation_passed = False
                failed_module = module_id
                failed_step_id = step_id
                failed_phase = "validation"
                failure = {
                    "ok": False,
                    "error": "Blueprint step validation failed",
                    "validation": validation,
                }
                executions.append({
                    "function": "execute_module",
                    "arguments": redact_args(arguments),
                    "module_id": module_id,
                    "step_id": step_id,
                    "ok": False,
                    "executed": False,
                    "phase": "validation",
                    "result_preview": _preview(failure),
                    "validation": redact_args(validation),
                    "blueprint_step": item["index"],
                })
                results.append({
                    "module_id": module_id,
                    "step_id": step_id,
                    "ok": False,
                    "executed": False,
                    "validation": validation,
                    "result": failure,
                })
                break
            attempts = []
            final_result: Dict[str, Any] = {}
            assertion_evidence: List[Dict[str, Any]] = []
            ok = False
            for attempt in range(item["retry"]["count"] + 1):
                final_result = await _call(dispatch, "execute_module", arguments)
                total_attempts += 1
                assertion_evidence = (
                    _evaluate_assertions(final_result, item["assertions"])
                    if _is_ok(final_result)
                    else []
                )
                assertions_ok = all(
                    assertion["ok"] for assertion in assertion_evidence
                )
                ok = _is_ok(final_result) and assertions_ok
                attempts.append({
                    "attempt": attempt + 1,
                    "ok": ok,
                    "result_preview": _preview(final_result),
                    "assertions": assertion_evidence,
                })
                if ok:
                    break
                if attempt < item["retry"]["count"]:
                    wait_seconds = _retry_wait_seconds(item["retry"], attempt)
                    if wait_seconds:
                        await asyncio.sleep(wait_seconds)
            source_module_id = module_id
            for repair_index in range(max_repairs):
                if ok:
                    break
                try:
                    if repair is not None:
                        decision = await repair(
                            {
                                "step_id": step_id,
                                "module_id": module_id,
                                "params": arguments["params"],
                            },
                            final_result,
                            {"attempts": copy.deepcopy(attempts)},
                        )
                    else:
                        decision = repair_from_result(
                            module_id,
                            arguments["params"],
                            final_result,
                        )
                except Exception as exc:
                    decision = None
                    repair_trace.append({
                        "step_id": step_id,
                        "from_module": module_id,
                        "to_module": None,
                        "ok": False,
                        "reason": "Repair callback failed: {}".format(exc),
                    })
                if isinstance(decision, dict):
                    candidate_params = decision.get("params", {})
                    if not isinstance(candidate_params, dict):
                        candidate_params = {}
                    decision = RepairDecision(
                        module_id=str(decision.get("module_id") or ""),
                        params=candidate_params,
                        reason=str(decision.get("reason") or "repair callback"),
                        retry=decision.get("retry"),
                        assertions=decision.get("assertions"),
                    )
                if not isinstance(decision, RepairDecision):
                    break
                if (
                    not decision.module_id
                    or (
                        decision.module_id == module_id
                        and decision.params == arguments["params"]
                    )
                ):
                    break
                repair_arguments = {
                    "module_id": decision.module_id,
                    "params": copy.deepcopy(decision.params),
                }
                repair_access: Dict[str, Any] = {"ok": True}
                if preflight is not None:
                    try:
                        repair_access = await preflight(repair_arguments)
                    except Exception as exc:
                        repair_access = {
                            "ok": False,
                            "error": str(exc) or type(exc).__name__,
                        }
                if not isinstance(repair_access, dict) or not repair_access.get("ok"):
                    final_result = {
                        "ok": False,
                        "error": (
                            repair_access.get("error", "Repair access denied")
                            if isinstance(repair_access, dict)
                            else "Repair access preflight returned a non-object result"
                        ),
                    }
                    repair_trace.append({
                        "step_id": step_id,
                        "from_module": module_id,
                        "to_module": decision.module_id,
                        "ok": False,
                        "reason": decision.reason,
                        "phase": "preflight",
                    })
                    continue
                repair_validation = await _call(
                    dispatch,
                    "validate_params",
                    repair_arguments,
                )
                if not _validation_ok(repair_validation):
                    final_result = {
                        "ok": False,
                        "error": "Repair step validation failed",
                        "validation": repair_validation,
                    }
                    repair_trace.append({
                        "step_id": step_id,
                        "from_module": module_id,
                        "to_module": decision.module_id,
                        "ok": False,
                        "reason": decision.reason,
                        "phase": "validation",
                    })
                    module_id = decision.module_id
                    arguments = repair_arguments
                    validation = repair_validation
                    continue
                try:
                    repair_retry = _normalize_retry(decision.retry)
                    repair_assertions = item["assertions"]
                    if decision.assertions is not None:
                        repair_assertions = _normalize_assertions({
                            "assertions": decision.assertions,
                        })
                except (TypeError, ValueError) as exc:
                    final_result = {
                        "ok": False,
                        "error": "Invalid repair contract: {}".format(exc),
                    }
                    repair_trace.append({
                        "step_id": step_id,
                        "from_module": module_id,
                        "to_module": decision.module_id,
                        "ok": False,
                        "reason": decision.reason,
                        "phase": "contract",
                    })
                    continue
                from_module = module_id
                module_id = decision.module_id
                arguments = repair_arguments
                validation = repair_validation
                for repair_attempt in range(repair_retry["count"] + 1):
                    final_result = await _call(
                        dispatch,
                        "execute_module",
                        arguments,
                    )
                    total_attempts += 1
                    assertion_evidence = (
                        _evaluate_assertions(final_result, repair_assertions)
                        if _is_ok(final_result)
                        else []
                    )
                    assertions_ok = all(
                        assertion["ok"] for assertion in assertion_evidence
                    )
                    ok = _is_ok(final_result) and assertions_ok
                    attempts.append({
                        "attempt": len(attempts) + 1,
                        "repair": repair_index + 1,
                        "repair_attempt": repair_attempt + 1,
                        "strategy": decision.reason,
                        "ok": ok,
                        "result_preview": _preview(final_result),
                        "assertions": assertion_evidence,
                    })
                    if ok:
                        break
                    if repair_attempt < repair_retry["count"]:
                        wait_seconds = _retry_wait_seconds(
                            repair_retry,
                            repair_attempt,
                        )
                        if wait_seconds:
                            await asyncio.sleep(wait_seconds)
                repair_trace.append({
                    "step_id": step_id,
                    "from_module": from_module,
                    "to_module": module_id,
                    "ok": ok,
                    "reason": decision.reason,
                    "phase": "execute",
                })
            if assertion_evidence and not all(
                assertion["ok"] for assertion in assertion_evidence
            ):
                assertion_passed = False
                failed_phase = "assertion"
            elif not ok:
                failed_phase = "execution"

            executions.append({
                "function": "execute_module",
                "arguments": redact_args(arguments),
                "module_id": module_id,
                "source_module_id": source_module_id,
                "step_id": step_id,
                "ok": ok,
                "executed": True,
                "phase": "execute",
                "attempt_count": len(attempts),
                "attempts": attempts,
                "assertions": assertion_evidence,
                "result_preview": _preview(final_result),
                "validation": redact_args(validation),
                "blueprint_step": item["index"],
            })
            results.append({
                "module_id": module_id,
                "source_module_id": source_module_id,
                "step_id": step_id,
                "ok": ok,
                "executed": True,
                "attempt_count": len(attempts),
                "assertions": assertion_evidence,
                "validation": validation,
                "result": final_result,
            })
            if not ok:
                failed_module = module_id
                failed_step_id = step_id
                break

            completed_step_ids.append(step_id)
            context[step_id] = final_result
            checkpoint_error = save_checkpoint("partial") or ""
            if checkpoint_error:
                failed_module = module_id
                failed_step_id = step_id
                failed_phase = "checkpoint"
                break

    executed_entries = [item for item in executions if item.get("executed")]
    success = (
        bool(prepared)
        and not preflight_errors
        and not failed_step_id
        and len(completed_step_ids) == len(prepared)
        and all(item["ok"] for item in executions)
    )
    if not success and checkpoint_store is not None and not preflight_errors:
        checkpoint_error = save_checkpoint("failed") or checkpoint_error
    if success and not failed_phase:
        failed_phase = ""
    if success and checkpoint_store is not None:
        checkpoint_error = save_checkpoint("awaiting_outcome") or checkpoint_error
    evidence = {
        "execution_id": execution_id,
        "blueprint_id": blueprint_id,
        "executor_version": _EXECUTOR_VERSION,
        "plan_ir_version": plan_ir.version,
        "plan_gate_passed": not preflight_errors,
        "workflow_hash": workflow_hash,
        "preflight_passed": not preflight_errors,
        "preflight_checked_steps": len(prepared),
        "preflight_errors": redact_args(preflight_errors),
        "deferred_validation_steps": deferred_validation_steps,
        "validation_passed": validation_passed,
        "assertion_passed": assertion_passed,
        "step_count": len(prepared),
        "executed_steps": len(executed_entries),
        "passed_steps": len(completed_step_ids),
        "total_attempts": total_attempts,
        "repair_count": len(repair_trace),
        "repairs": redact_args(repair_trace),
        "completed_step_ids": completed_step_ids,
        "resumed_step_ids": resumed_step_ids,
        "failed_module": failed_module or None,
        "failed_step_id": failed_step_id or None,
        "failed_phase": failed_phase or None,
        "resume_from_step_id": failed_step_id or None,
        "side_effects_started": bool(executed_entries),
        "checkpoint_enabled": checkpoint_store is not None,
        "checkpoint_loaded": checkpoint_loaded,
        "checkpoint_status": checkpoint_status or None,
        "checkpoint_source_execution_id": (
            checkpoint_source_execution_id or None
        ),
        "checkpoint_write_count": checkpoint_write_count,
        "checkpoint_error": checkpoint_error or None,
        "checkpoint_contains_step_results": checkpoint_store is not None,
        "duration_ms": int((time.monotonic() - started) * 1000),
        "selection_mode": selection_mode,
    }
    if selection_mode == "deterministic":
        evidence["planner_model_calls_used"] = 0
        evidence["model_call_scope"] = "planner"

    checks = {
        "validation_passed": validation_passed,
        "assertion_passed": assertion_passed,
        "workflow_succeeded": success,
        "outcome_success": success,
    }
    verification = try_build_closed_loop_verification_receipt(
        execution_id, executions, checks, len(executed_entries), workflow_hash,
    )
    runtime_evidence = {
        "execution_id": execution_id,
        "workflow_hash": workflow_hash,
        "executor_version": _EXECUTOR_VERSION,
        "selection_mode": selection_mode,
        "duration_ms": evidence["duration_ms"],
        "step_count": evidence["step_count"],
        "total_attempts": evidence["total_attempts"],
        "assertion_passed": assertion_passed,
    }
    outcome = await _call(dispatch, "report_blueprint_outcome", {
        "blueprint_id": blueprint_id,
        "success": success,
        "execution_id": execution_id,
        "_evidence_capability": _CLOSED_LOOP_EVIDENCE_CAPABILITY,
        "_execution_evidence": runtime_evidence,
        "_verification_receipt": verification,
    })
    outcome_reported = (
        type(outcome) is dict
        and outcome.get("ok") is True
        and outcome.get("blueprint_id") == blueprint_id
        and outcome.get("execution_id") == execution_id
        and outcome.get("evidence_tier") == "local_verified"
    )
    if success and outcome_reported and checkpoint_store is not None:
        try:
            checkpoint_store.delete(checkpoint_id)
            checkpoint_cleared = True
        except Exception as exc:
            checkpoint_error = str(exc) or type(exc).__name__
    evidence["checkpoint_cleared"] = checkpoint_cleared
    evidence["checkpoint_error"] = checkpoint_error or None
    return {
        "ok": success,
        "closed_loop_ok": success and outcome_reported,
        "blueprint_id": blueprint_id,
        "execution_id": execution_id,
        "workflow_executed": True,
        "outcome_reported": outcome_reported,
        "workflow_hash": workflow_hash,
        "evidence": evidence,
        "executions": executions,
        "results": results,
        "outcome": outcome,
    }
