# Copyright 2024 Flyto2
# Licensed under the Apache License, Version 2.0
"""Fail-closed contracts for adaptive security validation campaigns.

The LLM proposes bounded steps.  This module independently binds those steps
to scope, authorization, module and cost limits before the existing verified
MCP loop is allowed to dispatch anything to flyto-core.
"""
from __future__ import annotations

import hashlib
import inspect
import ipaddress
import json
import re
from datetime import datetime, timezone
from typing import Any, Awaitable, Callable, Dict, List, Optional, Union
from urllib.parse import urlparse

SECURITY_CAMPAIGN_VERSION = "flyto.security-campaign.v1"

_ACTION_RANK = {
    "passive_observation": 0,
    "active_probe": 1,
    "exploit_validation": 2,
    "credential_validation": 3,
}
_MODE_MAX_ACTION = {
    "footprint": "active_probe",
    "pentest": "exploit_validation",
    "redteam": "credential_validation",
}
_AUTH_MAX_ACTION = {
    "passive": "passive_observation",
    "active": "active_probe",
    "exploit": "exploit_validation",
    "credential": "credential_validation",
}
_ACTION_COST = {
    "passive_observation": 1,
    "active_probe": 2,
    "exploit_validation": 5,
    "credential_validation": 8,
}
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_MODULE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}$")
_TARGET_KEYS = frozenset({
    "base_url",
    "domain",
    "endpoint",
    "host",
    "hostname",
    "origin",
    "target",
    "target_url",
    "url",
})
_METADATA_HOSTS = frozenset({
    "169.254.169.254",
    "metadata.google.internal",
    "metadata.goog",
})
_SAFE_FACT_KEYS = frozenset({
    "certificate_valid",
    "cipher",
    "confidence",
    "finding_type",
    "matched",
    "protocol",
    "severity",
    "status",
    "status_code",
    "tls_version",
})
_PLANNER_KEYS = frozenset({
    "action",
    "assertion_passed",
    "attempt_count",
    "campaign_id",
    "checks",
    "closed_loop_ok",
    "cost_units",
    "cost_units_used",
    "error_class",
    "error_fingerprint",
    "evidence",
    "executed",
    "failed_phase",
    "failed_step_id",
    "facts",
    "mode",
    "module_id",
    "ok",
    "phase",
    "reason_code",
    "repair_count",
    "requests_used",
    "result_fingerprint",
    "round",
    "step_id",
    "verified",
    "verdict",
})
_DEFAULT_BUDGETS = {
    "max_steps": 10,
    "max_requests": 20,
    "max_rounds": 3,
    "max_planner_tokens": 50_000,
    "max_cost_units": 100,
}
_BUDGET_LIMITS = {
    "max_steps": (1, 50),
    "max_requests": (1, 500),
    "max_rounds": (1, 10),
    "max_planner_tokens": (1, 1_000_000),
    "max_cost_units": (1, 100_000),
}

Planner = Callable[
    [Dict[str, Any]],
    Union[Dict[str, Any], Awaitable[Dict[str, Any]]],
]


def _stable_hash(value: Any) -> str:
    raw = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return "sha256:" + hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _bounded_text(value: Any, limit: int) -> str:
    return str(value or "").strip()[:limit]


def _parse_expiry(value: Any) -> Optional[datetime]:
    raw = _bounded_text(value, 80)
    if not raw:
        return None
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _normalize_host(value: Any, *, allow_wildcard: bool = False) -> str:
    raw = _bounded_text(value, 512).lower()
    wildcard = allow_wildcard and raw.startswith("*.")
    if wildcard:
        raw = raw[2:]
    parsed = urlparse(raw if "://" in raw else "//" + raw)
    host = (parsed.hostname or "").rstrip(".")
    if not host or any(char.isspace() for char in host):
        return ""
    return "*." + host if wildcard else host


def _is_private_or_special(host: str) -> bool:
    if host in {"localhost", "0.0.0.0"}:
        return True
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return False
    return bool(
        address.is_private
        or address.is_loopback
        or address.is_link_local
        or address.is_multicast
        or address.is_reserved
        or address.is_unspecified
    )


def _in_scope(host: str, scope: List[str]) -> bool:
    for allowed in scope:
        if allowed.startswith("*."):
            suffix = allowed[1:]
            if host.endswith(suffix) and host != allowed[2:]:
                return True
        elif host == allowed:
            return True
    return False


def _looks_like_target(value: str) -> bool:
    if "://" in value:
        return True
    if value.startswith("${") or "/" in value or " " in value:
        return False
    return "." in value or value in {"localhost", "0.0.0.0"}


def _extract_targets(value: Any, key: str = "") -> List[str]:
    targets: List[str] = []
    if isinstance(value, dict):
        for nested_key, nested in value.items():
            targets.extend(_extract_targets(nested, str(nested_key).lower()))
    elif isinstance(value, (list, tuple)):
        for nested in value:
            targets.extend(_extract_targets(nested, key))
    elif isinstance(value, str):
        if "${" in value:
            return targets
        if key in _TARGET_KEYS or _looks_like_target(value):
            host = _normalize_host(value)
            if host:
                targets.append(host)
    return list(dict.fromkeys(targets))


def _declared_action(step: Dict[str, Any]) -> str:
    direct = step.get("security_action")
    contract = step.get("contract")
    nested = contract.get("security_action") if isinstance(contract, dict) else None
    action = _bounded_text(direct or nested, 40)
    return action if action in _ACTION_RANK else ""


def classify_security_action(
    module_id: str,
    declared_action: str = "",
) -> str:
    """Classify a Core module, never allowing a declaration to lower risk."""
    module = module_id.lower().replace("-", "_")
    inferred = "passive_observation"
    if any(token in module for token in (
        "credential", "password_spray", "stuffing", "bruteforce", "brute_force",
    )):
        inferred = "credential_validation"
    elif any(token in module for token in (
        "exploit", "injection", "payload", "rce", "sqli", "ssrf", "xss",
        "traversal", "deserialization", "auth_bypass",
    )):
        inferred = "exploit_validation"
    elif module.startswith((
        "browser.", "dns.", "http.", "https.", "network.", "port.", "tls.",
    )):
        inferred = "active_probe"

    if declared_action in _ACTION_RANK:
        return max(
            (inferred, declared_action),
            key=lambda item: _ACTION_RANK[item],
        )
    return inferred


def _normalize_usage(value: Any) -> Dict[str, Any]:
    raw = value if isinstance(value, dict) else {}

    def nonnegative(name: str) -> int:
        item = raw.get(name, 0)
        return item if isinstance(item, int) and item >= 0 else 0

    evidence = raw.get("evidence")
    normalized_evidence = (
        list(evidence[:100]) if isinstance(evidence, list) else []
    )
    evidence_count = nonnegative("evidence_count")
    return {
        "requests_used": nonnegative("requests_used"),
        "cost_units_used": nonnegative("cost_units_used"),
        "planner_tokens_used": nonnegative("planner_tokens_used"),
        "rounds_completed": nonnegative("rounds_completed"),
        "evidence_count": max(evidence_count, len(normalized_evidence)),
        "evidence": normalized_evidence,
    }


def _normalize_budget(
    raw: Dict[str, Any],
    name: str,
    errors: List[str],
) -> int:
    minimum, maximum = _BUDGET_LIMITS[name]
    value = raw.get(name)
    if value is None:
        errors.append("budgets.{} is required".format(name))
        return _DEFAULT_BUDGETS[name]
    if not isinstance(value, int) or isinstance(value, bool):
        errors.append("budgets.{} must be an integer".format(name))
        return _DEFAULT_BUDGETS[name]
    if not minimum <= value <= maximum:
        errors.append(
            "budgets.{} must be between {} and {}".format(
                name,
                minimum,
                maximum,
            ),
        )
        return max(minimum, min(value, maximum))
    return value


def compile_security_campaign(
    raw: Any,
    steps: Any,
) -> Dict[str, Any]:
    """Normalize and gate an LLM-proposed campaign without executing it."""
    errors: List[str] = []
    source = raw if isinstance(raw, dict) else {}
    if not isinstance(raw, dict):
        errors.append("security_campaign must be an object")
    proposed_steps = steps if isinstance(steps, list) else []
    if not isinstance(steps, list):
        errors.append("steps must be an array")

    campaign_id = _bounded_text(source.get("campaign_id"), 128)
    if not _ID_RE.fullmatch(campaign_id):
        errors.append("campaign_id is required and must be a stable identifier")
    mode = _bounded_text(source.get("mode"), 20)
    if mode not in _MODE_MAX_ACTION:
        errors.append("mode must be footprint, pentest, or redteam")
        mode = "footprint"
    objective = _bounded_text(source.get("objective"), 2000)
    if not objective:
        errors.append("objective is required")

    authorization_raw = source.get("authorization")
    authorization_raw = (
        authorization_raw if isinstance(authorization_raw, dict) else {}
    )
    if not isinstance(source.get("authorization"), dict):
        errors.append("authorization must be an object")
    auth_level = _bounded_text(authorization_raw.get("level"), 20)
    if auth_level not in _AUTH_MAX_ACTION:
        errors.append("authorization.level is invalid")
        auth_level = "passive"
    auth_reference = _bounded_text(
        authorization_raw.get("reference")
        or authorization_raw.get("authorization_id"),
        256,
    )
    approved_raw = authorization_raw.get("approved_actions")
    approved_actions = sorted({
        item for item in (approved_raw if isinstance(approved_raw, list) else [])
        if isinstance(item, str) and item in _ACTION_RANK
    })
    if not isinstance(approved_raw, list):
        errors.append("authorization.approved_actions must be an array")
    expiry = _parse_expiry(authorization_raw.get("expires_at"))
    expires_at = expiry.isoformat() if expiry else ""
    allow_private = authorization_raw.get("allow_private_targets") is True

    scope_raw = source.get("target_scope")
    scope_items = scope_raw if isinstance(scope_raw, list) else []
    if not isinstance(scope_raw, list) or not scope_items:
        errors.append("target_scope must be a non-empty array")
    scope: List[str] = []
    for item in scope_items[:100]:
        host = _normalize_host(item, allow_wildcard=True)
        if not host:
            errors.append("target_scope contains an invalid host")
            continue
        concrete = host[2:] if host.startswith("*.") else host
        if concrete in _METADATA_HOSTS:
            errors.append("metadata endpoints are never valid campaign targets")
        elif _is_private_or_special(concrete) and not allow_private:
            errors.append(
                "private or special targets require allow_private_targets",
            )
        scope.append(host)
    scope = list(dict.fromkeys(scope))

    modules_raw = source.get("module_allowlist")
    module_items = modules_raw if isinstance(modules_raw, list) else []
    if not isinstance(modules_raw, list) or not module_items:
        errors.append("module_allowlist must be a non-empty array")
    if len(module_items) > 50:
        errors.append("module_allowlist cannot exceed 50 entries")
    module_allowlist = list(dict.fromkeys(
        item for item in module_items[:50]
        if isinstance(item, str) and _MODULE_RE.fullmatch(item)
    ))
    if len(module_allowlist) != len(module_items[:50]):
        errors.append("module_allowlist contains an invalid module identifier")

    budgets_raw = source.get("budgets")
    budgets_raw = budgets_raw if isinstance(budgets_raw, dict) else {}
    if not isinstance(source.get("budgets"), dict):
        errors.append("budgets must be an object")
    budgets = {
        name: _normalize_budget(budgets_raw, name, errors)
        for name in _DEFAULT_BUDGETS
    }
    usage = _normalize_usage(source.get("prior_usage"))
    round_number = source.get("round", usage["rounds_completed"] + 1)
    if not isinstance(round_number, int) or round_number < 1:
        errors.append("round must be a positive integer")
        round_number = usage["rounds_completed"] + 1
    if round_number > budgets["max_rounds"]:
        errors.append("campaign round budget exceeded")
    if usage["planner_tokens_used"] > budgets["max_planner_tokens"]:
        errors.append("planner token budget exceeded")
    if usage["cost_units_used"] > budgets["max_cost_units"]:
        errors.append("campaign cost budget exceeded")
    if usage["requests_used"] > budgets["max_requests"]:
        errors.append("campaign request budget exceeded")
    if len(proposed_steps) > budgets["max_steps"]:
        errors.append("campaign step budget exceeded")

    module_actions: Dict[str, str] = {}
    highest_action = "passive_observation"
    for index, step in enumerate(proposed_steps):
        if not isinstance(step, dict):
            errors.append("step {} must be an object".format(index + 1))
            continue
        module_id = _bounded_text(
            step.get("module") or step.get("module_id"),
            128,
        )
        if not _MODULE_RE.fullmatch(module_id):
            errors.append("step {} has an invalid module".format(index + 1))
            continue
        declared = _declared_action(step)
        raw_declared = step.get("security_action")
        if raw_declared is None and isinstance(step.get("contract"), dict):
            raw_declared = step["contract"].get("security_action")
        if raw_declared is not None and not declared:
            errors.append(
                "step {} has an invalid security_action".format(index + 1),
            )
        action = classify_security_action(module_id, declared)
        previous = module_actions.get(module_id, "passive_observation")
        module_actions[module_id] = max(
            (previous, action),
            key=lambda item: _ACTION_RANK[item],
        )
        highest_action = max(
            (highest_action, action),
            key=lambda item: _ACTION_RANK[item],
        )
        if module_id not in module_allowlist:
            errors.append("module {} is not allowlisted".format(module_id))
        if _ACTION_RANK[action] > _ACTION_RANK[_MODE_MAX_ACTION[mode]]:
            errors.append("{} exceeds {} campaign authority".format(action, mode))
        if _ACTION_RANK[action] > _ACTION_RANK[_AUTH_MAX_ACTION[auth_level]]:
            errors.append("{} exceeds authorization level".format(action))
        if action != "passive_observation" and action not in approved_actions:
            errors.append("{} is not explicitly approved".format(action))
        if action != "passive_observation" and not (
            step.get("assertions") or step.get("assert")
        ):
            errors.append(
                "active step {} requires a proof assertion".format(index + 1),
            )
        targets = _extract_targets(step.get("params", {}))
        if action != "passive_observation" and not targets:
            errors.append(
                "active step {} requires an explicit target".format(index + 1),
            )
        for target in targets:
            if target in _METADATA_HOSTS:
                errors.append("metadata endpoints are never executable targets")
            elif not _in_scope(target, scope):
                errors.append("target {} is outside campaign scope".format(target))

    if highest_action != "passive_observation":
        if not auth_reference:
            errors.append("active authorization requires a reference")
        if expiry is None:
            errors.append("active authorization requires a valid expires_at")
        elif expiry <= datetime.now(timezone.utc):
            errors.append("authorization has expired")

    contract = {
        "version": SECURITY_CAMPAIGN_VERSION,
        "campaign_id": campaign_id,
        "mode": mode,
        "objective": objective,
        "target_scope": scope,
        "authorization": {
            "level": auth_level,
            "reference": auth_reference,
            "expires_at": expires_at,
            "approved_actions": approved_actions,
            "allow_private_targets": allow_private,
        },
        "module_allowlist": module_allowlist,
        "module_actions": module_actions,
        "budgets": budgets,
        "round": round_number,
        "parent_execution_id": _bounded_text(
            source.get("parent_execution_id"),
            128,
        ),
        "initial_usage": usage,
    }
    contract["contract_hash"] = _stable_hash(contract)
    contract["gate_errors"] = list(dict.fromkeys(errors))
    return contract


def evaluate_campaign_action(
    contract: Dict[str, Any],
    usage: Any,
    tool_name: str,
    arguments: Any,
) -> Dict[str, Any]:
    """Authorize one Core boundary call against the frozen campaign."""
    if contract.get("gate_errors"):
        return {
            "allowed": False,
            "reason": "security campaign gate failed",
            "reason_code": "campaign_gate_failed",
        }
    if tool_name == "report_blueprint_outcome":
        return {
            "allowed": True,
            "reason": "",
            "reason_code": "outcome_allowed",
            "action": "passive_observation",
            "cost_units": 0,
        }
    if tool_name not in {"execute_module", "validate_params"}:
        return {
            "allowed": False,
            "reason": "tool is outside the campaign execution boundary",
            "reason_code": "tool_not_allowed",
        }
    call = arguments if isinstance(arguments, dict) else {}
    module_id = _bounded_text(call.get("module_id"), 128)
    if module_id not in contract.get("module_allowlist", []):
        return {
            "allowed": False,
            "reason": "module is outside the campaign allowlist",
            "reason_code": "module_not_allowed",
        }
    declared = (contract.get("module_actions") or {}).get(module_id, "")
    action = classify_security_action(module_id, declared)
    mode_limit = _MODE_MAX_ACTION.get(contract.get("mode"), "passive_observation")
    authorization = contract.get("authorization") or {}
    auth_limit = _AUTH_MAX_ACTION.get(
        authorization.get("level"),
        "passive_observation",
    )
    if _ACTION_RANK[action] > _ACTION_RANK[mode_limit]:
        return {
            "allowed": False,
            "reason": "action exceeds campaign mode",
            "reason_code": "mode_exceeded",
        }
    if _ACTION_RANK[action] > _ACTION_RANK[auth_limit]:
        return {
            "allowed": False,
            "reason": "action exceeds authorization level",
            "reason_code": "authorization_exceeded",
        }
    if (
        action != "passive_observation"
        and action not in authorization.get("approved_actions", [])
    ):
        return {
            "allowed": False,
            "reason": "action is not explicitly approved",
            "reason_code": "action_not_approved",
        }
    if action != "passive_observation":
        expiry = _parse_expiry(authorization.get("expires_at"))
        if expiry is None or expiry <= datetime.now(timezone.utc):
            return {
                "allowed": False,
                "reason": "authorization is missing or expired",
                "reason_code": "authorization_expired",
            }

    targets = _extract_targets(call.get("params", {}))
    if action != "passive_observation" and not targets:
        return {
            "allowed": False,
            "reason": "active action has no explicit target",
            "reason_code": "target_missing",
        }
    for target in targets:
        if target in _METADATA_HOSTS:
            return {
                "allowed": False,
                "reason": "metadata endpoint is forbidden",
                "reason_code": "metadata_forbidden",
            }
        if not _in_scope(target, contract.get("target_scope", [])):
            return {
                "allowed": False,
                "reason": "target is outside campaign scope",
                "reason_code": "target_out_of_scope",
            }

    normalized_usage = _normalize_usage(usage)
    cost_units = _ACTION_COST[action]
    if tool_name == "execute_module":
        budgets = contract.get("budgets") or {}
        if normalized_usage["requests_used"] + 1 > budgets.get("max_requests", 0):
            return {
                "allowed": False,
                "reason": "campaign request budget exhausted",
                "reason_code": "request_budget_exhausted",
            }
        if (
            normalized_usage["cost_units_used"] + cost_units
            > budgets.get("max_cost_units", 0)
        ):
            return {
                "allowed": False,
                "reason": "campaign cost budget exhausted",
                "reason_code": "cost_budget_exhausted",
            }
    return {
        "allowed": True,
        "reason": "",
        "reason_code": "authorized",
        "action": action,
        "cost_units": cost_units if tool_name == "execute_module" else 0,
        "targets": targets,
    }


def _safe_facts(result: Any) -> Dict[str, Any]:
    facts: Dict[str, Any] = {}

    def visit(value: Any) -> None:
        if not isinstance(value, dict):
            return
        for key, nested in value.items():
            lowered = str(key).lower()
            if (
                lowered in _SAFE_FACT_KEYS
                and isinstance(nested, (str, int, float, bool, type(None)))
            ):
                facts[lowered] = (
                    nested[:128] if isinstance(nested, str) else nested
                )
            elif isinstance(nested, dict):
                visit(nested)

    visit(result)
    return facts


def record_campaign_result(
    contract: Dict[str, Any],
    usage: Any,
    tool_name: str,
    arguments: Any,
    result: Any,
) -> Dict[str, Any]:
    """Record compact, non-secret proof metadata for one executed module."""
    normalized = _normalize_usage(usage)
    if tool_name != "execute_module":
        return normalized
    decision = evaluate_campaign_action(
        contract,
        normalized,
        tool_name,
        arguments,
    )
    if not decision.get("allowed"):
        return normalized
    call = arguments if isinstance(arguments, dict) else {}
    response = result if isinstance(result, dict) else {}
    normalized["requests_used"] += 1
    normalized["cost_units_used"] += int(decision.get("cost_units") or 0)
    normalized["evidence_count"] += 1
    error_class = _bounded_text(
        response.get("exception_type") or response.get("error_code"),
        80,
    )
    evidence = {
        "module_id": _bounded_text(call.get("module_id"), 128),
        "action": decision.get("action"),
        "ok": bool(
            response.get("ok")
            if "ok" in response
            else response.get("status") == "success"
        ),
        "cost_units": decision.get("cost_units"),
        "facts": _safe_facts(response),
        "result_fingerprint": _stable_hash(response),
    }
    if error_class:
        evidence["error_class"] = error_class
    elif not evidence["ok"]:
        evidence["error_fingerprint"] = _stable_hash(
            response.get("error", "unknown failure"),
        )
    normalized["evidence"] = (normalized["evidence"] + [evidence])[-100:]
    return normalized


def project_evidence_for_planner(value: Any) -> Dict[str, Any]:
    """Produce a bounded allowlisted projection safe for an LLM prompt."""
    def project(item: Any, key: str = "") -> Any:
        if isinstance(item, dict):
            return {
                child_key: projected
                for child_key, child_value in item.items()
                if child_key in _PLANNER_KEYS or (
                    key == "facts" and child_key in _SAFE_FACT_KEYS
                )
                for projected in [project(child_value, child_key)]
                if projected is not None
            }
        if isinstance(item, list):
            return [
                projected
                for child in item[:20]
                for projected in [project(child, key)]
                if projected is not None
            ]
        if isinstance(item, bool) or item is None:
            return item
        if isinstance(item, (int, float)):
            return item
        if isinstance(item, str) and key in {
            "action",
            "campaign_id",
            "error_class",
            "error_fingerprint",
            "failed_phase",
            "failed_step_id",
            "finding_type",
            "mode",
            "module_id",
            "phase",
            "protocol",
            "reason_code",
            "result_fingerprint",
            "severity",
            "status",
            "step_id",
            "tls_version",
            "verdict",
        }:
            return item[:128]
        return None

    projected = project(value)
    if not isinstance(projected, dict):
        projected = {}
    raw = json.dumps(projected, sort_keys=True, separators=(",", ":"))
    if len(raw) > 12_000:
        projected["evidence"] = list(projected.get("evidence", []))[:5]
        projected["truncated"] = True
    return {
        "schema": "flyto.security-planner-evidence.v1",
        "trusted_projection": projected,
        "raw_target_content_included": False,
    }


def verify_security_campaign(
    contract: Dict[str, Any],
    usage: Any,
    execution_result: Any,
) -> Dict[str, Any]:
    """Issue a proof verdict without converting absence of evidence to success."""
    normalized = _normalize_usage(usage)
    result = execution_result if isinstance(execution_result, dict) else {}
    runtime = result.get("evidence")
    runtime = runtime if isinstance(runtime, dict) else {}
    budgets = contract.get("budgets") or {}
    checks = {
        "contract_gate": not bool(contract.get("gate_errors")),
        "runtime_closed_loop": bool(result.get("closed_loop_ok")),
        "assertions": runtime.get("assertion_passed") is True,
        "request_budget": (
            normalized["requests_used"] <= budgets.get("max_requests", -1)
        ),
        "cost_budget": (
            normalized["cost_units_used"] <= budgets.get("max_cost_units", -1)
        ),
        "proof_evidence": (
            normalized["requests_used"] > 0
            and normalized["evidence_count"] == normalized["requests_used"]
        ),
    }
    verified = all(checks.values())
    return {
        "verified": verified,
        "verdict": "proved" if verified else "not_proved",
        "checks": checks,
        "metrics": {
            "requests_used": normalized["requests_used"],
            "cost_units_used": normalized["cost_units_used"],
            "evidence_count": normalized["evidence_count"],
        },
        "next_action": "complete" if verified else "replan",
    }


async def _call_planner(
    planner: Planner,
    request: Dict[str, Any],
) -> Dict[str, Any]:
    proposed = planner(request)
    if inspect.isawaitable(proposed):
        proposed = await proposed
    if not isinstance(proposed, dict):
        raise TypeError("security campaign planner must return an object")
    return proposed


async def run_security_campaign(
    request: Dict[str, Any],
    planner: Planner,
    *,
    server: Any = None,
) -> Dict[str, Any]:
    """Run bounded LLM re-planning through the existing four-tool MCP loop."""
    if server is None:
        from flyto_ai.closed_loop_mcp import ClosedLoopMCPServer

        server = ClosedLoopMCPServer()
    source = dict(request)
    budgets = source.get("budgets")
    budgets = budgets if isinstance(budgets, dict) else {}
    max_rounds = budgets.get("max_rounds", _DEFAULT_BUDGETS["max_rounds"])
    if not isinstance(max_rounds, int):
        max_rounds = _DEFAULT_BUDGETS["max_rounds"]
    max_rounds = max(1, min(max_rounds, _BUDGET_LIMITS["max_rounds"][1]))
    usage = _normalize_usage(source.get("prior_usage"))
    planner_evidence = project_evidence_for_planner({})
    rounds: List[Dict[str, Any]] = []

    for round_number in range(usage["rounds_completed"] + 1, max_rounds + 1):
        planner_request = {
            "version": SECURITY_CAMPAIGN_VERSION,
            "campaign_id": _bounded_text(source.get("campaign_id"), 128),
            "mode": _bounded_text(source.get("mode"), 20),
            "objective": _bounded_text(source.get("objective"), 2000),
            "target_scope": list(source.get("target_scope") or [])[:100],
            "module_allowlist": list(source.get("module_allowlist") or [])[:50],
            "approved_actions": list(
                (source.get("authorization") or {}).get("approved_actions") or [],
            ),
            "round": round_number,
            "remaining": {
                "requests": max(
                    0,
                    budgets.get("max_requests", 0) - usage["requests_used"],
                ),
                "cost_units": max(
                    0,
                    budgets.get("max_cost_units", 0) - usage["cost_units_used"],
                ),
                "planner_tokens": max(
                    0,
                    budgets.get("max_planner_tokens", 0)
                    - usage["planner_tokens_used"],
                ),
            },
            "prior_evidence": planner_evidence,
        }
        try:
            proposal = await _call_planner(planner, planner_request)
        except Exception as exc:
            rounds.append({
                "round": round_number,
                "ok": False,
                "reason_code": "planner_failed",
                "error_class": type(exc).__name__,
            })
            break
        planner_usage = proposal.get("usage")
        planner_usage = planner_usage if isinstance(planner_usage, dict) else {}
        tokens = planner_usage.get("tokens", 0)
        if isinstance(tokens, int) and tokens >= 0:
            usage["planner_tokens_used"] += tokens
        usage["rounds_completed"] = round_number

        campaign_input = dict(source)
        campaign_input["round"] = round_number
        campaign_input["prior_usage"] = usage
        campaign_input["parent_execution_id"] = (
            rounds[-1].get("execution_id") if rounds else ""
        )
        planned = await server.call_tool("plan", {
            "message": _bounded_text(
                proposal.get("message") or source.get("objective"),
                2000,
            ),
            "blueprint_id": "{}-round-{}".format(
                _bounded_text(source.get("campaign_id"), 80),
                round_number,
            ),
            "steps": proposal.get("steps"),
            "model_candidates": proposal.get("model_candidates", []),
            "security_campaign": campaign_input,
        })
        plan_data = planned.get("structuredContent", {})
        round_record = {
            "round": round_number,
            "plan_id": plan_data.get("plan_id"),
            "plan_gate": bool((plan_data.get("gate") or {}).get("pass")),
        }
        if planned.get("isError"):
            round_record["ok"] = False
            round_record["reason_code"] = "plan_gate_failed"
            rounds.append(round_record)
            planner_evidence = project_evidence_for_planner(round_record)
            continue

        executed = await server.call_tool("execute", {
            "plan_id": plan_data["plan_id"],
            "max_repairs": 0,
        })
        execution = executed.get("structuredContent", {})
        campaign_summary = execution.get("security_campaign") or {}
        usage["requests_used"] = max(
            usage["requests_used"],
            int(campaign_summary.get("requests_used") or 0),
        )
        usage["cost_units_used"] = max(
            usage["cost_units_used"],
            int(campaign_summary.get("cost_units_used") or 0),
        )
        usage["evidence_count"] = max(
            usage["evidence_count"],
            int(campaign_summary.get("evidence_count") or 0),
        )
        verification_result = await server.call_tool("verify", {
            "execution_id": execution.get("execution_id"),
        })
        verification = verification_result.get("structuredContent", {})
        evidence_result = await server.call_tool("get_evidence", {
            "execution_id": execution.get("execution_id"),
            "section": "executions",
            "limit": 20,
        })
        evidence = evidence_result.get("structuredContent", {})
        round_record.update({
            "ok": bool(verification.get("verified")),
            "execution_id": execution.get("execution_id"),
            "closed_loop_ok": bool(execution.get("closed_loop_ok")),
            "verified": bool(verification.get("verified")),
            "failed_step_id": execution.get("failed_step_id"),
            "failed_phase": execution.get("failed_phase"),
        })
        rounds.append(round_record)
        if verification.get("verified"):
            return {
                "ok": True,
                "verified": True,
                "campaign_id": source.get("campaign_id"),
                "rounds": rounds,
                "usage": usage,
                "verification": verification,
            }
        planner_evidence = project_evidence_for_planner({
            **round_record,
            "checks": verification.get("checks"),
            "evidence": evidence.get("executions", []),
        })

    return {
        "ok": False,
        "verified": False,
        "campaign_id": source.get("campaign_id"),
        "rounds": rounds,
        "usage": usage,
        "verdict": "not_proved",
        "next_action": "human_review" if rounds else "fix_contract",
    }
