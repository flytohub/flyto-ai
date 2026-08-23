# Copyright 2024 Flyto2
# Licensed under the Apache License, Version 2.0
"""Token-bounded MCP facade for the verified Flyto closed loop.

The server intentionally exposes four high-level tools instead of mirroring
every internal Core call. Plans, checkpoints, and full evidence stay on the
server; MCP responses return compact summaries plus opaque identifiers.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from flyto_ai import __version__
from flyto_ai.blueprint_loop import execute_blueprint_loop
from flyto_ai.closed_loop_v3 import (
    CapabilityModelRouter,
    JsonCheckpointStore,
    ModelCandidate,
    PLAN_IR_VERSION,
    PlanIR,
    evaluate_distillation,
    stable_hash,
)
from flyto_ai.execution_verification import verified_learning_distillation
from flyto_ai.mcp_server import (
    DISCOVERY_TTL_MS,
    STATIC_LIST_TTL_MS,
    SUPPORTED_PROTOCOL_VERSIONS,
    build_modern_result,
    negotiate_legacy_protocol_version,
    request_protocol_era,
)
from flyto_ai.permissions import PermissionEnforcer, PermissionLevel
from flyto_ai.tools.core_tools import dispatch_core_tool

logger = logging.getLogger(__name__)

MCP_CONTRACT_VERSION = "flyto.closed-loop-mcp.v1"
_MAX_PLAN_STEPS = 50
_MAX_PLAN_CHARS = 128_000
_MAX_EVIDENCE_CHARS = 12_000
_SERVER_INSTRUCTIONS = (
    "Call plan before execute. Execute accepts only stored plan IDs and applies "
    "permission, PlanIR, Core validation, checkpoint, repair, assertion, and "
    "outcome gates. Call verify before claiming success. Use get_evidence only "
    "when compact summaries are insufficient; raw evidence is paginated."
)


def _annotations(
    *,
    read_only: bool,
    destructive: bool,
    idempotent: bool,
) -> Dict[str, bool]:
    return {
        "readOnlyHint": read_only,
        "destructiveHint": destructive,
        "idempotentHint": idempotent,
    }


TOOLS = [
    {
        "name": "plan",
        "description": (
            "Compile steps into a typed, hash-addressed PlanIR and run its "
            "structural gate. Stores the full plan server-side."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "message": {"type": "string"},
                "blueprint_id": {"type": "string"},
                "steps": {
                    "type": "array",
                    "items": {"type": "object"},
                    "minItems": 1,
                    "maxItems": _MAX_PLAN_STEPS,
                },
                "model_candidates": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "provider": {"type": "string"},
                            "model": {"type": "string"},
                            "cost_rank": {"type": "integer"},
                        },
                        "required": ["provider", "model"],
                    },
                    "maxItems": 8,
                },
                "security_campaign": {
                    "type": "object",
                    "description": (
                        "Optional fail-closed security campaign contract. "
                        "Binds scope, authorization, modules, budgets, and "
                        "planner rounds to the stored PlanIR."
                    ),
                    "properties": {
                        "campaign_id": {"type": "string"},
                        "mode": {
                            "type": "string",
                            "enum": ["footprint", "pentest", "redteam"],
                        },
                        "objective": {"type": "string"},
                        "target_scope": {
                            "type": "array",
                            "items": {"type": "string"},
                            "minItems": 1,
                            "maxItems": 100,
                        },
                        "authorization": {"type": "object"},
                        "module_allowlist": {
                            "type": "array",
                            "items": {"type": "string"},
                            "minItems": 1,
                            "maxItems": _MAX_PLAN_STEPS,
                        },
                        "budgets": {"type": "object"},
                        "round": {"type": "integer", "minimum": 1},
                        "parent_execution_id": {"type": "string"},
                    },
                    "required": [
                        "campaign_id",
                        "mode",
                        "target_scope",
                        "authorization",
                        "module_allowlist",
                        "budgets",
                    ],
                },
            },
            "required": ["steps"],
        },
        "annotations": _annotations(
            read_only=False,
            destructive=False,
            idempotent=True,
        ),
    },
    {
        "name": "execute",
        "description": (
            "Execute or resume one stored plan through real flyto-core. "
            "Returns compact verification evidence and an evidence ID."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "plan_id": {"type": "string"},
                "max_repairs": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 3,
                },
            },
            "required": ["plan_id"],
        },
        "annotations": _annotations(
            read_only=False,
            destructive=True,
            idempotent=False,
        ),
    },
    {
        "name": "verify",
        "description": (
            "Verify stored runtime evidence and, when eligible, distill it "
            "into a persistent verified Blueprint."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "execution_id": {"type": "string"},
                "plan_id": {"type": "string"},
                "min_steps": {
                    "type": "integer",
                    "minimum": 3,
                    "maximum": 20,
                    "default": 3,
                },
            },
        },
        "annotations": _annotations(
            read_only=False,
            destructive=False,
            idempotent=True,
        ),
    },
    {
        "name": "get_evidence",
        "description": (
            "Read stored evidence by section. Raw JSON is paginated and "
            "hard-capped to keep MCP context bounded."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "execution_id": {"type": "string"},
                "section": {
                    "type": "string",
                    "enum": ["summary", "executions", "raw"],
                    "default": "summary",
                },
                "offset": {"type": "integer", "minimum": 0, "default": 0},
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": _MAX_EVIDENCE_CHARS,
                    "default": 4000,
                },
            },
            "required": ["execution_id"],
        },
        "annotations": _annotations(
            read_only=True,
            destructive=False,
            idempotent=True,
        ),
    },
]
_TOOLS_BY_NAME = {item["name"]: item for item in TOOLS}


def _make_error(req_id: Any, code: int, message: str) -> Dict[str, Any]:
    return {
        "jsonrpc": "2.0",
        "id": req_id,
        "error": {"code": code, "message": message},
    }


def _make_result(req_id: Any, result: Any) -> Dict[str, Any]:
    return {"jsonrpc": "2.0", "id": req_id, "result": result}


def _server_info() -> Dict[str, Any]:
    return {
        "name": "flyto-closed-loop",
        "title": "Flyto2 Verified Closed Loop",
        "version": __version__,
        "description": (
            "Plan, execute, repair, verify, and preserve compact evidence for "
            "Flyto2 automation."
        ),
        "websiteUrl": "https://github.com/flytohub/flyto-ai",
    }


def _protocol_result(
    req_id: Any,
    result: Dict[str, Any],
    *,
    modern: bool,
    ttl_ms: Optional[int] = None,
    cache_scope: Optional[str] = None,
) -> Dict[str, Any]:
    if modern:
        result = build_modern_result(
            result,
            server_info=_server_info(),
            ttl_ms=ttl_ms,
            cache_scope=cache_scope,
        )
    return _make_result(req_id, result)


def _tool_result(
    data: Dict[str, Any],
    *,
    message: str,
    is_error: bool = False,
) -> Dict[str, Any]:
    """Return modern structured MCP content with a tiny legacy text fallback."""
    return {
        "content": [{"type": "text", "text": message[:300]}],
        "structuredContent": data,
        "isError": is_error,
    }


def _json_chars(value: Any) -> int:
    return len(json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ))


def _clamped_int(
    value: Any,
    *,
    default: int,
    minimum: int,
    maximum: int,
) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return max(minimum, min(parsed, maximum))


def _permission_level() -> PermissionLevel:
    raw = os.getenv(
        "FLYTO_CLOSED_LOOP_PERMISSION",
        "workspace_write",
    ).upper()
    try:
        return PermissionLevel[raw]
    except KeyError:
        return PermissionLevel.WORKSPACE_WRITE


def _candidate_list(raw: Any) -> List[ModelCandidate]:
    candidates: List[ModelCandidate] = []
    if not isinstance(raw, list):
        return candidates
    for index, item in enumerate(raw[:8]):
        if not isinstance(item, dict):
            continue
        provider = str(item.get("provider") or "")
        model = str(item.get("model") or "")
        if not provider or not model:
            continue
        cost_rank = _clamped_int(
            item.get("cost_rank"),
            default=index,
            minimum=0,
            maximum=100,
        )
        candidates.append(ModelCandidate.from_name(
            provider,
            model,
            cost_rank,
        ))
    return candidates


class ClosedLoopMCPServer:
    """Stateful, local-only closed-loop MCP server."""

    def __init__(
        self,
        state_dir: Optional[str] = None,
        *,
        trusted_campaign_scope: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Create a closed-loop server.

        ``trusted_campaign_scope`` is an in-process Runner integration hook,
        not part of the MCP tool schema.  It can narrow one attested security
        campaign to exact outbound hosts and ports without changing process
        environment variables.  The scope is ignored by ordinary plans and
        is fail-closed unless its contract hash matches the compiled campaign.
        """
        configured_dir = state_dir or os.getenv(
            "FLYTO_CLOSED_LOOP_STATE_DIR",
            "~/.flyto/closed-loop-mcp",
        )
        self._state_dir = Path(configured_dir).expanduser()
        self._state_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        try:
            self._state_dir.chmod(0o700)
        except OSError:
            pass
        self._records = JsonCheckpointStore(
            str(self._state_dir / "records"),
        )
        self._checkpoints = JsonCheckpointStore(
            str(self._state_dir / "checkpoints"),
        )
        self._enforcer = PermissionEnforcer(_permission_level())
        self._max_repairs = _clamped_int(
            os.getenv("FLYTO_CLOSED_LOOP_MAX_REPAIRS"),
            default=1,
            minimum=0,
            maximum=3,
        )
        self._router = CapabilityModelRouter()
        self._blueprint_engine = None
        self._test_fail_once_module = os.getenv(
            "FLYTO_CLOSED_LOOP_MCP_FAIL_ONCE_MODULE",
            "",
        )
        self._trusted_campaign_scope = self._normalize_trusted_campaign_scope(
            trusted_campaign_scope,
        )

    @staticmethod
    def _normalize_trusted_campaign_scope(
        value: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        if value is None:
            return None
        if not isinstance(value, dict):
            raise ValueError("trusted_campaign_scope must be an object")
        contract_hash = str(value.get("contract_hash") or "").strip()
        if not (
            contract_hash.startswith("sha256:")
            and len(contract_hash) == 71
        ):
            raise ValueError(
                "trusted_campaign_scope requires a sha256 contract_hash",
            )
        raw_hosts = value.get("allowed_hosts")
        raw_ports = value.get("allowed_ports")
        if not isinstance(raw_hosts, list) or not raw_hosts:
            raise ValueError(
                "trusted_campaign_scope.allowed_hosts must be non-empty",
            )
        if not isinstance(raw_ports, list) or not raw_ports:
            raise ValueError(
                "trusted_campaign_scope.allowed_ports must be non-empty",
            )
        hosts = []
        for raw_host in raw_hosts:
            host = str(raw_host or "").strip().lower().rstrip(".")
            if not host or "*" in host or any(char.isspace() for char in host):
                raise ValueError(
                    "trusted_campaign_scope hosts must be exact hostnames",
                )
            hosts.append(host)
        ports = []
        for raw_port in raw_ports:
            if (
                not isinstance(raw_port, int)
                or isinstance(raw_port, bool)
                or not 1 <= raw_port <= 65535
            ):
                raise ValueError(
                    "trusted_campaign_scope ports must be integers",
                )
            ports.append(raw_port)
        return {
            "contract_hash": contract_hash,
            "allowed_hosts": sorted(set(hosts)),
            "allowed_ports": sorted(set(ports)),
            "allow_private_targets": (
                value.get("allow_private_targets") is True
            ),
        }

    def _record_key(self, kind: str, record_id: str) -> str:
        return "{}:{}".format(kind, record_id)

    def _load(self, kind: str, record_id: str) -> Optional[Dict[str, Any]]:
        return self._records.load(self._record_key(kind, record_id))

    def _save(self, kind: str, record_id: str, value: Dict[str, Any]) -> None:
        self._records.save(self._record_key(kind, record_id), value)

    def _ensure_blueprint_engine(self):
        if self._blueprint_engine is not None:
            return self._blueprint_engine
        from flyto_blueprint import BlueprintEngine
        from flyto_blueprint.storage.sqlite import SQLiteBackend

        db_path = self._state_dir / "verified-blueprints.db"
        self._blueprint_engine = BlueprintEngine(
            storage=SQLiteBackend(db_path=str(db_path)),
        )
        try:
            db_path.chmod(0o600)
        except OSError:
            pass
        return self._blueprint_engine

    async def handle(self, request: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Handle one JSON-RPC request."""
        method = request.get("method", "")
        req_id = request.get("id")
        params = request.get("params", {})
        era, protocol_error = request_protocol_era(req_id, method, params)
        if protocol_error is not None:
            return protocol_error
        modern = era == "modern"

        if method == "initialize" and not modern:
            client_version = (
                params.get("protocolVersion")
                if isinstance(params, dict)
                else None
            )
            return _protocol_result(req_id, {
                "protocolVersion": negotiate_legacy_protocol_version(
                    client_version,
                ),
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": _server_info(),
                "instructions": _SERVER_INSTRUCTIONS,
            }, modern=False)
        if method == "server/discover" and modern:
            return _protocol_result(
                req_id,
                {
                    "supportedVersions": list(SUPPORTED_PROTOCOL_VERSIONS),
                    "capabilities": {"tools": {"listChanged": False}},
                    "instructions": _SERVER_INSTRUCTIONS,
                },
                modern=True,
                ttl_ms=DISCOVERY_TTL_MS,
                cache_scope="private",
            )
        if modern and method in {"initialize", "ping", "logging/setLevel"}:
            return _make_error(
                req_id,
                -32601,
                "Method not found: {}".format(method),
            )
        if method.startswith("notifications/"):
            return None
        if method == "ping":
            return _protocol_result(req_id, {}, modern=False)
        if method == "tools/list":
            return _protocol_result(
                req_id,
                {"tools": TOOLS},
                modern=modern,
                ttl_ms=STATIC_LIST_TTL_MS,
                cache_scope="public",
            )
        if method == "tools/call":
            if not isinstance(params, dict):
                return _make_error(req_id, -32602, "params must be an object")
            name = str(params.get("name") or "")
            arguments = params.get("arguments", {})
            if name not in _TOOLS_BY_NAME:
                return _make_error(req_id, -32602, "Unknown tool: {}".format(name))
            if not isinstance(arguments, dict):
                return _make_error(req_id, -32602, "arguments must be an object")
            try:
                result = await self.call_tool(name, arguments)
            except Exception as exc:
                logger.exception("Closed-loop MCP tool failed")
                result = _tool_result(
                    {"ok": False, "error": str(exc) or type(exc).__name__},
                    message="Tool failed: {}".format(str(exc)[:200]),
                    is_error=True,
                )
            return _protocol_result(req_id, result, modern=modern)
        return _make_error(req_id, -32601, "Method not found: {}".format(method))

    async def call_tool(
        self,
        name: str,
        arguments: Dict[str, Any],
    ) -> Dict[str, Any]:
        if name == "plan":
            return await self._plan(arguments)
        if name == "execute":
            return await self._execute(arguments)
        if name == "verify":
            return await self._verify(arguments)
        return await self._get_evidence(arguments)

    async def _plan(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        steps = arguments.get("steps")
        if not isinstance(steps, list) or not steps:
            return _tool_result(
                {"ok": False, "error": "steps must be a non-empty array"},
                message="Plan rejected: steps are required",
                is_error=True,
            )
        if len(steps) > _MAX_PLAN_STEPS or _json_chars(steps) > _MAX_PLAN_CHARS:
            return _tool_result(
                {"ok": False, "error": "plan exceeds bounded MCP limits"},
                message="Plan rejected: size limit exceeded",
                is_error=True,
            )

        requested_blueprint_id = str(arguments.get("blueprint_id") or "")
        initial_id = requested_blueprint_id or "mcp_plan"
        plan_ir = PlanIR.compile(initial_id, steps)
        campaign = None
        campaign_errors: List[str] = []
        campaign_raw = arguments.get("security_campaign")
        if campaign_raw is not None:
            from flyto_ai.security.campaign import compile_security_campaign

            campaign = compile_security_campaign(campaign_raw, steps)
            campaign_errors = list(campaign.get("gate_errors") or [])

        identity = {
            "blueprint_id": requested_blueprint_id,
            "workflow_hash": plan_ir.workflow_hash,
        }
        if campaign is not None:
            identity["security_campaign_hash"] = campaign.get("contract_hash")
        digest = stable_hash(identity).split(":", 1)[1]
        plan_id = "plan_{}".format(digest[:20])
        blueprint_id = requested_blueprint_id or plan_id
        if blueprint_id != initial_id:
            plan_ir = PlanIR.compile(blueprint_id, steps)

        gate_errors = list(plan_ir.gate()) + campaign_errors
        candidates = _candidate_list(arguments.get("model_candidates"))
        route = self._router.route(
            str(arguments.get("message") or ""),
            candidates,
            deterministic_available=not gate_errors,
            plan_steps=len(plan_ir.steps),
        )
        record = {
            "version": MCP_CONTRACT_VERSION,
            "plan_id": plan_id,
            "blueprint_id": blueprint_id,
            "message": str(arguments.get("message") or "")[:2000],
            "plan_ir_version": plan_ir.version,
            "workflow_hash": plan_ir.workflow_hash,
            "steps": plan_ir.to_steps(),
            "gate_errors": gate_errors,
            "model_route": route.to_dict(),
            "module_call_counts": {},
            "security_campaign": campaign,
            "campaign_usage": (
                dict(campaign.get("initial_usage") or {})
                if campaign is not None
                else {}
            ),
        }
        self._save("plan", plan_id, record)

        compact = {
            "ok": not gate_errors,
            "contract_version": MCP_CONTRACT_VERSION,
            "plan_id": plan_id,
            "blueprint_id": blueprint_id,
            "plan_ir_version": plan_ir.version,
            "workflow_hash": plan_ir.workflow_hash,
            "step_count": len(plan_ir.steps),
            "gate": {
                "pass": not gate_errors,
                "error_count": len(gate_errors),
                "errors": gate_errors[:5],
            },
            "model_route": route.to_dict(),
        }
        if campaign is not None:
            compact["security_campaign"] = {
                "campaign_id": campaign.get("campaign_id"),
                "mode": campaign.get("mode"),
                "round": campaign.get("round"),
                "contract_hash": campaign.get("contract_hash"),
                "gate_passed": not campaign_errors,
            }
        return _tool_result(
            compact,
            message="Plan {}: {} step(s), gate {}".format(
                plan_id,
                len(plan_ir.steps),
                "passed" if not gate_errors else "failed",
            ),
            is_error=bool(gate_errors),
        )

    async def _execute(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        plan_id = str(arguments.get("plan_id") or "")
        plan = self._load("plan", plan_id)
        if not plan:
            return _tool_result(
                {"ok": False, "error": "Unknown plan_id"},
                message="Execution rejected: unknown plan ID",
                is_error=True,
            )
        if plan.get("gate_errors"):
            return _tool_result(
                {
                    "ok": False,
                    "error": "Plan gate failed",
                    "gate_errors": plan["gate_errors"][:5],
                },
                message="Execution rejected: PlanIR gate failed",
                is_error=True,
            )

        counts = dict(plan.get("module_call_counts") or {})
        campaign = plan.get("security_campaign")
        campaign_usage = dict(plan.get("campaign_usage") or {})
        trusted_scope = self._trusted_campaign_scope
        if trusted_scope is not None:
            campaign_authorization = (
                campaign.get("authorization")
                if isinstance(campaign, dict)
                else {}
            )
            scope_matches = bool(
                isinstance(campaign, dict)
                and not campaign.get("gate_errors")
                and trusted_scope.get("contract_hash")
                == campaign.get("contract_hash")
                and trusted_scope.get("allow_private_targets")
                is (campaign_authorization or {}).get(
                    "allow_private_targets",
                    False,
                )
            )
            if not scope_matches:
                return _tool_result(
                    {
                        "ok": False,
                        "error": "Trusted campaign scope does not match plan",
                    },
                    message="Execution rejected: campaign scope mismatch",
                    is_error=True,
                )

        async def preflight(func_args: Dict[str, Any]) -> Dict[str, Any]:
            decision = self._enforcer.check("execute_module", func_args)
            if not decision.allowed:
                return {
                    "ok": False,
                    "error": decision.reason,
                }
            if campaign is not None:
                from flyto_ai.security.campaign import evaluate_campaign_action

                campaign_decision = evaluate_campaign_action(
                    campaign,
                    campaign_usage,
                    "execute_module",
                    func_args,
                )
                if not campaign_decision["allowed"]:
                    return {
                        "ok": False,
                        "error": campaign_decision["reason"],
                    }
            return {
                "ok": True,
                "error": "",
            }

        async def dispatch(name: str, tool_args: Dict[str, Any]) -> Dict[str, Any]:
            nonlocal campaign_usage
            decision = self._enforcer.check(name, tool_args)
            if not decision.allowed:
                return {"ok": False, "error": decision.reason}
            if campaign is not None:
                from flyto_ai.security.campaign import evaluate_campaign_action

                campaign_decision = evaluate_campaign_action(
                    campaign,
                    campaign_usage,
                    name,
                    tool_args,
                )
                if not campaign_decision["allowed"]:
                    return {
                        "ok": False,
                        "error": campaign_decision["reason"],
                    }
            if name == "report_blueprint_outcome":
                from flyto_ai.tools.blueprint_tools import dispatch_blueprint_tool
                validation = await dispatch_blueprint_tool("_validate_closed_loop_evidence", tool_args)
                if validation.get("ok") is not True:
                    return validation
                self._save("outcome", validation["execution_id"], {
                    "version": MCP_CONTRACT_VERSION, "plan_id": plan_id,
                    "blueprint_id": validation["blueprint_id"], "execution_id": validation["execution_id"], "success": tool_args.get("success"),
                })
                return {**validation, "recorded": True}
            if name == "execute_module":
                module_id = str(tool_args.get("module_id") or "")
                counts[module_id] = counts.get(module_id, 0) + 1
                if (
                    module_id
                    and module_id == self._test_fail_once_module
                    and self._load("test-failure", module_id) is None
                ):
                    self._save("test-failure", module_id, {"consumed": True})
                    return {
                        "ok": False,
                        "error": "intentional MCP checkpoint test interruption",
                    }
            if name == "execute_module" and trusted_scope is not None:
                core_result = await dispatch_core_tool(
                    name,
                    tool_args,
                    trusted_outbound_scope=trusted_scope,
                )
            else:
                core_result = await dispatch_core_tool(name, tool_args)
            if campaign is not None:
                from flyto_ai.security.campaign import record_campaign_result

                campaign_usage = record_campaign_result(
                    campaign,
                    campaign_usage,
                    name,
                    tool_args,
                    core_result,
                )
            return core_result

        max_repairs = _clamped_int(
            arguments.get("max_repairs"),
            default=self._max_repairs,
            minimum=0,
            maximum=self._max_repairs,
        )
        result = await execute_blueprint_loop(
            blueprint_id=str(plan["blueprint_id"]),
            steps=plan["steps"],
            dispatch=dispatch,
            preflight=preflight,
            checkpoint_store=self._checkpoints,
            max_repairs=max_repairs,
        )
        execution_id = str(result.get("execution_id") or "")
        evidence_id = "evidence_{}".format(execution_id)
        evidence = {
            "version": MCP_CONTRACT_VERSION,
            "evidence_id": evidence_id,
            "plan_id": plan_id,
            "execution_id": execution_id,
            "module_call_counts": counts,
            "security_campaign": campaign,
            "campaign_usage": campaign_usage,
            "result": result,
        }
        self._save("evidence", execution_id, evidence)
        plan["last_execution_id"] = execution_id
        plan["module_call_counts"] = counts
        plan["campaign_usage"] = campaign_usage
        self._save("plan", plan_id, plan)

        compact = self._execution_summary(evidence)
        full_chars = _json_chars(evidence)
        compact_chars = _json_chars(compact)
        compact["token_profile"] = {
            "compact_chars": compact_chars,
            "full_evidence_chars": full_chars,
            "compact_estimated_tokens": max(1, compact_chars // 4),
            "full_estimated_tokens": max(1, full_chars // 4),
            "reduction_percent": round(
                max(0.0, 1.0 - (compact_chars / max(1, full_chars))) * 100,
                1,
            ),
        }
        return _tool_result(
            compact,
            message="Execution {}: {}".format(
                execution_id,
                "verified runtime success"
                if result.get("closed_loop_ok")
                else "failed; checkpoint retained when possible",
            ),
            is_error=not bool(result.get("closed_loop_ok")),
        )

    def _execution_summary(self, evidence: Dict[str, Any]) -> Dict[str, Any]:
        result = evidence.get("result", {})
        runtime = result.get("evidence", {})
        summary = {
            "ok": bool(result.get("ok")),
            "closed_loop_ok": bool(result.get("closed_loop_ok")),
            "plan_id": evidence.get("plan_id"),
            "execution_id": evidence.get("execution_id"),
            "evidence_id": evidence.get("evidence_id"),
            "executor_version": runtime.get("executor_version"),
            "plan_ir_version": runtime.get("plan_ir_version"),
            "step_count": runtime.get("step_count", 0),
            "passed_steps": runtime.get("passed_steps", 0),
            "failed_step_id": runtime.get("failed_step_id"),
            "failed_phase": runtime.get("failed_phase"),
            "assertion_passed": runtime.get("assertion_passed"),
            "outcome_reported": bool(result.get("outcome_reported")),
            "checkpoint_loaded": bool(runtime.get("checkpoint_loaded")),
            "checkpoint_cleared": bool(runtime.get("checkpoint_cleared")),
            "resumed_step_ids": runtime.get("resumed_step_ids", []),
            "repair_count": runtime.get("repair_count", 0),
            "module_call_counts": evidence.get("module_call_counts", {}),
        }
        campaign = evidence.get("security_campaign")
        if isinstance(campaign, dict):
            usage = evidence.get("campaign_usage") or {}
            summary["security_campaign"] = {
                "campaign_id": campaign.get("campaign_id"),
                "mode": campaign.get("mode"),
                "round": campaign.get("round"),
                "requests_used": usage.get("requests_used", 0),
                "cost_units_used": usage.get("cost_units_used", 0),
                "evidence_count": usage.get(
                    "evidence_count",
                    len(usage.get("evidence") or []),
                ),
            }
        return summary

    async def _verify(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        execution_id = str(arguments.get("execution_id") or "")
        plan_id = str(arguments.get("plan_id") or "")
        if not execution_id and plan_id:
            plan = self._load("plan", plan_id)
            if not plan:
                return _tool_result(
                    {"ok": False, "error": "Unknown plan_id"},
                    message="Verification rejected: plan not found",
                    is_error=True,
                )
            execution_id = str(plan.get("last_execution_id") or "")
            if not execution_id:
                return _tool_result(
                    {"ok": False, "error": "Plan has no execution evidence"},
                    message="Verification rejected: plan has not been executed",
                    is_error=True,
                )
        if not execution_id:
            return _tool_result(
                {"ok": False, "error": "execution_id or plan_id is required"},
                message="Verification rejected: execution ID is required",
                is_error=True,
            )

        cached = self._load("verification", execution_id)
        if cached:
            return _tool_result(
                cached,
                message="Verification {} loaded from cache".format(execution_id),
                is_error=not bool(cached.get("verified")),
            )
        evidence = self._load("evidence", execution_id)
        if not evidence:
            return _tool_result(
                {"ok": False, "error": "Unknown execution_id"},
                message="Verification rejected: evidence not found",
                is_error=True,
            )

        result = evidence.get("result", {})
        runtime = result.get("evidence", {})
        checks = {
            "closed_loop": bool(result.get("closed_loop_ok")),
            "plan_gate": runtime.get("plan_gate_passed") is True,
            "validation": bool(runtime.get("validation_passed")),
            "assertions": runtime.get("assertion_passed") is not False,
            "outcome": bool(result.get("outcome_reported")),
            "checkpoint_finalized": bool(runtime.get("checkpoint_cleared")),
        }
        campaign_verification = None
        campaign = evidence.get("security_campaign")
        if isinstance(campaign, dict):
            from flyto_ai.security.campaign import verify_security_campaign

            campaign_verification = verify_security_campaign(
                campaign,
                evidence.get("campaign_usage") or {},
                result,
            )
            checks["security_campaign"] = bool(
                campaign_verification.get("verified"),
            )
        verified = all(checks.values())
        min_steps = _clamped_int(
            arguments.get("min_steps"),
            default=3,
            minimum=3,
            maximum=20,
        )
        decision = evaluate_distillation(
            [],
            result.get("executions", []),
            str((self._load("plan", evidence["plan_id"]) or {}).get("message") or ""),
            min_steps=min_steps,
        )
        distillation: Dict[str, Any] = {
            "eligible": bool(verified and decision.eligible),
            "reason": decision.reason,
            "evidence_count": decision.evidence_count,
        }
        if verified and decision.eligible and decision.workflow:
            learning_ok, learning_state = verified_learning_distillation(
                self._ensure_blueprint_engine(), decision.workflow, evidence,
                checks, decision.evidence_count, PLAN_IR_VERSION,
            )
            distillation.update(learning_state)
            verified = verified and learning_ok

        verification = {
            "ok": verified,
            "verified": verified,
            "contract_version": MCP_CONTRACT_VERSION,
            "execution_id": execution_id,
            "plan_id": evidence.get("plan_id"),
            "checks": checks,
            "distillation": distillation,
        }
        if campaign_verification is not None:
            verification["security_campaign"] = campaign_verification
        self._save("verification", execution_id, verification)
        return _tool_result(
            verification,
            message="Verification {}: {}".format(
                execution_id,
                "passed" if verified else "failed",
            ),
            is_error=not verified,
        )

    async def _get_evidence(
        self,
        arguments: Dict[str, Any],
    ) -> Dict[str, Any]:
        execution_id = str(arguments.get("execution_id") or "")
        evidence = self._load("evidence", execution_id)
        if not evidence:
            return _tool_result(
                {"ok": False, "error": "Unknown execution_id"},
                message="Evidence not found",
                is_error=True,
            )
        section = str(arguments.get("section") or "summary")
        if section == "summary":
            data = self._execution_summary(evidence)
        elif section == "executions":
            offset = _clamped_int(
                arguments.get("offset"),
                default=0,
                minimum=0,
                maximum=10_000,
            )
            limit = _clamped_int(
                arguments.get("limit"),
                default=10,
                minimum=1,
                maximum=20,
            )
            executions = evidence.get("result", {}).get("executions", [])
            data = {
                "ok": True,
                "execution_id": execution_id,
                "offset": offset,
                "next_offset": (
                    offset + limit
                    if offset + limit < len(executions)
                    else None
                ),
                "total": len(executions),
                "executions": executions[offset:offset + limit],
            }
        elif section == "raw":
            raw = json.dumps(
                evidence,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                default=str,
            )
            offset = _clamped_int(
                arguments.get("offset"),
                default=0,
                minimum=0,
                maximum=len(raw),
            )
            limit = _clamped_int(
                arguments.get("limit"),
                default=4000,
                minimum=1,
                maximum=_MAX_EVIDENCE_CHARS,
            )
            end = min(len(raw), offset + limit)
            data = {
                "ok": True,
                "execution_id": execution_id,
                "offset": offset,
                "next_offset": end if end < len(raw) else None,
                "total_chars": len(raw),
                "chunk": raw[offset:end],
            }
        else:
            return _tool_result(
                {"ok": False, "error": "Unknown evidence section"},
                message="Evidence request rejected: unknown section",
                is_error=True,
            )
        return _tool_result(
            data,
            message="Evidence {} section {}".format(execution_id, section),
        )


async def async_main() -> None:
    """Run the newline-delimited JSON-RPC STDIO loop."""
    server = ClosedLoopMCPServer()
    reader = asyncio.StreamReader()
    protocol = asyncio.StreamReaderProtocol(reader)
    await asyncio.get_event_loop().connect_read_pipe(lambda: protocol, sys.stdin)
    write_transport, write_protocol = (
        await asyncio.get_event_loop().connect_write_pipe(
            asyncio.streams.FlowControlMixin,
            sys.stdout,
        )
    )
    writer = asyncio.StreamWriter(
        write_transport,
        write_protocol,
        reader,
        asyncio.get_event_loop(),
    )

    while True:
        line = await reader.readline()
        if not line:
            break
        try:
            request = json.loads(
                line.decode("utf-8", errors="replace").strip(),
            )
            response = await server.handle(request)
        except json.JSONDecodeError:
            response = _make_error(None, -32700, "Parse error")
        except Exception as exc:
            logger.exception("Closed-loop MCP handler failed")
            response = _make_error(None, -32603, str(exc))
        if response is not None:
            writer.write((
                json.dumps(response, ensure_ascii=False, default=str) + "\n"
            ).encode("utf-8"))
            await writer.drain()


def main() -> None:
    """Entry point for ``flyto-closed-loop-mcp``."""
    logging.basicConfig(level=logging.WARNING, stream=sys.stderr)
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
