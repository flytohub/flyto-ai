# Copyright 2024 Flyto2
# Licensed under the Apache License, Version 2.0
"""Three-tier permission model — runtime enforcement for tool and module access.

Inspired by claw-code's ``PermissionLevel::ReadOnly | WorkspaceWrite | DangerFullAccess``
pattern with ``PermissionEnforcer`` checked at every tool dispatch.

Usage::

    enforcer = PermissionEnforcer(PermissionLevel.WORKSPACE_WRITE)
    decision = enforcer.check("execute_module", {"module_id": "shell.run"})
    if not decision.allowed:
        return {"ok": False, "error": decision.reason}
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum, IntEnum
from pathlib import Path
from typing import Any, Callable, Dict, FrozenSet, Mapping, Optional

from flyto_ai.workspace_permissions import is_workspace_file_call

logger = logging.getLogger(__name__)


class PermissionLevel(IntEnum):
    """Permission tiers — higher value grants more access."""
    READ_ONLY = 0           # list/search/get (discovery only)
    WORKSPACE_WRITE = 1     # execute safe modules, use blueprints
    DANGER_FULL = 2         # shell, docker, k8s, unbounded filesystem access


class PermissionOutcome(str, Enum):
    """Machine-readable policy result for UI, audit, and evaluations."""

    ALLOW = "allow"
    REQUIRE_CONFIRMATION = "require_confirmation"
    BLOCK = "block"


@dataclass(frozen=True)
class PermissionDecision:
    """Result of a permission check."""
    allowed: bool
    reason: str = ""
    outcome: PermissionOutcome = PermissionOutcome.ALLOW


# ── Per-tool permission requirements ──────────────────────────────────

TOOL_PERMISSION_MAP: Dict[str, PermissionLevel] = {
    # Discovery — READ_ONLY
    "list_modules": PermissionLevel.READ_ONLY,
    "search_modules": PermissionLevel.READ_ONLY,
    "get_module_info": PermissionLevel.READ_ONLY,
    "get_module_examples": PermissionLevel.READ_ONLY,
    "get_core_capability_manifest": PermissionLevel.READ_ONLY,
    "list_recipes": PermissionLevel.READ_ONLY,
    "list_blueprints": PermissionLevel.READ_ONLY,
    "inspect_page": PermissionLevel.READ_ONLY,
    "validate_params": PermissionLevel.READ_ONLY,
    # Workspace — WORKSPACE_WRITE
    "execute_module": PermissionLevel.WORKSPACE_WRITE,
    "run_recipe": PermissionLevel.WORKSPACE_WRITE,
    "use_blueprint": PermissionLevel.WORKSPACE_WRITE,
    "save_as_blueprint": PermissionLevel.WORKSPACE_WRITE,
    "report_blueprint_outcome": PermissionLevel.WORKSPACE_WRITE,
    "navigate_website": PermissionLevel.WORKSPACE_WRITE,
    "ask_user": PermissionLevel.READ_ONLY,
}

# Module categories that require DANGER_FULL
DANGER_MODULE_CATEGORIES = frozenset({
    "shell", "process", "docker", "k8s",
    "ssh", "network", "port", "dns",
    "file", "path", "env",
    "git",
})


def _required_level_for_module(
    module_id: str, arguments: Optional[Dict[str, Any]] = None,
    workspace_root: Optional[Path] = None,
) -> PermissionLevel:
    """Determine the permission level required for a specific module."""
    if workspace_root is not None and is_workspace_file_call(
        module_id, arguments or {}, workspace_root,
    ):
        return PermissionLevel.WORKSPACE_WRITE
    category = module_id.split(".")[0] if "." in module_id else module_id
    if category in DANGER_MODULE_CATEGORIES:
        return PermissionLevel.DANGER_FULL
    return PermissionLevel.WORKSPACE_WRITE


# ── Capability-contract grading ───────────────────────────────────────
#
# A module registered with a ``flyto.capability-contract.v1`` contract says
# what it does to the world as data. That declaration -- not the module's
# category prefix -- decides its grade, so a provider package needs no entry in
# any host table to be graded correctly. The scale mirrors the five
# consequence levels hosts already use: 1 observe, 2 reversible, 3 external
# write, 4 real-world effect, 5 irreversible or high-impact. Lowering a level
# for a simulated deployment is the host's job (it knows the deployment mode);
# this grade is always the real-world one.

RISK_OBSERVE = 1
RISK_REVERSIBLE = 2
RISK_EXTERNAL_WRITE = 3
RISK_REAL_WORLD = 4
RISK_IRREVERSIBLE = 5

_CONTRACT_SAFETY_RISK = {
    "read_only": RISK_OBSERVE,
    "controlled": RISK_EXTERNAL_WRITE,
    "movement": RISK_REAL_WORLD,
    "dangerous": RISK_IRREVERSIBLE,
}

_CONTRACT_REQUIRED_KEYS = frozenset({
    "actuates", "safety_class", "requires_safe_stop", "cancellable",
})

#: Capabilities that ARE the stop. Their risk is in not running them, so they
#: are never graded into a confirmation and a host keeps them on its immediate
#: path (never a proposal or approval queue). The name alone is not enough:
#: the contract must also describe a stop -- not movement or dangerous, no
#: safe stop of its own, not cancellable -- so a package cannot borrow the id
#: to slip an actuating capability past confirmation.
IMMEDIATE_STOP_CAPABILITIES: FrozenSet[str] = frozenset({"motion.halt"})


@dataclass(frozen=True)
class ContractGrade:
    """What one module's capability contract (or its absence) costs.

    ``permission_level`` is the flyto-ai tier; ``risk_level`` is the 1-5
    consequence level; ``actuating`` is True when the module may change the
    world (always True for a fail-closed grade); ``immediate`` marks a stop
    capability that must bypass proposals and approvals; ``source`` is one of
    ``contract``, ``immediate_stop``, ``missing_contract`` or
    ``invalid_contract``.
    """

    permission_level: PermissionLevel
    risk_level: int
    actuating: bool
    immediate: bool
    source: str
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "permission_level": self.permission_level.name,
            "risk_level": self.risk_level,
            "actuating": self.actuating,
            "immediate": self.immediate,
            "source": self.source,
            "reason": self.reason,
        }


def _fail_closed(source: str, reason: str) -> ContractGrade:
    return ContractGrade(
        permission_level=PermissionLevel.DANGER_FULL,
        risk_level=RISK_REAL_WORLD,
        actuating=True,
        immediate=False,
        source=source,
        reason=reason,
    )


def _is_stop_contract(contract: Mapping[str, Any]) -> bool:
    return (
        contract.get("safety_class") in ("read_only", "controlled")
        and contract.get("requires_safe_stop") is False
        and contract.get("cancellable") is False
    )


def grade_module_contract(
    module_info: Optional[Mapping[str, Any]],
    *,
    immediate_capabilities: FrozenSet[str] = IMMEDIATE_STOP_CAPABILITIES,
) -> Optional[ContractGrade]:
    """Grade one module from its catalog detail (``get_module_info`` shape).

    Reads ``plugin``, ``provides_capability`` and ``contract``. Pure: no Core
    import, so a host that only holds reported data can call it.

    Returns None only for a module flyto-core itself registered (empty
    ``plugin``) that declares no contract -- legacy grading applies there.
    A module from a non-core package without a valid contract fails closed:
    actuating, real-world, ``DANGER_FULL``. An actuating contract, or one whose
    ``safety_class`` is ``movement`` / ``dangerous``, is ``DANGER_FULL`` with
    risk 4 / 5 and is never ``WORKSPACE_WRITE``.
    """
    if not isinstance(module_info, Mapping):
        return None
    plugin = module_info.get("plugin")
    plugin = plugin.strip() if isinstance(plugin, str) else ""
    capability = module_info.get("provides_capability")
    capability = capability if isinstance(capability, str) else ""
    contract = module_info.get("contract")

    if contract is None:
        if not plugin:
            return None
        return _fail_closed(
            "missing_contract",
            "module from package '{}' declares no capability contract; "
            "treated as actuating".format(plugin),
        )
    if (
        not isinstance(contract, Mapping)
        or not _CONTRACT_REQUIRED_KEYS.issubset(contract)
        or contract.get("safety_class") not in _CONTRACT_SAFETY_RISK
        or not isinstance(contract.get("actuates"), bool)
    ):
        return _fail_closed(
            "invalid_contract", "capability contract is unreadable; treated as actuating",
        )

    safety_class = contract["safety_class"]
    actuates = contract["actuates"]
    if capability in immediate_capabilities and _is_stop_contract(contract):
        return ContractGrade(
            permission_level=PermissionLevel.READ_ONLY,
            risk_level=RISK_OBSERVE,
            actuating=actuates,
            immediate=True,
            source="immediate_stop",
            reason="{} is a stop; its risk is in not running it".format(capability),
        )

    risk = _CONTRACT_SAFETY_RISK[safety_class]
    if actuates and risk < RISK_REAL_WORLD:
        risk = RISK_REAL_WORLD
    if risk >= RISK_REAL_WORLD:
        level = PermissionLevel.DANGER_FULL
    elif risk == RISK_OBSERVE:
        level = PermissionLevel.READ_ONLY
    else:
        level = PermissionLevel.WORKSPACE_WRITE
    return ContractGrade(
        permission_level=level,
        risk_level=risk,
        actuating=bool(actuates or risk >= RISK_REAL_WORLD),
        immediate=False,
        source="contract",
        reason="contract safety_class={} actuates={}".format(
            safety_class, str(actuates).lower(),
        ),
    )


ModuleInfoResolver = Callable[[str], Optional[Mapping[str, Any]]]


def core_module_info(module_id: str) -> Optional[Mapping[str, Any]]:
    """Catalog detail for ``module_id`` from the installed flyto-core.

    Returns None when Core is absent or does not know the module. Imported
    lazily so this module stays importable on hosts without Core.
    """
    from flyto_ai.tools.core_tools import _get_mcp_handler

    handler = _get_mcp_handler()
    if not handler or not callable(handler.get("get_module_info")):
        return None
    detail = handler["get_module_info"](module_id=module_id)
    if not isinstance(detail, Mapping) or detail.get("error"):
        return None
    return detail


def resolve_module_grade(
    module_id: str,
    resolver: Optional[ModuleInfoResolver] = None,
) -> Optional[ContractGrade]:
    """Grade ``module_id`` by looking its contract up through ``resolver``.

    ``resolver`` defaults to :func:`core_module_info`. A lookup that raises is
    graded fail-closed rather than silently falling back to category rules.
    """
    if not module_id:
        return None
    lookup = resolver or core_module_info
    try:
        info = lookup(module_id)
    except Exception as exc:
        logger.warning("Contract lookup for %s failed: %s", module_id, exc)
        return _fail_closed(
            "invalid_contract", "contract lookup failed; treated as actuating",
        )
    return grade_module_contract(info)


class PermissionEnforcer:
    """Runtime permission gate — checked at tool dispatch time.

    Wraps around the existing policy enforcement (``is_tool_allowed`` / ``is_module_allowed``)
    and adds tier-based access control.

    Parameters
    ----------
    level : PermissionLevel
        The maximum permission level for this session.
    overrides : dict, optional
        Per-tool overrides: ``{"tool_name": PermissionLevel}``.
    module_info_resolver : callable, optional
        ``module_id -> catalog detail``, used to read a module's capability
        contract for ``execute_module``. Defaults to the installed flyto-core.
    """

    def __init__(
        self,
        level: PermissionLevel = PermissionLevel.WORKSPACE_WRITE,
        overrides: Optional[Dict[str, PermissionLevel]] = None,
        module_info_resolver: Optional[ModuleInfoResolver] = None,
    ) -> None:
        self._level = level
        self._overrides = overrides or {}
        self._workspace_root = Path.cwd().resolve()
        self._module_info_resolver = module_info_resolver

    @property
    def level(self) -> PermissionLevel:
        return self._level

    @property
    def workspace_root(self) -> Path:
        """Host working directory captured when the session policy is created."""
        return self._workspace_root

    def required_level(
        self,
        tool_name: str,
        arguments: Optional[Dict[str, Any]] = None,
    ) -> PermissionLevel:
        """Return the effective permission requirement for an exact call."""
        arguments = arguments or {}
        required = self._overrides.get(tool_name)
        if required is None:
            required = TOOL_PERMISSION_MAP.get(
                tool_name, PermissionLevel.WORKSPACE_WRITE,
            )

        if tool_name == "execute_module":
            module_id = str(arguments.get("module_id", ""))
            module_level = _required_level_for_module(
                module_id, arguments, self._workspace_root,
            )
            # The contract can only raise the grade. An actuating module is
            # DANGER_FULL even when its category or a workspace path would
            # have read as WORKSPACE_WRITE.
            grade = resolve_module_grade(module_id, self._module_info_resolver)
            if grade is not None and grade.permission_level > module_level:
                module_level = grade.permission_level
            if module_level > required:
                required = module_level
        return required

    def check(self, tool_name: str, arguments: Dict[str, Any] = None) -> PermissionDecision:
        """Check whether the current session level allows this tool call.

        Returns ``PermissionDecision(allowed=True)`` if permitted, otherwise
        a decision with ``allowed=False`` and a human-readable ``reason``.
        """
        arguments = arguments or {}
        required = self.required_level(tool_name, arguments)

        if self._level >= required:
            return PermissionDecision(
                allowed=True, outcome=PermissionOutcome.ALLOW,
            )

        return PermissionDecision(
            allowed=False,
            reason="Permission denied: '{}' requires {} but session is {}".format(
                tool_name, required.name, self._level.name,
            ),
            outcome=(
                PermissionOutcome.REQUIRE_CONFIRMATION
                if required == PermissionLevel.DANGER_FULL
                else PermissionOutcome.BLOCK
            ),
        )

    def check_route(
        self,
        tool_name: str,
        arguments: Optional[Dict[str, Any]],
        route_mode: str,
    ) -> PermissionDecision:
        """Apply conversation intent before the regular permission tier.

        Tool metadata is not trusted for authorization: the effective
        requirement is calculated locally from the exact tool and arguments.
        """
        if route_mode == "answer_only":
            return PermissionDecision(
                allowed=False,
                reason="Tool blocked: this turn is an answer-only conversation.",
                outcome=PermissionOutcome.BLOCK,
            )

        if route_mode == "ambiguous":
            required = self.required_level(tool_name, arguments)
            if required > PermissionLevel.READ_ONLY:
                return PermissionDecision(
                    allowed=False,
                    reason=(
                        "Confirmation required: the user did not make an "
                        "explicit action request."
                    ),
                    outcome=PermissionOutcome.REQUIRE_CONFIRMATION,
                )

        if route_mode not in {"ambiguous", "action"}:
            return PermissionDecision(
                allowed=False,
                reason="Tool blocked: unknown conversation route.",
                outcome=PermissionOutcome.BLOCK,
            )

        return self.check(tool_name, arguments)
