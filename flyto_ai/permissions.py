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
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class PermissionLevel(IntEnum):
    """Permission tiers — higher value grants more access."""
    READ_ONLY = 0           # list/search/get (discovery only)
    WORKSPACE_WRITE = 1     # execute safe modules, use blueprints
    DANGER_FULL = 2         # shell, docker, k8s, file write, unconstrained


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


def _required_level_for_module(module_id: str) -> PermissionLevel:
    """Determine the permission level required for a specific module."""
    category = module_id.split(".")[0] if "." in module_id else module_id
    if category in DANGER_MODULE_CATEGORIES:
        return PermissionLevel.DANGER_FULL
    return PermissionLevel.WORKSPACE_WRITE


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
    """

    def __init__(
        self,
        level: PermissionLevel = PermissionLevel.WORKSPACE_WRITE,
        overrides: Optional[Dict[str, PermissionLevel]] = None,
    ) -> None:
        self._level = level
        self._overrides = overrides or {}

    @property
    def level(self) -> PermissionLevel:
        return self._level

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
            module_level = _required_level_for_module(
                str(arguments.get("module_id", "")),
            )
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
