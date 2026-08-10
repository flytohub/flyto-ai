# Copyright 2024 Flyto2
# Licensed under the Apache License, Version 2.0
"""Detachable built-in capability presets for the Flyto2 agent stack."""
from __future__ import annotations

import sys
from typing import Dict, Iterable, Sequence, Tuple

from flyto_ai.coding.contracts import CapabilitySpec


DEFAULT_COMPONENTS = (
    "flyto-indexer",
    "flyto-blueprint",
    "flyto-page-inspector",
    "flyto-core",
)

# Strict verification can rebuild a large workspace index and legitimately run
# longer than the interactive MCP default. Keep every host-owned Indexer entry
# on one bound so the public CLI cannot silently drift from the stack preset.
INDEXER_CAPABILITY_TIMEOUT_SECONDS = 60


def _preset_factories(python: str, required: set[str]) -> Dict[str, CapabilitySpec]:
    """Build the complete preset catalog before applying caller selection."""
    return {
        "flyto-indexer": CapabilitySpec(
            name="flyto-indexer",
            argv=(python, "-m", "flyto_indexer.mcp_server"),
            required="flyto-indexer" in required,
            contract_version="flyto-indexer.mcp.v1",
            required_tools=("search", "impact", "call_hierarchy", "structure", "task", "verify"),
            allowed_tools=("search", "impact", "call_hierarchy", "structure", "task", "verify"),
            tool_permissions=(
                ("call_hierarchy", "read_only"),
                ("impact", "read_only"),
                ("search", "read_only"),
                ("structure", "read_only"),
                ("task", "workspace_write"),
                ("verify", "workspace_write"),
            ),
            timeout_seconds=INDEXER_CAPABILITY_TIMEOUT_SECONDS,
        ),
        "flyto-blueprint": CapabilitySpec(
            name="flyto-blueprint",
            argv=(python, "-m", "flyto_ai.mcp_server"),
            required="flyto-blueprint" in required,
            contract_version="flyto-blueprint.mcp.v1",
            required_tools=(
                "list_blueprints", "use_blueprint", "save_as_blueprint",
                "report_blueprint_outcome", "export_blueprint", "import_blueprint",
            ),
            allowed_tools=(
                "list_blueprints", "use_blueprint", "save_as_blueprint",
                "report_blueprint_outcome", "export_blueprint", "import_blueprint",
            ),
            tool_permissions=(
                ("export_blueprint", "workspace_write"),
                ("import_blueprint", "workspace_write"),
                ("list_blueprints", "read_only"),
                ("report_blueprint_outcome", "workspace_write"),
                ("save_as_blueprint", "workspace_write"),
                ("use_blueprint", "workspace_write"),
            ),
            timeout_seconds=30,
        ),
        "flyto-page-inspector": CapabilitySpec(
            name="flyto-page-inspector",
            argv=(python, "-m", "flyto_ai.mcp_server"),
            required="flyto-page-inspector" in required,
            contract_version="flyto-page-inspector.mcp.v1",
            required_tools=("inspect_page",),
            allowed_tools=("inspect_page",),
            tool_permissions=(("inspect_page", "read_only"),),
            timeout_seconds=30,
        ),
        "flyto-core": CapabilitySpec(
            name="flyto-core",
            argv=(python, "-m", "core.mcp_server"),
            required="flyto-core" in required,
            contract_version="flyto-core.mcp.v1",
            required_tools=(
                "list_modules", "search_modules", "get_module_info",
                "get_module_examples", "validate_params", "execute_module",
                "list_recipes", "run_recipe",
            ),
            allowed_tools=(
                "list_modules", "search_modules", "get_module_info",
                "get_module_examples", "validate_params", "execute_module",
                "list_recipes", "run_recipe",
            ),
            tool_permissions=(
                ("execute_module", "workspace_write"),
                ("get_module_examples", "read_only"),
                ("get_module_info", "read_only"),
                ("list_modules", "read_only"),
                ("list_recipes", "read_only"),
                ("run_recipe", "danger_full"),
                ("search_modules", "read_only"),
                ("validate_params", "read_only"),
            ),
            timeout_seconds=30,
        ),
    }


def build_agent_stack_capabilities(
    components: Sequence[str] = DEFAULT_COMPONENTS,
    *,
    required_components: Iterable[str] | None = None,
    python_executable: str | None = None,
) -> Tuple[CapabilitySpec, ...]:
    """Build independently detachable, least-privilege MCP capability specs."""
    selected = tuple(components)
    if len(set(selected)) != len(selected):
        raise ValueError("agent stack components contain duplicates")
    required = set(selected if required_components is None else required_components)
    unknown_required = required - set(selected)
    if unknown_required:
        raise ValueError(
            "required agent stack components were not selected: {}".format(
                ", ".join(sorted(unknown_required)),
            )
        )
    factories = _preset_factories(python_executable or sys.executable, required)
    unknown = set(selected) - set(factories)
    if unknown:
        raise ValueError("unknown agent stack components: {}".format(", ".join(sorted(unknown))))
    return tuple(factories[name] for name in selected)
