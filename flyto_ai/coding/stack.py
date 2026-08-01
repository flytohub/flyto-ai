# Copyright 2024 Flyto2
# Licensed under the Apache License, Version 2.0
"""Composable preflight for the full Flyto2 native agent stack."""
from __future__ import annotations

import argparse
import asyncio
import dataclasses
import hashlib
import json
import sys
from pathlib import Path
from typing import Dict, Iterable, Sequence, Tuple

from flyto_ai.coding.capabilities import CapabilityManager
from flyto_ai.coding.contracts import CapabilitySpec


AGENT_STACK_CONTRACT_VERSION = "flyto.agent-stack.v1"
DEFAULT_COMPONENTS = (
    "flyto-indexer",
    "flyto-blueprint",
    "flyto-page-inspector",
    "flyto-core",
)


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

    python = python_executable or sys.executable
    factories: Dict[str, CapabilitySpec] = {
        "flyto-indexer": CapabilitySpec(
            name="flyto-indexer",
            argv=(python, "-m", "flyto_indexer.mcp_server"),
            required="flyto-indexer" in required,
            contract_version="flyto-indexer.mcp.v1",
            required_tools=("search", "impact", "call_hierarchy", "structure", "task", "verify"),
            allowed_tools=("search", "impact", "call_hierarchy", "structure", "task", "verify"),
            timeout_seconds=30,
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
            timeout_seconds=30,
        ),
        "flyto-page-inspector": CapabilitySpec(
            name="flyto-page-inspector",
            argv=(python, "-m", "flyto_ai.mcp_server"),
            required="flyto-page-inspector" in required,
            contract_version="flyto-page-inspector.mcp.v1",
            required_tools=("inspect_page",),
            allowed_tools=("inspect_page",),
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
            timeout_seconds=30,
        ),
    }
    unknown = set(selected) - set(factories)
    if unknown:
        raise ValueError("unknown agent stack components: {}".format(", ".join(sorted(unknown))))
    return tuple(factories[name] for name in selected)


async def probe_agent_stack(
    workspace: str,
    components: Sequence[str] = DEFAULT_COMPONENTS,
    *,
    required_components: Iterable[str] | None = None,
) -> dict:
    """Negotiate every selected server and attest its actually exposed tools."""
    root = str(Path(workspace).expanduser().resolve(strict=True))
    specs = build_agent_stack_capabilities(
        components,
        required_components=required_components,
    )
    manager = CapabilityManager(root)
    try:
        statuses = await manager.start(specs)
    finally:
        await manager.close()

    projected = [dataclasses.asdict(status) for status in statuses]
    identity = [
        {
            "name": item["name"],
            "available": item["available"],
            "required": item["required"],
            "server_name": item["server_name"],
            "protocol": item["negotiated_protocol_version"],
            "tools": item["tools"],
        }
        for item in projected
    ]
    fingerprint = hashlib.sha256(
        json.dumps(identity, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(),
    ).hexdigest()
    return {
        "contract_version": AGENT_STACK_CONTRACT_VERSION,
        "ok": all(item["available"] for item in projected if item["required"]),
        "workspace": root,
        "composition_fingerprint": fingerprint,
        "components": projected,
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Run the credential-free full-stack preflight."""
    parser = argparse.ArgumentParser(description="Probe the composable Flyto2 agent stack")
    parser.add_argument("--workspace", default=".", help="Workspace used as the MCP process cwd")
    parser.add_argument(
        "--components",
        nargs="+",
        choices=DEFAULT_COMPONENTS,
        default=list(DEFAULT_COMPONENTS),
        help="Components to attach; omitted components remain detached",
    )
    parser.add_argument("--json", action="store_true", help="Emit the full evidence object")
    args = parser.parse_args(argv)
    result = asyncio.run(
        probe_agent_stack(
            args.workspace,
            args.components,
            required_components=args.components,
        )
    )
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        for component in result["components"]:
            state = "ready" if component["available"] else "unavailable"
            print("{}: {} ({} tools)".format(component["name"], state, component["tool_count"]))
        print("stack: {}".format("ready" if result["ok"] else "blocked"))
        print("fingerprint: {}".format(result["composition_fingerprint"]))
    return 0 if result["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
