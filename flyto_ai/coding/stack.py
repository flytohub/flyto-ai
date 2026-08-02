# Copyright 2024 Flyto2
# Licensed under the Apache License, Version 2.0
"""Public compatibility facade and CLI for the composable agent stack."""
from __future__ import annotations

import argparse
import asyncio
import json
from typing import Sequence

from flyto_ai.coding.stack_manifest import (
    AGENT_STACK_CONTRACT_VERSION,
    AGENT_STACK_POLICY_VERSION,
    MAX_AGENT_STACK_MANIFEST_BYTES,
    SUPPORTED_AGENT_STACK_MANIFEST_VERSIONS,
    AgentStackManifest,
    compose_capability_stack,
    load_agent_stack_manifest,
)
from flyto_ai.coding.stack_presets import (
    DEFAULT_COMPONENTS,
    build_agent_stack_capabilities,
)
from flyto_ai.coding.stack_probe import (
    probe_agent_stack,
    probe_capability_stack,
)


__all__ = [
    "AGENT_STACK_CONTRACT_VERSION",
    "AGENT_STACK_POLICY_VERSION",
    "MAX_AGENT_STACK_MANIFEST_BYTES",
    "SUPPORTED_AGENT_STACK_MANIFEST_VERSIONS",
    "AgentStackManifest",
    "DEFAULT_COMPONENTS",
    "build_agent_stack_capabilities",
    "compose_capability_stack",
    "load_agent_stack_manifest",
    "main",
    "probe_agent_stack",
    "probe_capability_stack",
]


def main(argv: Sequence[str] | None = None) -> int:
    """Run the credential-free full-stack preflight."""
    parser = argparse.ArgumentParser(description="Probe the composable Flyto2 agent stack")
    parser.add_argument("--workspace", default=".", help="Workspace used as the MCP process cwd")
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--components",
        nargs="+",
        help="Built-in preset components to attach; omitted components remain detached",
    )
    selection.add_argument(
        "--manifest",
        help="Workspace-relative flyto.agent-stack.v1/v2 YAML profile",
    )
    parser.add_argument("--json", action="store_true", help="Emit the full evidence object")
    args = parser.parse_args(argv)
    if args.manifest:
        manifest = load_agent_stack_manifest(args.workspace, args.manifest)
        result = asyncio.run(
            probe_capability_stack(
                args.workspace,
                manifest.capabilities,
                profile=manifest.profile,
                manifest_fingerprint=manifest.manifest_fingerprint,
                contract_version=manifest.contract_version,
            )
        )
    else:
        components = args.components or list(DEFAULT_COMPONENTS)
        result = asyncio.run(
            probe_agent_stack(
                args.workspace,
                components,
                required_components=components,
            )
        )
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print("profile: {}".format(result["profile"]))
        for component in result["components"]:
            state = "ready" if component["available"] else "unavailable"
            print("{}: {} ({} tools)".format(component["name"], state, component["tool_count"]))
        print("stack: {}".format("ready" if result["ok"] else "blocked"))
        print("fingerprint: {}".format(result["composition_fingerprint"]))
    return 0 if result["ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
