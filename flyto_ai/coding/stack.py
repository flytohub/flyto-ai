# Copyright 2024 Flyto2
# Licensed under the Apache License, Version 2.0
"""Composable preflight for the full Flyto2 native agent stack."""
from __future__ import annotations

import argparse
import asyncio
import dataclasses
import hashlib
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, Sequence, Tuple

import yaml

from flyto_ai.coding.capabilities import CapabilityManager
from flyto_ai.coding.contracts import CapabilitySpec


AGENT_STACK_CONTRACT_VERSION = "flyto.agent-stack.v1"
MAX_AGENT_STACK_MANIFEST_BYTES = 256 * 1024
DEFAULT_COMPONENTS = (
    "flyto-indexer",
    "flyto-blueprint",
    "flyto-page-inspector",
    "flyto-core",
)
_PROFILE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
_MANIFEST_KEYS = frozenset({"version", "profile", "capabilities"})
_CAPABILITY_KEYS = frozenset({field.name for field in dataclasses.fields(CapabilitySpec)})


@dataclass(frozen=True)
class AgentStackManifest:
    """Validated, source-controlled composition for one agent domain profile."""

    profile: str
    capabilities: Tuple[CapabilitySpec, ...]
    manifest_fingerprint: str
    contract_version: str = AGENT_STACK_CONTRACT_VERSION


def compose_capability_stack(*groups: Sequence[CapabilitySpec]) -> Tuple[CapabilitySpec, ...]:
    """Compose arbitrary capability groups without encoding their task domain."""
    capabilities = tuple(spec for group in groups for spec in group)
    if not capabilities:
        raise ValueError("agent stack must contain at least one capability")
    if len(capabilities) > 64:
        raise ValueError("agent stack cannot exceed 64 capabilities")
    if any(not isinstance(spec, CapabilitySpec) for spec in capabilities):
        raise TypeError("agent stack accepts only CapabilitySpec values")
    names = tuple(spec.name for spec in capabilities)
    if len(set(names)) != len(names):
        raise ValueError("agent stack capabilities contain duplicate names")
    return capabilities


def _canonical_manifest_fingerprint(
    profile: str,
    capabilities: Sequence[CapabilitySpec],
) -> str:
    normalized = {
        "version": AGENT_STACK_CONTRACT_VERSION,
        "profile": profile,
        "capabilities": [dataclasses.asdict(spec) for spec in capabilities],
    }
    return hashlib.sha256(
        json.dumps(normalized, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(),
    ).hexdigest()


def _validate_manifest_capability_mapping(index: int, item: Mapping) -> None:
    string_fields = ("name", "kind", "contract_version", "protocol_version")
    list_fields = ("argv", "required_tools", "allowed_tools", "env_passthrough")
    for field in string_fields:
        if field in item and not isinstance(item[field], str):
            raise ValueError(
                "capability {} field {} must be a string".format(index, field),
            )
    if "required" in item and not isinstance(item["required"], bool):
        raise ValueError("capability {} field required must be a boolean".format(index))
    if "timeout_seconds" in item and (
        isinstance(item["timeout_seconds"], bool)
        or not isinstance(item["timeout_seconds"], int)
    ):
        raise ValueError("capability {} field timeout_seconds must be an integer".format(index))
    for field in list_fields:
        if field not in item:
            continue
        value = item[field]
        if not isinstance(value, list) or any(not isinstance(member, str) for member in value):
            raise ValueError(
                "capability {} field {} must be an array of strings".format(index, field),
            )


def load_agent_stack_manifest(
    workspace: str,
    manifest_path: str = ".flyto/agent-stack.yaml",
) -> AgentStackManifest:
    """Load a bounded workspace-local stack profile and fail closed on tool scope."""
    root = Path(workspace).expanduser().resolve(strict=True)
    requested = Path(manifest_path)
    if requested.is_absolute():
        raise ValueError("agent stack manifest path must be relative to the workspace")
    try:
        path = (root / requested).resolve(strict=True)
    except OSError as exc:
        raise ValueError("agent stack manifest must be a regular file") from exc
    if root != path and root not in path.parents:
        raise ValueError("agent stack manifest must remain inside the workspace")
    if not path.is_file():
        raise ValueError("agent stack manifest must be a regular file")
    raw = path.read_bytes()
    if len(raw) > MAX_AGENT_STACK_MANIFEST_BYTES:
        raise ValueError("agent stack manifest exceeds the 256 KiB limit")
    try:
        value = yaml.safe_load(raw.decode("utf-8"))
    except (UnicodeDecodeError, yaml.YAMLError) as exc:
        raise ValueError("agent stack manifest is not valid UTF-8 YAML") from exc
    if not isinstance(value, Mapping):
        raise ValueError("agent stack manifest must be a YAML object")
    unknown = set(value) - _MANIFEST_KEYS
    if unknown:
        raise ValueError("unknown agent stack manifest fields: {}".format(", ".join(sorted(unknown))))
    if value.get("version") != AGENT_STACK_CONTRACT_VERSION:
        raise ValueError("unsupported agent stack manifest version")
    profile = value.get("profile")
    if not isinstance(profile, str) or not _PROFILE_NAME.fullmatch(profile):
        raise ValueError("agent stack profile must be a safe identifier")
    raw_capabilities = value.get("capabilities")
    if not isinstance(raw_capabilities, Sequence) or isinstance(raw_capabilities, (str, bytes)):
        raise ValueError("agent stack capabilities must be a YAML array")

    specs = []
    for index, item in enumerate(raw_capabilities):
        if not isinstance(item, Mapping):
            raise ValueError("agent stack capability {} must be an object".format(index))
        unknown_capability_fields = set(item) - _CAPABILITY_KEYS
        if unknown_capability_fields:
            raise ValueError(
                "unknown capability fields at index {}: {}".format(
                    index, ", ".join(sorted(unknown_capability_fields)),
                )
            )
        _validate_manifest_capability_mapping(index, item)
        spec = CapabilitySpec.from_mapping(item)
        if spec.kind == "mcp-stdio" and (
            "allowed_tools" not in item or not spec.allowed_tools
        ):
            raise ValueError(
                "manifest MCP capability {} must declare a non-empty allowed_tools list".format(
                    spec.name,
                )
            )
        specs.append(spec)
    capabilities = compose_capability_stack(specs)
    return AgentStackManifest(
        profile=profile,
        capabilities=capabilities,
        manifest_fingerprint=_canonical_manifest_fingerprint(profile, capabilities),
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


async def probe_capability_stack(
    workspace: str,
    capabilities: Sequence[CapabilitySpec],
    *,
    profile: str = "custom",
    manifest_fingerprint: str | None = None,
) -> dict:
    """Negotiate an arbitrary profile and attest its actually exposed tools."""
    if not _PROFILE_NAME.fullmatch(profile):
        raise ValueError("agent stack profile must be a safe identifier")
    root = str(Path(workspace).expanduser().resolve(strict=True))
    specs = compose_capability_stack(capabilities)
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
        "profile": profile,
        "manifest_fingerprint": manifest_fingerprint,
        "composition_fingerprint": fingerprint,
        "components": projected,
    }


async def probe_agent_stack(
    workspace: str,
    components: Sequence[str] = DEFAULT_COMPONENTS,
    *,
    required_components: Iterable[str] | None = None,
) -> dict:
    """Probe the built-in Flyto preset; callers may detach any preset lane."""
    specs = build_agent_stack_capabilities(
        components,
        required_components=required_components,
    )
    return await probe_capability_stack(workspace, specs, profile="flyto-default")


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
        help="Workspace-relative flyto.agent-stack.v1 YAML profile",
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
