# Copyright 2024 Flyto2
# Licensed under the Apache License, Version 2.0
"""Runtime negotiation and evidence attestation for composed agent stacks."""
from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path
from typing import Iterable, Sequence

from flyto_ai.coding.capabilities import CapabilityManager
from flyto_ai.coding.contracts import CapabilitySpec
from flyto_ai.coding.stack_manifest import (
    AGENT_STACK_CONTRACT_VERSION,
    compose_capability_stack,
    validate_contract_version,
    validate_profile_name,
)
from flyto_ai.coding.stack_presets import (
    DEFAULT_COMPONENTS,
    build_agent_stack_capabilities,
)


def composition_fingerprint(components: Sequence[dict]) -> str:
    """Hash only observed runtime identity, never mutable status prose."""
    identity = [
        {
            "name": item["name"],
            "available": item["available"],
            "required": item["required"],
            "server_name": item["server_name"],
            "protocol": item["negotiated_protocol_version"],
            "tools": item["tools"],
        }
        for item in components
    ]
    return hashlib.sha256(
        json.dumps(identity, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(),
    ).hexdigest()


async def probe_capability_stack(
    workspace: str,
    capabilities: Sequence[CapabilitySpec],
    *,
    profile: str = "custom",
    manifest_fingerprint: str | None = None,
    contract_version: str = AGENT_STACK_CONTRACT_VERSION,
) -> dict:
    """Negotiate an arbitrary profile and attest its actually exposed tools."""
    validate_profile_name(profile)
    validate_contract_version(contract_version)
    root = str(Path(workspace).expanduser().resolve(strict=True))
    specs = compose_capability_stack(capabilities)
    manager = CapabilityManager(root)
    try:
        statuses = await manager.start(specs)
    finally:
        await manager.close()

    projected = [dataclasses.asdict(status) for status in statuses]
    return {
        "contract_version": contract_version,
        "ok": all(item["available"] for item in projected if item["required"]),
        "workspace": root,
        "profile": profile,
        "manifest_fingerprint": manifest_fingerprint,
        "composition_fingerprint": composition_fingerprint(projected),
        "components": projected,
    }


async def probe_agent_stack(
    workspace: str,
    components: Sequence[str] = DEFAULT_COMPONENTS,
    *,
    required_components: Iterable[str] | None = None,
) -> dict:
    """Probe the built-in Flyto2 preset; callers may detach any preset lane."""
    specs = build_agent_stack_capabilities(
        components,
        required_components=required_components,
    )
    return await probe_capability_stack(workspace, specs, profile="flyto-default")
