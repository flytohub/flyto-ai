# Copyright 2024 Flyto2
# Licensed under the Apache License, Version 2.0
"""Pure MCP tool catalog scoping, naming, and domain-status normalization."""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Tuple

from flyto_ai.coding.contracts import CapabilitySpec
from flyto_ai.coding.mcp_transport import MAX_MCP_MESSAGE_BYTES


_SAFE_NAME = re.compile(r"[^A-Za-z0-9_-]+")


def provider_tool_name(capability: str, tool_name: str) -> str:
    """Create a deterministic provider-safe name bounded to 64 characters."""
    base = "cap_{}_{}".format(
        _SAFE_NAME.sub("_", capability), _SAFE_NAME.sub("_", tool_name),
    ).strip("_")
    if len(base) <= 64:
        return base
    digest = hashlib.sha256(base.encode()).hexdigest()[:10]
    return "{}_{}".format(base[:53], digest)


def mcp_domain_status(result: Any) -> tuple[bool, Optional[str]]:
    """Normalize transport and nested domain status without trusting text prose."""
    if not isinstance(result, dict):
        return False, "capability result must be an object"
    if result.get("isError") is True:
        return False, "capability returned an MCP error result"

    candidates: List[Dict[str, Any]] = []
    structured = result.get("structuredContent")
    if isinstance(structured, dict):
        candidates.append(structured)
    content = result.get("content")
    if isinstance(content, list) and len(content) == 1 and isinstance(content[0], dict):
        text = content[0].get("text")
        if isinstance(text, str) and len(text) <= MAX_MCP_MESSAGE_BYTES:
            try:
                decoded = json.loads(text)
            except json.JSONDecodeError:
                decoded = None
            if isinstance(decoded, dict):
                candidates.append(decoded)

    for candidate in candidates:
        if candidate.get("ok") is False or candidate.get("status") in {"error", "failed"}:
            error = (
                candidate.get("error")
                or candidate.get("message")
                or "capability domain operation failed"
            )
            return False, str(error)[:1000]
    return True, None


def _raw_catalog(raw_tools: object) -> Dict[str, Dict[str, Any]]:
    if not isinstance(raw_tools, list) or len(raw_tools) > 2000:
        raise RuntimeError("capability returned an invalid tool catalog")
    catalog: Dict[str, Dict[str, Any]] = {}
    for item in raw_tools:
        if not isinstance(item, dict) or not isinstance(item.get("name"), str):
            continue
        catalog.setdefault(item["name"], item)
    return catalog


def catalog_tool_names(raw_tools: object) -> Tuple[str, ...]:
    """Return bounded discovered names before required/allowed validation."""
    return tuple(sorted(_raw_catalog(raw_tools)))


@dataclass(frozen=True)
class McpToolCatalog:
    """Scoped provider definitions plus a reversible name mapping."""

    definitions: Tuple[Dict[str, Any], ...]
    tool_map: Mapping[str, str]
    remote_names: Tuple[str, ...]


def build_mcp_tool_catalog(spec: CapabilitySpec, raw_tools: object) -> McpToolCatalog:
    """Scope one remote catalog to the exact contract-declared tool surface."""
    catalog = _raw_catalog(raw_tools)
    catalog_names = set(catalog)
    missing = sorted(set(spec.required_tools) - catalog_names)
    if missing:
        raise RuntimeError("capability is missing required tools: {}".format(", ".join(missing)))
    unavailable_allowed = sorted(set(spec.allowed_tools) - catalog_names)
    if unavailable_allowed:
        raise RuntimeError(
            "capability is missing allowed tools: {}".format(", ".join(unavailable_allowed)),
        )

    selected_names = set(spec.allowed_tools) if spec.allowed_tools else catalog_names
    definitions = []
    tool_map = {}
    for remote_name in sorted(selected_names):
        item = catalog[remote_name]
        provider_name = provider_tool_name(spec.name, remote_name)
        schema = (
            item.get("inputSchema")
            if isinstance(item.get("inputSchema"), dict)
            else {"type": "object", "properties": {}}
        )
        definitions.append({
            "name": provider_name,
            "description": "[{}] {}".format(
                spec.name, str(item.get("description", ""))[:2000],
            ),
            "inputSchema": schema,
        })
        tool_map[provider_name] = remote_name
    return McpToolCatalog(
        definitions=tuple(definitions),
        tool_map=tool_map,
        remote_names=tuple(sorted(selected_names)),
    )
