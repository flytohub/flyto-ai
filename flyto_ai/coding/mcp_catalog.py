# Copyright 2024 Flyto2
# Licensed under the Apache License, Version 2.0
"""Pure MCP tool catalog scoping, naming, and domain-status normalization."""
from __future__ import annotations

import hashlib
import json
import re
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Tuple

from flyto_ai.coding.contracts import CapabilitySpec
from flyto_ai.coding.mcp_transport import MAX_MCP_MESSAGE_BYTES


_SAFE_NAME = re.compile(r"[^A-Za-z0-9_-]+")
_REMOTE_TOOL_NAME = re.compile(r"^[^\s\x00-\x1f\x7f]{1,256}$")
_FAILURE_STATUSES = frozenset({"denied", "error", "failed", "failure", "not_proved"})


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
    if "isError" in result and not isinstance(result["isError"], bool):
        return False, "capability returned an invalid MCP error status"
    if result.get("isError") is True:
        return False, "capability returned an MCP error result"

    candidates: List[Dict[str, Any]] = []
    structured = result.get("structuredContent")
    if isinstance(structured, dict):
        candidates.append(structured)
    content = result.get("content")
    if isinstance(content, list) and len(content) <= 64:
        remaining = MAX_MCP_MESSAGE_BYTES
        for block in content:
            if not isinstance(block, dict):
                continue
            text = block.get("text")
            if isinstance(text, str) and len(text.encode("utf-8")) <= remaining:
                remaining -= len(text.encode("utf-8"))
            else:
                continue
            try:
                decoded = json.loads(text)
            except json.JSONDecodeError:
                decoded = None
            if isinstance(decoded, dict):
                candidates.append(decoded)

    for candidate in candidates:
        status = candidate.get("status")
        failed = (
            candidate.get("ok") is False
            or candidate.get("success") is False
            or candidate.get("passed") is False
            or (isinstance(status, str) and status.lower() in _FAILURE_STATUSES)
        )
        if failed:
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
    for index, item in enumerate(raw_tools):
        if not isinstance(item, dict):
            raise RuntimeError(
                "capability tool catalog entry {} must be an object".format(index),
            )
        name = item.get("name")
        if not isinstance(name, str) or not _REMOTE_TOOL_NAME.fullmatch(name):
            raise RuntimeError(
                "capability tool catalog entry {} has an invalid name".format(index),
            )
        if name in catalog:
            raise RuntimeError("capability tool catalog contains duplicate names")
        if "description" in item and not isinstance(item["description"], str):
            raise RuntimeError("capability tool description must be a string")
        schema = item.get("inputSchema")
        if not isinstance(schema, dict):
            raise RuntimeError("capability tool inputSchema must be an object")
        if "type" in schema and schema["type"] != "object":
            raise RuntimeError("capability tool inputSchema type must be object")
        catalog[name] = item
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
    if len(selected_names) > 100:
        raise RuntimeError("capability scoped tool catalog cannot exceed 100 tools")
    definitions = []
    tool_map = {}
    for remote_name in sorted(selected_names):
        item = catalog[remote_name]
        provider_name = provider_tool_name(spec.name, remote_name)
        schema = deepcopy(item["inputSchema"])
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
