# Copyright 2024 Flyto2
# Licensed under the Apache License, Version 2.0
"""Composition tests for the detachable Flyto2 full-agent stack."""
from __future__ import annotations

import importlib.util

import pytest

from flyto_ai.coding.stack import (
    AGENT_STACK_CONTRACT_VERSION,
    DEFAULT_COMPONENTS,
    build_agent_stack_capabilities,
    probe_agent_stack,
)
from flyto_ai.coding import build_agent_stack_capabilities as public_stack_builder


def test_full_stack_is_split_into_least_privilege_components():
    specs = build_agent_stack_capabilities(python_executable="python")
    assert tuple(spec.name for spec in specs) == DEFAULT_COMPONENTS
    assert all(spec.required for spec in specs)
    assert all(spec.required_tools == spec.allowed_tools for spec in specs)

    by_name = {spec.name: spec for spec in specs}
    assert by_name["flyto-page-inspector"].allowed_tools == ("inspect_page",)
    assert "chat" not in by_name["flyto-blueprint"].allowed_tools
    assert "execute_module" not in by_name["flyto-blueprint"].allowed_tools
    assert "inspect_page" not in by_name["flyto-core"].allowed_tools
    assert by_name["flyto-indexer"].argv == (
        "python", "-m", "flyto_indexer.mcp_server",
    )
    assert by_name["flyto-core"].argv == ("python", "-m", "core.mcp_server")


def test_stack_components_can_be_detached_or_made_optional():
    selected = build_agent_stack_capabilities(
        ("flyto-indexer", "flyto-core"),
        python_executable="python",
    )
    assert all(spec.required for spec in selected)

    specs = build_agent_stack_capabilities(
        ("flyto-indexer", "flyto-core"),
        required_components=("flyto-indexer",),
        python_executable="python",
    )
    assert [spec.name for spec in specs] == ["flyto-indexer", "flyto-core"]
    assert specs[0].required is True
    assert specs[1].required is False


def test_stack_builder_is_available_from_the_public_coding_api():
    assert public_stack_builder is build_agent_stack_capabilities


def test_stack_rejects_unknown_duplicate_and_unselected_required_components():
    with pytest.raises(ValueError, match="unknown agent stack"):
        build_agent_stack_capabilities(("unknown",), required_components=())
    with pytest.raises(ValueError, match="duplicates"):
        build_agent_stack_capabilities(
            ("flyto-core", "flyto-core"), required_components=("flyto-core",),
        )
    with pytest.raises(ValueError, match="were not selected"):
        build_agent_stack_capabilities(
            ("flyto-core",), required_components=("flyto-indexer",),
        )


@pytest.mark.skipif(
    importlib.util.find_spec("flyto_indexer") is None,
    reason="flyto-indexer is an independently installed stack component",
)
def test_real_full_stack_preflight_negotiates_isolated_tool_surfaces(tmp_path):
    import asyncio

    result = asyncio.run(probe_agent_stack(str(tmp_path)))
    assert result["contract_version"] == AGENT_STACK_CONTRACT_VERSION
    assert result["ok"] is True
    assert len(result["composition_fingerprint"]) == 64
    components = {item["name"]: item for item in result["components"]}
    assert tuple(components) == DEFAULT_COMPONENTS
    assert components["flyto-page-inspector"]["tools"] == ("inspect_page",)
    assert "chat" not in components["flyto-blueprint"]["tools"]
    assert all(item["negotiated_protocol_version"] == "2025-06-18" for item in components.values())
