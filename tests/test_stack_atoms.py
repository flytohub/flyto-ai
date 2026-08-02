# Copyright 2024 Flyto2
# Licensed under the Apache License, Version 2.0
"""Pure tests for manifest, preset, probe, and facade stack atoms."""
from __future__ import annotations

import pytest

from flyto_ai.coding.contracts import CapabilitySpec
from flyto_ai.coding import stack as facade
from flyto_ai.coding.stack_manifest import (
    AGENT_STACK_CONTRACT_VERSION,
    AGENT_STACK_POLICY_VERSION,
    AgentStackManifest,
    canonical_manifest_fingerprint,
    compose_capability_stack,
    load_agent_stack_manifest,
    validate_contract_version,
    validate_profile_name,
)
from flyto_ai.coding.stack_presets import (
    DEFAULT_COMPONENTS,
    build_agent_stack_capabilities,
)
from flyto_ai.coding.stack_probe import (
    composition_fingerprint,
    probe_agent_stack,
    probe_capability_stack,
)


def test_stack_facade_reexports_each_atomic_contract_without_wrapping():
    assert facade.AgentStackManifest is AgentStackManifest
    assert facade.compose_capability_stack is compose_capability_stack
    assert facade.load_agent_stack_manifest is load_agent_stack_manifest
    assert facade.build_agent_stack_capabilities is build_agent_stack_capabilities
    assert facade.probe_capability_stack is probe_capability_stack
    assert facade.probe_agent_stack is probe_agent_stack


@pytest.mark.parametrize(
    "profile", ["coding", "robotics.lab", "red-team_v2", "ops-2026"],
)
def test_profile_identity_accepts_domain_neutral_safe_names(profile):
    assert validate_profile_name(profile) == profile


@pytest.mark.parametrize("profile", ["", "unsafe/name", " space", "x" * 65, None])
def test_profile_identity_rejects_unsafe_or_unbounded_names(profile):
    with pytest.raises(ValueError, match="safe identifier"):
        validate_profile_name(profile)


def test_contract_version_atom_accepts_only_v1_and_v2():
    assert validate_contract_version(AGENT_STACK_CONTRACT_VERSION) == AGENT_STACK_CONTRACT_VERSION
    assert validate_contract_version(AGENT_STACK_POLICY_VERSION) == AGENT_STACK_POLICY_VERSION
    with pytest.raises(ValueError, match="unsupported"):
        validate_contract_version("flyto.agent-stack.v3")


def test_composition_atom_enforces_nonempty_type_count_and_unique_name():
    spec = CapabilitySpec(name="one", argv=("one",), kind="command")
    assert compose_capability_stack((spec,)) == (spec,)
    with pytest.raises(ValueError, match="at least one"):
        compose_capability_stack(())
    with pytest.raises(TypeError, match="CapabilitySpec"):
        compose_capability_stack((object(),))
    with pytest.raises(ValueError, match="duplicate"):
        compose_capability_stack((spec, spec))
    many = tuple(
        CapabilitySpec(name="cap-{}".format(index), argv=("run",), kind="command")
        for index in range(65)
    )
    with pytest.raises(ValueError, match="cannot exceed 64"):
        compose_capability_stack(many)


def test_manifest_fingerprint_is_format_independent_for_same_normalized_contract():
    spec = CapabilitySpec(
        name="observer",
        argv=("observer",),
        allowed_tools=("observe",),
        tool_permissions=(("observe", "read_only"),),
    )
    first = canonical_manifest_fingerprint(
        AGENT_STACK_POLICY_VERSION, "profile", (spec,),
    )
    second = canonical_manifest_fingerprint(
        AGENT_STACK_POLICY_VERSION, "profile", tuple([spec]),
    )
    assert first == second
    assert len(first) == 64


def test_preset_atom_keeps_each_lane_detachable_and_exhaustively_classified():
    specs = build_agent_stack_capabilities(
        ("flyto-indexer", "flyto-core"),
        required_components=("flyto-indexer",),
        python_executable="python",
    )
    assert tuple(spec.name for spec in specs) == ("flyto-indexer", "flyto-core")
    assert specs[0].required is True
    assert specs[1].required is False
    assert all(
        {name for name, _level in spec.tool_permissions} == set(spec.allowed_tools)
        for spec in specs
    )
    assert DEFAULT_COMPONENTS == (
        "flyto-indexer", "flyto-blueprint", "flyto-page-inspector", "flyto-core",
    )


def test_runtime_fingerprint_ignores_error_prose_but_tracks_observed_identity():
    base = [{
        "name": "observer",
        "available": True,
        "required": True,
        "server_name": "fixture",
        "negotiated_protocol_version": "2025-06-18",
        "tools": ("observe",),
        "error": "first prose",
    }]
    changed_prose = [dict(base[0], error="different prose")]
    changed_tools = [dict(base[0], tools=("observe", "move"))]
    assert composition_fingerprint(base) == composition_fingerprint(changed_prose)
    assert composition_fingerprint(base) != composition_fingerprint(changed_tools)
