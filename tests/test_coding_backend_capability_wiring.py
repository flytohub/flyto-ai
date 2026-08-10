# Copyright 2024 Flyto2
# Licensed under the Apache License, Version 2.0
"""Startup must tell preflight what the selected backend can really bridge.

Preflight refuses a contract whose required capability the chosen implementer
cannot attach. That is only correct if startup actually hands it the truth, and
only maintainable if the truth comes from the implementer itself rather than
from a name check inside preflight - a name check would have to be edited every
time a backend is added, and would be wrong until somebody remembered.
"""
import pytest

from flyto_ai.agents.claude_code import ClaudeCodingAgent
from flyto_ai.cli import _declared_capability_kinds
from flyto_ai.coding import FlytoCodingAgent


def test_each_implementer_answers_for_itself():
    assert _declared_capability_kinds(FlytoCodingAgent) == ("command", "mcp-stdio")
    # Truthful, not a placeholder: this adapter refuses every required
    # capability the moment it reads a contract.
    assert _declared_capability_kinds(ClaudeCodingAgent) == ()


@pytest.mark.parametrize(
    "declared",
    [
        None,                       # a backend that never declared anything
        "mcp-stdio",                # a bare string is not a set of kinds
        123,
        {"mcp-stdio", 7},           # partially typed
        {"mcp-stdio", ""},          # empty token
        {True},                     # bools are not capability kinds
    ],
)
def test_an_undeclared_or_malformed_bridge_fails_closed(declared):
    backend = type("Backend", (), {"attachable_capability_kinds": declared})
    resolved = _declared_capability_kinds(backend)
    assert all(isinstance(item, str) and item for item in resolved)
    assert 7 not in resolved and True not in resolved and "" not in resolved


def test_preflight_holds_no_backend_names():
    """The rule lives with the backend; preflight stays domain-neutral."""

    import pathlib
    import re

    from flyto_ai.coding import preflight

    source = pathlib.Path(preflight.__file__).read_text(encoding="utf-8").lower()
    for name in ("claude", "codex", "native", "anthropic", "openai", "flytocodingagent"):
        # Whole words only: "alternative" is not a backend.
        assert not re.search(r"\b{}\b".format(name), source), name


def test_the_service_stores_the_declaration_it_was_given(tmp_path):
    from flyto_ai.coding.service import CodingService

    workspace = tmp_path / "ws"
    workspace.mkdir()
    service = CodingService(
        lambda store: None,
        state_root=str(tmp_path / "state"),
        workspace_roots=(str(workspace),),
        attachable_capability_kinds=_declared_capability_kinds(FlytoCodingAgent),
    )
    try:
        assert service.attachable_capability_kinds == frozenset({"command", "mcp-stdio"})
    finally:
        service.close()

    # Omitted entirely means "unproven", which is treated as nothing.
    bare = CodingService(
        lambda store: None,
        state_root=str(tmp_path / "state-2"),
        workspace_roots=(str(workspace),),
    )
    try:
        assert bare.attachable_capability_kinds == frozenset()
    finally:
        bare.close()
