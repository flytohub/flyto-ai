# Copyright 2024 Flyto2
# Licensed under the Apache License, Version 2.0
"""Guard the values flyto-ai borrows from flyto-core against silent drift.

flyto-ai reads a handful of primitives from Core rather than restating them,
because Core owns what a browser module is and how long a tool result may be.
Each import site also carries an ImportError fallback, and a fallback is a
second copy: `assistant/resilience.py` shipped one that had quietly lost
`browser.detect_list` and `browser.readability` from the snapshot set and
carried `browser.extract` in the interact set that Core does not, with nothing
in the suite touching either name.

Two rules, then:

* Where the fallback is empty or inert, assert it is, so a later edit cannot
  reintroduce a copy without this test noticing.
* Where a fallback must hold a real value (result truncation applies to every
  provider, Core installed or not), assert it equals Core's value whenever Core
  is importable.

Every import here is inside a test on purpose. Importing `flyto_ai.tools`
pulls in the whole Core module registry, and this file sorts early enough in
the suite that doing so at collection time changes the import state later tests
run against.
"""
from pathlib import Path

import pytest

RESILIENCE_SOURCE = (
    Path(__file__).resolve().parents[1] / "flyto_ai" / "assistant" / "resilience.py"
)


def _core_resilience():
    return pytest.importorskip(
        "core.modules.atomic.llm._resilience",
        reason="flyto-core is an optional extra; parity needs it installed",
    )


def test_browser_module_sets_come_from_core_unchanged() -> None:
    core_resilience = _core_resilience()
    from flyto_ai.assistant import resilience

    assert resilience._HAS_CORE_RESILIENCE is True
    assert set(resilience._SNAPSHOT_MODULES) == set(core_resilience._SNAPSHOT_MODULES)
    assert set(resilience._INTERACT_MODULES) == set(core_resilience._INTERACT_MODULES)


def test_result_truncation_constants_match_core() -> None:
    core_resilience = _core_resilience()
    from flyto_ai.providers import base

    assert base.MAX_RESULT_LEN == core_resilience.MAX_TOOL_RESULT_LEN
    assert base.TRUNCATION_NOTE == core_resilience.TRUNCATION_MARKER


def test_error_classifiers_are_cores_own() -> None:
    core_resilience = _core_resilience()
    from flyto_ai.tools import core_tools

    assert core_tools._is_transient_error is core_resilience.is_transient_error
    assert core_tools._is_session_dead is core_resilience.is_session_dead


def test_no_module_set_fallback_is_reintroduced() -> None:
    """The fallback branches must stay empty rather than restating Core.

    Read as text, not by import: the point is what the fallback contains, and
    on a machine with Core installed the fallback never executes.
    """
    text = RESILIENCE_SOURCE.read_text(encoding="utf-8")
    fallback = text.split("except ImportError:", 1)[1].split("logger = ", 1)[0]
    assert "_INTERACT_MODULES = frozenset()" in fallback
    assert "_SNAPSHOT_MODULES = frozenset()" in fallback
    assert "browser." not in fallback
