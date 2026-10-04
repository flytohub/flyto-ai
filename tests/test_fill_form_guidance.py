"""The agent is told to fill a form with one browser.fill_form call.

Filling a form one browser.type/select/upload/click call per field costs a model
round-trip per field (about 100 s for one demo form). The guidance reaches the
agent in two places it always reads -- the execute_module tool description and
the running-browser prompt hint -- not only through search_modules, and only
when the installed Core actually provides the module.
"""

import pytest

from flyto_ai.tools import core_tools, form_guidance


def _handler(description):
    return {
        "TOOLS": [
            {"name": "execute_module", "description": description,
             "inputSchema": {"type": "object", "properties": {"module_id": {"type": "string"}}}},
            {"name": "search_modules", "description": "Search modules",
             "inputSchema": {"type": "object", "properties": {}}},
        ],
    }


def _execute_tool(monkeypatch, description, provides):
    monkeypatch.setattr(core_tools, "_get_mcp_handler", lambda: _handler(description))
    monkeypatch.setattr(form_guidance, "core_provides_module", lambda module_id: provides)
    tools = core_tools.get_core_tool_defs()
    return next(t for t in tools if t["name"] == "execute_module"), tools


def test_execute_module_description_offers_fill_form_when_core_has_it(monkeypatch):
    tool, _ = _execute_tool(monkeypatch, "Execute a module.", provides=True)
    assert "browser.fill_form ONCE" in tool["description"]
    assert tool["description"].startswith("Execute a module.")


def test_no_guidance_for_a_core_without_the_module(monkeypatch):
    tool, _ = _execute_tool(monkeypatch, "Execute a module.", provides=False)
    assert tool["description"] == "Execute a module."


def test_a_core_description_that_already_names_it_is_left_alone(monkeypatch):
    own = "Execute a module. FORMS: call browser.fill_form once."
    tool, _ = _execute_tool(monkeypatch, own, provides=True)
    assert tool["description"] == own


def test_only_execute_module_is_touched_and_schema_is_unchanged(monkeypatch):
    _, tools = _execute_tool(monkeypatch, "Execute a module.", provides=True)
    search = next(t for t in tools if t["name"] == "search_modules")
    assert search["description"] == "Search modules"
    execute = next(t for t in tools if t["name"] == "execute_module")
    assert execute["inputSchema"] == {"type": "object", "properties": {"module_id": {"type": "string"}}}


@pytest.mark.parametrize("provides", [True, False])
def test_running_browser_hint_names_fill_form_only_when_available(monkeypatch, provides):
    monkeypatch.setattr(core_tools, "_active_browser_sessions", lambda: {"one": object()})
    monkeypatch.setattr(form_guidance, "core_provides_module", lambda module_id: provides)
    hint = core_tools.get_browser_status()
    assert "BROWSER IS ALREADY RUNNING" in hint
    assert ("browser.fill_form ONCE" in hint) is provides


def test_no_browser_means_no_hint_at_all(monkeypatch):
    monkeypatch.setattr(core_tools, "_active_browser_sessions", lambda: {})
    monkeypatch.setattr(form_guidance, "core_provides_module", lambda module_id: True)
    assert core_tools.get_browser_status() == ""


def test_presence_check_without_core_is_false(monkeypatch):
    monkeypatch.setattr(form_guidance, "_core_module_presence", {})
    monkeypatch.setattr(core_tools, "_get_mcp_handler", lambda: None)
    assert form_guidance.core_provides_module(form_guidance.FILL_FORM_MODULE) is False


@pytest.mark.parametrize("detail, expected", [
    ({"module_id": "browser.fill_form", "params_schema": {}}, True),
    ({"error": "Module not found: browser.fill_form"}, False),
    (None, False),
])
def test_presence_check_reads_core_module_info(monkeypatch, detail, expected):
    monkeypatch.setattr(form_guidance, "_core_module_presence", {})
    monkeypatch.setattr(core_tools, "_get_mcp_handler",
                        lambda: {"get_module_info": lambda module_id: detail})
    assert form_guidance.core_provides_module(form_guidance.FILL_FORM_MODULE) is expected


def test_installed_core_answers_from_its_catalog():
    """Against the Core installed here, whichever version that is."""
    form_guidance._core_module_presence.pop(form_guidance.FILL_FORM_MODULE, None)
    handler = core_tools._get_mcp_handler()
    if handler is None:
        pytest.skip("flyto-core is not installed")
    detail = handler["get_module_info"](module_id=form_guidance.FILL_FORM_MODULE)
    expected = isinstance(detail, dict) and not detail.get("error") and bool(detail)
    assert form_guidance.core_provides_module(form_guidance.FILL_FORM_MODULE) is expected
    form_guidance._core_module_presence.pop(form_guidance.FILL_FORM_MODULE, None)


def test_fill_form_contract_grades_like_the_browser_calls_it_replaces():
    """A form submit is an external write (level 3), not a real-world actuation.

    browser.fill_form declares safety_class ``controlled`` without ``actuates``.
    If it graded DANGER_FULL, every form fill would wait on a confirmation the
    per-field browser.type calls never needed, and the one-call path would be
    the slow one.
    """
    from flyto_ai.permissions import PermissionLevel, grade_module_contract

    grade = grade_module_contract({
        "plugin": "",
        "provides_capability": "browser.fill_form",
        "contract": {
            "actuates": False, "safety_class": "controlled", "requires_safe_stop": False,
            "cancellable": True, "idempotent": False, "effects": ["external.record.changed"],
        },
    })
    assert grade.permission_level == PermissionLevel.WORKSPACE_WRITE
    assert grade.risk_level == 3
