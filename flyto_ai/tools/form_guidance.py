# Copyright 2024 Flyto2
# Licensed under the Apache License, Version 2.0
"""Agent guidance for filling a web form in one Core call.

Lives beside ``core_tools`` rather than in it: the guidance is text plus one
presence check, and Core access still goes through ``core_tools``'s handler.
"""

from typing import Dict

#: The Core module that fills a whole web form in one call. Filling a form one
#: browser.type/select/upload/click call per field costs one model round-trip
#: per field; this module costs one for the whole form.
FILL_FORM_MODULE = "browser.fill_form"

#: Agent-facing guidance for forms, offered only when the installed Core
#: provides FILL_FORM_MODULE: advertising a module Core cannot run would send
#: the agent into an unknown-module failure.
FILL_FORM_GUIDANCE = (
    "FORMS: read the form once (browser.snapshot), then call browser.fill_form ONCE "
    "with every field ({\"label\": visible label text, \"value\": ...}), every upload "
    "({\"label\": ..., \"path\": ...}), the submit ({\"label\": button text}) and an "
    "optional confirm ({\"text\": text shown after saving}). Do not fill a form with one "
    "browser.type/select/upload/click call per field: each call is a full model "
    "round-trip. If a label is not found, fill_form fills nothing and lists it."
)

_core_module_presence: Dict[str, bool] = {}


def core_provides_module(module_id: str) -> bool:
    """Whether the installed flyto-core knows ``module_id`` (its get_module_info). Cached."""
    if module_id in _core_module_presence:
        return _core_module_presence[module_id]
    present = False
    from flyto_ai.tools import core_tools

    handler = core_tools._get_mcp_handler()
    lookup = handler.get("get_module_info") if handler else None
    if callable(lookup):
        try:
            detail = lookup(module_id=module_id)
            present = isinstance(detail, dict) and not detail.get("error") and bool(detail)
        except Exception:  # noqa: BLE001 - an unreadable catalog offers nothing
            present = False
    _core_module_presence[module_id] = present
    return present


def execute_module_description(description: str) -> str:
    """The execute_module description, with the one-call form path guaranteed.

    A Core that ships browser.fill_form also says so in its own description;
    this only fills the gap for one that registers the module but whose tool
    text predates it, so the agent never learns of it only through search.
    """
    if FILL_FORM_MODULE in description or not core_provides_module(FILL_FORM_MODULE):
        return description
    return f"{description}\n{FILL_FORM_GUIDANCE}" if description else FILL_FORM_GUIDANCE
