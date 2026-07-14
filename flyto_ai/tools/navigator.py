# Copyright 2024 Flyto2
# Licensed under the Apache License, Version 2.0
"""Web navigator tool — system-driven multi-step website navigation.

Instead of LLM guessing URLs round by round, this tool:
1. Opens the website
2. Snapshots the page → detects interactive choices
3. Returns choices to user via ask_user
4. User picks → system clicks → snapshots next page
5. Repeats until task is done or login form found

The LLM calls this ONCE. The system handles the loop.
No patches, no middleware layers, no context loss.
"""
import asyncio
import json
import logging
from typing import Any, Dict

from flyto_ai.assistant.choice_detector import detect_choices

logger = logging.getLogger(__name__)


def _is_success(result: dict) -> bool:
    """Check if a module result indicates success (handles both ok and status)."""
    return bool(result.get("ok") or result.get("status") == "success")

TOOL_DEF = {
    "name": "navigate_website",
    "description": (
        "Navigate a website step by step. Opens the URL, detects interactive "
        "choices (buttons, links, forms), and asks the user to pick. "
        "Handles multi-step navigation automatically. "
        "Use this instead of manually calling browser.launch + browser.goto + browser.click."
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "Starting URL to navigate to",
            },
            "goal": {
                "type": "string",
                "description": "What the user wants to accomplish (e.g. 'book appointment for pediatrics')",
            },
        },
        "required": ["url"],
    },
}


# Module-level dispatch reference — set by agent during tool registration
_dispatch_ref = None


def set_dispatch(dispatch_fn):
    """Set the dispatch function for navigator to use."""
    global _dispatch_ref
    _dispatch_ref = dispatch_fn


async def dispatch_navigator(name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Navigate a website, detect choices, and return them for user selection."""

    url = arguments.get("url", "")
    if not url:
        return {"ok": False, "error": "URL is required"}

    dispatch = _dispatch_ref
    if not dispatch:
        return {"ok": False, "error": "Navigator dispatch not initialized"}

    # Step 1: Launch browser (clean stale locks first)
    from pathlib import Path
    profile_dir = Path.home() / '.flyto' / 'chrome-profile'
    for lock_name in ('SingletonLock', 'SingletonSocket', 'SingletonCookie'):
        lock_file = profile_dir / lock_name
        if lock_file.exists():
            try:
                lock_file.unlink()
            except OSError:
                pass

    await dispatch("execute_module", {
        "module_id": "browser.launch",
        "params": {"headless": False},
    })

    # Step 2: Navigate to URL
    goto_result = await dispatch("execute_module", {
        "module_id": "browser.goto",
        "params": {"url": url},
    })

    if not _is_success(goto_result):
        return {
            "ok": False,
            "error": "Failed to navigate to {}".format(url),
            "details": goto_result,
        }

    await asyncio.sleep(1)

    # Step 3: Snapshot to detect interactive elements
    snap_result = await dispatch("execute_module", {
        "module_id": "browser.snapshot",
        "params": {},
    })

    snap_data = snap_result.get("result", snap_result)
    if isinstance(snap_data, str):
        try:
            snap_data = json.loads(snap_data)
        except (ValueError, TypeError):
            snap_data = snap_result

    # Step 4: Detect choices
    choices = detect_choices(snap_data)

    # Step 5: Check for login forms
    inputs = snap_data.get("inputs", [])
    has_password = any(
        i.get("type") == "password"
        for i in inputs
        if isinstance(i, dict)
    )

    if has_password:
        # Login form detected — call ask_user via dispatch
        fields = []
        for inp in inputs:
            if not isinstance(inp, dict):
                continue
            itype = inp.get("type", "text")
            label = inp.get("label", inp.get("placeholder", inp.get("name", "")))
            if not label:
                continue
            fields.append({
                "id": inp.get("name", inp.get("id", label)),
                "type": "password" if itype == "password" else "text",
                "label": label,
            })

        if fields:
            return await dispatch("ask_user", {
                "question": "Login required. Please enter your credentials.",
                "fields": fields,
                "context_key": "",
            })

    if choices:
        # Choices detected — call ask_user via dispatch
        return await dispatch("ask_user", {
            "question": choices.get("question", "Please select an option"),
            "fields": choices.get("fields", []),
            "context_key": "",
        })

    # No choices, no login — just return page info
    page_text = snap_data.get("text", "")
    return {
        "ok": True,
        "message": "Page loaded successfully",
        "page_title": snap_data.get("title", ""),
        "page_url": goto_result.get("url", url),
        "page_text": page_text[:500] if page_text else "",
        "links_count": len(snap_data.get("links", [])),
        "inputs_count": len(snap_data.get("inputs", [])),
    }
