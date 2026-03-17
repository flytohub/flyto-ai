# Copyright 2024 Flyto
# Licensed under the Apache License, Version 2.0
"""Choice detector — auto-convert page choices into ask_user interactive forms.

When a browser snapshot/inspect returns interactive elements (buttons, links,
selects) that form a logical "choice group", this module detects them and
generates an ask_user-compatible field definition.

General purpose: works for any website with choices — hospital departments,
airline seats, shopping filters, form options, etc.
"""
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def detect_choices(result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Detect choice groups in any browser result (snapshot, inspect_page, goto).

    Supports multiple result formats:
    - snapshot: {buttons: [], links: [], selects: []}
    - inspect_page: {data: {elements: [{tag, text, href}]}}
    - goto: {links: [{text, href}]}

    Returns:
        ask_user-compatible dict with question + fields, or None.
    """
    if not isinstance(result, dict):
        return None

    fields = []

    # Normalize: extract items from various formats
    buttons, links, selects = _extract_interactive(result)

    # Source 1: <select> elements with options
    for sel in selects:
        options = sel.get("options", [])
        if len(options) >= 2:
            label = sel.get("label", sel.get("name", sel.get("id", "Selection")))
            fields.append({
                "id": _safe_id(label),
                "type": "select",
                "label": label,
                "options": [o.get("text", o) if isinstance(o, dict) else str(o) for o in options],
                "selector": sel.get("selector", ""),
            })

    # Source 2: Groups of buttons/links
    btn_group = _find_choice_group(buttons, min_size=3, max_size=20)
    if btn_group:
        opts = [b.get("text", "").strip() for b in btn_group if b.get("text", "").strip()]
        if opts:
            fields.append({
                "id": "button_choice",
                "type": "select",
                "label": "Please select",
                "options": opts,
                "_selectors": {b.get("text", "").strip(): b.get("selector", b.get("hint", b.get("href", ""))) for b in btn_group},
            })

    link_group = _find_choice_group(links, min_size=3, max_size=30)
    if link_group and not btn_group:
        opts = [l.get("text", "").strip() for l in link_group if l.get("text", "").strip()]
        if opts:
            fields.append({
                "id": "link_choice",
                "type": "select",
                "label": "Please select",
                "options": opts,
                "_selectors": {l.get("text", "").strip(): l.get("href", l.get("selector", "")) for l in link_group},
            })

    if not fields:
        return None

    return {
        "question": "Please make a selection to continue",
        "fields": fields,
    }


def _extract_interactive(result: Dict[str, Any]) -> Tuple[list, list, list]:
    """Extract buttons, links, selects from various result formats."""
    buttons = result.get("buttons", [])
    links = result.get("links", [])
    selects = result.get("selects", [])

    # inspect_page format: {data: {elements: [...]}}
    data = result.get("data", {})
    if isinstance(data, dict):
        elements = data.get("elements", [])
        if elements and not links:
            for el in elements:
                tag = el.get("tag", "")
                text = (el.get("text", "") or "").strip()
                if not text:
                    continue
                if tag == "a":
                    links.append({"text": text, "href": el.get("href", ""), "selector": el.get("selector", "")})
                elif tag in ("button", "input"):
                    buttons.append({"text": text, "selector": el.get("selector", "")})

    # goto format: {links: [{text, href}]}
    # Already handled by result.get("links")

    return buttons, links, selects


def _find_choice_group(
    items: List[Dict[str, Any]],
    min_size: int = 3,
    max_size: int = 20,
) -> Optional[List[Dict[str, Any]]]:
    """Find a group of similar items that represent choices.

    Heuristics:
    - Items with similar text length (within 3x of each other)
    - Items at similar DOM depth
    - Exclude navigation items (Home, Back, Menu, etc.)
    """
    if len(items) < min_size:
        return None

    # Filter out common navigation/UI items
    nav_words = {
        "home", "back", "menu", "close", "cancel", "login", "logout",
        "search", "submit", "ok", "yes", "no", "next", "previous",
        "首頁", "返回", "選單", "關閉", "取消", "登入", "登出", "搜尋",
    }

    candidates = []
    for item in items:
        text = (item.get("text", "") or "").strip()
        if not text or len(text) > 30:
            continue
        if text.lower() in nav_words:
            continue
        candidates.append(item)

    if len(candidates) < min_size:
        return None

    # Group by similar text length (within 3x ratio)
    if candidates:
        lengths = [len(c.get("text", "").strip()) for c in candidates]
        avg_len = sum(lengths) / len(lengths)
        if avg_len > 0:
            similar = [
                c for c, l in zip(candidates, lengths)
                if 0.3 * avg_len <= l <= 3 * avg_len
            ]
            if len(similar) >= min_size:
                return similar[:max_size]

    return candidates[:max_size] if len(candidates) >= min_size else None


def build_click_action(field_id: str, selected_value: str, fields: List[Dict]) -> Optional[str]:
    """Convert a user's selection back to a CSS selector for clicking.

    Returns the selector string, or None if not found.
    """
    for field in fields:
        if field.get("id") != field_id:
            continue
        selectors_map = field.get("_selectors", {})
        return selectors_map.get(selected_value)
    return None


def _safe_id(text: str) -> str:
    """Convert text to a safe field ID."""
    clean = re.sub(r'[^a-zA-Z0-9\u4e00-\u9fff]', '_', text.strip().lower())
    return clean[:30] or "field"
