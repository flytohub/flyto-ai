# Copyright 2024 Flyto2
# Licensed under the Apache License, Version 2.0
"""Narrow prose-to-path authority projections for the coding route."""
from __future__ import annotations

import re


# A task may declare an exact numbered path allowlist before describing each
# item. In that form the mutation authority belongs to the list declaration,
# not to a verb repeated beside every path. Keep this deliberately narrower
# than generic ``include`` prose: a target/intent-ledger cue, a strong exact-set
# phrase, a colon, a numbered item and a short item-local description are all
# required. The route still owns filesystem, suffix, polarity, symlink, parent
# and count validation.
_EXPLICIT_PATH_LIST_CUE_RE = re.compile(
    r"\b(?:target|intent)(?:\s*/\s*(?:target|intent))?\s+ledger\b"
    r"[^;.\n\r]{0,64}"
    r"\b(?:include[sd]?|contain[sd]?|allow(?:s|ed)?|target(?:s|ed)?)\b"
    r"[^;.\n\r]{0,80}"
    r"\b(?:exactly|only|each\s+of\s+these|all\s+of\s+these)\b"
    r"[^;.\n\r]{0,96}"
    r"\b(?:(?:repository|repo)[-\s]+relative\s+)?(?:paths?|files?)\b"
    r"[^;.\n\r:]{0,64}:",
    re.IGNORECASE,
)
_NUMBERED_PATH_ITEM_RE = re.compile(r"(?<![A-Za-z0-9])\(\d{1,2}\)\s*")
_EXPLICIT_PATH_LIST_SPAN = 2048
_EXPLICIT_PATH_ITEM_WINDOW = 160


def is_numbered_exact_path_item(message: str, position: int) -> bool:
    """Whether ``position`` is in a bounded item of an exact target ledger."""
    text = str(message or "")
    if position < 0 or position > len(text):
        return False
    list_starts = tuple(
        match.end() for match in _EXPLICIT_PATH_LIST_CUE_RE.finditer(text)
    )
    for list_start in reversed(list_starts):
        if list_start > position:
            continue
        if position - list_start > _EXPLICIT_PATH_LIST_SPAN:
            break
        item_start = max(list_start, position - _EXPLICIT_PATH_ITEM_WINDOW)
        if any(_NUMBERED_PATH_ITEM_RE.finditer(text, item_start, position)):
            return True
    return False
