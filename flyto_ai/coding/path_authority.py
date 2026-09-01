# Copyright 2024 Flyto2
# Licensed under the Apache License, Version 2.0
"""Narrow prose-to-path authority projections for the coding route."""
from __future__ import annotations

import re
from typing import Mapping, Sequence


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


def amendment_delta_targets(
    parent_contract: Mapping[str, object],
    prior_scope: Sequence[str],
    explicit_targets: Sequence[str],
) -> list[str]:
    """Return a bounded amendment delta without replaying parent authority."""
    ledger = parent_contract.get("intent_ledger")
    ledger = ledger if isinstance(ledger, Mapping) else {}
    raw_parent_paths = ledger.get("allowed_paths")
    if not isinstance(raw_parent_paths, (list, tuple)):
        raw_parent_paths = ledger.get("targets")
    parent_order = [
        item for item in (raw_parent_paths or ()) if isinstance(item, str)
    ]
    parent_paths = set(parent_order)
    novel = list(dict.fromkeys(
        [item for item in prior_scope if item not in parent_paths]
        + [item for item in explicit_targets if item not in parent_paths]
    ))
    named_parent = [item for item in explicit_targets if item in parent_paths]
    # An audit can name several existing parent-owned files in one rework.
    # They are the active analysis set even though they add no new authority.
    # Keep them all (the caller applies the unchanged per-amendment bound),
    # while still refusing to replay unnamed cumulative parent scope.
    # Preserve the existing new-scope priority: when this round genuinely
    # widens authority, only that novel delta is planned.  A same-scope audit
    # has no novel path, so all explicitly named parent targets become its
    # active analysis set instead of the historical single-path fallback.
    return novel or list(dict.fromkeys(named_parent)) or parent_order[:1]
