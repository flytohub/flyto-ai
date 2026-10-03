# Copyright 2024 Flyto2
# Licensed under the Apache License, Version 2.0
"""The handoff records are read as fact by the next agent; keep them parseable."""

from __future__ import annotations

import re
from pathlib import Path

HANDOFFS = Path(__file__).resolve().parents[1] / "handoffs"
_MARKER = re.compile(r"^(<<<<<<< |=======$|>>>>>>> )", re.MULTILINE)


def test_no_merge_conflict_markers_in_handoffs():
    offenders = [
        path.name for path in sorted(HANDOFFS.glob("*.md"))
        if _MARKER.search(path.read_text(encoding="utf-8"))
    ]
    assert offenders == []


def test_registry_rows_name_existing_files():
    rows = (HANDOFFS / "_registry.md").read_text(encoding="utf-8").splitlines()
    names = [
        cells[3].strip() for cells in (row.split("|") for row in rows)
        if len(cells) >= 5 and cells[3].strip().endswith(".md")
    ]
    assert names
    missing = [name for name in names if not (HANDOFFS / name).is_file()]
    assert missing == []
