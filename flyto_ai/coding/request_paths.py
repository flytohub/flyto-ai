# Copyright 2024 Flyto2
# Licensed under the Apache License, Version 2.0
"""Bounded first-round request-path authority for the coding route."""
from __future__ import annotations

import re
from pathlib import Path, PurePosixPath
from typing import Tuple

from flyto_ai.coding.path_authority import is_numbered_exact_path_item


EXPLICIT_PATH_RE = re.compile(
    r"(?<![A-Za-z0-9_.:/\\()[\]\-])"
    r"((?:[A-Za-z0-9_.()[\]\-]+/)*[A-Za-z0-9_.()[\]\-]+)"
    r"(?![A-Za-z0-9_./\\()[\]\-])"
)
MAX_EXPLICIT_REQUEST_TARGETS = 64
CLAUSE_BOUNDARY_RE = re.compile(r"[;\n\r\x0b\x0c]|[.!?](?=\s|$)")
_DRIVE_PREFIX_RE = re.compile(r"^[A-Za-z]:")
_NEW_FILE_SUFFIX_RE = re.compile(r"^\.[A-Za-z0-9]{1,8}$")
_VERSION_LABEL_SUFFIX_RE = re.compile(r"^\.[0-9]+$")
_MUTATION_VERB_RE = re.compile(
    r"\b(add|create|new|write|generate|regenerate|emit|produce|introduce|implement|"
    r"update|edit|modify|change|rewrite|replace|amend|patch|fix|repair|"
    r"rename|move|delete|remove|drop|touch|append|extend)\b[^A-Za-z0-9]*$",
    re.IGNORECASE,
)
_MUTATION_VERB_WINDOW = 48
_NEGATIVE_LEADING_RE = re.compile(
    r"\b(?:do(?:es)?\s+not|do(?:es)?n[’']?t|did\s+not|didn[’']?t|"
    r"must\s+not|mustn[’']?t|must\s+never|may\s+not|might\s+not|"
    r"shall\s+not|should\s+not|shouldn[’']?t|will\s+not|won[’']?t|"
    r"would\s+not|wouldn[’']?t|cannot|can\s+not|can[’']?t|"
    r"could\s+not|couldn[’']?t|never|without|avoid(?:s|ed|ing)?|refrain|"
    r"leave\s+alone|hands\s+off|no\s+changes?\s+to|no\s+edits?\s+to|"
    r"exclude|excluding|not\s+allowed|forbidden|prohibited|off[\s-]limits|"
    r"out\s+of\s+scope)\b",
    re.IGNORECASE,
)
_NEGATIVE_TRAILING_RE = re.compile(
    r"\b(?:must\s+not|mustn[’']?t|must\s+never|may\s+not|might\s+not|"
    r"shall\s+not|should\s+not|shouldn[’']?t|will\s+not|won[’']?t|"
    r"would\s+not|wouldn[’']?t|cannot|can\s+not|can[’']?t|"
    r"is\s+not|are\s+not|isn[’']?t|aren[’']?t|unchanged|untouched|"
    r"unmodified|unaltered|stays?\s+the\s+same|read[\s-]only|off[\s-]limits|"
    r"out\s+of\s+scope|as[\s-]is|not\s+allowed|forbidden|prohibited)\b",
    re.IGNORECASE,
)
_MUTATION_HEADING_RE = re.compile(
    r"\b(?:add|create|write|generate|regenerate|emit|produce|introduce|implement)"
    r"(?:\s+(?![:\n\r.!?])[^:\n\r.!?]{0,80})?\s*:\s*",
    re.IGNORECASE,
)
_EXACT_NEW_HEADING_RE = re.compile(
    r"(?:add|create|write|generate|regenerate|emit|produce|introduce|implement)"
    r"\s+(?:(?:only|these)\s+)*exact\s+new\s+(?:files?|paths?)\s*:\s*",
    re.IGNORECASE,
)
_COMMAND_OR_EVIDENCE_RE = re.compile(
    r"\b(?:run|execute|invoke|call|command|check|proof|evidence|receipt|passed|"
    r"verified|via|through|using)\b",
    re.IGNORECASE,
)


def _relative_path(value: str) -> bool:
    if not value or "\\" in value or _DRIVE_PREFIX_RE.match(value):
        return False
    if value.startswith(("/", "~")) or len(value) > 200:
        return False
    if any(char < " " or char == "\x7f" for char in value):
        return False
    return ".." not in PurePosixPath(value).parts


def _admissible(value: str, root: Path, *, new_authority: bool) -> bool:
    if not _relative_path(value) or EXPLICIT_PATH_RE.fullmatch(value) is None:
        return False
    relative = PurePosixPath(value)
    spelled = root / relative
    try:
        cursor = root
        for component in relative.parts:
            next_path = cursor / component
            if next_path.is_symlink():
                return False
            if not next_path.exists():
                break
            cursor = next_path
            if cursor != spelled and not cursor.is_dir():
                return False
        if spelled.exists():
            return spelled.is_file()
        suffix = relative.suffix
        return bool(
            new_authority
            and _NEW_FILE_SUFFIX_RE.fullmatch(suffix)
            and not _VERSION_LABEL_SUFFIX_RE.fullmatch(suffix)
        )
    except (OSError, RuntimeError):
        return False


def prohibited_spans(text: str) -> Tuple[Tuple[int, int], ...]:
    """Return clause-local spans whose paths are explicitly fenced off."""
    clauses = []
    start = 0
    for boundary in CLAUSE_BOUNDARY_RE.finditer(text):
        clauses.append((start, boundary.start()))
        start = boundary.end()
    clauses.append((start, len(text)))
    spans = []
    for begin, end in clauses:
        clause = text[begin:end]
        leading = _NEGATIVE_LEADING_RE.search(clause)
        if leading is not None:
            spans.append((begin + leading.start(), end))
        last_trailing = 0
        for trailing in _NEGATIVE_TRAILING_RE.finditer(clause):
            last_trailing = trailing.end()
        if last_trailing:
            spans.append((begin, begin + last_trailing))
    return tuple(spans)


def _statement_end(text: str, start: int) -> int:
    boundary = re.search(r"[\n\r\x0b\x0c]|[.!?](?=\s|$)", text[start:])
    return len(text) if boundary is None else start + boundary.start()


def _statement_start(text: str, end: int) -> int:
    boundaries = list(re.finditer(r"[\n\r\x0b\x0c]|[.!?](?=\s|$)", text[:end]))
    return 0 if not boundaries else boundaries[-1].end()


def _headed_lists(
    text: str, root: Path,
) -> tuple[list[tuple[int, tuple[str, ...], bool]], tuple[tuple[int, int], ...]]:
    """Project atomic affirmative headed lists and mask every recognized one."""
    projected = []
    spans = []
    for heading in _MUTATION_HEADING_RE.finditer(text):
        begin = _statement_start(text, heading.start())
        end = _statement_end(text, heading.end())
        spans.append((begin, end))
        body = text[heading.end():end].strip()
        items = [item.strip() for item in re.split(r"[,;]", body)]
        invalid = (
            _EXACT_NEW_HEADING_RE.fullmatch(heading.group()) is None
            or not body
            or not items
            or len(items) > MAX_EXPLICIT_REQUEST_TARGETS
            or any(not item for item in items)
            or _NEGATIVE_LEADING_RE.search(text[begin:end]) is not None
            or _NEGATIVE_TRAILING_RE.search(text[begin:end]) is not None
            or _COMMAND_OR_EVIDENCE_RE.search(body) is not None
            or any(not _admissible(item, root, new_authority=True) for item in items)
        )
        if not invalid:
            projected.append((heading.start(), tuple(items), True))
    return projected, tuple(spans)


def explicit_request_targets(message: str, working_dir: str) -> list[str]:
    """Return bounded, ordered path authority from one first-round request."""
    try:
        root = Path(working_dir).resolve(strict=True)
    except (OSError, RuntimeError):
        return []
    text = str(message or "")
    headed, headed_spans = _headed_lists(text, root)
    forbidden = prohibited_spans(text)
    projected = list(headed)
    for match in EXPLICIT_PATH_RE.finditer(text):
        position = match.start(1)
        if any(begin <= position < end for begin, end in headed_spans + forbidden):
            continue
        raw = match.group(1)
        for value in dict.fromkeys((raw, raw.rstrip("."))):
            if not value or (value == raw and raw.rstrip(".") != raw and not (root / raw).exists()):
                continue
            new_authority = bool(
                _MUTATION_VERB_RE.search(
                    text[max(0, position - _MUTATION_VERB_WINDOW):position]
                )
                or is_numbered_exact_path_item(text, position)
            )
            if _admissible(value, root, new_authority=new_authority):
                projected.append((position, (value,), False))
                break
    targets = []
    for _, values, atomic in sorted(projected, key=lambda item: item[0]):
        additions = [value for value in values if value not in targets]
        additions = list(dict.fromkeys(additions))
        if len(targets) + len(additions) > MAX_EXPLICIT_REQUEST_TARGETS:
            if atomic:
                continue
            break
        targets.extend(additions)
    return targets
