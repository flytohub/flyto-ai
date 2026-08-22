# Copyright 2024 Flyto2
# Licensed under the Apache License, Version 2.0
"""Pure validation for Indexer-issued cumulative task amendments.

This module is deliberately stdlib-only and has no dependency on the coding
route, service, provider, or sibling Indexer package.  It reproduces the small
published ``task-amendment.v1`` digest and ancestry contract so the route can
derive a bounded executable delta from persisted evidence without importing a
producer implementation.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

AMENDMENT_VERSION = "task-amendment.v1"
PARENT_DIGEST_TAG = "task-amendment.parent.v1"
ENTRY_DIGEST_TAG = "task-amendment.entry.v1"
CONTRACT_DIGEST_TAG = "task-amendment.contract.v1"
ROOT_DIGEST_TAG = "task-amendment.root.v1"
INTENT_LEDGER_VERSIONS = frozenset(
    {"intent-ledger.v1", "task-context.v1"}
)
MAX_AMENDMENT_CHAIN = 8
MAX_CUMULATIVE_PATHS = 64
MAX_TARGET_LENGTH = 512
MAX_TARGET_SEGMENTS = 24
MAX_TARGET_SEGMENT_LENGTH = 255
TEST_COVERAGE_QUERY_PREFIX = "tests covering "
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_CONTROL_CHAR_RE = re.compile(r"[\x00-\x1f\x7f]")
_WINDOWS_DRIVE_RE = re.compile(r"^[A-Za-z]:")
_GLOB_METACHARACTERS = frozenset("*?[]")


class AmendmentContractError(ValueError):
    """One stable, content-free amendment contract refusal."""

    def __init__(self, code: str):
        super().__init__(code)
        self.code = code


@dataclass(frozen=True)
class AmendmentBoundary:
    """The exact path partition and finite successor generation."""

    original: frozenset[str]
    added: frozenset[str]
    amendment_index: int


def _fail(code: str) -> None:
    raise AmendmentContractError(code)


def _digest(*parts: Any) -> str:
    encoded = json.dumps(
        list(parts), sort_keys=True, separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _text(value: Any, limit: int) -> str:
    return value[:limit] if isinstance(value, str) else ""


def _index(value: Any) -> int:
    return value if not isinstance(value, bool) and isinstance(value, int) else -1


def parent_contract_digest(parent: Mapping[str, Any]) -> str:
    """Reproduce Indexer's bounded ``task-amendment.parent.v1`` digest."""

    def section(name: str) -> Mapping[str, Any]:
        value = parent.get(name)
        return value if isinstance(value, Mapping) else {}

    profile = section("task_profile")
    ledger = section("intent_ledger")
    instruction = section("instruction_context")
    amendment = section("task_amendment")
    return _digest(
        PARENT_DIGEST_TAG,
        _text(profile.get("version"), 64),
        _text(profile.get("task_id"), 160),
        _text(profile.get("intent"), 32)
        or _text(profile.get("original_intent"), 32),
        _text(profile.get("project"), 512),
        _text(ledger.get("version"), 64),
        _text(ledger.get("fingerprint"), 64),
        _text(instruction.get("version"), 64),
        _text(instruction.get("fingerprint"), 64),
        _text(amendment.get("version"), 64),
        _text(amendment.get("contract_id"), 64),
        _index(amendment.get("amendment_index")),
    )


def root_contract_id(
    root_task_id: str, objective: str, paths: Sequence[str],
) -> str:
    return "amd_root_" + _digest(
        ROOT_DIGEST_TAG, root_task_id, objective, list(paths),
    )[:20]


def amendment_contract_id(
    root_task_id: str,
    amendment_index: int,
    objective: str,
    cumulative_paths: Sequence[str],
    parent_contract_id: str,
    parent_digest: str,
) -> str:
    return "amd_" + _digest(
        CONTRACT_DIGEST_TAG,
        root_task_id,
        amendment_index,
        objective,
        list(cumulative_paths),
        parent_contract_id,
        parent_digest,
    )[:24]


def chain_entry_digest(entry: Mapping[str, Any]) -> str:
    return _digest(
        ENTRY_DIGEST_TAG,
        entry.get("contract_id"),
        entry.get("parent_contract_id") or "",
        entry.get("root_task_id"),
        entry.get("project") or "",
        entry.get("amendment_index"),
        entry.get("target_count"),
        entry.get("contract_digest"),
    )


def chain_entry(
    *,
    contract_id: str,
    parent_contract_id: str | None,
    root_task_id: str,
    project: str,
    amendment_index: int,
    target_count: int,
    contract_digest: str,
) -> dict[str, Any]:
    entry = {
        "contract_id": contract_id,
        "parent_contract_id": parent_contract_id,
        "root_task_id": root_task_id,
        "project": project,
        "amendment_index": amendment_index,
        "target_count": target_count,
        "contract_digest": contract_digest,
    }
    entry["entry_digest"] = chain_entry_digest(entry)
    return entry


def _bounded_index(value: Any) -> int:
    index = _index(value)
    if not 1 <= index <= MAX_AMENDMENT_CHAIN:
        _fail("amendment_parent_proof_mismatch")
    return index


def _canonical_path(value: Any) -> str:
    if not isinstance(value, str) or not value or len(value) > MAX_TARGET_LENGTH:
        return ""
    if (
        value.startswith(("/", "~"))
        or "\\" in value
        or _WINDOWS_DRIVE_RE.match(value)
        or _CONTROL_CHAR_RE.search(value)
        or any(character in _GLOB_METACHARACTERS for character in value)
    ):
        return ""
    segments = value.split("/")
    if len(segments) > MAX_TARGET_SEGMENTS or any(
        segment in {"", ".", ".."} or len(segment) > MAX_TARGET_SEGMENT_LENGTH
        for segment in segments
    ):
        return ""
    return "/".join(segments)


def _bounded_path_list(value: Any, *, allow_empty: bool) -> list[str]:
    if not isinstance(value, (list, tuple)):
        _fail("amendment_path_boundary_invalid")
    bounded = [_canonical_path(item) for item in value]
    if (not allow_empty and not bounded) or any(not item for item in bounded):
        _fail("amendment_path_boundary_invalid")
    if len(bounded) > MAX_CUMULATIVE_PATHS or len(set(bounded)) != len(bounded):
        _fail("amendment_path_boundary_invalid")
    return bounded


def _bounded_paths(amendment: Mapping[str, Any], name: str) -> list[str]:
    return _bounded_path_list(
        amendment.get(name), allow_empty=name == "added_paths",
    )


def _contract_sections(
    successor: Mapping[str, Any], parent: Mapping[str, Any], host_project: str,
) -> tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]:
    amendment = successor.get("task_amendment")
    parent_profile = parent.get("task_profile")
    successor_profile = successor.get("task_profile")
    parent_ledger = parent.get("intent_ledger")
    parent_instruction = parent.get("instruction_context")
    successor_ledger = successor.get("intent_ledger")
    successor_instruction = successor.get("instruction_context")
    if not isinstance(amendment, Mapping):
        _fail("amendment_parent_proof_missing")
    sections = (
        parent_profile, successor_profile, parent_ledger,
        parent_instruction, successor_ledger, successor_instruction,
    )
    if not all(isinstance(item, Mapping) for item in sections):
        _fail("amendment_parent_proof_missing")
    _validate_context_contract(
        parent_profile, successor_profile, parent_ledger, successor_ledger,
        parent_instruction, successor_instruction,
    )
    if not isinstance(host_project, str) or not host_project:
        _fail("amendment_parent_proof_mismatch")
    return amendment, parent_profile, parent_ledger


def _profile_context_mirror(
    profile: Mapping[str, Any],
    ledger: Mapping[str, Any],
    instruction: Mapping[str, Any],
) -> bool:
    return (
        profile.get("intent_fingerprint") == ledger.get("fingerprint")
        and profile.get("instruction_fingerprint")
        == instruction.get("fingerprint")
    )


def _validate_context_contract(
    parent_profile: Mapping[str, Any],
    successor_profile: Mapping[str, Any],
    parent_ledger: Mapping[str, Any],
    successor_ledger: Mapping[str, Any],
    parent_instruction: Mapping[str, Any],
    successor_instruction: Mapping[str, Any],
) -> None:
    """Validate the pinned v2 context schema and all profile mirrors."""

    profile_versions = (
        parent_profile.get("version"), successor_profile.get("version"),
    )
    ledger_versions = (
        parent_ledger.get("version"), successor_ledger.get("version"),
    )
    instruction_versions = (
        parent_instruction.get("version"), successor_instruction.get("version"),
    )
    if (
        profile_versions != ("task-contract.v2", "task-contract.v2")
        or any(version not in INTENT_LEDGER_VERSIONS for version in ledger_versions)
        or instruction_versions != ("task-context.v1", "task-context.v1")
    ):
        _fail("amendment_parent_proof_mismatch")
    fingerprints = (
        parent_ledger.get("fingerprint"), successor_ledger.get("fingerprint"),
        parent_instruction.get("fingerprint"), successor_instruction.get("fingerprint"),
    )
    if any(not isinstance(item, str) or not _SHA256_RE.fullmatch(item) for item in fingerprints):
        _fail("amendment_parent_proof_mismatch")
    intents = (parent_profile.get("intent"), successor_profile.get("intent"))
    if not isinstance(intents[0], str) or not intents[0] or intents[0] != intents[1]:
        _fail("amendment_parent_proof_mismatch")
    if not _profile_context_mirror(parent_profile, parent_ledger, parent_instruction):
        _fail("amendment_profile_mismatch")
    if not _profile_context_mirror(successor_profile, successor_ledger, successor_instruction):
        _fail("amendment_profile_mismatch")
    descriptions = (parent_ledger.get("description"), successor_ledger.get("description"))
    if not isinstance(descriptions[0], str) or descriptions[1] != descriptions[0]:
        _fail("amendment_chain_identity_mismatch")


def _validate_task_identity(
    amendment: Mapping[str, Any],
    parent_profile: Mapping[str, Any],
    successor: Mapping[str, Any],
    host_project: str,
) -> str:
    successor_profile = successor["task_profile"]
    root_task_id = _text(amendment.get("root_task_id"), 160)
    identity = (
        amendment.get("version"), amendment.get("status"), root_task_id,
        parent_profile.get("task_id"), successor_profile.get("task_id"),
    )
    if identity != (
        AMENDMENT_VERSION, "amended", root_task_id, root_task_id, root_task_id,
    ) or not root_task_id:
        _fail("amendment_parent_proof_mismatch")
    projects = (parent_profile.get("project"), successor_profile.get("project"))
    if projects != (host_project, host_project):
        _fail("amendment_parent_proof_mismatch")
    return root_task_id


def _parent_amendment_index(
    amendment: Any, parent_profile: Mapping[str, Any],
) -> int:
    if amendment is None:
        return 0
    if not isinstance(amendment, Mapping):
        _fail("amendment_parent_proof_mismatch")
    identity = (
        amendment.get("version"), amendment.get("status"),
        amendment.get("root_task_id"),
    )
    expected = (AMENDMENT_VERSION, "amended", parent_profile.get("task_id"))
    contract_id = amendment.get("contract_id")
    if identity != expected or not isinstance(contract_id, str) or not contract_id:
        _fail("amendment_parent_proof_mismatch")
    return _bounded_index(amendment.get("amendment_index"))


def _successor_index(
    amendment: Mapping[str, Any], parent: Mapping[str, Any], parent_index: int,
) -> int:
    index = _bounded_index(amendment.get("amendment_index"))
    if index != parent_index + 1:
        _fail("amendment_parent_proof_mismatch")
    digest = amendment.get("parent_contract_digest")
    if digest != parent_contract_digest(parent):
        _fail("amendment_parent_proof_mismatch")
    return index


def _validate_path_partition(
    successor: Mapping[str, Any],
    parent_ledger: Mapping[str, Any],
    original: Sequence[str],
    added: Sequence[str],
    cumulative: Sequence[str],
    host_requested_paths: Sequence[str],
) -> None:
    parent_paths = _bounded_path_list(
        parent_ledger.get("allowed_paths"), allow_empty=False,
    )
    successor_paths = _bounded_path_list(
        successor["intent_ledger"].get("allowed_paths"), allow_empty=False,
    )
    host_paths = _bounded_path_list(host_requested_paths, allow_empty=True)
    expected = list(dict.fromkeys(list(original) + list(added)))
    host_expected = list(dict.fromkeys(parent_paths + host_paths))
    if parent_paths != list(original) or set(original) & set(added):
        _fail("amendment_path_boundary_invalid")
    if (
        list(cumulative) != expected
        or successor_paths != expected
        or host_expected != expected
    ):
        _fail("amendment_path_boundary_invalid")


def _entry_shape(item: Any) -> dict[str, Any]:
    if not isinstance(item, Mapping):
        _fail("amendment_chain_malformed")
    contract_id = item.get("contract_id")
    parent_id = item.get("parent_contract_id")
    contract_digest = item.get("contract_digest")
    index = _index(item.get("amendment_index"))
    count = _index(item.get("target_count"))
    if not isinstance(contract_id, str) or not contract_id:
        _fail("amendment_chain_malformed")
    if parent_id is not None and not isinstance(parent_id, str):
        _fail("amendment_chain_malformed")
    if not isinstance(contract_digest, str) or not _SHA256_RE.fullmatch(contract_digest):
        _fail("amendment_chain_malformed")
    if index < 0 or count < 0:
        _fail("amendment_chain_malformed")
    return {
        "contract_id": contract_id, "parent_contract_id": parent_id,
        "root_task_id": item.get("root_task_id"), "project": item.get("project"),
        "amendment_index": index, "target_count": count,
        "contract_digest": contract_digest,
    }


def _validated_entry(
    item: Any,
    *,
    position: int,
    root_task_id: str,
    project: str,
    previous_id: str | None,
    previous_count: int,
) -> dict[str, Any]:
    entry = _entry_shape(item)
    if (entry["root_task_id"], entry["project"]) != (root_task_id, project):
        _fail("amendment_chain_identity_mismatch")
    if entry["amendment_index"] != position:
        _fail("amendment_chain_index_invalid")
    if entry["parent_contract_id"] != previous_id:
        _fail("amendment_chain_linkage_invalid")
    count = entry["target_count"]
    if count < previous_count or count > MAX_CUMULATIVE_PATHS:
        _fail("amendment_chain_count_invalid")
    if item.get("entry_digest") != chain_entry_digest(entry):
        _fail("amendment_chain_tampered")
    entry["entry_digest"] = item.get("entry_digest")
    return entry


def _validated_chain(
    raw_chain: Any,
    *,
    root_task_id: str,
    project: str,
    own_contract_id: Any,
) -> list[dict[str, Any]]:
    if not isinstance(raw_chain, (list, tuple)):
        _fail("amendment_chain_malformed")
    # Pinned Indexer verify_chain refuses len >= 8. Generation eight therefore
    # remains fail-closed until the producer/verifier off-by-one is resolved.
    if len(raw_chain) >= MAX_AMENDMENT_CHAIN:
        _fail("amendment_chain_oversized")
    chain = []
    seen = set()
    for position, item in enumerate(raw_chain):
        previous_id = chain[-1]["contract_id"] if chain else None
        previous_count = chain[-1]["target_count"] if chain else 0
        entry = _validated_entry(
            item, position=position, root_task_id=root_task_id, project=project,
            previous_id=previous_id, previous_count=previous_count,
        )
        if entry["contract_id"] in seen:
            _fail("amendment_chain_cyclic")
        seen.add(entry["contract_id"])
        chain.append(entry)
    if isinstance(own_contract_id, str) and own_contract_id in seen:
        _fail("amendment_chain_cyclic")
    return chain


def _objective(
    amendment: Mapping[str, Any],
    parent_ledger: Mapping[str, Any],
) -> str:
    objective = parent_ledger.get("description")
    if not isinstance(objective, str) or not objective or len(objective) > 4096:
        _fail("amendment_chain_identity_mismatch")
    if amendment.get("objective") != objective:
        _fail("amendment_chain_identity_mismatch")
    return objective


def _validate_profile_mirror(
    profile: Mapping[str, Any],
    amendment: Mapping[str, Any],
    objective: str,
    amendment_index: int,
) -> None:
    """Require the producer's task-profile amendment mirror to be exact."""

    expected = (
        amendment.get("root_task_id"), objective, amendment_index,
        amendment.get("contract_id"),
    )
    actual = (
        profile.get("root_task_id"), profile.get("description"),
        profile.get("amendment_index"), profile.get("amendment_contract_id"),
    )
    if actual != expected:
        _fail("amendment_profile_mismatch")


def _parent_chain_authority(
    *,
    parent: Mapping[str, Any],
    parent_index: int,
    root_task_id: str,
    project: str,
    objective: str,
    original: Sequence[str],
) -> tuple[str, list[dict[str, Any]], str]:
    parent_digest = parent_contract_digest(parent)
    if parent_index == 0:
        return root_contract_id(root_task_id, objective, original), [], parent_digest
    amendment = parent.get("task_amendment")
    if not isinstance(amendment, Mapping):
        _fail("amendment_chain_malformed")
    contract_id = amendment.get("contract_id")
    chain = _validated_chain(
        amendment.get("chain"), root_task_id=root_task_id, project=project,
        own_contract_id=contract_id,
    )
    if len(chain) != parent_index:
        _fail("amendment_chain_index_invalid")
    predecessor = chain[-1]
    linkage = (
        amendment.get("parent_contract_id"), amendment.get("parent_contract_digest"),
    )
    expected_linkage = (predecessor["contract_id"], predecessor["contract_digest"])
    if linkage != expected_linkage:
        _fail("amendment_chain_linkage_invalid")
    expected_id = amendment_contract_id(
        root_task_id, parent_index, objective, original,
        predecessor["contract_id"], predecessor["contract_digest"],
    )
    if contract_id != expected_id or amendment.get("objective") != objective:
        _fail("amendment_chain_tampered")
    return contract_id, chain, parent_digest


def _validate_successor_chain(
    *,
    amendment: Mapping[str, Any],
    parent: Mapping[str, Any],
    parent_profile: Mapping[str, Any],
    successor_profile: Mapping[str, Any],
    parent_ledger: Mapping[str, Any],
    host_project: str,
    parent_index: int,
    amendment_index: int,
    original: Sequence[str],
    cumulative: Sequence[str],
) -> None:
    root_task_id = str(amendment.get("root_task_id") or "")
    objective = _objective(amendment, parent_ledger)
    _validate_profile_mirror(
        successor_profile, amendment, objective, amendment_index,
    )
    parent_amendment = parent.get("task_amendment")
    if parent_index >= 1 and isinstance(parent_amendment, Mapping):
        _validate_profile_mirror(
            parent_profile, parent_amendment, objective, parent_index,
        )
    elif parent_profile.get("description") != objective:
        _fail("amendment_chain_identity_mismatch")
    parent_id, parent_chain, parent_digest = _parent_chain_authority(
        parent=parent, parent_index=parent_index, root_task_id=root_task_id,
        project=host_project, objective=objective, original=original,
    )
    contract_id = amendment.get("contract_id")
    if amendment.get("parent_contract_id") != parent_id:
        _fail("amendment_chain_linkage_invalid")
    chain = _validated_chain(
        amendment.get("chain"), root_task_id=root_task_id, project=host_project,
        own_contract_id=contract_id,
    )
    if len(chain) != amendment_index or chain[:-1] != parent_chain:
        _fail("amendment_chain_index_invalid")
    expected_parent = chain_entry(
        contract_id=parent_id,
        parent_contract_id=parent_chain[-1]["contract_id"] if parent_chain else None,
        root_task_id=root_task_id, project=host_project,
        amendment_index=parent_index, target_count=len(original),
        contract_digest=parent_digest,
    )
    if chain[-1] != expected_parent:
        _fail("amendment_chain_tampered")
    expected_id = amendment_contract_id(
        root_task_id, amendment_index, objective, cumulative,
        parent_id, parent_digest,
    )
    if contract_id != expected_id:
        _fail("amendment_chain_tampered")


def validate_amendment_contract(
    successor: Mapping[str, Any],
    parent: Mapping[str, Any],
    host_project: str,
    host_requested_paths: Sequence[str],
) -> AmendmentBoundary:
    """Validate one complete successor and return its exact path boundary."""

    amendment, parent_profile, parent_ledger = _contract_sections(
        successor, parent, host_project,
    )
    _validate_task_identity(amendment, parent_profile, successor, host_project)
    parent_index = _parent_amendment_index(
        parent.get("task_amendment"), parent_profile,
    )
    amendment_index = _successor_index(amendment, parent, parent_index)
    original = _bounded_paths(amendment, "original_paths")
    added = _bounded_paths(amendment, "added_paths")
    cumulative = _bounded_paths(amendment, "cumulative_paths")
    _validate_path_partition(
        successor, parent_ledger, original, added, cumulative,
        host_requested_paths,
    )
    _validate_successor_chain(
        amendment=amendment, parent=parent, parent_profile=parent_profile,
        successor_profile=successor["task_profile"],
        parent_ledger=parent_ledger, host_project=host_project,
        parent_index=parent_index, amendment_index=amendment_index,
        original=original, cumulative=cumulative,
    )
    return AmendmentBoundary(
        original=frozenset(original), added=frozenset(added),
        amendment_index=amendment_index,
    )


def plan_source_coordinates(
    source: Mapping[str, Any],
) -> tuple[Sequence[Any], Sequence[Any]]:
    resolved = source.get("resolved_targets")
    targets = source.get("targets")
    if resolved is None or targets is None:
        profile = source.get("task_profile")
        if isinstance(profile, Mapping):
            resolved = profile.get("resolved_targets")
            targets = profile.get("targets")
    if not isinstance(resolved, (list, tuple)) or not isinstance(targets, (list, tuple)):
        _fail("amendment_plan_boundary_missing")
    return resolved, targets


def covered_amendment_paths(sources: Sequence[Mapping[str, Any]]) -> frozenset[str]:
    covered = set()
    for source in sources:
        resolved, _targets = plan_source_coordinates(source)
        for item in resolved:
            value = item.get("input") if isinstance(item, Mapping) else None
            if isinstance(value, str) and value:
                covered.add(value)
    return frozenset(covered)


def reusable_step_identity(scope: str, step: Mapping[str, Any]) -> str:
    args = step.get("args")
    if args is not None and not isinstance(args, Mapping):
        _fail("malformed_evidence")
    identity = [
        scope, step.get("tool"), dict(args or {}),
        step.get("required"), step.get("purpose"),
    ]
    try:
        return json.dumps(
            identity, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        )
    except (TypeError, ValueError) as exc:
        raise AmendmentContractError("malformed_evidence") from exc


def reusable_parent_step_counts(
    groups: Sequence[tuple[str, Sequence[Mapping[str, Any]]]],
    gate_tools: frozenset[str],
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for scope, steps in groups:
        for step in steps:
            if step.get("tool") in gate_tools:
                continue
            identity = reusable_step_identity(scope, step)
            counts[identity] = counts.get(identity, 0) + 1
    return counts


def _coordinate_owners(
    resolved: Sequence[Any],
    original: frozenset[str],
    added: frozenset[str],
) -> tuple[frozenset[str], Mapping[str, frozenset[str]]]:
    declared = set()
    owners: dict[str, set[str]] = {}
    for item in resolved:
        if not isinstance(item, Mapping):
            _fail("amendment_plan_boundary_missing")
        value = item.get("input")
        if not isinstance(value, str) or value not in original | added:
            _fail("amendment_plan_boundary_mismatch")
        declared.add(value)
        owner = "added" if value in added else "original"
        for key in ("input", "path", "symbol_id"):
            coordinate = item.get(key)
            if isinstance(coordinate, str) and coordinate:
                owners.setdefault(coordinate, set()).add(owner)
    return frozenset(declared), {
        coordinate: frozenset(values) for coordinate, values in owners.items()
    }


def _step_owner(
    step: Mapping[str, Any], coordinate_owners: Mapping[str, frozenset[str]],
) -> str:
    args = step.get("args")
    if args is not None and not isinstance(args, Mapping):
        _fail("malformed_evidence")
    args = args or {}
    coordinates = [
        value
        for key in ("target", "symbol_id", "path", "file_path")
        for value in (args.get(key),)
        if isinstance(value, str) and value
    ]
    query = args.get("query")
    if isinstance(query, str) and query.startswith(TEST_COVERAGE_QUERY_PREFIX):
        coordinates.append(query[len(TEST_COVERAGE_QUERY_PREFIX):])
    owners = {
        owner for coordinate in coordinates
        for owner in coordinate_owners.get(coordinate, ())
    }
    if len(owners) != 1:
        _fail("amendment_plan_step_unattributable")
    return next(iter(owners))


def amendment_delta_steps(
    *,
    scope: str,
    steps: Sequence[Mapping[str, Any]],
    source: Mapping[str, Any],
    boundary: AmendmentBoundary,
    reusable_counts: dict[str, int],
    gate_tools: frozenset[str],
) -> list[Mapping[str, Any]]:
    """Derive exact novel work while rerunning every successor gate."""

    resolved, targets = plan_source_coordinates(source)
    declared, owners = _coordinate_owners(
        resolved, boundary.original, boundary.added,
    )
    if any(not isinstance(item, str) for item in targets) or set(targets) != declared:
        _fail("amendment_plan_boundary_mismatch")
    delta = []
    for step in steps:
        if step.get("tool") in gate_tools:
            delta.append(step)
            continue
        owner = _step_owner(step, owners)
        identity = reusable_step_identity(scope, step)
        if owner == "original" and reusable_counts.get(identity, 0) > 0:
            reusable_counts[identity] -= 1
            continue
        delta.append(step)
    return delta
