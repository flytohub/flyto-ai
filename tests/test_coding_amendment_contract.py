# Copyright 2024 Flyto2
# Licensed under the Apache License, Version 2.0
"""Parity and adversarial tests for the pure Indexer amendment boundary."""
import copy

import pytest

from flyto_ai.coding.amendment_contract import (
    AmendmentContractError,
    amendment_contract_id,
    chain_entry,
    parent_contract_digest,
    root_contract_id,
    validate_amendment_contract,
)


def _root(project="repo"):
    objective = "Repair the exact audited implementation"
    return {
        "task_profile": {
            "version": "task-contract.v2", "task_id": "task_contract_parity",
            "intent": "feature", "project": project,
            "description": objective,
            "intent_fingerprint": "1" * 64,
            "instruction_fingerprint": "2" * 64,
        },
        "intent_ledger": {
            "version": "intent-ledger.v1", "fingerprint": "1" * 64,
            "description": objective, "allowed_paths": ["src/root.py"],
        },
        "instruction_context": {
            "version": "task-context.v1", "fingerprint": "2" * 64,
        },
        "execution_plan": [],
    }


def _successor(parent, added_path):
    profile = parent["task_profile"]
    parent_state = parent.get("task_amendment") or {}
    parent_index = int(parent_state.get("amendment_index") or 0)
    index = parent_index + 1
    original = list(parent["intent_ledger"]["allowed_paths"])
    cumulative = original + [added_path]
    objective = parent["intent_ledger"]["description"]
    root = profile["task_id"]
    project = profile["project"]
    digest = parent_contract_digest(parent)
    parent_id = parent_state.get("contract_id") or root_contract_id(
        root, objective, original,
    )
    contract_id = amendment_contract_id(
        root, index, objective, cumulative, parent_id, digest,
    )
    prior_chain = list(parent_state.get("chain") or [])
    return {
        "task_profile": {
            "version": "task-contract.v2", "task_id": root,
            "intent": "feature", "project": project,
            "description": objective,
            "root_task_id": root, "amendment_index": index,
            "amendment_contract_id": contract_id,
            "intent_fingerprint": "3" * 64,
            "instruction_fingerprint": "4" * 64,
        },
        "intent_ledger": {
            "version": "intent-ledger.v1", "fingerprint": "3" * 64,
            "description": objective, "allowed_paths": cumulative,
        },
        "instruction_context": {
            "version": "task-context.v1", "fingerprint": "4" * 64,
        },
        "task_amendment": {
            "version": "task-amendment.v1", "status": "amended",
            "root_task_id": root, "objective": objective,
            "amendment_index": index, "contract_id": contract_id,
            "parent_contract_id": parent_id,
            "parent_contract_digest": digest,
            "chain": prior_chain + [
                chain_entry(
                    contract_id=parent_id,
                    parent_contract_id=(
                        prior_chain[-1]["contract_id"] if prior_chain else None
                    ),
                    root_task_id=root, project=project,
                    amendment_index=parent_index, target_count=len(original),
                    contract_digest=digest,
                )
            ],
            "original_paths": original, "added_paths": [added_path],
            "cumulative_paths": cumulative,
        },
        "execution_plan": [],
    }


def _validate(successor, parent, host_paths=None):
    if host_paths is None:
        host_paths = successor["task_amendment"]["added_paths"]
    return validate_amendment_contract(
        successor, parent, "repo", host_paths,
    )


def test_pinned_content_address_and_second_generation_chain_are_stable():
    root = _root()
    first = _successor(root, "src/first.py")
    second = _successor(first, "src/second.py")

    assert parent_contract_digest(root) == (
        "0ad0d2e7a6b7fdddef35738b56b99bb2dc5152c2d8f1a33ac8a5a65d71cab69c"
    )
    assert first["task_amendment"]["contract_id"] == "amd_dbdc88461dcf3bcdd7611d3a"
    boundary = _validate(second, first)
    assert boundary.amendment_index == 2
    assert boundary.original == frozenset({"src/root.py", "src/first.py"})
    assert boundary.added == frozenset({"src/second.py"})


@pytest.mark.parametrize(
    ("parent_version", "successor_version"),
    [
        ("task-context.v1", "task-context.v1"),
        ("task-context.v1", "intent-ledger.v1"),
        ("intent-ledger.v1", "intent-ledger.v1"),
    ],
)
def test_ledger_version_transition_preserves_parent_proof(
    parent_version, successor_version,
):
    root = _root()
    root["intent_ledger"]["version"] = parent_version
    successor = _successor(root, "src/first.py")
    successor["intent_ledger"]["version"] = successor_version

    boundary = _validate(successor, root)

    assert boundary.amendment_index == 1


def test_unknown_ledger_version_remains_fail_closed():
    root = _root()
    successor = _successor(root, "src/first.py")
    successor["intent_ledger"]["version"] = "intent-ledger.v99"

    with pytest.raises(
        AmendmentContractError, match="amendment_parent_proof_mismatch",
    ):
        _validate(successor, root)


@pytest.mark.parametrize(
    "drift",
    [
        "missing_contract_id", "forged_contract_id", "wrong_parent_contract_id",
        "empty_chain", "truncated_chain", "cyclic_chain",
        "entry_digest", "entry_root", "entry_project", "entry_index",
    ],
)
def test_successor_chain_refuses_missing_forged_truncated_or_cyclic_proof(drift):
    root = _root()
    first = _successor(root, "src/first.py")
    second = _successor(first, "src/second.py")
    candidate = copy.deepcopy(second)
    state = candidate["task_amendment"]

    if drift == "missing_contract_id":
        state.pop("contract_id")
    elif drift == "forged_contract_id":
        state["contract_id"] = "amd_" + "f" * 24
    elif drift == "wrong_parent_contract_id":
        state["parent_contract_id"] = "amd_" + "e" * 24
    elif drift == "empty_chain":
        state["chain"] = []
    elif drift == "truncated_chain":
        state["chain"] = state["chain"][1:]
    elif drift == "cyclic_chain":
        state["contract_id"] = state["chain"][0]["contract_id"]
    elif drift == "entry_digest":
        state["chain"][-1]["entry_digest"] = "0" * 64
    elif drift == "entry_root":
        state["chain"][-1]["root_task_id"] = "task_foreign"
    elif drift == "entry_project":
        state["chain"][-1]["project"] = "foreign"
    else:
        state["chain"][-1]["amendment_index"] = 7

    with pytest.raises(AmendmentContractError):
        _validate(candidate, first)


@pytest.mark.parametrize("generation", [1, 2])
@pytest.mark.parametrize(
    "field",
    ["root_task_id", "description", "amendment_index", "amendment_contract_id"],
)
def test_generation_profiles_require_every_exact_amendment_mirror(field, generation):
    root = _root()
    first = _successor(root, "src/first.py")
    parent = root if generation == 1 else first
    candidate = _successor(parent, "src/current.py")
    candidate["task_profile"].pop(field)

    with pytest.raises(AmendmentContractError, match="amendment_profile_mismatch"):
        _validate(candidate, parent)


@pytest.mark.parametrize("generation", [1, 2])
@pytest.mark.parametrize(
    "field",
    ["root_task_id", "description", "amendment_index", "amendment_contract_id"],
)
def test_generation_profiles_refuse_contradictory_amendment_mirrors(
    field, generation,
):
    root = _root()
    first = _successor(root, "src/first.py")
    parent = root if generation == 1 else first
    candidate = _successor(parent, "src/current.py")
    candidate["task_profile"][field] = (
        99 if field == "amendment_index" else "contradictory"
    )

    with pytest.raises(AmendmentContractError, match="amendment_profile_mismatch"):
        _validate(candidate, parent)


@pytest.mark.parametrize("mode", ["missing", "contradictory"])
@pytest.mark.parametrize(
    "field",
    ["root_task_id", "description", "amendment_index", "amendment_contract_id"],
)
def test_generation_two_revalidates_the_parent_profile_mirror(field, mode):
    root = _root()
    first = _successor(root, "src/first.py")
    if mode == "missing":
        first["task_profile"].pop(field)
    else:
        first["task_profile"][field] = (
            99 if field == "amendment_index" else "contradictory"
        )
    second = _successor(first, "src/second.py")

    with pytest.raises(AmendmentContractError, match="amendment_profile_mismatch"):
        _validate(second, first)


def test_successor_requires_a_complete_instruction_context():
    root = _root()
    successor = _successor(root, "src/first.py")
    successor.pop("instruction_context")

    with pytest.raises(AmendmentContractError, match="amendment_parent_proof_missing"):
        _validate(successor, root)


def test_successor_refuses_a_non_sha_ledger_fingerprint():
    root = _root()
    successor = _successor(root, "src/first.py")
    successor["intent_ledger"]["fingerprint"] = "not-a-sha"
    successor["task_profile"]["intent_fingerprint"] = "not-a-sha"

    with pytest.raises(AmendmentContractError, match="amendment_parent_proof_mismatch"):
        _validate(successor, root)


@pytest.mark.parametrize("side", ["parent", "successor"])
@pytest.mark.parametrize(
    "field",
    ["intent", "intent_fingerprint", "instruction_fingerprint"],
)
def test_profiles_refuse_conflicting_intent_and_context_mirrors(side, field):
    root = _root()
    if side == "parent":
        root["task_profile"][field] = "conflicting"
        successor = _successor(root, "src/first.py")
    else:
        successor = _successor(root, "src/first.py")
        successor["task_profile"][field] = "conflicting"

    with pytest.raises(AmendmentContractError):
        _validate(successor, root)


def test_successor_ledger_description_must_equal_the_immutable_objective():
    root = _root()
    successor = _successor(root, "src/first.py")
    successor["intent_ledger"]["description"] = "different objective"

    with pytest.raises(AmendmentContractError, match="amendment_chain_identity_mismatch"):
        _validate(successor, root)


def test_generation_eight_chain_is_refused_at_pinned_verifier_limit():
    parent = _root()
    for index in range(1, 8):
        previous = parent
        parent = _successor(previous, f"src/g{index}.py")
        _validate(parent, previous)
    generation_eight = _successor(parent, "src/g8.py")
    with pytest.raises(AmendmentContractError, match="amendment_chain_oversized"):
        _validate(generation_eight, parent)


def test_root_amendment_refuses_an_empty_parent_scope():
    root = _root()
    root["intent_ledger"]["allowed_paths"] = []
    successor = _successor(root, "src/first.py")

    with pytest.raises(AmendmentContractError, match="amendment_path_boundary_invalid"):
        _validate(successor, root)


def test_descendant_amendment_refuses_a_parent_scope_shrunk_to_empty():
    root = _root()
    parent = _successor(root, "src/first.py")
    parent["intent_ledger"]["allowed_paths"] = []
    successor = _successor(parent, "src/second.py")

    with pytest.raises(AmendmentContractError, match="amendment_path_boundary_invalid"):
        _validate(successor, parent)


@pytest.mark.parametrize(
    "unsafe",
    [
        "../escape.py", "/etc/passwd", "**", "C:\\secret.py",
        "src//double.py", "./src/a.py", "src/./a.py", "src/../a.py",
        "src/*.py", "src/a\x00.py",
    ],
)
def test_amendment_paths_require_exact_canonical_repo_relative_spelling(unsafe):
    root = _root()
    successor = _successor(root, unsafe)

    with pytest.raises(AmendmentContractError, match="amendment_path_boundary_invalid"):
        _validate(successor, root)


def test_host_cumulative_authority_must_match_the_successor_exactly():
    root = _root()
    successor = _successor(root, "src/unrequested.py")

    with pytest.raises(AmendmentContractError, match="amendment_path_boundary_invalid"):
        _validate(successor, root, ["src/root.py"])


@pytest.mark.parametrize("literal", ["M1.1", "artifacts/archive.7z"])
def test_existing_numeric_suffix_paths_remain_lexically_valid(literal):
    root = _root()
    successor = _successor(root, literal)

    assert _validate(successor, root, [literal]).added == frozenset({literal})


def test_successor_cannot_omit_an_audited_prior_scope_path():
    root = _root()
    successor = _successor(root, "src/current.py")

    with pytest.raises(AmendmentContractError, match="amendment_path_boundary_invalid"):
        _validate(
            successor, root, ["src/audited-prior.py", "src/current.py"],
        )
