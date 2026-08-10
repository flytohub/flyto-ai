# Copyright 2024 Flyto2
# Licensed under the Apache License, Version 2.0
"""Tests for the domain-neutral durable resource-claim kernel.

Every namespace, kind, identity and lifecycle token used here is derived at
runtime from SHA-256, so no test can accidentally encode a domain vocabulary or
depend on a literal name.  The four properties the suite exists to pin down are
the four the kernel is built around: owner *identity* is separate from owner
*lifecycle state*, every unresolved or ambiguous answer fails closed, durable
storage is private to the account that owns it and stays inside the configured
store, and a host that cannot supply the primitives the exclusion contract
rests on is refused outright instead of being served a weaker guarantee under
the same API.

The host-capability tests substitute the primitive, never the kernel: they
withdraw ``os.link``, break ``os.chmod`` or ``os.fchmod``, or replace the
``fcntl`` module the store reaches for, and then assert on the store's public
behaviour, so they stay deterministic on a hard-link-capable POSIX developer
machine and still describe the host the kernel refuses.  The storage-privacy
tests plant a real symbolic link and assert on the *external* target - that it
was neither re-moded nor written into - because a containment claim that only
inspects the store cannot tell adoption from escape.  Mode assertions are exact
rather than masked, and are skipped off POSIX, where the bits do not mean what
they say.
"""
import errno
import hashlib
import json
import multiprocessing
import os
import stat
import subprocess
import sys
import threading
import time
import types
from pathlib import Path

import pytest

import flyto_ai
from flyto_ai import orchestration
from flyto_ai.orchestration import resource_claims
from flyto_ai.orchestration.resource_claims import (
    DIRECTORY_MODE,
    FILE_MODE,
    MAX_FIELD_CHARS,
    MAX_TRANSITIONS,
    RECORD_VERSION,
    VERDICT_HELD,
    VERDICT_MISSING,
    VERDICT_RELEASED,
    VERDICT_UNKNOWN,
    ClaimTransition,
    OwnerRef,
    ResourceClaimConflict,
    ResourceClaimError,
    ResourceClaimRejected,
    ResourceClaimStore,
    ResourceClaimUnresolved,
    ResourceRef,
)

_REPO_ROOT = str(Path(flyto_ai.__file__).resolve().parents[1])

#: POSIX mode bits are the subject of these assertions, so skip where the
#: filesystem does not implement them rather than assert something untrue.
_POSIX_MODES = pytest.mark.skipif(
    os.name != "posix", reason="POSIX permission bits are not meaningful on this host"
)


def _derive(*parts: str) -> str:
    """Derive a bounded opaque value; never a hand-written domain word."""

    seed = "\x1f".join(parts).encode("utf-8")
    return hashlib.sha256(seed).hexdigest()[:32]


def _ref(seed: str) -> ResourceRef:
    return ResourceRef(
        namespace=_derive("namespace", seed),
        kind=_derive("kind", seed),
        identity=_derive("identity", seed),
    )


def _owner(seed: str) -> OwnerRef:
    return OwnerRef(scope=_derive("scope", seed), id=_derive("owner", seed))


def _state(seed: str) -> str:
    """An opaque lifecycle token: the caller's word, never this kernel's."""

    return _derive("state", seed)


def _claim_file(store: ResourceClaimStore, resource: ResourceRef) -> Path:
    digest = resource.digest
    return store.root / digest[:2] / (digest + ".claim.json")


def _store(tmp_path: Path) -> ResourceClaimStore:
    return ResourceClaimStore(tmp_path)


def _shard(store: ResourceClaimStore, resource: ResourceRef) -> Path:
    return store.root / resource.digest[:2]


def _mode(path: Path) -> int:
    """The permission bits alone, so an assertion reads as an exact mode."""

    return stat.S_IMODE(path.stat().st_mode)


def _authority(verdict):
    """An authority that always answers the same way, recording what it saw."""

    seen = []

    def resolver(owner, state):
        seen.append((owner, state))
        return verdict

    resolver.seen = seen
    return resolver


# --------------------------------------------------------------------------
# addressing, bounds and durable-data hygiene
# --------------------------------------------------------------------------


def test_lookup_is_content_addressed_and_carries_no_caller_vocabulary(tmp_path):
    store = _store(tmp_path)
    resource = _ref("addressing")
    store.acquire(resource, _owner("addressing"), _state("addressing"))

    path = _claim_file(store, resource)
    assert path.is_file()
    assert resource.digest in path.name
    for part in (resource.namespace, resource.kind, resource.identity):
        assert part not in str(path)

    text = path.read_text()
    stored = json.loads(text)
    assert set(stored) == {
        "version",
        "binding",
        "owner_scope",
        "owner_id",
        "state",
        "sequence",
        "transitions",
    }
    assert stored["version"] == RECORD_VERSION
    assert stored["binding"] == resource.digest
    assert stored["transitions"] == [{"sequence": 1, "state": _state("addressing")}]
    for part in (resource.namespace, resource.kind, resource.identity):
        assert part not in text


def test_only_validated_identifiers_and_lifecycle_metadata_are_persisted(tmp_path):
    store = _store(tmp_path)
    resource = _ref("hygiene")
    owner = _owner("hygiene")
    store.acquire(resource, owner, _state("hygiene"))

    stored = json.loads(_claim_file(store, resource).read_text())
    assert stored["owner_scope"] == owner.scope
    assert stored["owner_id"] == owner.id
    for value in (stored["owner_scope"], stored["owner_id"], stored["state"]):
        assert value.isprintable()
        assert not any(char.isspace() for char in value)
        assert len(value) <= MAX_FIELD_CHARS
    assert _claim_file(store, resource).stat().st_mode & 0o077 == 0


def test_digest_is_unambiguous_across_field_boundaries():
    seed = _derive("boundary")
    tail = _derive("boundary", "tail")
    left = ResourceRef(namespace=seed[:8], kind=seed[8:], identity=tail)
    right = ResourceRef(namespace=seed[:9], kind=seed[9:], identity=tail)
    assert left.digest != right.digest
    assert OwnerRef("a" + seed[:8], seed[8:]).digest != OwnerRef("a", seed).digest


@pytest.mark.parametrize(
    "bad",
    [
        "",
        "x" * (MAX_FIELD_CHARS + 1),
        "a\nb",
        "a\x00b",
        "a\x7fb",
        "a b",
        " leading",
        "trailing ",
        "\t",
    ],
)
def test_bounds_reject_unsafe_identifiers(tmp_path, bad):
    good = _derive("bounds", "good")
    with pytest.raises(ResourceClaimRejected):
        ResourceRef(namespace=bad, kind=good, identity=good)
    with pytest.raises(ResourceClaimRejected):
        OwnerRef(scope=good, id=bad)
    with pytest.raises(ResourceClaimRejected):
        ClaimTransition(sequence=1, state=bad)

    store = _store(tmp_path)
    with pytest.raises(ResourceClaimRejected):
        store.acquire(_ref("bounds"), _owner("bounds"), bad)
    assert store.inspect(_ref("bounds")).status == "free"


def test_arbitrary_namespaces_are_accepted_without_a_sanctioned_list(tmp_path):
    store = _store(tmp_path)
    for index in range(4):
        resource = ResourceRef(
            namespace=_derive("arbitrary-namespace", str(index)),
            kind=_derive("arbitrary-kind", str(index)),
            identity=_derive("arbitrary-identity", str(index)),
        )
        record = store.acquire(
            resource, _owner(f"arbitrary-{index}"), _state(f"arbitrary-{index}")
        )
        assert record.binding == resource.digest


# --------------------------------------------------------------------------
# exclusion and independence
# --------------------------------------------------------------------------


def test_free_then_held_then_free(tmp_path):
    store = _store(tmp_path)
    resource = _ref("lifecycle")
    owner = _owner("lifecycle")

    assert store.inspect(resource).status == "free"
    record = store.acquire(resource, owner, _state("lifecycle"))
    assert record.sequence == 1

    found = store.inspect(resource)
    assert found.status == "held"
    assert found.record is not None
    assert found.record.owner == owner

    assert store.release(resource, owner) is True
    assert store.inspect(resource).status == "free"
    assert store.release(resource, owner) is False


def test_same_resource_excludes_a_competitor(tmp_path):
    store = _store(tmp_path)
    resource = _ref("exclusion")
    holder = _owner("exclusion-holder")
    competitor = _owner("exclusion-competitor")

    store.acquire(resource, holder, _state("exclusion"))
    with pytest.raises(ResourceClaimConflict):
        store.acquire(resource, competitor, _state("exclusion"))
    # Even the holder cannot re-acquire; continuity goes through reassert.
    with pytest.raises(ResourceClaimConflict):
        store.acquire(resource, holder, _state("exclusion"))

    found = store.inspect(resource)
    assert found.record is not None
    assert found.record.owner == holder


def test_different_resources_stay_independent(tmp_path):
    store = _store(tmp_path)
    base = _ref("independent")
    variants = [
        base,
        ResourceRef(base.namespace, base.kind, _derive("identity", "other")),
        ResourceRef(base.namespace, _derive("kind", "other"), base.identity),
        ResourceRef(_derive("namespace", "other"), base.kind, base.identity),
    ]
    owners = [_owner(f"independent-{index}") for index in range(len(variants))]

    for resource, owner in zip(variants, owners):
        store.acquire(resource, owner, _state("independent"))
    for resource, owner in zip(variants, owners):
        found = store.inspect(resource)
        assert found.record is not None
        assert found.record.owner == owner

    store.release(variants[0], owners[0])
    assert store.inspect(variants[0]).status == "free"
    for resource in variants[1:]:
        assert store.inspect(resource).status == "held"


# --------------------------------------------------------------------------
# identity is not lifecycle state
# --------------------------------------------------------------------------


def test_one_owner_moves_through_states_without_becoming_another_owner(tmp_path):
    store = _store(tmp_path)
    resource = _ref("transition")
    owner = _owner("transition")
    phases = [_state(f"phase-{index}") for index in range(3)]

    opened = store.acquire(resource, owner, phases[0])
    assert (opened.sequence, opened.state) == (1, phases[0])

    for index, phase in enumerate(phases[1:], start=2):
        record = store.reassert(resource, owner, phase)
        assert record.sequence == index
        assert record.state == phase
        assert record.owner == owner
        assert record.binding == resource.digest

    found = store.inspect(resource)
    assert found.status == "held"
    assert found.record is not None
    assert found.record.state == phases[-1]
    assert [step.state for step in found.record.transitions] == phases
    assert [step.sequence for step in found.record.transitions] == [1, 2, 3]

    # The same identity still owns it and can still release it, having changed
    # state twice since it acquired.
    assert store.release(resource, owner) is True


def test_a_different_identity_conflicts_whatever_state_it_presents(tmp_path):
    store = _store(tmp_path)
    resource = _ref("identity")
    owner = _owner("identity")
    store.acquire(resource, owner, _state("identity"))

    strangers = [
        OwnerRef(scope=owner.scope, id=_derive("owner", "stranger")),
        OwnerRef(scope=_derive("scope", "stranger"), id=owner.id),
    ]
    for stranger in strangers:
        assert stranger != owner
        for state in (_state("identity"), _state("stranger")):
            with pytest.raises(ResourceClaimConflict):
                store.reassert(resource, stranger, state)
            with pytest.raises(ResourceClaimConflict):
                store.acquire(resource, stranger, state)
        with pytest.raises(ResourceClaimConflict):
            store.release(resource, stranger)

    still = store.inspect(resource)
    assert still.record is not None
    assert still.record.owner == owner
    assert still.record.sequence == 1


def test_transition_evidence_stays_bounded(tmp_path):
    store = _store(tmp_path)
    resource = _ref("bounded-evidence")
    owner = _owner("bounded-evidence")
    total = MAX_TRANSITIONS + 5

    store.acquire(resource, owner, _state("step-0"))
    for index in range(1, total):
        record = store.reassert(resource, owner, _state(f"step-{index}"))

    assert record.sequence == total
    assert len(record.transitions) == MAX_TRANSITIONS
    sequences = [step.sequence for step in record.transitions]
    assert sequences == list(range(total - MAX_TRANSITIONS + 1, total + 1))
    assert store.inspect(resource).record == record


def test_reassert_on_a_free_resource_conflicts(tmp_path):
    store = _store(tmp_path)
    resource = _ref("reassert-free")
    with pytest.raises(ResourceClaimConflict):
        store.reassert(resource, _owner("reassert-free"), _state("reassert-free"))


# --------------------------------------------------------------------------
# authority-aware resolution
# --------------------------------------------------------------------------


def test_resolve_reports_every_authority_answer(tmp_path):
    store = _store(tmp_path)
    resource = _ref("resolve")
    owner = _owner("resolve")
    state = _state("resolve")
    store.acquire(resource, owner, state)

    for verdict, expected in (
        (VERDICT_HELD, "held"),
        (VERDICT_RELEASED, "released"),
        (VERDICT_MISSING, "missing"),
        (VERDICT_UNKNOWN, "unresolved"),
    ):
        resolver = _authority(verdict)
        resolution = store.resolve(resource, resolver)
        assert resolution.status == expected, verdict
        assert resolver.seen == [(owner, state)], verdict
        assert resolution.record is not None
        assert resolution.record.owner == owner
        assert resolution.reclaimable is (expected == "released")
        # Resolution never mutates anything by itself.
        assert store.inspect(resource).status == "held"


def test_resolve_fails_closed_on_authority_errors(tmp_path):
    store = _store(tmp_path)
    resource = _ref("authority-error")
    owner = _owner("authority-error")
    store.acquire(resource, owner, _state("authority-error"))

    def explodes(candidate, state):
        raise RuntimeError("the authority is unavailable")

    for resolver in (
        explodes,
        _authority(None),
        _authority(True),
        _authority("released "),
        _authority(_derive("verdict", "junk")),
    ):
        resolution = store.resolve(resource, resolver)
        assert resolution.status == "unresolved"
        assert resolution.verdict is None
        assert store.sweep(resource, resolver).applied is False
        assert store.repair(resource, resolver).applied is False
    assert store.inspect(resource).status == "held"


def test_resolve_on_free_and_unresolved_records(tmp_path):
    store = _store(tmp_path)
    resource = _ref("resolve-edges")
    resolver = _authority(VERDICT_RELEASED)

    resolution = store.resolve(resource, resolver)
    assert (resolution.status, resolution.record, resolver.seen) == ("free", None, [])

    _claim_file(store, resource).parent.mkdir(parents=True, exist_ok=True)
    _claim_file(store, resource).write_bytes(b"{truncated")
    resolution = store.resolve(resource, resolver)
    assert resolution.status == "unresolved"
    # A record that cannot be parsed names nobody, so nobody is asked.
    assert resolver.seen == []


# --------------------------------------------------------------------------
# fail-closed damaged records
# --------------------------------------------------------------------------


def _corrupt(store: ResourceClaimStore, resource: ResourceRef, payload) -> Path:
    path = _claim_file(store, resource)
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(payload, bytes):
        path.write_bytes(payload)
    else:
        path.write_text(json.dumps(payload))
    return path


def _valid_payload(resource: ResourceRef, seed: str) -> dict:
    owner = _owner(seed)
    state = _state(seed)
    return {
        "version": RECORD_VERSION,
        "binding": resource.digest,
        "owner_scope": owner.scope,
        "owner_id": owner.id,
        "state": state,
        "sequence": 1,
        "transitions": [{"sequence": 1, "state": state}],
    }


def _damaged_cases(resource: ResourceRef, other: ResourceRef):
    missing_owner = _valid_payload(resource, "missing-owner")
    missing_owner["owner_id"] = ""

    orphaned = _valid_payload(resource, "orphaned")
    del orphaned["owner_scope"]

    unknown_version = _valid_payload(resource, "version")
    unknown_version["version"] = RECORD_VERSION + 1

    wrong_binding = _valid_payload(resource, "binding")
    wrong_binding["binding"] = other.digest

    extra_key = _valid_payload(resource, "extra")
    extra_key["surplus"] = "1"

    bad_sequence = _valid_payload(resource, "sequence")
    bad_sequence["sequence"] = 0

    bad_state = _valid_payload(resource, "state")
    bad_state["state"] = "a\nb"

    # The counter advanced but the evidence did not: a stale record.
    stale = _valid_payload(resource, "stale")
    stale["sequence"] = 5

    # The evidence claims a step the counter never reached.
    ahead = _valid_payload(resource, "ahead")
    ahead["transitions"] = [{"sequence": 9, "state": ahead["state"]}]

    # The recorded state disagrees with its own latest evidence.
    divergent = _valid_payload(resource, "divergent")
    divergent["state"] = _state("divergent-other")

    non_monotonic = _valid_payload(resource, "monotonic")
    non_monotonic["sequence"] = 2
    non_monotonic["transitions"] = [
        {"sequence": 2, "state": non_monotonic["state"]},
        {"sequence": 2, "state": non_monotonic["state"]},
    ]

    empty_evidence = _valid_payload(resource, "empty-evidence")
    empty_evidence["transitions"] = []

    overlong_evidence = _valid_payload(resource, "overlong-evidence")
    overlong_evidence["sequence"] = MAX_TRANSITIONS + 1
    overlong_evidence["transitions"] = [
        {"sequence": index, "state": overlong_evidence["state"]}
        for index in range(1, MAX_TRANSITIONS + 2)
    ]

    malformed_evidence = _valid_payload(resource, "malformed-evidence")
    malformed_evidence["transitions"] = [{"sequence": 1}]

    return {
        "malformed": b"{not json",
        "not-an-object": [1, 2, 3],
        "missing-owner": missing_owner,
        "orphaned": orphaned,
        "unknown-version": unknown_version,
        "wrong-binding": wrong_binding,
        "extra-key": extra_key,
        "bad-sequence": bad_sequence,
        "bad-state": bad_state,
        "stale": stale,
        "evidence-ahead": ahead,
        "divergent-state": divergent,
        "non-monotonic": non_monotonic,
        "empty-evidence": empty_evidence,
        "overlong-evidence": overlong_evidence,
        "malformed-evidence": malformed_evidence,
        "oversized": b"{" + b"x" * 8192,
    }


def test_damaged_records_are_unresolved_and_never_auto_cleared(tmp_path):
    other = _ref("damage-other")
    cases = _damaged_cases(_ref("damage"), other)
    for index, (label, payload) in enumerate(cases.items()):
        store = _store(tmp_path / f"case-{index}")
        resource = _ref("damage")
        path = _corrupt(store, resource, payload)

        found = store.inspect(resource)
        assert found.status == "unresolved", label
        assert found.record is None, label

        owner = _owner("damage")
        state = _state("damage")
        with pytest.raises(ResourceClaimUnresolved):
            store.acquire(resource, owner, state)
        with pytest.raises(ResourceClaimUnresolved):
            store.release(resource, owner)
        with pytest.raises(ResourceClaimUnresolved):
            store.reassert(resource, owner, state)

        # No authority answer, not even a positive release, can sweep a record
        # that could not be attributed to an owner in the first place.
        for verdict in (VERDICT_RELEASED, VERDICT_MISSING, VERDICT_HELD, VERDICT_UNKNOWN):
            resolver = _authority(verdict)
            assert store.sweep(resource, resolver).applied is False, label
            assert resolver.seen == [], label
        assert path.exists(), label


def test_stale_evidence_is_distinguished_from_plain_corruption(tmp_path):
    store = _store(tmp_path)
    resource = _ref("stale-reason")
    payload = _valid_payload(resource, "stale-reason")
    payload["sequence"] = 4
    _corrupt(store, resource, payload)

    found = store.inspect(resource)
    assert found.status == "unresolved"
    assert "stale" in found.reason


def test_unreadable_record_is_unresolved(tmp_path):
    store = _store(tmp_path)
    resource = _ref("unreadable")
    path = _claim_file(store, resource)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.mkdir()

    found = store.inspect(resource)
    assert found.status == "unresolved"
    assert path.exists()


def test_a_record_copied_onto_another_resource_is_unresolved(tmp_path):
    store = _store(tmp_path)
    source = _ref("binding-source")
    target = _ref("binding-target")
    store.acquire(source, _owner("binding"), _state("binding"))

    target_path = _claim_file(store, target)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_bytes(_claim_file(store, source).read_bytes())

    assert store.inspect(source).status == "held"
    assert store.inspect(target).status == "unresolved"


# --------------------------------------------------------------------------
# sweep and repair
# --------------------------------------------------------------------------


def test_sweep_only_when_the_authority_proves_release(tmp_path):
    store = _store(tmp_path)
    resource = _ref("sweep")
    owner = _owner("sweep")
    state = _state("sweep")
    store.acquire(resource, owner, state)

    for verdict, expected in (
        (VERDICT_HELD, "held"),
        (VERDICT_MISSING, "missing"),
        (VERDICT_UNKNOWN, "unresolved"),
    ):
        resolver = _authority(verdict)
        outcome = store.sweep(resource, resolver)
        assert (outcome.applied, outcome.status) == (False, expected), verdict
        assert resolver.seen == [(owner, state)], verdict
        assert store.inspect(resource).status == "held", verdict

    resolver = _authority(VERDICT_RELEASED)
    outcome = store.sweep(resource, resolver)
    assert outcome.applied is True
    assert outcome.status == "free"
    assert outcome.record is not None
    assert outcome.record.owner == owner
    assert store.inspect(resource).status == "free"

    successor = _owner("sweep-successor")
    assert store.acquire(resource, successor, _state("sweep-successor")).sequence == 1


def test_sweep_on_a_free_resource_is_a_no_op(tmp_path):
    store = _store(tmp_path)
    resource = _ref("sweep-free")
    resolver = _authority(VERDICT_RELEASED)
    outcome = store.sweep(resource, resolver)
    assert (outcome.applied, outcome.status, resolver.seen) == (False, "free", [])


def test_repair_clears_an_unparseable_record_without_an_authority(tmp_path):
    store = _store(tmp_path)
    resource = _ref("repair")
    owner = _owner("repair")

    assert store.repair(resource).applied is False  # free

    store.acquire(resource, owner, _state("repair"))
    outcome = store.repair(resource)
    assert (outcome.applied, outcome.status) == (False, "held")
    assert store.inspect(resource).status == "held"

    _claim_file(store, resource).write_bytes(b"{truncated")
    assert store.inspect(resource).status == "unresolved"
    assert store.repair(resource).applied is True
    assert store.inspect(resource).status == "free"
    assert store.repair(resource).applied is False

    assert store.acquire(resource, _owner("repair-successor"), _state("repair")).sequence == 1


def test_repair_clears_an_orphan_only_on_a_missing_verdict(tmp_path):
    store = _store(tmp_path)
    resource = _ref("orphan-repair")
    owner = _owner("orphan-repair")
    state = _state("orphan-repair")
    store.acquire(resource, owner, state)

    def explodes(candidate, ignored):
        raise RuntimeError("the authority is unavailable")

    # Ambiguity is not a licence to repair: a well-formed record whose owner
    # cannot be resolved stays exactly where it is.
    for resolver in (
        _authority(VERDICT_HELD),
        _authority(VERDICT_UNKNOWN),
        _authority(VERDICT_RELEASED),
        explodes,
    ):
        outcome = store.repair(resource, resolver)
        assert outcome.applied is False
        assert store.inspect(resource).status == "held"

    resolver = _authority(VERDICT_MISSING)
    outcome = store.repair(resource, resolver)
    assert outcome.applied is True
    assert outcome.status == "free"
    assert outcome.record is not None
    assert outcome.record.owner == owner
    assert resolver.seen == [(owner, state)]
    assert store.inspect(resource).status == "free"


# --------------------------------------------------------------------------
# hosts that cannot keep the exclusion contract
# --------------------------------------------------------------------------


def _shard_names(store: ResourceClaimStore, resource: ResourceRef):
    """Every filename the store has left in one resource's shard directory."""

    shard = store.root / resource.digest[:2]
    return sorted(entry.name for entry in shard.iterdir()) if shard.is_dir() else []


def _residue(store: ResourceClaimStore, resource: ResourceRef):
    """Shard contents that are neither the lock file nor a legitimate claim."""

    return [name for name in _shard_names(store, resource) if not name.endswith(".lock")]


def _withdraw_hard_links(patcher):
    """Present a filesystem on which ``os.link`` can never succeed."""

    def refuse(source, target, *args, **kwargs):
        raise OSError(errno.EPERM, "hard links are not supported here")

    patcher.setattr(os, "link", refuse)


#: Hosts whose ``fcntl`` cannot deliver a real inter-process lock.  Partial
#: support is not support: a module carrying the constants but no ``flock``, or
#: ``flock`` but no ``LOCK_EX``, is exactly as unusable as no module at all.
_UNLOCKABLE_HOSTS = (
    None,
    types.SimpleNamespace(),
    types.SimpleNamespace(LOCK_EX=2, LOCK_UN=8),
    types.SimpleNamespace(flock=lambda *_: None, LOCK_UN=8),
)


def test_acquire_fails_closed_when_hard_link_publication_is_unsupported(tmp_path, monkeypatch):
    """No hard link means no claim - never a copy of the bytes into the claim."""

    store = _store(tmp_path)
    resource = _ref("no-hard-links")
    owner = _owner("no-hard-links")
    state = _state("no-hard-links")

    _withdraw_hard_links(monkeypatch)
    with pytest.raises(ResourceClaimError) as raised:
        store.acquire(resource, owner, state)

    # The host is at fault, so this is not a conflict, not a rejection of the
    # caller's arguments, and not an unresolved stored record.
    assert not isinstance(
        raised.value,
        (ResourceClaimConflict, ResourceClaimRejected, ResourceClaimUnresolved),
    )

    # Nothing final, and nothing partial hiding under any other name either.
    assert not _claim_file(store, resource).exists()
    assert store.inspect(resource).status == "free"
    assert _residue(store, resource) == []


def test_a_refused_publication_leaves_the_resource_cleanly_claimable(tmp_path, monkeypatch):
    """A refusal is not a half-claim: it leaves nothing to repair or take over."""

    store = _store(tmp_path)
    resource = _ref("publication-recovery")
    owner = _owner("publication-recovery")
    state = _state("publication-recovery")

    with monkeypatch.context() as patcher:
        _withdraw_hard_links(patcher)
        for _ in range(3):
            with pytest.raises(ResourceClaimError):
                store.acquire(resource, owner, state)
            # A reader looking in between two refusals never sees a record at
            # all, let alone a truncated one.
            assert store.inspect(resource).status == "free"
            assert _residue(store, resource) == []

    # The host recovers.  What lands is a first claim at sequence 1, not a
    # resumption of anything the refused attempts left behind.
    record = store.acquire(resource, owner, state)
    assert (record.sequence, record.owner, record.state) == (1, owner, state)
    assert record.transitions == (ClaimTransition(1, state),)
    assert store.inspect(resource).record == record


def test_mutations_fail_closed_without_an_interprocess_lock(tmp_path, monkeypatch):
    """A process-local lock alone is not cross-process safety, so refuse to act."""

    store = _store(tmp_path)
    resource = _ref("unlockable")
    unclaimed = _ref("unlockable-free")
    owner = _owner("unlockable")
    state = _state("unlockable")
    held = store.acquire(resource, owner, state)

    resolver = _authority(VERDICT_RELEASED)
    for host in _UNLOCKABLE_HOSTS:
        with monkeypatch.context() as patcher:
            patcher.setattr(resource_claims, "fcntl", host)
            mutations = (
                lambda: store.acquire(unclaimed, owner, state),
                lambda: store.reassert(resource, owner, _state("unlockable-next")),
                lambda: store.release(resource, owner),
                lambda: store.sweep(resource, resolver),
                lambda: store.repair(resource, resolver),
            )
            for mutation in mutations:
                with pytest.raises(ResourceClaimError):
                    mutation()

    # A positive release verdict would have swept the claim on a sound host.
    # The authority was never even consulted, because the refusal happens
    # before the store touches anything.
    assert resolver.seen == []
    assert not _claim_file(store, unclaimed).exists()
    assert store.inspect(resource).record == held


def test_a_lock_that_cannot_be_taken_fails_closed(tmp_path, monkeypatch):
    """A host that advertises flock but cannot grant it is refused just the same."""

    store = _store(tmp_path)
    resource = _ref("lock-refused")

    def refuse(descriptor, operation):
        raise OSError(errno.ENOLCK, "no locks available")

    monkeypatch.setattr(
        resource_claims,
        "fcntl",
        types.SimpleNamespace(flock=refuse, LOCK_EX=2, LOCK_UN=8),
    )
    with pytest.raises(ResourceClaimError):
        store.acquire(resource, _owner("lock-refused"), _state("lock-refused"))

    assert not _claim_file(store, resource).exists()
    assert _residue(store, resource) == []


def test_inspection_stays_usable_without_an_interprocess_lock(tmp_path, monkeypatch):
    """Reporting what is written down needs no exclusion, so it keeps working."""

    store = _store(tmp_path)
    resource = _ref("read-only-host")
    absent = _ref("read-only-host-absent")
    owner = _owner("read-only-host")
    state = _state("read-only-host")
    record = store.acquire(resource, owner, state)

    monkeypatch.setattr(resource_claims, "fcntl", None)

    found = store.inspect(resource)
    assert found.status == "held"
    assert found.record == record
    assert store.inspect(absent).status == "free"

    resolver = _authority(VERDICT_HELD)
    resolution = store.resolve(resource, resolver)
    assert resolution.status == "held"
    assert resolution.record == record
    assert resolver.seen == [(owner, state)]


# --------------------------------------------------------------------------
# durable storage is private to the account that owns it
# --------------------------------------------------------------------------


@_POSIX_MODES
def test_a_fresh_store_creates_only_private_directories_and_files(tmp_path):
    """Everything the kernel lays down on a clean host is owner-only."""

    store = _store(tmp_path)
    resource = _ref("privacy-fresh")
    # Constructing a store is not a side effect; the first mutation builds it.
    assert not store.root.exists()

    store.acquire(resource, _owner("privacy-fresh"), _state("privacy-fresh"))

    assert DIRECTORY_MODE == 0o700
    assert FILE_MODE == 0o600
    assert _mode(store.root) == DIRECTORY_MODE
    assert _mode(_shard(store, resource)) == DIRECTORY_MODE
    # The claim and the lock file both: neither is a place to leak a record.
    names = _shard_names(store, resource)
    assert names, "the acquire left nothing behind to check"
    for name in names:
        assert _mode(_shard(store, resource) / name) == FILE_MODE, name


@_POSIX_MODES
@pytest.mark.parametrize("inherited", [0o777, 0o775, 0o755, 0o750, 0o701])
def test_a_permissive_directory_is_tightened_before_anything_is_written(tmp_path, inherited):
    """An inherited root is adopted at the kernel's mode, not at its own.

    A group- or world-writable claim root lets a bystander unlink a live claim
    or substitute a record, so finding one already standing is not a reason to
    keep using it as found.
    """

    store = _store(tmp_path)
    resource = _ref("privacy-inherited")
    shard = _shard(store, resource)
    shard.mkdir(parents=True)
    os.chmod(store.root, inherited)
    os.chmod(shard, inherited)

    record = store.acquire(resource, _owner("privacy-inherited"), _state("privacy-inherited"))

    assert _mode(store.root) == DIRECTORY_MODE
    assert _mode(shard) == DIRECTORY_MODE
    assert store.inspect(resource).record == record
    assert _mode(_claim_file(store, resource)) == FILE_MODE

    # A shard created later, under the now-private root, is private too.
    later = _ref("privacy-inherited-later")
    store.acquire(later, _owner("privacy-inherited-later"), _state("privacy-inherited-later"))
    assert _mode(_shard(store, later)) == DIRECTORY_MODE
    assert _mode(store.root) == DIRECTORY_MODE


@_POSIX_MODES
@pytest.mark.parametrize("failure", ["refused", "ignored"])
def test_a_directory_that_cannot_be_made_private_fails_closed(tmp_path, monkeypatch, failure):
    """No private directory, no claim - and nothing half-written either.

    Both host failures are covered: a ``chmod`` that refuses outright, and one
    that reports success while leaving the mode exactly as permissive as it
    found it.  Trusting the second would store a claim in a directory anyone
    can write to, so the mode is re-read rather than assumed.
    """

    store = _store(tmp_path)
    resource = _ref("privacy-refused")
    shard = _shard(store, resource)
    shard.mkdir(parents=True)
    os.chmod(store.root, 0o755)
    os.chmod(shard, 0o755)

    def refuse(path, mode, *args, **kwargs):
        raise OSError(errno.EPERM, "the mode cannot be changed here")

    def ignore(path, mode, *args, **kwargs):
        return None

    monkeypatch.setattr(os, "chmod", refuse if failure == "refused" else ignore)

    with pytest.raises(ResourceClaimError) as raised:
        store.acquire(resource, _owner("privacy-refused"), _state("privacy-refused"))

    # The host is at fault: not a conflict, not a rejected argument, and not a
    # damaged stored record.
    assert not isinstance(
        raised.value,
        (ResourceClaimConflict, ResourceClaimRejected, ResourceClaimUnresolved),
    )

    # Refusing happens before the lock file is opened, so the directory whose
    # mode could not be settled holds nothing at all - no claim, no lock, no
    # staged temporary.
    assert not _claim_file(store, resource).exists()
    assert _shard_names(store, resource) == []
    assert store.inspect(resource).status == "free"


@_POSIX_MODES
def test_a_claim_root_that_cannot_be_created_fails_closed(tmp_path):
    """A root path occupied by a non-directory is a host fault, not a claim."""

    store = _store(tmp_path)
    store.root.write_bytes(b"")
    resource = _ref("privacy-occupied")

    with pytest.raises(ResourceClaimError) as raised:
        store.acquire(resource, _owner("privacy-occupied"), _state("privacy-occupied"))
    assert not isinstance(raised.value, (ResourceClaimConflict, ResourceClaimRejected))
    assert not _claim_file(store, resource).exists()
    assert store.inspect(resource).record is None


@_POSIX_MODES
@pytest.mark.parametrize("linked", ["root", "shard"])
def test_a_symlinked_directory_is_refused_instead_of_followed(tmp_path, linked):
    """A planted link must not become the store, and must not be re-moded.

    ``mkdir(exist_ok=True)`` resolves a link before deciding a directory is
    already there, so adopting the path as found would chmod a directory
    outside the configured store to the kernel's own mode and then write claims
    into it.  Both the root and a shard are checked: a store that guards only
    its root can still be diverted one level down.
    """

    outside = tmp_path / "outside"
    outside.mkdir()
    os.chmod(outside, 0o755)
    (outside / "bystander").write_bytes(b"")

    store = _store(tmp_path / "inside")
    resource = _ref(f"symlink-{linked}")
    if linked == "root":
        store.root.parent.mkdir(parents=True, exist_ok=True)
        store.root.symlink_to(outside, target_is_directory=True)
    else:
        store.root.mkdir(parents=True)
        _shard(store, resource).symlink_to(outside, target_is_directory=True)

    with pytest.raises(ResourceClaimError) as raised:
        store.acquire(resource, _owner("symlink"), _state("symlink"))
    assert not isinstance(
        raised.value,
        (ResourceClaimConflict, ResourceClaimRejected, ResourceClaimUnresolved),
    )

    # The link is still a link - it was never replaced - and the directory it
    # pointed at was neither re-moded nor written into.
    assert (store.root if linked == "root" else _shard(store, resource)).is_symlink()
    assert _mode(outside) == 0o755
    assert sorted(entry.name for entry in outside.iterdir()) == ["bystander"]
    assert not store.inspect(resource).record


@_POSIX_MODES
def test_a_pre_existing_lock_file_is_tightened_before_the_lock_is_taken(tmp_path):
    """A lock anyone may write to is not exclusion, so its mode is settled too.

    ``os.open`` applies its mode argument only when it creates the file, so a
    lock left behind at a wider mode would keep it forever.
    """

    store = _store(tmp_path)
    resource = _ref("lock-mode")
    shard = _shard(store, resource)
    shard.mkdir(parents=True)
    lock = shard / (resource.digest + ".lock")
    lock.write_bytes(b"")
    os.chmod(lock, 0o644)

    record = store.acquire(resource, _owner("lock-mode"), _state("lock-mode"))

    assert _mode(lock) == FILE_MODE
    assert _mode(_claim_file(store, resource)) == FILE_MODE
    assert store.inspect(resource).record == record

    # Still owner-only after a second operation that reuses the same lock.
    store.reassert(resource, _owner("lock-mode"), _state("lock-mode-next"))
    assert _mode(lock) == FILE_MODE


@_POSIX_MODES
@pytest.mark.parametrize("failure", ["refused", "ignored"])
def test_a_lock_file_that_cannot_be_made_private_fails_closed(tmp_path, monkeypatch, failure):
    """No private lock, no claim - and the authority is never even consulted."""

    store = _store(tmp_path)
    resource = _ref("lock-mode-refused")
    shard = _shard(store, resource)
    shard.mkdir(parents=True)
    lock = shard / (resource.digest + ".lock")
    lock.write_bytes(b"")
    os.chmod(lock, 0o644)

    def refuse(descriptor, mode, *args, **kwargs):
        raise OSError(errno.EPERM, "the mode cannot be changed here")

    def ignore(descriptor, mode, *args, **kwargs):
        return None

    monkeypatch.setattr(os, "fchmod", refuse if failure == "refused" else ignore)

    with pytest.raises(ResourceClaimError) as raised:
        store.acquire(resource, _owner("lock-mode-refused"), _state("lock-mode-refused"))
    assert not isinstance(
        raised.value,
        (ResourceClaimConflict, ResourceClaimRejected, ResourceClaimUnresolved),
    )

    # Refusing happens before flock and before publication, so no record and no
    # staged temporary were left behind.
    assert not _claim_file(store, resource).exists()
    assert _residue(store, resource) == []
    assert store.inspect(resource).status == "free"


@_POSIX_MODES
def test_a_symlinked_lock_file_is_refused_instead_of_followed(tmp_path):
    """``O_NOFOLLOW``: a lock path planted as a link is never opened at all."""

    outside = tmp_path / "outside-lock"
    outside.write_bytes(b"")
    os.chmod(outside, 0o644)

    store = _store(tmp_path)
    resource = _ref("lock-symlink")
    shard = _shard(store, resource)
    shard.mkdir(parents=True)
    (shard / (resource.digest + ".lock")).symlink_to(outside)

    with pytest.raises(ResourceClaimError):
        store.acquire(resource, _owner("lock-symlink"), _state("lock-symlink"))

    assert _mode(outside) == 0o644
    assert outside.read_bytes() == b""
    assert not _claim_file(store, resource).exists()
    assert _residue(store, resource) == []


@_POSIX_MODES
def test_read_only_lookup_creates_nothing_and_re_modes_nothing(tmp_path):
    """Looking is safe on a store this process may only read.

    Reporting what is written down needs no exclusion and no directory of its
    own, so neither :meth:`inspect` nor :meth:`resolve` may build one or touch
    the mode of one it finds - that is what keeps them usable for an observer
    that has no business tightening somebody else's storage.
    """

    store = _store(tmp_path)
    resource = _ref("privacy-read-only")
    resolver = _authority(VERDICT_HELD)

    assert store.inspect(resource).status == "free"
    assert store.resolve(resource, resolver).status == "free"
    assert not store.root.exists()
    assert resolver.seen == []

    # A well-formed claim laid down by somebody else, at a wider mode than this
    # kernel would ever create.
    _corrupt(store, resource, _valid_payload(resource, "privacy-read-only"))
    shard = _shard(store, resource)
    os.chmod(shard, 0o755)
    os.chmod(store.root, 0o755)

    assert store.inspect(resource).status == "held"
    assert store.resolve(resource, resolver).status == "held"
    assert len(resolver.seen) == 1
    assert _mode(store.root) == 0o755
    assert _mode(shard) == 0o755


@_POSIX_MODES
def test_a_symlinked_claim_record_is_refused_instead_of_followed(tmp_path):
    """A planted link must not be read, and must never pass as a real record.

    Following it would report a claim this store never wrote, sourced from a
    file outside the configured root - and ``acquire`` would then refuse the
    resource on the strength of somebody else's bytes.
    """

    outside = tmp_path / "outside.json"
    outside.write_text(json.dumps(_valid_payload(_ref("record-symlink"), "outside")))

    store = _store(tmp_path)
    resource = _ref("record-symlink")
    shard = _shard(store, resource)
    shard.mkdir(parents=True)
    _claim_file(store, resource).symlink_to(outside)

    found = store.inspect(resource)
    assert found.status == "unresolved"
    assert found.record is None
    # Not silently treated as free either: something *is* there, and a host has
    # to look at it.
    with pytest.raises(ResourceClaimUnresolved):
        store.acquire(resource, _owner("record-symlink"), _state("record-symlink"))
    assert _claim_file(store, resource).is_symlink()
    assert outside.exists()


@_POSIX_MODES
def test_a_claim_record_that_is_not_a_regular_file_is_refused(tmp_path):
    """A FIFO at a claim path is refused rather than opened and read from."""

    store = _store(tmp_path)
    resource = _ref("record-fifo")
    shard = _shard(store, resource)
    shard.mkdir(parents=True)
    os.mkfifo(_claim_file(store, resource))

    found = store.inspect(resource)
    assert found.status == "unresolved"
    assert found.record is None


def test_an_enormous_claim_record_is_bounded_before_it_is_parsed(tmp_path, monkeypatch):
    """Reading stops one byte past the bound, however large the file is."""

    store = _store(tmp_path)
    resource = _ref("record-huge")
    shard = _shard(store, resource)
    shard.mkdir(parents=True)
    _claim_file(store, resource).write_bytes(b"{" + b"x" * (4 * 1024 * 1024))

    requested = []
    real_read = os.read

    def counting_read(descriptor, size):
        requested.append(size)
        return real_read(descriptor, size)

    monkeypatch.setattr(os, "read", counting_read)
    found = store.inspect(resource)

    assert found.status == "unresolved"
    assert requested, "the record was not read through os.read"
    # Never asks for more than the bound plus the one discriminating byte, and
    # never keeps asking once that budget is spent.
    assert max(requested) <= resource_claims.MAX_RECORD_BYTES + 1
    assert sum(requested) <= 2 * (resource_claims.MAX_RECORD_BYTES + 1)


def test_a_record_exactly_at_the_bound_still_parses(tmp_path):
    """The bound is inclusive: MAX_RECORD_BYTES is legal, one more is not."""

    store = _store(tmp_path)
    resource = _ref("record-at-bound")
    owner = _owner("record-at-bound")
    store.acquire(resource, owner, _state("record-at-bound"))

    raw = _claim_file(store, resource).read_bytes()
    assert 0 < len(raw) <= resource_claims.MAX_RECORD_BYTES
    assert store.inspect(resource).record is not None

    _claim_file(store, resource).write_bytes(
        b"{" + b" " * resource_claims.MAX_RECORD_BYTES
    )
    assert store.inspect(resource).status == "unresolved"


# --------------------------------------------------------------------------
# the process-local lock registry stays bounded
# --------------------------------------------------------------------------


def test_the_local_lock_registry_does_not_grow_with_resources_seen(tmp_path):
    """A long-lived process must not leak one mutex per resource touched."""

    store = _store(tmp_path)
    before = len(resource_claims._LOCAL_LOCKS)

    for index in range(50):
        resource = _ref(f"registry-{index}")
        owner = _owner(f"registry-{index}")
        store.acquire(resource, owner, _state("registry"))
        store.reassert(resource, owner, _state("registry-next"))
        assert store.release(resource, owner) is True

    # Every entry retired itself; nothing accumulated across 50 resources.
    assert len(resource_claims._LOCAL_LOCKS) == before == 0


def test_the_registry_entry_survives_while_a_waiter_still_needs_it(tmp_path):
    """Retiring an entry must never let two threads into one resource.

    The evicting thread and the waiting thread race by construction here: the
    holder is still inside its critical section when a second thread arrives,
    so an implementation that dropped the entry on the way out - or handed the
    late arrival a fresh mutex - would admit both at once.
    """

    store = _store(tmp_path)
    resource = _ref("registry-race")
    inside = []
    overlapped = []
    started = threading.Event()

    def worker(seed):
        owner = _owner(f"registry-race-{seed}")
        for _ in range(20):
            try:
                store.acquire(resource, owner, _state("registry-race"))
            except ResourceClaimConflict:
                continue
            inside.append(seed)
            if len(inside) > 1:
                overlapped.append(tuple(inside))
            started.set()
            time.sleep(0.001)
            inside.remove(seed)
            store.release(resource, owner)

    threads = [threading.Thread(target=worker, args=(seed,)) for seed in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=60)
        assert not thread.is_alive()

    assert started.is_set(), "no worker ever entered the critical section"
    assert overlapped == [], f"two owners held one resource at once: {overlapped}"
    assert len(resource_claims._LOCAL_LOCKS) == 0
    assert store.inspect(resource).status == "free"


def test_distinct_resources_do_not_share_a_local_lock(tmp_path):
    """Bounding the registry must not accidentally serialize the whole store."""

    store = _store(tmp_path)
    first = _ref("registry-distinct-a")
    second = _ref("registry-distinct-b")
    seen = {}

    def observe(resource, seed):
        owner = _owner(seed)
        store.acquire(resource, owner, _state("registry-distinct"))

        def authority(candidate, recorded_state):
            # ``sweep`` consults the authority from *inside* the resource's
            # critical section, which is the only place the registry entry is
            # guaranteed to still exist.
            with resource_claims._LOCAL_LOCKS_GUARD:
                seen[seed] = sorted(resource_claims._LOCAL_LOCKS)
            return VERDICT_HELD

        store.sweep(resource, authority)
        store.release(resource, owner)

    observe(first, "a")
    observe(second, "b")

    assert len(seen["a"]) == len(seen["b"]) == 1
    assert seen["a"] != seen["b"], "two resources shared one local lock key"
    assert len(resource_claims._LOCAL_LOCKS) == 0


# --------------------------------------------------------------------------
# the package-level public surface
# --------------------------------------------------------------------------


def test_the_kernel_api_is_re_exported_without_private_helpers():
    """``flyto_ai.orchestration`` publishes the kernel and nothing internal."""

    exported = set(orchestration.__all__)
    assert len(exported) == len(orchestration.__all__), "__all__ repeats a name"

    # Every public kernel name, bound to the very same object.
    assert set(resource_claims.__all__) <= exported
    for name in resource_claims.__all__:
        assert getattr(orchestration, name) is getattr(resource_claims, name), name

    # The pre-existing sub-agent surface still resolves from the same package.
    for name in ("SubAgent", "AgentOrchestrator", "OrchestrationPolicy"):
        assert name in exported, name
        assert getattr(orchestration, name) is not None, name

    # Internals stay internal: they exist on the module and nowhere else.
    for name in (
        "_secure_directory",
        "_secure_open_lock",
        "_own_directory",
        "_exclusive",
        "_encode",
        "_decode",
        "_digest",
    ):
        assert hasattr(resource_claims, name), name
        assert name not in exported, name
    assert not [name for name in exported if name.startswith("_")]


def test_the_re_exported_store_is_the_same_working_kernel(tmp_path):
    """The package alias is the kernel itself, not a thin lookalike."""

    resource = orchestration.ResourceRef(
        namespace=_derive("namespace", "exported"),
        kind=_derive("kind", "exported"),
        identity=_derive("identity", "exported"),
    )
    owner = orchestration.OwnerRef(
        scope=_derive("scope", "exported"), id=_derive("owner", "exported")
    )
    store = orchestration.ResourceClaimStore(tmp_path)

    record = store.acquire(resource, owner, _state("exported"))
    assert isinstance(record, orchestration.ClaimRecord)
    with pytest.raises(orchestration.ResourceClaimConflict):
        store.acquire(resource, _owner("exported-stranger"), _state("exported"))
    assert store.resolve(resource, _authority(orchestration.VERDICT_HELD)).status == "held"
    assert store.release(resource, owner) is True


# --------------------------------------------------------------------------
# real cross-process behaviour
# --------------------------------------------------------------------------


_CHILD_ACQUIRE = """
import json, sys
from flyto_ai.orchestration.resource_claims import (
    OwnerRef, ResourceClaimStore, ResourceRef,
)

root, ns, kind, ident, scope, oid, state = sys.argv[1:8]
store = ResourceClaimStore(root)
record = store.acquire(ResourceRef(ns, kind, ident), OwnerRef(scope, oid), state)
print(json.dumps({"sequence": record.sequence, "binding": record.binding}))
"""


def _child_env():
    env = dict(os.environ)
    existing = env.get("PYTHONPATH")
    env["PYTHONPATH"] = _REPO_ROOT + (os.pathsep + existing if existing else "")
    return env


def test_claim_survives_the_process_that_took_it(tmp_path):
    store = _store(tmp_path)
    resource = _ref("durability")
    owner = _owner("durability")
    state = _state("durability")

    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            _CHILD_ACQUIRE,
            str(tmp_path),
            resource.namespace,
            resource.kind,
            resource.identity,
            owner.scope,
            owner.id,
            state,
        ],
        capture_output=True,
        text=True,
        env=_child_env(),
        timeout=120,
    )
    assert completed.returncode == 0, completed.stderr
    emitted = json.loads(completed.stdout.strip())
    assert emitted == {"sequence": 1, "binding": resource.digest}

    # The child is gone; no daemon and no TTL renewed anything.
    found = store.inspect(resource)
    assert found.status == "held"
    assert found.record is not None
    assert found.record.owner == owner
    assert found.record.state == state

    with pytest.raises(ResourceClaimConflict):
        store.acquire(resource, _owner("durability-successor"), state)

    # Continuity still belongs to the identity the dead child recorded, and it
    # may carry on under a different state.
    moved = store.reassert(resource, owner, _state("durability-next"))
    assert (moved.sequence, moved.owner) == (2, owner)
    assert store.release(resource, owner) is True


def _contend(root, fields, owner_fields, state, barrier, results):
    """Runs in a spawned child process; must stay importable at module level."""

    store = ResourceClaimStore(root)
    resource = ResourceRef(*fields)
    owner = OwnerRef(*owner_fields)
    try:
        barrier.wait(timeout=60)
        store.acquire(resource, owner, state)
    except ResourceClaimConflict:
        results.put(("conflict", owner.id))
    except Exception as exc:  # pragma: no cover - surfaced as a test failure
        results.put(("error", repr(exc)))
    else:
        results.put(("acquired", owner.id))


def test_concurrent_processes_produce_exactly_one_holder(tmp_path):
    context = multiprocessing.get_context("spawn")
    workers = 8
    resource = _ref("contention")
    fields = (resource.namespace, resource.kind, resource.identity)
    barrier = context.Barrier(workers)
    results = context.Queue()

    processes = []
    for index in range(workers):
        owner = _owner(f"contention-{index}")
        process = context.Process(
            target=_contend,
            args=(
                str(tmp_path),
                fields,
                (owner.scope, owner.id),
                _state(f"contention-{index}"),
                barrier,
                results,
            ),
        )
        process.start()
        processes.append(process)

    outcomes = [results.get(timeout=120) for _ in range(workers)]
    for process in processes:
        process.join(timeout=120)
        assert process.exitcode == 0

    assert [outcome for outcome, _ in outcomes].count("acquired") == 1
    assert [outcome for outcome, _ in outcomes].count("conflict") == workers - 1
    assert not [detail for outcome, detail in outcomes if outcome == "error"]

    winner = next(detail for outcome, detail in outcomes if outcome == "acquired")
    found = _store(tmp_path).inspect(resource)
    assert found.status == "held"
    assert found.record is not None
    assert found.record.owner.id == winner
    assert found.record.sequence == 1


def _progress(root, fields, owner_fields, state, barrier, results):
    """Cross the barrier from *inside* the store's own critical section.

    ``sweep`` calls the authority while holding the resource lock, so a worker
    can only reach the barrier if its peers are simultaneously inside their own
    locks.  If exclusion were global rather than per-resource, the barrier would
    never fill and every worker would report a timeout.
    """

    store = ResourceClaimStore(root)
    resource = ResourceRef(*fields)
    owner = OwnerRef(*owner_fields)

    def authority(candidate, recorded_state):
        barrier.wait(timeout=30)
        return "held"

    try:
        store.acquire(resource, owner, state)
        outcome = store.sweep(resource, authority)
    except Exception as exc:  # pragma: no cover - surfaced as a test failure
        results.put(("error", owner.id, repr(exc)))
    else:
        results.put(("progressed", owner.id, outcome.status))


def test_distinct_resources_make_simultaneous_progress(tmp_path):
    context = multiprocessing.get_context("spawn")
    workers = 4
    barrier = context.Barrier(workers)
    results = context.Queue()

    resources = [_ref(f"parallel-{index}") for index in range(workers)]
    owners = [_owner(f"parallel-{index}") for index in range(workers)]
    processes = []
    for resource, owner in zip(resources, owners):
        process = context.Process(
            target=_progress,
            args=(
                str(tmp_path),
                (resource.namespace, resource.kind, resource.identity),
                (owner.scope, owner.id),
                _state("parallel"),
                barrier,
                results,
            ),
        )
        process.start()
        processes.append(process)

    outcomes = [results.get(timeout=120) for _ in range(workers)]
    for process in processes:
        process.join(timeout=120)
        assert process.exitcode == 0

    assert not [detail for outcome, _, detail in outcomes if outcome == "error"], outcomes
    assert {outcome for outcome, _, _ in outcomes} == {"progressed"}
    assert {status for _, _, status in outcomes} == {"held"}

    store = _store(tmp_path)
    for resource, owner in zip(resources, owners):
        found = store.inspect(resource)
        assert found.status == "held"
        assert found.record is not None
        assert found.record.owner == owner


# --------------------------------------------------------------------------
# authority boundary: this kernel is a primitive, not the workspace claim
# --------------------------------------------------------------------------


def test_the_kernel_is_not_wired_into_the_audited_workspace_claim():
    """Guard against a silent second authority over one worktree.

    The audited route's workspace claim is authoritative. This kernel is a
    general primitive with no production consumer yet, and wiring it in
    casually is how two stores come to disagree - the dangerous direction
    being the generic store reporting `free` for a worktree the service still
    owns, letting a second job edit a tree whose revision an auditor is about
    to read.

    If you are here because you deliberately integrated the two, this test is
    the checklist: shadow the existing claim, prove the two never disagree
    across the ownership suites, keep the service claim deciding, and only then
    change this assertion. See `docs/CODING_CONTROL_PLANE.md`.
    """

    package = Path(flyto_ai.__file__).resolve().parent
    consumers = []
    for path in sorted(package.rglob("*.py")):
        if path.parent.name == "orchestration":
            continue  # the kernel and its own package export
        text = path.read_text(encoding="utf-8", errors="replace")
        if "resource_claims" in text or "ResourceClaimStore" in text:
            consumers.append(path.relative_to(package).as_posix())

    assert consumers == [], (
        "the claim kernel gained a consumer outside flyto_ai/orchestration: "
        + ", ".join(consumers)
    )
