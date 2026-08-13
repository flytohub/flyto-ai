import copy
import dataclasses
import json
from collections.abc import Mapping

import pytest

from flyto_ai.capability_catalog import (
    CAPABILITY_CARD_VERSION,
    CAPABILITY_CLAIM_VERSION,
    CAPABILITY_SEARCH_VERSION,
    CapabilityAuthority,
    CapabilityCatalogError,
    build_capability_card,
    capability_claim_digest,
    project_capability_search,
)


def claim(origin="declared", kind="workflow"):
    return {
        "claim_version": CAPABILITY_CLAIM_VERSION,
        "semantic_origin": origin,
        "source": {"kind": kind, "reference": "catalog/item"},
        "display": {
            "title": "Convert records",
            "summary": "Convert tabular records to JSON.",
        },
        "semantics": {
            "intents": ["data.convert"],
            "affordances": ["records.read"],
            "effects": ["json.create"],
            "events": [],
        },
    }


def authority(value, **changes):
    values = dict(
        tenant_id="tenant-1",
        space_id="space-1",
        capability_id="cap-1",
        claim_digest=capability_claim_digest(value),
        host_verified=True,
        approved=True,
        verified=True,
        active=True,
        retired=False,
    )
    values.update(changes)
    return CapabilityAuthority(**values)


class HostileMapping(Mapping):
    def __init__(self, value, fault):
        self.value = value
        self.fault = fault

    def __len__(self):
        return len(self.value)

    def __iter__(self):
        if self.fault == "iter":
            raise RuntimeError("secret-runtime")
        return iter(self.value)

    def __getitem__(self, key):
        if self.fault == "getitem":
            raise RuntimeError("secret-runtime")
        return self.value[key]

    def items(self):
        if self.fault == "items":
            raise RuntimeError("secret-runtime")
        return super().items()


def test_versions_round_trip_and_detachment():
    value = claim()
    frozen_authority = authority(value)
    card = build_capability_card(value, frozen_authority)
    search = project_capability_search(card, frozen_authority)
    assert (card["claim_version"], card["card_version"], search["search_version"]) == (
        CAPABILITY_CLAIM_VERSION,
        CAPABILITY_CARD_VERSION,
        CAPABILITY_SEARCH_VERSION,
    )
    assert json.loads(json.dumps(card)) == card
    assert json.loads(json.dumps(search)) == search
    value["display"]["title"] = "tampered"
    card["display"]["title"] = "also tampered"
    assert search["title"] == "Convert records"


def test_digest_is_order_independent_and_every_claim_field_is_governed():
    value = claim()
    reordered = dict(reversed(list(value.items())))
    reordered["semantics"] = dict(reversed(list(value["semantics"].items())))
    assert capability_claim_digest(value) == capability_claim_digest(reordered)
    mutations = [
        ("semantic_origin", "static_derived"),
        ("source", None),
        ("display", {"title": "Other", "summary": value["display"]["summary"]}),
        ("semantics", {**value["semantics"], "events": ["done"]}),
    ]
    for field, replacement in mutations:
        changed = copy.deepcopy(value)
        changed[field] = replacement
        assert capability_claim_digest(changed) != capability_claim_digest(value)


def test_host_ownership_digest_binding_and_frozen_authority():
    value = claim()
    with pytest.raises(dataclasses.FrozenInstanceError):
        authority(value).active = False
    injected = copy.deepcopy(value)
    injected["tenant_id"] = "attacker"
    with pytest.raises(CapabilityCatalogError):
        capability_claim_digest(injected)
    tampered = copy.deepcopy(value)
    tampered["display"]["summary"] = "changed"
    with pytest.raises(CapabilityCatalogError):
        build_capability_card(tampered, authority(value))


@pytest.mark.parametrize(
    ("changes", "state"),
    [
        ({"approved": False}, "draft_unapproved"),
        ({"verified": False}, "draft_unverified"),
        ({"active": False}, "inactive"),
        ({"retired": True}, "retired"),
    ],
)
def test_nonapproved_transition_matrix_is_never_routable(changes, state):
    value = claim("static_derived")
    card = build_capability_card(value, authority(value, **changes))
    assert card["trust_state"] == state
    assert card["autonomous_routable"] is False
    assert project_capability_search(card, authority(value, **changes))["audit_visible"] is True


def test_only_complete_exact_approved_verified_active_card_routes():
    value = claim()
    card = build_capability_card(value, authority(value))
    assert card["trust_state"] == "approved_verified"
    assert card["autonomous_routable"] is True


@pytest.mark.parametrize(
    "replacement",
    [
        {"title": "", "summary": ""},
        {"title": " ", "summary": "   "},
        {"title": "Identifier only", "summary": ""},
    ],
)
def test_empty_or_identifier_only_claims_remain_incomplete(replacement):
    value = claim()
    value["source"] = None
    value["display"] = replacement
    value["semantics"] = {field: [] for field in value["semantics"]}
    card = build_capability_card(value, authority(value))
    assert card["complete"] is False
    assert card["autonomous_routable"] is False
    assert "Run workflow" not in json.dumps(card)


@pytest.mark.parametrize(
    "kind", ["workflow", "mcp", "software.package", "hardware.sensor"]
)
def test_source_kinds_are_domain_neutral(kind):
    value = claim(kind=kind)
    assert build_capability_card(value, authority(value))["source_kind"] == kind


def test_search_projection_is_an_exact_safe_allowlist():
    value = claim()
    value["source"]["reference"] = "https://endpoint.invalid?token=secret"
    frozen_authority = authority(value)
    projection = project_capability_search(
        build_capability_card(value, frozen_authority), frozen_authority
    )
    encoded = json.dumps(projection)
    assert set(projection) == {
        "search_version",
        "card_version",
        "tenant_id",
        "space_id",
        "capability_id",
        "content_digest",
        "semantic_origin",
        "source_kind",
        "title",
        "summary",
        "semantic_ids",
        "approved",
        "host_verified",
        "verified",
        "active",
        "retired",
        "complete",
        "trust_state",
        "autonomous_routable",
        "audit_visible",
    }
    assert "endpoint.invalid" not in encoded and "secret" not in encoded


@pytest.mark.parametrize(
    "field", ["params", "credentials", "token", "endpoint", "raw_body"]
)
def test_unknown_sensitive_claim_fields_fail_closed(field):
    value = claim()
    value[field] = {"anything": "secret"}
    with pytest.raises(CapabilityCatalogError):
        capability_claim_digest(value)


@pytest.mark.parametrize(
    "bad", ["control\n", "bidi\u202e", "zero\u200bwidth", "surrogate\ud800"]
)
def test_unsafe_text_fails_with_stable_exception(bad):
    value = claim()
    value["display"]["title"] = bad
    with pytest.raises(CapabilityCatalogError):
        capability_claim_digest(value)


def test_duplicates_and_resource_exhaustion_fail_closed():
    cases = []
    duplicate = claim()
    duplicate["semantics"]["events"] = ["done", "done"]
    cases.append(duplicate)
    long_list = claim()
    long_list["semantics"]["events"] = [f"event.{i}" for i in range(33)]
    cases.append(long_list)
    long_text = claim()
    long_text["display"]["summary"] = "x" * 2001
    cases.append(long_text)
    deep = claim()
    nested = {}
    deep["extra"] = nested
    for _ in range(10):
        child = {}
        nested["x"] = child
        nested = child
    cases.append(deep)
    for value in cases:
        with pytest.raises(CapabilityCatalogError):
            capability_claim_digest(value)


@pytest.mark.parametrize("bad", [object(), {1: "x"}, {"x": 2**100}, {"x": 1.2}])
def test_non_json_and_huge_integer_inputs_fail_closed(bad):
    with pytest.raises(CapabilityCatalogError):
        capability_claim_digest(bad)


def test_malformed_authority_fails_with_boundary_exception():
    value = claim()
    for changes in (
        {"approved": 1},
        {"claim_digest": "718c5a24bd4cf123"},
        {"tenant_id": "bad id"},
        {"host_verified": False},
    ):
        with pytest.raises(CapabilityCatalogError):
            authority(value, **changes)


@pytest.mark.parametrize(
    "source",
    [
        None,
        {"kind": "workflow", "reference": None},
        {"kind": "workflow", "reference": ""},
        {"kind": "workflow", "reference": "   "},
    ],
)
def test_missing_or_empty_source_reference_is_auditable_but_incomplete(source):
    value = claim()
    value["source"] = source
    card = build_capability_card(value, authority(value))
    assert card["complete"] is False
    assert card["autonomous_routable"] is False
    assert "Run workflow" not in json.dumps(card)


def test_identifiers_and_empty_semantics_never_synthesize_claim_content():
    value = claim()
    value["display"] = {"title": "", "summary": ""}
    value["semantics"] = {field: [] for field in value["semantics"]}
    card = build_capability_card(value, authority(value))
    assert card["display"] == value["display"]
    assert card["semantics"] == value["semantics"]
    assert card["complete"] is False
    assert card["autonomous_routable"] is False


def test_projection_rejects_every_tampered_claim_or_authority_field():
    value = claim()
    frozen_authority = authority(value)
    card = build_capability_card(value, frozen_authority)
    mutations = {
        "card_version": "flyto.capability-card.v0",
        "claim_version": "flyto.capability-claim.v0",
        "tenant_id": "tenant-2",
        "space_id": "space-2",
        "capability_id": "cap-2",
        "content_digest": "sha256:" + "0" * 64,
        "semantic_origin": "static_derived",
        "source_kind": "other",
        "display": {"title": "Changed", "summary": card["display"]["summary"]},
        "semantics": {**card["semantics"], "events": ["changed"]},
        "approved": False,
        "host_verified": False,
        "verified": False,
        "active": False,
        "retired": True,
        "complete": False,
        "trust_state": "inactive",
        "autonomous_routable": False,
        "audit_visible": False,
    }
    for field, replacement in mutations.items():
        changed = copy.deepcopy(card)
        changed[field] = replacement
        with pytest.raises(CapabilityCatalogError):
            project_capability_search(changed, frozen_authority)

    changed = copy.deepcopy(card)
    changed["canonical_claim"]["source"]["reference"] = "catalog/other"
    with pytest.raises(CapabilityCatalogError):
        project_capability_search(changed, frozen_authority)


def test_projection_excludes_canonical_claim_and_sensitive_runtime_shapes():
    value = claim()
    value["source"]["reference"] = "endpoint/token/secret"
    frozen_authority = authority(value)
    projection = project_capability_search(
        build_capability_card(value, frozen_authority), frozen_authority
    )
    encoded = json.dumps(projection)
    for forbidden in (
        "canonical_claim",
        "reference",
        "params",
        "defaults",
        "payloads",
        "prompts",
        "headers",
        "credentials",
        "tokens",
        "secrets",
        "MCP args",
        "endpoint",
        "raw body",
        "endpoint/token/secret",
    ):
        assert forbidden not in encoded


@pytest.mark.parametrize("fault", ["iter", "items", "getitem"])
def test_hostile_claim_mapping_faults_have_stable_content_free_error(fault):
    with pytest.raises(CapabilityCatalogError) as caught:
        capability_claim_digest(HostileMapping(claim(), fault))
    assert str(caught.value) == "claim is malformed"
    assert "secret-runtime" not in str(caught.value)


@pytest.mark.parametrize("fault", ["iter", "items", "getitem"])
def test_hostile_card_mapping_faults_have_stable_content_free_error(fault):
    value = claim()
    frozen_authority = authority(value)
    card = build_capability_card(value, frozen_authority)
    with pytest.raises(CapabilityCatalogError) as caught:
        project_capability_search(HostileMapping(card, fault), frozen_authority)
    assert str(caught.value) == "card is malformed"
    assert "secret-runtime" not in str(caught.value)
