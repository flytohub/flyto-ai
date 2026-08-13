# Copyright 2024 Flyto2
# Licensed under the Apache License, Version 2.0
"""Provider-neutral Capability Card and search-projection trust boundary.

This phase-one module validates data only.  It owns no persistence, retrieval,
approval, verification, installation, routing, or execution behavior.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


CAPABILITY_CLAIM_VERSION = "flyto.capability-claim.v1"
CAPABILITY_CARD_VERSION = "flyto.capability-card.v1"
CAPABILITY_SEARCH_VERSION = "flyto.capability-search.v1"

_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:@/-]{0,191}$")
_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_ORIGINS = frozenset({"declared", "static_derived"})
_SEMANTIC_FIELDS = ("intents", "affordances", "effects", "events")
_MAX_DEPTH = 8
_MAX_NODES = 512
_MAX_CLAIM_BYTES = 65_536
_MAX_OUTPUT_BYTES = 65_536
_MAX_DISPLAY_TEXT = 2_000
_MAX_SOURCE_TEXT = 1_000
_MAX_SEMANTICS = 32

__all__ = [
    "CAPABILITY_CLAIM_VERSION",
    "CAPABILITY_CARD_VERSION",
    "CAPABILITY_SEARCH_VERSION",
    "CapabilityAuthority",
    "CapabilityCatalogError",
    "build_capability_card",
    "capability_claim_digest",
    "project_capability_search",
]


class CapabilityCatalogError(ValueError):
    """Stable fail-closed exception for the capability catalog boundary."""


def _safe_text(value: object, path: str, maximum: int, *, empty: bool) -> str:
    if not isinstance(value, str) or len(value) > maximum:
        raise CapabilityCatalogError(f"{path} must be bounded text")
    if not empty and not value.strip():
        raise CapabilityCatalogError(f"{path} must not be empty")
    if any(unicodedata.category(char) in {"Cc", "Cf", "Cs"} for char in value):
        raise CapabilityCatalogError(f"{path} contains unsafe text")
    normalized = unicodedata.normalize("NFC", value)
    if not empty and not normalized.strip():
        raise CapabilityCatalogError(f"{path} must not be empty")
    return normalized


def _identifier(value: object, path: str) -> str:
    result = _safe_text(value, path, 192, empty=False)
    if not _SAFE_ID.fullmatch(result):
        raise CapabilityCatalogError(f"{path} must be a safe identifier")
    return result


def _exact_object(value: object, fields: set[str], path: str) -> dict[str, Any]:
    if type(value) is not dict:
        raise CapabilityCatalogError(f"{path} must be an object")
    keys = set(value)
    if any(not isinstance(key, str) for key in keys) or keys != fields:
        raise CapabilityCatalogError(f"{path} has unknown or missing fields")
    return value


def _snapshot_json(value: object, path: str) -> Any:
    """Detach bounded exact JSON without trusting Mapping getters afterward."""
    nodes = 0

    def snapshot(item: object, depth: int) -> Any:
        nonlocal nodes
        nodes += 1
        if nodes > _MAX_NODES or depth > _MAX_DEPTH:
            raise CapabilityCatalogError(f"{path} exceeds structural limits")
        if item is None or type(item) in {bool, str}:
            return item
        if type(item) is int:
            if not -(2**63 - 1) <= item <= 2**63 - 1:
                raise CapabilityCatalogError(f"{path} contains an unsafe integer")
            return item
        if isinstance(item, Mapping):
            try:
                entries = list(item.items())
            except Exception as exc:
                raise CapabilityCatalogError(f"{path} is malformed") from exc
            detached: dict[str, Any] = {}
            try:
                for entry in entries:
                    key, child = entry
                    if not isinstance(key, str) or key in detached:
                        raise CapabilityCatalogError(f"{path} is malformed")
                    detached[key] = snapshot(child, depth + 1)
            except CapabilityCatalogError:
                raise
            except Exception as exc:
                raise CapabilityCatalogError(f"{path} is malformed") from exc
            return detached
        if isinstance(item, list):
            try:
                return [snapshot(child, depth + 1) for child in item]
            except CapabilityCatalogError:
                raise
            except Exception as exc:
                raise CapabilityCatalogError(f"{path} is malformed") from exc
        raise CapabilityCatalogError(f"{path} must contain only exact JSON values")

    return snapshot(value, 0)


def _encode(value: object, path: str, maximum: int) -> bytes:
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (RecursionError, TypeError, ValueError, OverflowError) as exc:
        raise CapabilityCatalogError(f"{path} is not canonical JSON") from exc
    if len(encoded) > maximum:
        raise CapabilityCatalogError(f"{path} exceeds its byte limit")
    return encoded


def _canonical_claim(claim: object) -> dict[str, Any]:
    claim = _snapshot_json(claim, "claim")
    obj = _exact_object(
        claim,
        {"claim_version", "semantic_origin", "source", "display", "semantics"},
        "claim",
    )
    if obj["claim_version"] != CAPABILITY_CLAIM_VERSION:
        raise CapabilityCatalogError("claim.claim_version is unsupported")
    origin = obj["semantic_origin"]
    if type(origin) is not str or origin not in _ORIGINS:
        raise CapabilityCatalogError("claim.semantic_origin is unsupported")

    source_value = obj["source"]
    source: dict[str, Any] | None
    if source_value is None:
        source = None
    else:
        source_obj = _exact_object(source_value, {"kind", "reference"}, "claim.source")
        kind = _identifier(source_obj["kind"], "claim.source.kind")
        reference_value = source_obj["reference"]
        reference = (
            None
            if reference_value is None
            else _safe_text(
                reference_value, "claim.source.reference", _MAX_SOURCE_TEXT, empty=True
            )
        )
        source = {"kind": kind, "reference": reference}

    display_obj = _exact_object(obj["display"], {"title", "summary"}, "claim.display")
    display = {
        "title": _safe_text(
            display_obj["title"], "claim.display.title", _MAX_DISPLAY_TEXT, empty=True
        ),
        "summary": _safe_text(
            display_obj["summary"],
            "claim.display.summary",
            _MAX_DISPLAY_TEXT,
            empty=True,
        ),
    }
    semantics_obj = _exact_object(
        obj["semantics"], set(_SEMANTIC_FIELDS), "claim.semantics"
    )
    semantics: dict[str, list[str]] = {}
    for field in _SEMANTIC_FIELDS:
        raw = semantics_obj[field]
        if not isinstance(raw, list) or len(raw) > _MAX_SEMANTICS:
            raise CapabilityCatalogError(
                f"claim.semantics.{field} must be a bounded list"
            )
        values = [_identifier(item, f"claim.semantics.{field}[]") for item in raw]
        if len(set(values)) != len(values):
            raise CapabilityCatalogError(f"claim.semantics.{field} contains duplicates")
        semantics[field] = sorted(values)

    canonical = {
        "claim_version": CAPABILITY_CLAIM_VERSION,
        "display": display,
        "semantic_origin": origin,
        "semantics": semantics,
        "source": source,
    }
    _encode(canonical, "claim", _MAX_CLAIM_BYTES)
    return canonical


def capability_claim_digest(claim: object) -> str:
    """Return the stable digest of one exact, canonical v1 untrusted claim."""
    canonical = _canonical_claim(claim)
    return (
        "sha256:"
        + hashlib.sha256(_encode(canonical, "claim", _MAX_CLAIM_BYTES)).hexdigest()
    )


@dataclass(frozen=True, slots=True)
class CapabilityAuthority:
    """Frozen host-owned state bound to an exact approved claim digest."""

    tenant_id: str
    space_id: str
    capability_id: str
    claim_digest: str
    host_verified: bool
    approved: bool
    verified: bool
    active: bool
    retired: bool

    def __post_init__(self) -> None:
        for field in ("tenant_id", "space_id", "capability_id"):
            object.__setattr__(
                self, field, _identifier(getattr(self, field), f"authority.{field}")
            )
        if not isinstance(self.claim_digest, str) or not _DIGEST.fullmatch(
            self.claim_digest
        ):
            raise CapabilityCatalogError("authority.claim_digest is invalid")
        for field in ("host_verified", "approved", "verified", "active", "retired"):
            if type(getattr(self, field)) is not bool:
                raise CapabilityCatalogError(f"authority.{field} must be a boolean")
        if self.host_verified is not True:
            raise CapabilityCatalogError("authority.host_verified must be true")


def _trust_state(authority: CapabilityAuthority, complete: bool) -> str:
    if authority.retired:
        return "retired"
    if not authority.active:
        return "inactive"
    if not complete:
        return "incomplete"
    if not authority.approved:
        return "draft_unapproved"
    if not authority.verified:
        return "draft_unverified"
    return "approved_verified"


def build_capability_card(
    claim: object, authority: CapabilityAuthority
) -> dict[str, Any]:
    """Build a detached v1 Capability Card from claim plus verified host state."""
    if not isinstance(authority, CapabilityAuthority):
        raise CapabilityCatalogError("authority must be CapabilityAuthority")
    canonical = _canonical_claim(claim)
    digest = capability_claim_digest(canonical)
    if digest != authority.claim_digest:
        raise CapabilityCatalogError("authority is not bound to this claim digest")
    display = canonical["display"]
    semantics = canonical["semantics"]
    source = canonical["source"]
    complete = bool(
        display["title"].strip()
        and display["summary"].strip()
        and any(semantics[field] for field in _SEMANTIC_FIELDS)
        and source is not None
        and source["reference"] is not None
        and source["reference"].strip()
    )
    routable = bool(
        complete
        and authority.approved
        and authority.verified
        and authority.active
        and not authority.retired
    )
    card = {
        "card_version": CAPABILITY_CARD_VERSION,
        "claim_version": CAPABILITY_CLAIM_VERSION,
        "tenant_id": authority.tenant_id,
        "space_id": authority.space_id,
        "capability_id": authority.capability_id,
        "content_digest": digest,
        "canonical_claim": canonical,
        "semantic_origin": canonical["semantic_origin"],
        "source_kind": None if source is None else source["kind"],
        "display": {"title": display["title"], "summary": display["summary"]},
        "semantics": {field: list(semantics[field]) for field in _SEMANTIC_FIELDS},
        "approved": authority.approved,
        "host_verified": authority.host_verified,
        "verified": authority.verified,
        "active": authority.active,
        "retired": authority.retired,
        "complete": complete,
        "trust_state": _trust_state(authority, complete),
        "autonomous_routable": routable,
        "audit_visible": True,
    }
    card = _snapshot_json(card, "card")
    _encode(card, "card", _MAX_OUTPUT_BYTES)
    return card


def project_capability_search(
    card: object, authority: CapabilityAuthority
) -> dict[str, Any]:
    """Create the exact bounded v1 audit/search projection from a valid card."""
    if not isinstance(authority, CapabilityAuthority):
        raise CapabilityCatalogError("authority must be CapabilityAuthority")
    fields = {
        "card_version",
        "claim_version",
        "tenant_id",
        "space_id",
        "capability_id",
        "content_digest",
        "canonical_claim",
        "semantic_origin",
        "source_kind",
        "display",
        "semantics",
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
    obj = _exact_object(_snapshot_json(card, "card"), fields, "card")
    # Rebuild through the same strict boundary so caller-authored card fields
    # cannot smuggle values into the projection.
    expected = build_capability_card(obj["canonical_claim"], authority)
    if obj != expected:
        raise CapabilityCatalogError("card is not bound to its canonical claim")
    display = expected["display"]
    projected_semantics = expected["semantics"]
    projection = {
        "search_version": CAPABILITY_SEARCH_VERSION,
        "card_version": CAPABILITY_CARD_VERSION,
        "tenant_id": authority.tenant_id,
        "space_id": authority.space_id,
        "capability_id": authority.capability_id,
        "content_digest": authority.claim_digest,
        "semantic_origin": expected["semantic_origin"],
        "source_kind": expected["source_kind"],
        "title": display["title"],
        "summary": display["summary"],
        "semantic_ids": projected_semantics,
        "approved": authority.approved,
        "host_verified": authority.host_verified,
        "verified": authority.verified,
        "active": authority.active,
        "retired": authority.retired,
        "complete": expected["complete"],
        "trust_state": expected["trust_state"],
        "autonomous_routable": expected["autonomous_routable"],
        "audit_visible": True,
    }
    projection = _snapshot_json(projection, "search projection")
    _encode(projection, "search projection", _MAX_OUTPUT_BYTES)
    return projection
