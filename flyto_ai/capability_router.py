# Copyright 2024 Flyto2
# Licensed under the Apache License, Version 2.0
"""Provider-neutral, deterministic-first routing for large capability catalogs.

This module is deliberately independent of robot source trees.  It consumes
versioned JSON manifests, optionally enriches discovery through the existing
Flyto Core and Blueprint bridges, and returns a bounded shortlist for an LLM.
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import math
import re
import unicodedata
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from itertools import islice
from typing import Any

CAPABILITY_ROUTE_VERSION = "flyto.capability-route.v1"
ROUTING_DECISION_VERSION = "flyto.capability-routing-decision.v1"
GOAL_FRAME_VERSION = "flyto.goal-frame.v1"
GOAL_FRAME_REQUEST_VERSION = "flyto.goal-frame-request.v1"
CAPABILITY_RETRIEVAL_VERSION = "flyto.ai.capability-retrieval-handoff.v2"
CAPABILITY_RETRIEVAL_EVIDENCE_VERSION = "flyto.capability-retrieval-evidence.v1"
_WORD = re.compile(r"[^\W_]+", re.UNICODE)
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:@-]{0,191}$")
_RETRIEVAL_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/@-]*$")
_DIGEST = re.compile(r"^sha256:[0-9a-f]{64}$")
_RETRIEVAL_ERROR = "invalid capability retrieval handoff"
_RETRIEVAL_MAX_BYTES = 131_072
_RETRIEVAL_MAX_DEPTH = 8
_RETRIEVAL_MAX_NODES = 2_048
_RETRIEVAL_RISK_LEVELS = ("minimal", "low", "medium", "high", "critical")

# Bounded, machine-readable discovery outcomes.  A lane outcome is derived from
# a completed bridge call, never from model prose, and never carries raw
# exception text across the trust boundary.
DISCOVERY_APPLIED = "applied"
DISCOVERY_NOT_APPLICABLE = "not_applicable"
DISCOVERY_UNAVAILABLE = "unavailable"
DISCOVERY_FAILED = "failed"
DISCOVERY_STATUSES = (
    DISCOVERY_APPLIED,
    DISCOVERY_NOT_APPLICABLE,
    DISCOVERY_UNAVAILABLE,
    DISCOVERY_FAILED,
)
_DISCOVERY_READY = frozenset({DISCOVERY_APPLIED, DISCOVERY_NOT_APPLICABLE})

# Core runtime evidence is only trusted when the bridge returns this exact
# contract plus an explicit boolean ok=true.  Anything else is an absent or
# malformed runtime, never a legitimate no-match.
CORE_RUNTIME_CONTRACT = "flyto-core-mcp.v1"
# Conservative ceiling on read-only Blueprint discovery.  Blueprint search is
# relevance ordered and only one trusted procedure is ever consumed, so an
# over-bound result is a broken bridge rather than richer evidence.
BLUEPRINT_CANDIDATE_LIMIT = 32
# Public routing bounds have different units even though their current values
# are equal.  Keep them separately named so changing the number of capability
# choices cannot silently widen the provider-row projection (or vice versa).
CAPABILITY_GROUP_LIMIT = 32
EMITTED_PROVIDER_ROW_LIMIT = 32
_MISSING = object()

CoreDispatch = Callable[[str, dict[str, Any]], Awaitable[dict[str, Any]]]
BlueprintSearch = Callable[[str], Sequence[Mapping[str, Any]]]


class CapabilityRoutingError(ValueError):
    """Raised when a capability catalog or routing policy is unsafe."""


def capability_routing_bounds() -> dict[str, int]:
    """Return detached public bounds with their distinct routing units."""
    return {
        "capability_groups": CAPABILITY_GROUP_LIMIT,
        "emitted_provider_rows": EMITTED_PROVIDER_ROW_LIMIT,
    }


def _retrieval_fail() -> None:
    """Raise the one content-free error exposed by the retrieval boundary."""
    raise CapabilityRoutingError(_RETRIEVAL_ERROR)


def _retrieval_id(value: object) -> str:
    if (
        type(value) is not str
        or not value
        or len(value) > 192
        or value != unicodedata.normalize("NFC", value)
        or value != value.strip()
        or not _RETRIEVAL_ID.fullmatch(value)
        or any(
            unicodedata.category(character) in {"Cc", "Cf", "Cs", "Co", "Cn"}
            for character in value
        )
    ):
        _retrieval_fail()
    return value


def _retrieval_ids(value: object, *, maximum: int, item_maximum: int) -> list[str]:
    if type(value) is not list or len(value) > maximum:
        _retrieval_fail()
    result = [_retrieval_id(item) for item in value]
    if any(len(item) > item_maximum for item in result) or result != sorted(
        set(result)
    ):
        _retrieval_fail()
    return result


def _retrieval_optional_id(value: object) -> str:
    if value == "":
        return ""
    return _retrieval_id(value)


def _retrieval_digest(value: object) -> str:
    if type(value) is not str or not _DIGEST.fullmatch(value):
        _retrieval_fail()
    return value


@dataclass(frozen=True, slots=True)
class CapabilityRetrievalAuthority:
    """Frozen host authority for one exact, already-validated search result."""

    tenant_id: str
    workspace_id: str
    space_id: str
    request_digest: str
    model_digest: str
    index_digest: str
    snapshot_digest: str
    query_context_digest: str
    requirements_digest: str
    result_digest: str
    goal_digest: str
    routing_context_digest: str
    goal_frame_digest: str
    handoff_digest: str
    host_verified: bool

    def __post_init__(self) -> None:
        try:
            for field in (
                "tenant_id",
                "workspace_id",
                "space_id",
            ):
                object.__setattr__(self, field, _retrieval_id(getattr(self, field)))
            for field in (
                "request_digest",
                "model_digest",
                "index_digest",
                "snapshot_digest",
                "query_context_digest",
                "requirements_digest",
                "result_digest",
                "goal_digest",
                "routing_context_digest",
                "goal_frame_digest",
                "handoff_digest",
            ):
                object.__setattr__(self, field, _retrieval_digest(getattr(self, field)))
            if type(self.host_verified) is not bool or self.host_verified is not True:
                _retrieval_fail()
        except CapabilityRoutingError:
            raise
        except Exception:
            _retrieval_fail()


def _retrieval_json(value: object) -> Any:
    """Detach exact JSON while bounding structure before encoding or recursion."""
    nodes = 0

    def copy(item: object, depth: int) -> Any:
        nonlocal nodes
        nodes += 1
        if nodes > _RETRIEVAL_MAX_NODES or depth > _RETRIEVAL_MAX_DEPTH:
            _retrieval_fail()
        if item is None or type(item) in {bool, str}:
            return item
        if type(item) is int:
            if not -(2**63 - 1) <= item <= 2**63 - 1:
                _retrieval_fail()
            return item
        if type(item) is float:
            if not math.isfinite(item):
                _retrieval_fail()
            return item
        if type(item) is dict:
            if any(type(key) is not str for key in item):
                _retrieval_fail()
            return {key: copy(child, depth + 1) for key, child in item.items()}
        if type(item) is list:
            return [copy(child, depth + 1) for child in item]
        _retrieval_fail()

    try:
        detached = copy(value, 0)
        encoded = json.dumps(
            detached,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except CapabilityRoutingError:
        raise
    except Exception:
        _retrieval_fail()
    if len(encoded) > _RETRIEVAL_MAX_BYTES:
        _retrieval_fail()
    return detached


def _exact_retrieval_object(value: object, fields: set[str]) -> dict[str, Any]:
    if type(value) is not dict or set(value) != fields:
        _retrieval_fail()
    return value


def _sha(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def validate_capability_retrieval(
    handoff: Mapping[str, Any], authority: CapabilityRetrievalAuthority
) -> dict[str, Any]:
    """Validate and detach one terminal, candidate-only bounded search handoff."""
    if type(authority) is not CapabilityRetrievalAuthority:
        _retrieval_fail()
    obj = _exact_retrieval_object(
        _retrieval_json(handoff),
        {
            "contract_version",
            "tenant_id",
            "workspace_id",
            "space_id",
            "goal_digest",
            "routing_context_digest",
            "goal_frame_digest",
            "request",
            "result",
            "candidate_only",
            "execution_authority",
            "handoff_digest",
        },
    )
    if (
        obj["contract_version"] != CAPABILITY_RETRIEVAL_VERSION
        or any(
            obj[field] != getattr(authority, field)
            for field in (
                "tenant_id",
                "workspace_id",
                "space_id",
                "goal_digest",
                "routing_context_digest",
                "goal_frame_digest",
            )
        )
        or obj["candidate_only"] is not True
        or obj["execution_authority"] is not False
    ):
        _retrieval_fail()
    request = _exact_retrieval_object(
        obj["request"],
        {
            "request_version",
            "query",
            "top_k",
            "page_size",
            "hard_filters",
            "prefilter_required",
            "retrieval_order",
            "model",
            "index_digest",
            "snapshot_digest",
            "weights",
            "cursor",
            "request_digest",
        },
    )
    top_k = request["top_k"]
    if (
        request["request_version"] != "flyto.capability-search-query.v1"
        or type(top_k) is not int
        or not 1 <= top_k <= 32
        or type(request["page_size"]) is not int
        or request["page_size"] != top_k
        or request["cursor"] is not None
        or request["prefilter_required"] is not True
        or request["retrieval_order"] != ["hard_filter", "lexical", "ann", "fuse"]
    ):
        _retrieval_fail()
    filters = _exact_retrieval_object(
        request["hard_filters"],
        {
            "tenant_id",
            "space_id",
            "status",
            "acl_principals",
            "acl_scopes",
            "risk_classification",
            "resource_ids",
            "capability_ids",
        },
    )
    model = _exact_retrieval_object(
        request["model"],
        {"model_id", "model_version", "dimensions", "model_digest"},
    )
    weights = _exact_retrieval_object(request["weights"], {"lexical", "vector"})
    if (
        type(request["query"]) is not str
        or not request["query"]
        or len(request["query"]) > 2_048
        or request["query"] != unicodedata.normalize("NFC", request["query"])
        or request["query"] != request["query"].strip()
        or any(
            unicodedata.category(character) in {"Cc", "Cf", "Cs", "Co", "Cn"}
            for character in request["query"]
        )
        or _retrieval_id(model["model_id"]) != model["model_id"]
        or _retrieval_id(model["model_version"]) != model["model_version"]
        or len(model["model_id"]) > 128
        or len(model["model_version"]) > 128
        or type(model["dimensions"]) is not int
        or not 1 <= model["dimensions"] <= 65_536
        or any(type(weights[field]) is not int for field in ("lexical", "vector"))
        or not 0 <= weights["lexical"] <= 100
        or not 0 <= weights["vector"] <= 100
        or weights["lexical"] + weights["vector"] != 100
    ):
        _retrieval_fail()
    principals = _retrieval_ids(
        filters["acl_principals"], maximum=128, item_maximum=128
    )
    scopes = _retrieval_ids(filters["acl_scopes"], maximum=128, item_maximum=128)
    resources_filter = _retrieval_ids(
        filters["resource_ids"], maximum=128, item_maximum=128
    )
    capabilities_filter = _retrieval_ids(
        filters["capability_ids"], maximum=128, item_maximum=192
    )
    if (
        _retrieval_id(filters["tenant_id"]) != authority.tenant_id
        or _retrieval_id(filters["space_id"]) != authority.space_id
        or filters["status"] != "active"
        or not principals
        or not scopes
        or filters["risk_classification"] not in _RETRIEVAL_RISK_LEVELS
        or request["request_digest"] != authority.request_digest
        or model["model_digest"] != authority.model_digest
        or request["index_digest"] != authority.index_digest
        or request["snapshot_digest"] != authority.snapshot_digest
        or request["request_digest"]
        != _sha(
            {k: v for k, v in request.items() if k not in {"cursor", "request_digest"}}
        )
    ):
        _retrieval_fail()
    result = _exact_retrieval_object(
        obj["result"],
        {
            "result_version",
            "query_context_digest",
            "cloud_next_continuation",
            "page",
            "feasibility",
            "candidate_only",
            "execution_authority",
            "result_digest",
        },
    )
    page = _exact_retrieval_object(
        result["page"],
        {
            "page_version",
            "request_digest",
            "candidates",
            "next_cursor",
            "candidate_only",
            "execution_authority",
        },
    )
    if (
        result["result_version"] != "flyto.cloud.capability-index-result.v1"
        or page["page_version"] != "flyto.capability-search-page.v1"
        or result["query_context_digest"] != authority.query_context_digest
        or result["cloud_next_continuation"] is not None
        or result["candidate_only"] is not True
        or result["execution_authority"] is not False
        or page["request_digest"] != authority.request_digest
        or page["next_cursor"] is not None
        or page["candidate_only"] is not True
        or page["execution_authority"] is not False
    ):
        _retrieval_fail()
    candidates = page["candidates"]
    if type(candidates) is not list or len(candidates) > top_k:
        _retrieval_fail()
    candidate_fields = {
        "tenant_id",
        "space_id",
        "capability_id",
        "status",
        "acl_principals",
        "acl_scopes",
        "risk_classification",
        "resource_ids",
        "score",
        "source_projection_digest",
        "upstream_content_digest",
        "document_digest",
        "model_digest",
        "index_digest",
        "snapshot_digest",
        "candidate_digest",
        "candidate_only",
        "execution_authority",
    }
    seen: set[str] = set()
    previous: tuple[int, str] | None = None
    for raw in candidates:
        candidate = _exact_retrieval_object(raw, candidate_fields)
        capability_id = _retrieval_id(candidate["capability_id"])
        if capability_id in seen or candidate["status"] != "active":
            _retrieval_fail()
        seen.add(capability_id)
        candidate_principals = _retrieval_ids(
            candidate["acl_principals"], maximum=128, item_maximum=128
        )
        candidate_scopes = _retrieval_ids(
            candidate["acl_scopes"], maximum=128, item_maximum=128
        )
        candidate_resources = _retrieval_ids(
            candidate["resource_ids"], maximum=128, item_maximum=128
        )
        if (
            candidate["tenant_id"] != authority.tenant_id
            or candidate["space_id"] != authority.space_id
            or candidate["status"] != filters["status"]
            or not set(principals).issubset(candidate_principals)
            or not set(scopes).issubset(candidate_scopes)
            or candidate["risk_classification"] not in _RETRIEVAL_RISK_LEVELS
            or _RETRIEVAL_RISK_LEVELS.index(candidate["risk_classification"])
            > _RETRIEVAL_RISK_LEVELS.index(filters["risk_classification"])
            or (
                resources_filter
                and not set(resources_filter).issubset(candidate_resources)
            )
            or (capabilities_filter and capability_id not in capabilities_filter)
            or candidate["model_digest"] != authority.model_digest
            or candidate["index_digest"] != authority.index_digest
            or candidate["snapshot_digest"] != authority.snapshot_digest
            or candidate["candidate_only"] is not True
            or candidate["execution_authority"] is not False
        ):
            _retrieval_fail()
        for field in (
            "source_projection_digest",
            "upstream_content_digest",
            "document_digest",
            "model_digest",
            "index_digest",
            "snapshot_digest",
        ):
            _retrieval_digest(candidate[field])
        score = candidate["score"]
        if type(score) is not int or not -1_000_000_000 <= score <= 1_000_000_000:
            _retrieval_fail()
        canonical_candidate = {
            key: candidate[key] for key in candidate if key != "candidate_digest"
        }
        if _retrieval_digest(candidate["candidate_digest"]) != _sha(
            canonical_candidate
        ):
            _retrieval_fail()
        order = (-score, capability_id)
        if previous is not None and order <= previous:
            _retrieval_fail()
        previous = order
    feasibility = _exact_retrieval_object(
        result["feasibility"],
        {
            "result_version",
            "requirements_digest",
            "candidate_resources",
            "feasible",
            "candidate_only",
            "execution_authority",
            "result_digest",
        },
    )
    resources = feasibility["candidate_resources"]
    if (
        feasibility["requirements_digest"] != authority.requirements_digest
        or feasibility["result_version"]
        != "flyto.cloud.capability-resource-feasibility.v1"
        or feasibility["feasible"] is not True
        or feasibility["candidate_only"] is not True
        or feasibility["execution_authority"] is not False
        or type(resources) is not dict
    ):
        _retrieval_fail()
    if len(resources) > 128:
        _retrieval_fail()
    clean_resources: dict[str, list[str]] = {}
    for capability_id, resource_ids in resources.items():
        clean_id = _retrieval_id(capability_id)
        clean_resources[clean_id] = _retrieval_ids(
            resource_ids, maximum=128, item_maximum=128
        )
    if resources != dict(sorted(clean_resources.items())):
        _retrieval_fail()
    if (
        feasibility["result_digest"]
        != _sha({k: v for k, v in feasibility.items() if k != "result_digest"})
        or result["result_digest"] != authority.result_digest
        or result["result_digest"]
        != _sha({k: v for k, v in result.items() if k != "result_digest"})
        or obj["handoff_digest"] != authority.handoff_digest
        or obj["handoff_digest"]
        != _sha({k: v for k, v in obj.items() if k != "handoff_digest"})
    ):
        _retrieval_fail()
    return obj


def _empty_blueprint_search(_goal: str) -> tuple[Mapping[str, Any], ...]:
    return ()


def _normalized(value: object) -> str:
    return unicodedata.normalize("NFKC", str(value)).casefold()


def _words(value: object) -> set[str]:
    return set(_WORD.findall(_normalized(value)))


def _string_list(value: object) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple, set, frozenset)):
        return ()
    return tuple(str(item) for item in value if str(item))


def _semantic_ids(value: object, field_name: str) -> tuple[str, ...]:
    values = _string_list(value)
    if len(values) > 128:
        raise CapabilityRoutingError(f"goal_frame.{field_name} exceeds 128 items")
    if any(not _SAFE_ID.fullmatch(item) for item in values):
        raise CapabilityRoutingError(
            f"goal_frame.{field_name} contains an unsafe semantic identifier"
        )
    return tuple(sorted(set(values)))


def normalize_goal_frame(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a language-neutral semantic frame at the trust boundary."""
    if not isinstance(value, Mapping):
        raise CapabilityRoutingError("goal_frame must be an object")
    allowed = {
        "contract_version",
        "intent_ids",
        "required_affordances",
        "desired_effects",
        "trigger_events",
        "constraints",
    }
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise CapabilityRoutingError(
            "goal_frame contains unsupported fields: " + ", ".join(unknown)
        )
    if value.get("contract_version") != GOAL_FRAME_VERSION:
        raise CapabilityRoutingError("unsupported goal_frame contract_version")

    constraints = value.get("constraints", [])
    if not isinstance(constraints, list) or len(constraints) > 64:
        raise CapabilityRoutingError(
            "goal_frame.constraints must contain at most 64 items"
        )
    for constraint in constraints:
        if not isinstance(constraint, Mapping):
            raise CapabilityRoutingError("goal_frame constraint must be an object")
        if set(constraint) != {"key", "operator", "value"}:
            raise CapabilityRoutingError(
                "goal_frame constraint requires only key, operator, and value"
            )
        if not _SAFE_ID.fullmatch(str(constraint["key"])) or not _SAFE_ID.fullmatch(
            str(constraint["operator"])
        ):
            raise CapabilityRoutingError(
                "goal_frame constraint key and operator must be safe identifiers"
            )
    try:
        encoded_constraints = json.dumps(
            constraints,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as exc:
        raise CapabilityRoutingError(
            "goal_frame.constraints must be JSON serializable"
        ) from exc
    if len(encoded_constraints.encode("utf-8")) > 16_384:
        raise CapabilityRoutingError("goal_frame.constraints exceeds 16384 bytes")

    return {
        "contract_version": GOAL_FRAME_VERSION,
        "intent_ids": list(_semantic_ids(value.get("intent_ids", ()), "intent_ids")),
        "required_affordances": list(
            _semantic_ids(
                value.get("required_affordances", ()),
                "required_affordances",
            )
        ),
        "desired_effects": list(
            _semantic_ids(value.get("desired_effects", ()), "desired_effects")
        ),
        "trigger_events": list(
            _semantic_ids(value.get("trigger_events", ()), "trigger_events")
        ),
        "constraints": [dict(item) for item in constraints],
    }


def goal_frame_request(
    goal: str,
    manifests: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Build a provider-neutral request that maps any language to ontology IDs."""
    if not isinstance(goal, str) or not goal.strip() or len(goal) > 4000:
        raise CapabilityRoutingError("goal must be 1 to 4000 characters")
    vocabulary = {
        "intent_ids": sorted(
            {
                item
                for manifest in manifests
                for item in _string_list(manifest.get("intent_ids", ()))
            }
        ),
        "affordances": sorted(
            {
                item
                for manifest in manifests
                for item in _string_list(manifest.get("affordances", ()))
            }
        ),
        "effects": sorted(
            {
                item
                for manifest in manifests
                for item in _string_list(manifest.get("effects", ()))
            }
        ),
        "events": sorted(
            {
                item
                for manifest in manifests
                for item in _string_list(manifest.get("handled_events", ()))
            }
        ),
    }
    for field_name, items in vocabulary.items():
        _semantic_ids(items, field_name)
    return {
        "contract_version": GOAL_FRAME_REQUEST_VERSION,
        "instructions": (
            "Map the goal's meaning, regardless of language, to one "
            "flyto.goal-frame.v1 JSON object. Use only ontology IDs from vocabulary. "
            "Preserve route order and other non-linguistic values in constraints. "
            "Do not choose capability runtime names."
        ),
        "goal": goal,
        "vocabulary": vocabulary,
    }


def _semantic_query(goal_frame: Mapping[str, Any]) -> str:
    return " ".join(
        (
            *_string_list(goal_frame.get("intent_ids", ())),
            *_string_list(goal_frame.get("required_affordances", ())),
            *_string_list(goal_frame.get("desired_effects", ())),
            *_string_list(goal_frame.get("trigger_events", ())),
        )
    )


def _runtime_name(manifest: Mapping[str, Any]) -> str:
    return str(
        manifest.get("runtime_name")
        or manifest.get("name")
        or manifest.get("module_id")
        or ""
    )


def _canonical_id(manifest: Mapping[str, Any]) -> str:
    """Return the declared capability ID, never a derived or repaired one.

    A capability ID is an authority-bearing identity.  It is only ever the
    registry-declared value carried by the manifest.  When a manifest declares
    no usable ID this returns ``""`` so the caller excludes it, because a
    synthesized ID would hand execution authority to an identity that no
    registry ever published.
    """
    if "canonical_id" in manifest:
        explicit = manifest["canonical_id"]
        return explicit if isinstance(explicit, str) else ""
    # ``flyto-core`` identities are stamped by ModuleRegistry, so AI must not
    # reconstruct one from module_id, category, prefix, or plugin name.
    if manifest.get("source") == "flyto-core":
        return ""
    return f"capability.{_runtime_name(manifest)}@1"


def _manifest_source(manifest: Mapping[str, Any]) -> str:
    explicit = manifest.get("source", "")
    if isinstance(explicit, str) and explicit:
        return explicit
    canonical_id = _canonical_id(manifest)
    if canonical_id.startswith("robotics."):
        return "flyto-robotics"
    if canonical_id.startswith("core."):
        return "flyto-core"
    return "external"


def _manifest_plugin(manifest: Mapping[str, Any]) -> str:
    """Return the registry-stamped plugin identity, or ``""`` when unstamped."""
    plugin = manifest.get("plugin", "")
    return plugin if isinstance(plugin, str) else ""


def _provider_identity(manifest: Mapping[str, Any]) -> tuple[str, str, str, str]:
    """Return the full provider identity of one manifest.

    A capability ID answers "what can be done"; a provider identity answers
    "which installed module would do it".  Several modules or plugins may
    legitimately provide one capability, so only the full tuple may be used for
    duplicate rejection, tiebreaks, and planner propagation.
    """
    return (
        _canonical_id(manifest),
        _runtime_name(manifest),
        _manifest_plugin(manifest),
        _manifest_source(manifest),
    )


def _retrieval_candidate_matches_manifest(
    candidate: Mapping[str, Any], manifest: Mapping[str, Any]
) -> bool:
    """Resolve retrieval evidence only to an exact installed provider record."""
    return (
        candidate["capability_id"] == _canonical_id(manifest)
        and candidate["source_projection_digest"]
        == manifest.get("source_projection_digest")
        and candidate["upstream_content_digest"]
        == manifest.get("upstream_content_digest")
        and candidate["document_digest"] == manifest.get("document_digest")
    )


def _catalog_snapshot(manifests: Sequence[Mapping[str, Any]]) -> str:
    payload = json.dumps(
        [dict(item) for item in manifests],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _context_set(
    context: Mapping[str, object],
    name: str,
) -> frozenset[str] | None:
    if name not in context:
        return None
    raw = context[name]
    if not isinstance(raw, (list, tuple, set, frozenset)):
        raise CapabilityRoutingError(f"context.{name} must be an array")
    parsed = frozenset(str(item) for item in raw)
    if any(not _SAFE_ID.fullmatch(item) for item in parsed):
        raise CapabilityRoutingError(f"context.{name} contains an unsafe identifier")
    return parsed


def _hard_filter(
    manifest: Mapping[str, Any],
    context: Mapping[str, object],
) -> tuple[str, ...]:
    failures: list[str] = []
    runtime_name = _runtime_name(manifest)
    source = _manifest_source(manifest)
    domain = str(manifest.get("domain") or manifest.get("category") or "")
    robot_model = str(context.get("robot_model", ""))
    compatible_robots = _string_list(manifest.get("compatible_robots", ("*",)))

    allowed_sources = _context_set(context, "allowed_sources")
    allowed_domains = _context_set(context, "allowed_domains")
    enabled = _context_set(context, "enabled_capabilities")
    if allowed_sources is not None and source not in allowed_sources:
        failures.append("source_out_of_scope")
    if allowed_domains is not None and domain not in allowed_domains:
        failures.append("domain_out_of_scope")
    if enabled is not None and runtime_name not in enabled:
        failures.append("not_enabled")
    if (
        robot_model
        and "*" not in compatible_robots
        and robot_model not in compatible_robots
    ):
        failures.append("robot_incompatible")

    checks = (
        (
            _context_set(context, "available_observations"),
            _string_list(manifest.get("required_observations", ())),
            "missing_observation",
        ),
        (
            _context_set(context, "available_resources"),
            _string_list(manifest.get("required_resources", ())),
            "missing_resource",
        ),
        (
            _context_set(context, "granted_permissions"),
            _string_list(manifest.get("required_permissions", ())),
            "permission_denied",
        ),
    )
    for available, required, reason in checks:
        if available is not None and not set(required).issubset(available):
            failures.append(reason)
    return tuple(failures)


def _trusted_blueprint_module_hints(
    blueprints: Sequence[Mapping[str, Any]],
) -> frozenset[str]:
    if not blueprints:
        return frozenset()
    from flyto_ai.intelligence.planner import blueprint_is_trusted

    hints: set[str] = set()
    for candidate in blueprints:
        materialized = dict(candidate)
        if not blueprint_is_trusted(materialized):
            continue
        hints.update(_string_list(materialized.get("module_ids", ())))
        step_groups = (
            materialized.get("steps"),
            materialized.get("workflow", {}).get("steps")
            if isinstance(materialized.get("workflow"), dict)
            else None,
        )
        for steps in step_groups:
            if not isinstance(steps, list):
                continue
            for step in steps:
                if not isinstance(step, dict):
                    continue
                module_id = step.get("module") or step.get("capability")
                if module_id:
                    hints.add(str(module_id))
        # Blueprint search is relevance ordered. One trusted procedure is
        # evidence; merging every match recreates the large-catalog drift this
        # router is designed to prevent.
        break
    return frozenset(hints)


def _score(
    goal: str,
    manifest: Mapping[str, Any],
    blueprint_hints: frozenset[str],
    goal_frame: Mapping[str, Any] | None,
    retrieval_score: float | None = None,
) -> tuple[float, tuple[str, ...]]:
    runtime_name = _runtime_name(manifest)
    canonical_id = _canonical_id(manifest)
    score = 0.0
    reasons: list[str] = []

    if goal_frame is not None:
        semantic_pairs = (
            ("intent_ids", "intent_ids", 12.0, "intent_match"),
            (
                "required_affordances",
                "affordances",
                16.0,
                "affordance_match",
            ),
            ("desired_effects", "effects", 8.0, "effect_match"),
            ("trigger_events", "handled_events", 10.0, "event_match"),
        )
        for frame_field, manifest_field, weight, reason in semantic_pairs:
            overlap = set(_string_list(goal_frame.get(frame_field, ()))) & set(
                _string_list(manifest.get(manifest_field, ()))
            )
            if overlap:
                score += weight * len(overlap)
                reasons.append(reason)
        if not reasons:
            return 0.0, ()
    else:
        query = _normalized(goal)
        query_words = _words(goal)
        aliases = _string_list(manifest.get("aliases", ()))
        tags = _string_list(manifest.get("tags", ()))
        examples = _string_list(manifest.get("positive_examples", ()))
        negative_examples = _string_list(manifest.get("negative_examples", ()))
        description = str(manifest.get("description", ""))
        label = str(manifest.get("label", ""))
        identifiers = (runtime_name, runtime_name.replace("_", " "), canonical_id)
        if any(term and _normalized(term) in query for term in identifiers):
            score += 8.0
            reasons.append("identifier_match")
        alias_hits = [alias for alias in aliases if _normalized(alias) in query]
        if alias_hits:
            score += min(12.0, 6.0 + 2.0 * (len(alias_hits) - 1))
            reasons.append("alias_phrase_match")
        tag_hits = [tag for tag in tags if _normalized(tag) in query]
        if tag_hits:
            score += min(6.0, 2.0 * len(tag_hits))
            reasons.append("tag_match")

        candidate_words = _words(
            " ".join(
                (
                    runtime_name,
                    canonical_id,
                    label,
                    description,
                    *aliases,
                    *tags,
                    *examples,
                )
            )
        )
        overlap = query_words & candidate_words
        if overlap:
            score += min(6.0, 1.25 * len(overlap))
            reasons.append("token_overlap")
        if any(_normalized(example) in query for example in examples):
            score += 3.0
            reasons.append("positive_example_match")
        if any(_normalized(example) in query for example in negative_examples):
            score -= 8.0
            reasons.append("negative_example_match")
    if runtime_name in blueprint_hints or canonical_id in blueprint_hints:
        score += 2.5
        reasons.append("trusted_blueprint_hint")

    # Retrieval is candidate-only relevance evidence.  Its deliberately small
    # cap cannot replace a semantic match or any hard-filter decision.
    if retrieval_score is not None:
        score += min(1.0, max(0.0, retrieval_score))
        reasons.append("bounded_retrieval_hint")

    upstream_score = manifest.get("discovery_score", manifest.get("score", 0.0))
    if isinstance(upstream_score, (int, float)) and math.isfinite(
        float(upstream_score)
    ):
        score += min(2.0, max(0.0, float(upstream_score)) / 10.0)
        if upstream_score:
            reasons.append("runtime_discovery_score")
    return max(0.0, score), tuple(reasons)


def route_capabilities(
    goal: str,
    manifests: Sequence[Mapping[str, Any]],
    *,
    goal_frame: Mapping[str, Any] | None = None,
    context: Mapping[str, object] | None = None,
    limit: int = 8,
    blueprint_candidates: Sequence[Mapping[str, Any]] = (),
    retrieval_handoff: Mapping[str, Any] | None = None,
    retrieval_authority: CapabilityRetrievalAuthority | None = None,
) -> dict[str, Any]:
    """Return a deterministic, bounded shortlist from versioned JSON manifests."""
    if not isinstance(goal, str) or not goal.strip() or len(goal) > 4000:
        raise CapabilityRoutingError("goal must be 1 to 4000 characters")
    if not 1 <= limit <= CAPABILITY_GROUP_LIMIT:
        raise CapabilityRoutingError(
            f"limit must be between 1 and {CAPABILITY_GROUP_LIMIT} capability groups"
        )
    if len(manifests) > 10_000:
        raise CapabilityRoutingError("catalog exceeds the 10000 capability limit")

    uses_retrieval = retrieval_handoff is not None or retrieval_authority is not None
    if uses_retrieval:
        # AI-local digest inputs are an exact bounded JSON boundary too.  Take
        # the snapshot before Goal Frame normalization or context use so hostile
        # containers and recursive/oversized values cannot escape as raw Python
        # exceptions or be returned by the route.
        bounded_context = _retrieval_json({} if context is None else context)
        bounded_goal_frame = None if goal_frame is None else _retrieval_json(goal_frame)
        if type(bounded_context) is not dict:
            _retrieval_fail()
        try:
            active_goal_frame = (
                normalize_goal_frame(bounded_goal_frame)
                if bounded_goal_frame is not None
                else None
            )
        except CapabilityRoutingError:
            _retrieval_fail()
        active_context: dict[str, object] = bounded_context
    else:
        active_goal_frame = (
            normalize_goal_frame(goal_frame) if goal_frame is not None else None
        )
        active_context = dict(context or {})
    blueprint_hints = _trusted_blueprint_module_hints(blueprint_candidates)
    retrieval: dict[str, Any] | None = None
    retrieval_scores: dict[tuple[str, str, str, str], float] = {}
    if uses_retrieval:
        if retrieval_handoff is None or retrieval_authority is None:
            _retrieval_fail()
        retrieval = validate_capability_retrieval(
            retrieval_handoff, retrieval_authority
        )
        if (
            retrieval["goal_digest"]
            != _sha({"digest_version": "flyto.ai.goal-digest.v1", "goal": goal})
            or retrieval["routing_context_digest"]
            != _sha(
                {
                    "digest_version": "flyto.ai.routing-context-digest.v1",
                    "routing_context": active_context,
                }
            )
            or retrieval["goal_frame_digest"]
            != _sha(
                {
                    "digest_version": "flyto.ai.goal-frame-digest.v1",
                    "goal_frame": active_goal_frame,
                }
            )
        ):
            _retrieval_fail()
        try:
            retrieval_scores = {
                _provider_identity(manifest): item["score"] / 1_000_000_000
                for item in retrieval["result"]["page"]["candidates"]
                for manifest in manifests
                if isinstance(manifest, Mapping)
                and _retrieval_candidate_matches_manifest(item, manifest)
            }
        except Exception:
            _retrieval_fail()
    valid: list[Mapping[str, Any]] = []
    excluded: list[dict[str, object]] = []
    seen_providers: set[tuple[str, str, str, str]] = set()
    for manifest in manifests:
        if not isinstance(manifest, Mapping):
            excluded.append({"runtime_name": "", "reasons": ["invalid_manifest"]})
            continue
        runtime_name = _runtime_name(manifest)
        canonical_id = _canonical_id(manifest)
        plugin = _manifest_plugin(manifest)
        identity = _provider_identity(manifest)
        if (
            not _SAFE_ID.fullmatch(runtime_name)
            or not _RETRIEVAL_ID.fullmatch(canonical_id)
            or len(canonical_id) > 192
            or (plugin and not _SAFE_ID.fullmatch(plugin))
            # Two modules may legitimately provide one capability, so only an
            # identical provider identity is a duplicate.
            or identity in seen_providers
        ):
            excluded.append(
                {
                    "runtime_name": runtime_name,
                    "reasons": ["invalid_or_duplicate_identity"],
                }
            )
            continue
        seen_providers.add(identity)
        failures = _hard_filter(manifest, active_context)
        if failures:
            excluded.append({"runtime_name": runtime_name, "reasons": list(failures)})
            continue
        if retrieval is not None and identity not in retrieval_scores:
            # Keep installed safety and human-gate controls available around a
            # retrieved task shortlist; they still need semantic selection.
            if not retrieval_scores or str(manifest.get("control_class", "")) not in {
                "safety",
                "human_gate",
            }:
                continue
        valid.append(manifest)

    if retrieval is not None:
        for candidate in retrieval["result"]["page"]["candidates"]:
            matches = [
                manifest
                for manifest in manifests
                if isinstance(manifest, Mapping)
                and _retrieval_candidate_matches_manifest(candidate, manifest)
            ]
            identities = [_provider_identity(manifest) for manifest in matches]
            if not matches or len(identities) != len(set(identities)):
                _retrieval_fail()
            if any(_hard_filter(manifest, active_context) for manifest in matches):
                _retrieval_fail()

    ranked: list[tuple[float, str, Mapping[str, Any], tuple[str, ...]]] = []
    for manifest in valid:
        score, reasons = _score(
            goal,
            manifest,
            blueprint_hints,
            active_goal_frame,
            retrieval_scores.get(_provider_identity(manifest)),
        )
        ranked.append((score, _canonical_id(manifest), manifest, reasons))
    # Full provider identity is the tiebreak so co-providers of one capability
    # keep a stable, auditable order instead of an arbitrary catalog order.
    ranked.sort(key=lambda item: (-item[0], item[1], *_provider_identity(item[2])[1:]))
    selection_pool = (
        [item for item in ranked if item[0] > 0.0]
        if active_goal_frame is not None
        else ranked
    )
    # ``limit`` bounds canonical capability groups, not provider rows.  A
    # selected capability always expands to every exact installed co-provider;
    # the independent row ceiling prevents a large provider group from turning
    # a bounded route into unbounded planner input.
    groups: dict[str, list[tuple[float, str, Mapping[str, Any], tuple[str, ...]]]] = {}
    for item in ranked:
        canonical_id = item[1]
        if canonical_id not in groups:
            groups[canonical_id] = []
        groups[canonical_id].append(item)
    group_order: list[str] = []
    for item in selection_pool:
        if item[1] not in group_order:
            group_order.append(item[1])
    selected_group_ids = group_order[:limit]

    def expanded_selection() -> list[
        tuple[float, str, Mapping[str, Any], tuple[str, ...]]
    ]:
        expanded = [item for group_id in selected_group_ids for item in groups[group_id]]
        if len(expanded) > EMITTED_PROVIDER_ROW_LIMIT:
            raise CapabilityRoutingError(
                "selected capability provider groups exceed the "
                f"{EMITTED_PROVIDER_ROW_LIMIT} emitted provider row limit"
            )
        return expanded

    selected = expanded_selection()

    # Confidence and ambiguity describe capability choices, not the number of
    # installed providers for one choice.  Collapse non-control provider rows
    # to their canonical group and use the group's highest provider score.
    relevant_group_scores: dict[str, float] = {}
    for score, canonical_id, manifest, _reasons in ranked:
        if str(manifest.get("control_class", "")) in {"safety", "human_gate"}:
            continue
        relevant_group_scores[canonical_id] = max(
            score, relevant_group_scores.get(canonical_id, score)
        )
    relevance_scores = sorted(relevant_group_scores.values(), reverse=True)
    top_relevant = (
        relevance_scores[0]
        if relevance_scores
        else (ranked[0][0] if ranked else 0.0)
    )
    second_relevant = relevance_scores[1] if len(relevance_scores) > 1 else 0.0

    required_names: list[str] = []
    if any(
        score > 0 and str(manifest.get("control_class", "")) == "motion"
        for score, _canonical_id_value, manifest, _reasons in selected
    ):
        required_names.append("safe_stop")
    for required_name in required_names:
        if any(_runtime_name(item[2]) == required_name for item in selected):
            continue
        required = next(
            (item for item in ranked if _runtime_name(item[2]) == required_name),
            None,
        )
        if required is None:
            continue
        required_group = required[1]
        if required_group not in selected_group_ids:
            if len(selected_group_ids) >= limit:
                selected_group_ids = selected_group_ids[:-1]
            selected_group_ids.append(required_group)
        selected = expanded_selection()

    semantic_coverage = _semantic_coverage(active_goal_frame, selected)
    if active_goal_frame is not None:
        needs_clarification = not selected or bool(semantic_coverage["missing"])
    else:
        needs_clarification = not selected or top_relevant < 2.0
        if top_relevant >= 2.0 and second_relevant >= 2.0:
            needs_clarification = top_relevant - second_relevant < 0.35
    if needs_clarification and limit >= 2:
        for required_name in ("ask_human", "resume"):
            if any(_runtime_name(item[2]) == required_name for item in selected):
                continue
            required = next(
                (item for item in ranked if _runtime_name(item[2]) == required_name),
                None,
            )
            if required is None:
                continue
            required_group = required[1]
            if required_group not in selected_group_ids:
                if len(selected_group_ids) >= limit:
                    selected_group_ids = selected_group_ids[:-1]
                selected_group_ids.append(required_group)
            selected = expanded_selection()
    semantic_coverage = _semantic_coverage(active_goal_frame, selected)
    if active_goal_frame is not None and semantic_coverage["missing"]:
        needs_clarification = True

    candidates = [
        {
            "canonical_id": canonical_id,
            "runtime_name": _runtime_name(manifest),
            "plugin": _manifest_plugin(manifest),
            "source": _manifest_source(manifest),
            "score": round(score, 4),
            "reasons": list(reasons or ("deterministic_tiebreak",)),
            "selected_by": (
                "deterministic_semantic_frame_v1"
                if active_goal_frame is not None
                else "deterministic_hybrid_v1"
            ),
        }
        for score, canonical_id, manifest, reasons in selected
    ]
    result = {
        "contract_version": CAPABILITY_ROUTE_VERSION,
        "registry_snapshot": _catalog_snapshot(manifests),
        "selection_method": (
            "hard_filter_then_semantic_frame_rank_v1"
            if active_goal_frame is not None
            else "hard_filter_then_deterministic_hybrid_rank_v1"
        ),
        "confidence": round(
            semantic_coverage["ratio"]
            if active_goal_frame is not None
            else min(1.0, max(0.0, top_relevant / 10.0)),
            4,
        ),
        "needs_clarification": needs_clarification,
        "candidates": candidates,
        "excluded_count": len(excluded),
        "excluded": excluded,
        "routing_context": active_context,
        "goal_frame": active_goal_frame,
        "semantic_coverage": semantic_coverage,
    }
    if retrieval is not None:
        result["retrieval_evidence"] = {
            "contract_version": CAPABILITY_RETRIEVAL_EVIDENCE_VERSION,
            "request_digest": retrieval["request"]["request_digest"],
            "index_digest": retrieval["request"]["index_digest"],
            "snapshot_digest": retrieval["request"]["snapshot_digest"],
            "query_context_digest": retrieval["result"]["query_context_digest"],
            "requirements_digest": retrieval["result"]["feasibility"][
                "requirements_digest"
            ],
            "result_digest": retrieval["result"]["result_digest"],
            "goal_digest": retrieval["goal_digest"],
            "routing_context_digest": retrieval["routing_context_digest"],
            "goal_frame_digest": retrieval["goal_frame_digest"],
            "handoff_digest": retrieval["handoff_digest"],
            "candidate_count": len(retrieval["result"]["page"]["candidates"]),
            "candidate_only": True,
            "execution_authority": False,
            "planning_required": True,
            "permission_required": True,
            "execution_closure_required": True,
        }
    return result


def _semantic_coverage(
    goal_frame: Mapping[str, Any] | None,
    selected: Sequence[tuple[float, str, Mapping[str, Any], tuple[str, ...]]],
) -> dict[str, Any]:
    if goal_frame is None:
        return {"required": [], "matched": [], "missing": [], "ratio": 0.0}
    requirements = {
        *(f"intent:{item}" for item in _string_list(goal_frame.get("intent_ids", ()))),
        *(
            f"affordance:{item}"
            for item in _string_list(goal_frame.get("required_affordances", ()))
        ),
        *(
            f"effect:{item}"
            for item in _string_list(goal_frame.get("desired_effects", ()))
        ),
        *(
            f"event:{item}"
            for item in _string_list(goal_frame.get("trigger_events", ()))
        ),
    }
    provided: set[str] = set()
    for _score_value, _canonical_id_value, manifest, _reasons in selected:
        provided.update(
            f"intent:{item}" for item in _string_list(manifest.get("intent_ids", ()))
        )
        provided.update(
            f"affordance:{item}"
            for item in _string_list(manifest.get("affordances", ()))
        )
        provided.update(
            f"effect:{item}" for item in _string_list(manifest.get("effects", ()))
        )
        provided.update(
            f"event:{item}" for item in _string_list(manifest.get("handled_events", ()))
        )
    matched = requirements & provided
    missing = requirements - provided
    ratio = len(matched) / len(requirements) if requirements else 1.0
    return {
        "required": sorted(requirements),
        "matched": sorted(matched),
        "missing": sorted(missing),
        "ratio": round(ratio, 4),
    }


def _core_text(item: Mapping[str, Any], field: str, default: str = "") -> str:
    value = item.get(field, default)
    return value if isinstance(value, str) else default


def _core_capability_providers(result: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Project Core search results into registry-declared capability providers.

    Only a result that carries a module identity *and* a non-empty, safe
    registry-declared ``provides_capability`` becomes a manifest, and its
    ``canonical_id`` is exactly that declared value.  An ordinary Core module
    reports ``provides_capability=""``; that is a legitimate non-capability
    search hit, so it is skipped rather than turned into an invented
    capability.  Unsafe declared identities are excluded deterministically and
    are never stripped or repaired here.
    """
    manifests: list[dict[str, Any]] = []
    raw_results = result.get("results", [])
    if not isinstance(raw_results, list):
        return manifests
    for item in raw_results:
        if not isinstance(item, Mapping):
            continue
        module_id = item.get("module_id", "")
        if not isinstance(module_id, str) or not _SAFE_ID.fullmatch(module_id):
            continue
        capability_id = item.get("provides_capability", "")
        if not isinstance(capability_id, str) or not capability_id:
            # An ordinary module, not a capability provider.
            continue
        if not _SAFE_ID.fullmatch(capability_id):
            continue
        plugin = item.get("plugin", "")
        if not isinstance(plugin, str) or (plugin and not _SAFE_ID.fullmatch(plugin)):
            continue
        category = _core_text(item, "category")
        manifests.append(
            {
                "manifest_contract": "flyto.capability-manifest.v1",
                "canonical_id": capability_id,
                "provides_capability": capability_id,
                "plugin": plugin,
                "runtime_name": module_id,
                "name": module_id,
                "version": "runtime",
                "source": "flyto-core",
                "domain": category or "core",
                "description": _core_text(item, "description"),
                "label": _core_text(item, "label"),
                "aliases": [],
                "tags": [category],
                "control_class": "module",
                "required_observations": [],
                "required_resources": [],
                "required_permissions": [],
                "compatible_robots": [],
                "positive_examples": [],
                "negative_examples": [],
                "intent_ids": list(item.get("semantics", {}).get("intent_ids", [])),
                "affordances": list(item.get("semantics", {}).get("affordances", [])),
                "effects": list(item.get("semantics", {}).get("effects", [])),
                "handled_events": list(item.get("semantics", {}).get("handled_events", [])),
                "discovery_score": item.get("score", 0.0),
            }
        )
    return manifests


def _resolve_blueprint_search(
    blueprint_search: BlueprintSearch | None,
) -> tuple[BlueprintSearch, str | None]:
    """Resolve the read-only Blueprint discovery callable.

    Returns the callable plus a bounded reason code when the Blueprint package
    or engine is absent.  Absence is a lane outcome, not an error string.
    """
    if blueprint_search is not None:
        return blueprint_search, None
    try:
        from flyto_blueprint import get_engine

        engine_search = get_engine().search
    except Exception:
        return _empty_blueprint_search, "engine_unavailable"
    if not callable(engine_search):
        return _empty_blueprint_search, "engine_unavailable"
    return engine_search, None


def _blueprint_discovery(
    blueprint_search: BlueprintSearch | None,
    query: str,
) -> tuple[list[Mapping[str, Any]], str, str]:
    """Run read-only Blueprint discovery and classify the outcome."""
    search, absent_reason = _resolve_blueprint_search(blueprint_search)
    if absent_reason is not None:
        return [], DISCOVERY_UNAVAILABLE, absent_reason
    try:
        raw = search(query)
    except Exception:
        # Never surface raw exception text; the status is the contract.
        return [], DISCOVERY_FAILED, "search_failed"
    if isinstance(raw, (str, bytes, bytearray, Mapping)):
        return [], DISCOVERY_FAILED, "invalid_result"
    try:
        # Read at most one item past the ceiling so an unbounded or streaming
        # bridge cannot force this router to materialize an unbounded list.
        candidates = list(islice(raw, BLUEPRINT_CANDIDATE_LIMIT + 1))
    except TypeError:
        return [], DISCOVERY_FAILED, "invalid_result"
    if len(candidates) > BLUEPRINT_CANDIDATE_LIMIT:
        return [], DISCOVERY_FAILED, "result_over_bound"
    if any(not isinstance(item, Mapping) for item in candidates):
        return [], DISCOVERY_FAILED, "invalid_result"
    if not candidates:
        return [], DISCOVERY_NOT_APPLICABLE, "search_empty"
    return candidates, DISCOVERY_APPLIED, "search_matched"


def _core_runtime_issue(
    core_runtime_manifest: Mapping[str, Any],
) -> tuple[str, str] | None:
    """Return a bounded (status, reason) when Core runtime evidence is untrusted."""
    if not isinstance(core_runtime_manifest, Mapping):
        return DISCOVERY_UNAVAILABLE, "runtime_missing"
    ok = core_runtime_manifest.get("ok", _MISSING)
    if ok is not _MISSING and not isinstance(ok, bool):
        return DISCOVERY_FAILED, "runtime_malformed"
    if ok is False:
        return DISCOVERY_UNAVAILABLE, "runtime_unavailable"
    if ok is _MISSING:
        return DISCOVERY_UNAVAILABLE, "runtime_missing"
    contract_version = core_runtime_manifest.get("contract_version", "")
    if not isinstance(contract_version, str):
        return DISCOVERY_FAILED, "runtime_malformed"
    if contract_version != CORE_RUNTIME_CONTRACT:
        return DISCOVERY_UNAVAILABLE, "runtime_contract_mismatch"
    return None


def _core_results_malformed(core_result: Mapping[str, Any]) -> bool:
    """Report whether the Core search response violates its bounded shape."""
    semantic_fields = ("intent_ids", "affordances", "effects", "handled_events")
    def valid_semantics(value: object) -> bool:
        if type(value) is not dict or set(value) != set(semantic_fields):
            return False
        fields = [value[name] for name in semantic_fields]
        return all(
            type(field) is list and 0 < len(field) <= 16
            and all(type(item) is str and len(item) <= 96 and re.fullmatch(r"[a-z][a-z0-9]*(?:[._-][a-z0-9]+)*", item) for item in field) and len(field) == len(set(field))
            for field in fields
        ) and sum(map(len, fields)) <= 48

    def valid_item(item: object) -> bool:
        if type(item) is not dict or not {"module_id", "provides_capability", "plugin"} <= item.keys():
            return False
        identities = tuple(item.get(name, "") for name in ("module_id", "provides_capability", "plugin"))
        semantics = item.get("semantics", _MISSING)
        return bool(identities[0]) and all(
            type(value) is str and len(value) <= 96 and (not value or _SAFE_ID.fullmatch(value)) for value in identities
        ) and (semantics is _MISSING or (valid_semantics(semantics) if identities[1] else semantics == {}))
    if type(core_result) is not dict:
        return True
    ok, raw_results = core_result.get("ok", _MISSING), core_result.get("results", _MISSING)
    if type(raw_results) is not list:
        return True
    providers = [tuple(item.get(name, "") for name in ("module_id", "provides_capability", "plugin")) for item in raw_results if type(item) is dict and item.get("provides_capability", "")]
    return (ok is not _MISSING and (type(ok) is not bool or ok is False)) or not all(valid_item(item) for item in raw_results) or len(providers) != len(set(providers))


def _core_discovery_status(
    core_result: Mapping[str, Any],
    core_runtime_manifest: Mapping[str, Any],
    core_candidate_count: int,
) -> tuple[str, str]:
    """Classify Core discovery from the bridge responses alone."""
    # Runtime evidence takes precedence over the bounded search result.
    runtime_issue = _core_runtime_issue(core_runtime_manifest)
    if runtime_issue:
        return runtime_issue
    if _core_results_malformed(core_result):
        return DISCOVERY_FAILED, "search_malformed"
    if core_candidate_count:
        return DISCOVERY_APPLIED, "discovery_matched"
    raw_results = core_result.get("results", [])
    if isinstance(raw_results, list) and raw_results:
        # Well-formed hits without capabilities are ordinary Core modules.
        return DISCOVERY_NOT_APPLICABLE, "no_capability_providers"
    return DISCOVERY_NOT_APPLICABLE, "discovery_empty"


async def route_with_flyto(
    goal: str,
    manifests: Sequence[Mapping[str, Any]],
    *,
    goal_frame: Mapping[str, Any] | None = None,
    context: Mapping[str, object] | None = None,
    limit: int = 8,
    core_limit: int = 12,
    core_dispatch: CoreDispatch | None = None,
    blueprint_search: BlueprintSearch | None = None,
) -> dict[str, Any]:
    """Route with trusted Blueprint hints and Core discovery through existing bridges."""
    decision, _catalog = await _route_with_flyto_catalog(
        goal,
        manifests,
        goal_frame=goal_frame,
        context=context,
        limit=limit,
        core_limit=core_limit,
        core_dispatch=core_dispatch,
        blueprint_search=blueprint_search,
    )
    return decision


async def _route_with_flyto_catalog(
    goal: str,
    manifests: Sequence[Mapping[str, Any]],
    *,
    goal_frame: Mapping[str, Any] | None = None,
    context: Mapping[str, object] | None = None,
    limit: int = 8,
    core_limit: int = 12,
    core_dispatch: CoreDispatch | None = None,
    blueprint_search: BlueprintSearch | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Route and also return the exact catalog the route was decided over."""
    if not 1 <= core_limit <= 100:
        raise CapabilityRoutingError("core_limit must be between 1 and 100")
    active_goal_frame = (
        normalize_goal_frame(goal_frame) if goal_frame is not None else None
    )
    discovery_query = (
        _semantic_query(active_goal_frame) if active_goal_frame is not None else goal
    )
    active_context: dict[str, object] = dict(context or {})

    (
        blueprint_candidates,
        blueprint_status,
        blueprint_status_reason,
    ) = _blueprint_discovery(blueprint_search, discovery_query)

    if core_dispatch is None:
        try:
            from flyto_ai.tools.core_tools import dispatch_core_tool
        except Exception as exc:
            raise CapabilityRoutingError(
                "flyto-core discovery bridge is unavailable"
            ) from exc
        core_dispatch = dispatch_core_tool
    try:
        core_search = core_dispatch(
            "search_modules",
            {"query": discovery_query, "limit": core_limit},
        )
        core_manifest_call = core_dispatch(
            "get_core_capability_manifest",
            {"include_tools": False, "include_categories": True},
        )
        core_result, core_runtime_manifest = await _gather_calls(
            core_search,
            core_manifest_call,
        )
    except CapabilityRoutingError:
        raise
    except Exception as exc:
        # Bounded, typed failure; the underlying text stays out of the contract.
        raise CapabilityRoutingError("flyto-core discovery bridge call failed") from exc
    combined = [dict(item) for item in manifests]
    def invalid_core_result(
        result: Mapping[str, Any], result_limit: int, require_semantics: bool
    ) -> bool:
        if _core_results_malformed(result):
            return True
        results, total = result["results"], result.get("total")
        return type(total) is not int or total < len(results) or len(results) > result_limit or require_semantics and any(
            item.get("provides_capability", "") and "semantics" not in item for item in results
        )
    core_results_malformed = invalid_core_result(
        core_result, core_limit, active_goal_frame is not None
    )
    core_candidates = [] if core_results_malformed else _core_capability_providers(core_result)
    combined.extend(core_candidates)
    if "allowed_sources" not in active_context:
        sources = {_manifest_source(item) for item in manifests}
        if core_candidates:
            sources.add("flyto-core")
        active_context["allowed_sources"] = sorted(sources or {"external"})
    core_status, core_status_reason = (DISCOVERY_FAILED, "search_malformed") if core_results_malformed else _core_discovery_status(
        core_result, core_runtime_manifest, len(core_candidates)
    )
    route = route_capabilities(
        goal,
        combined,
        goal_frame=active_goal_frame,
        context=active_context,
        limit=limit,
        blueprint_candidates=blueprint_candidates,
    )
    trusted_hints = _trusted_blueprint_module_hints(blueprint_candidates)
    decision = {
        "contract_version": ROUTING_DECISION_VERSION,
        "route": route,
        "discovery_evidence": {
            "blueprint": {
                "status": blueprint_status,
                "status_reason": blueprint_status_reason,
                "query_mode": (
                    "semantic_frame" if active_goal_frame is not None else "raw_goal"
                ),
                "candidate_count": len(blueprint_candidates),
                "trusted_module_hints": sorted(trusted_hints),
            },
            "core": {
                "status": core_status,
                "status_reason": core_status_reason,
                "query_mode": (
                    "semantic_frame" if active_goal_frame is not None else "raw_goal"
                ),
                "contract_version": str(
                    core_runtime_manifest.get("contract_version", "")
                ),
                "core_version": str(core_runtime_manifest.get("core_version", "")),
                "fingerprint": str(
                    core_runtime_manifest.get("tool_fingerprint", "")
                    or core_runtime_manifest.get("fingerprint", "")
                ),
                "candidate_count": len(core_candidates),
                "used_bridge": "flyto_ai.tools.core_tools.dispatch_core_tool",
            },
        },
    }
    return decision, combined


def _selected_manifests(
    candidates: Sequence[Mapping[str, Any]],
    catalog: Sequence[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    """Resolve every selected candidate back to exactly one catalog manifest.

    The catalog is the exact combined view the route was decided over — the
    supplied manifests plus the verified Core-discovered providers.  Matching on
    the full provider identity keeps two providers of one capability distinct
    and prevents an unselected or unregistered module from inheriting execution
    authority via a shared runtime name or a shared capability ID.

    Resolution fails closed.  A selected candidate that matches no manifest, or
    that matches more than one, means the route and the catalog disagree about
    who would execute it.  Dropping such a candidate would hand the planner a
    silently narrower shortlist, and picking one of several same-identity
    manifests would grant authority by catalog order, so both raise a bounded
    ``CapabilityRoutingError`` that carries counts only.
    """
    by_identity: dict[tuple[str, str, str, str], list[Mapping[str, Any]]] = {}
    for manifest in catalog:
        by_identity.setdefault(_provider_identity(manifest), []).append(manifest)
    selected: list[Mapping[str, Any]] = []
    unresolved = 0
    ambiguous = 0
    for candidate in candidates:
        key = (
            str(candidate.get("canonical_id", "")),
            str(candidate.get("runtime_name", "")),
            str(candidate.get("plugin", "")),
            str(candidate.get("source", "")),
        )
        matches = by_identity.get(key, ())
        if len(matches) == 1:
            selected.append(matches[0])
        elif not matches:
            unresolved += 1
        else:
            ambiguous += 1
    if unresolved or ambiguous:
        # Raised before any provider could run; identities stay out of the
        # message so no catalog-controlled text crosses this boundary.
        raise CapabilityRoutingError(
            "selected route candidates must resolve to exactly one catalog "
            f"manifest by provider identity (unresolved={unresolved}, "
            f"ambiguous={ambiguous})"
        )
    return selected


async def prepare_planner_request(
    request: Mapping[str, Any],
    *,
    context: Mapping[str, object] | None = None,
    limit: int = 8,
    require_goal_frame: bool = False,
    require_discovery: bool = False,
    core_dispatch: CoreDispatch | None = None,
    blueprint_search: BlueprintSearch | None = None,
) -> dict[str, Any]:
    """Apply Flyto2 routing to a Robotics planner request before provider dispatch."""
    if request.get("planner_contract") != "flyto.robotics.planner-request.v1":
        raise CapabilityRoutingError("unsupported planner_contract")
    goal = request.get("goal")
    goal_frame = request.get("goal_frame")
    manifests = request.get("capabilities")
    if not isinstance(goal, str):
        raise CapabilityRoutingError("planner request goal must be a string")
    if not isinstance(manifests, list) or not all(
        isinstance(item, dict) for item in manifests
    ):
        raise CapabilityRoutingError("planner request capabilities must be an array")
    if goal_frame is not None and not isinstance(goal_frame, Mapping):
        raise CapabilityRoutingError("planner request goal_frame must be an object")
    if require_goal_frame and goal_frame is None:
        raise CapabilityRoutingError(
            "planner request requires flyto.goal-frame.v1 for language-neutral routing"
        )

    decision, catalog = await _route_with_flyto_catalog(
        goal,
        manifests,
        goal_frame=goal_frame,
        context=context,
        limit=limit,
        core_dispatch=core_dispatch,
        blueprint_search=blueprint_search,
    )
    if require_discovery:
        evidence = decision["discovery_evidence"]
        blueprint_status = str(evidence["blueprint"]["status"])
        core_status = str(evidence["core"]["status"])
        if (
            blueprint_status not in _DISCOVERY_READY
            or core_status not in _DISCOVERY_READY
        ):
            # Raised before any provider could run; message stays bounded.
            raise CapabilityRoutingError(
                "planner request requires resolved Flyto discovery "
                f"(blueprint={blueprint_status}, core={core_status})"
            )

    route = decision["route"]
    prepared = dict(request)
    prepared["capabilities"] = _selected_manifests(route["candidates"], catalog)
    prepared["capability_route"] = route
    prepared["flyto_routing_decision"] = decision
    return prepared


async def _gather_calls(
    first: Awaitable[dict[str, Any]],
    second: Awaitable[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Await bridge calls without requiring a specific async framework."""
    if not inspect.isawaitable(first) or not inspect.isawaitable(second):
        raise CapabilityRoutingError("core_dispatch must return awaitable results")
    first_result, second_result = await asyncio.gather(first, second)
    return (
        first_result if isinstance(first_result, dict) else {},
        second_result if isinstance(second_result, dict) else {},
    )


__all__ = [
    "BLUEPRINT_CANDIDATE_LIMIT",
    "CAPABILITY_GROUP_LIMIT",
    "CAPABILITY_ROUTE_VERSION",
    "CAPABILITY_RETRIEVAL_VERSION",
    "CAPABILITY_RETRIEVAL_EVIDENCE_VERSION",
    "CORE_RUNTIME_CONTRACT",
    "DISCOVERY_STATUSES",
    "EMITTED_PROVIDER_ROW_LIMIT",
    "GOAL_FRAME_REQUEST_VERSION",
    "GOAL_FRAME_VERSION",
    "ROUTING_DECISION_VERSION",
    "CapabilityRoutingError",
    "CapabilityRetrievalAuthority",
    "capability_routing_bounds",
    "goal_frame_request",
    "normalize_goal_frame",
    "prepare_planner_request",
    "route_capabilities",
    "validate_capability_retrieval",
    "route_with_flyto",
]
