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
from itertools import islice
from typing import Any

CAPABILITY_ROUTE_VERSION = "flyto.capability-route.v1"
ROUTING_DECISION_VERSION = "flyto.capability-routing-decision.v1"
GOAL_FRAME_VERSION = "flyto.goal-frame.v1"
GOAL_FRAME_REQUEST_VERSION = "flyto.goal-frame-request.v1"
_WORD = re.compile(r"[^\W_]+", re.UNICODE)
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:@-]{0,191}$")

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
_MISSING = object()

CoreDispatch = Callable[[str, dict[str, Any]], Awaitable[dict[str, Any]]]
BlueprintSearch = Callable[[str], Sequence[Mapping[str, Any]]]


class CapabilityRoutingError(ValueError):
    """Raised when a capability catalog or routing policy is unsafe."""


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
        raise CapabilityRoutingError("goal_frame.constraints must contain at most 64 items")
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

    upstream_score = manifest.get("discovery_score", manifest.get("score", 0.0))
    if isinstance(upstream_score, (int, float)) and math.isfinite(float(upstream_score)):
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
) -> dict[str, Any]:
    """Return a deterministic, bounded shortlist from versioned JSON manifests."""
    if not isinstance(goal, str) or not goal.strip() or len(goal) > 4000:
        raise CapabilityRoutingError("goal must be 1 to 4000 characters")
    if not 1 <= limit <= 32:
        raise CapabilityRoutingError("limit must be between 1 and 32")
    if len(manifests) > 10_000:
        raise CapabilityRoutingError("catalog exceeds the 10000 capability limit")

    active_goal_frame = (
        normalize_goal_frame(goal_frame) if goal_frame is not None else None
    )
    active_context: dict[str, object] = dict(context or {})
    blueprint_hints = _trusted_blueprint_module_hints(blueprint_candidates)
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
            or not _SAFE_ID.fullmatch(canonical_id)
            or (plugin and not _SAFE_ID.fullmatch(plugin))
            # Two modules may legitimately provide one capability, so only an
            # identical provider identity is a duplicate.
            or identity in seen_providers
        ):
            excluded.append(
                {"runtime_name": runtime_name, "reasons": ["invalid_or_duplicate_identity"]}
            )
            continue
        seen_providers.add(identity)
        failures = _hard_filter(manifest, active_context)
        if failures:
            excluded.append({"runtime_name": runtime_name, "reasons": list(failures)})
            continue
        valid.append(manifest)

    ranked: list[tuple[float, str, Mapping[str, Any], tuple[str, ...]]] = []
    for manifest in valid:
        score, reasons = _score(goal, manifest, blueprint_hints, active_goal_frame)
        ranked.append((score, _canonical_id(manifest), manifest, reasons))
    # Full provider identity is the tiebreak so co-providers of one capability
    # keep a stable, auditable order instead of an arbitrary catalog order.
    ranked.sort(key=lambda item: (-item[0], item[1], *_provider_identity(item[2])[1:]))
    selection_pool = (
        [item for item in ranked if item[0] > 0.0]
        if active_goal_frame is not None
        else ranked
    )
    selected = selection_pool[:limit]

    relevant = [
        item
        for item in ranked
        if str(item[2].get("control_class", "")) not in {"safety", "human_gate"}
    ]
    top_relevant = relevant[0][0] if relevant else (ranked[0][0] if ranked else 0.0)
    second_relevant = relevant[1][0] if len(relevant) > 1 else 0.0

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
        if len(selected) >= limit:
            selected = selected[:-1]
        selected.append(required)

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
            if len(selected) >= limit:
                selected = selected[:-1]
            selected.append(required)
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
    return {
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
        provided.update(f"intent:{item}" for item in _string_list(manifest.get("intent_ids", ())))
        provided.update(
            f"affordance:{item}" for item in _string_list(manifest.get("affordances", ()))
        )
        provided.update(f"effect:{item}" for item in _string_list(manifest.get("effects", ())))
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
                "intent_ids": [],
                "affordances": [],
                "effects": [],
                "handled_events": [],
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
    """Report whether the Core search response violates its bounded shape.

    A malformed entry is a broken bridge, so it must surface as ``failed``.  It
    must never be silently filtered out and reported as a legitimate no-match.
    """
    if not isinstance(core_result, Mapping):
        return True
    ok = core_result.get("ok", _MISSING)
    if ok is not _MISSING and (not isinstance(ok, bool) or ok is False):
        return True
    raw_results = core_result.get("results", _MISSING)
    if raw_results is _MISSING or not isinstance(raw_results, list):
        return True
    for item in raw_results:
        if not isinstance(item, Mapping):
            return True
        module_id = item.get("module_id", "")
        if not isinstance(module_id, str) or not _SAFE_ID.fullmatch(module_id):
            return True
        # Core normalizes both identity fields to a registry value or "".  A
        # present non-string is a broken bridge, and coercing it with str()
        # would manufacture an identity, so it fails the search contract.
        for field in ("provides_capability", "plugin"):
            value = item.get(field, _MISSING)
            if value is not _MISSING and not isinstance(value, str):
                return True
    return False


def _core_discovery_status(
    core_result: Mapping[str, Any],
    core_runtime_manifest: Mapping[str, Any],
    core_candidate_count: int,
) -> tuple[str, str]:
    """Classify Core discovery from the bridge responses alone."""
    runtime_issue = _core_runtime_issue(core_runtime_manifest)
    if runtime_issue is not None:
        return runtime_issue
    if _core_results_malformed(core_result):
        return DISCOVERY_FAILED, "search_malformed"
    if core_candidate_count:
        return DISCOVERY_APPLIED, "discovery_matched"
    raw_results = core_result.get("results", [])
    if isinstance(raw_results, list) and raw_results:
        # Well-formed hits that declare no capability are ordinary modules, so
        # the lane resolved cleanly with nothing routable to contribute.
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
    """Route and also return the exact catalog the route was decided over.

    The catalog is the supplied manifests plus the verified Core-discovered
    capability providers.  Callers that must resolve a selected candidate back
    to a manifest need this combined view; resolving against the request's
    original list alone would silently drop a legitimately discovered provider.
    """
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
    core_candidates = _core_capability_providers(core_result)
    combined.extend(core_candidates)
    if "allowed_sources" not in active_context:
        # An explicit ceiling is preserved exactly.  Only when the caller gave
        # none may installed, registry-declared Core providers join the default
        # scope; an ordinary Core search hit never reaches this point because it
        # is not projected into a manifest at all.
        sources = {_manifest_source(item) for item in manifests}
        if core_candidates:
            sources.add("flyto-core")
        active_context["allowed_sources"] = sorted(sources or {"external"})
    core_status, core_status_reason = _core_discovery_status(
        core_result,
        core_runtime_manifest,
        len(core_candidates),
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
    "CAPABILITY_ROUTE_VERSION",
    "CORE_RUNTIME_CONTRACT",
    "DISCOVERY_STATUSES",
    "GOAL_FRAME_REQUEST_VERSION",
    "GOAL_FRAME_VERSION",
    "ROUTING_DECISION_VERSION",
    "CapabilityRoutingError",
    "goal_frame_request",
    "normalize_goal_frame",
    "prepare_planner_request",
    "route_capabilities",
    "route_with_flyto",
]
