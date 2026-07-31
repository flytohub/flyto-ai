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
from typing import Any

CAPABILITY_ROUTE_VERSION = "flyto.capability-route.v1"
ROUTING_DECISION_VERSION = "flyto.capability-routing-decision.v1"
GOAL_FRAME_VERSION = "flyto.goal-frame.v1"
GOAL_FRAME_REQUEST_VERSION = "flyto.goal-frame-request.v1"
_WORD = re.compile(r"[^\W_]+", re.UNICODE)
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:@-]{0,191}$")

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
    runtime_name = _runtime_name(manifest)
    return str(
        manifest.get("canonical_id")
        or (
            f"core.module.{runtime_name}@1"
            if manifest.get("source") == "flyto-core"
            else f"capability.{runtime_name}@1"
        )
    )


def _manifest_source(manifest: Mapping[str, Any]) -> str:
    explicit = str(manifest.get("source", ""))
    if explicit:
        return explicit
    canonical_id = _canonical_id(manifest)
    if canonical_id.startswith("robotics."):
        return "flyto-robotics"
    if canonical_id.startswith("core."):
        return "flyto-core"
    return "external"


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
    seen_names: set[str] = set()
    seen_ids: set[str] = set()
    for manifest in manifests:
        if not isinstance(manifest, Mapping):
            excluded.append({"runtime_name": "", "reasons": ["invalid_manifest"]})
            continue
        runtime_name = _runtime_name(manifest)
        canonical_id = _canonical_id(manifest)
        if (
            not _SAFE_ID.fullmatch(runtime_name)
            or not _SAFE_ID.fullmatch(canonical_id)
            or runtime_name in seen_names
            or canonical_id in seen_ids
        ):
            excluded.append(
                {"runtime_name": runtime_name, "reasons": ["invalid_or_duplicate_identity"]}
            )
            continue
        seen_names.add(runtime_name)
        seen_ids.add(canonical_id)
        failures = _hard_filter(manifest, active_context)
        if failures:
            excluded.append({"runtime_name": runtime_name, "reasons": list(failures)})
            continue
        valid.append(manifest)

    ranked: list[tuple[float, str, Mapping[str, Any], tuple[str, ...]]] = []
    for manifest in valid:
        score, reasons = _score(goal, manifest, blueprint_hints, active_goal_frame)
        ranked.append((score, _canonical_id(manifest), manifest, reasons))
    ranked.sort(key=lambda item: (-item[0], item[1]))
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


def _core_manifests(result: Mapping[str, Any]) -> list[dict[str, Any]]:
    manifests: list[dict[str, Any]] = []
    raw_results = result.get("results", [])
    if not isinstance(raw_results, list):
        return manifests
    for item in raw_results:
        if not isinstance(item, dict):
            continue
        module_id = str(item.get("module_id", ""))
        if not module_id:
            continue
        manifests.append(
            {
                "manifest_contract": "flyto.capability-manifest.v1",
                "canonical_id": f"core.module.{module_id}@1",
                "runtime_name": module_id,
                "name": module_id,
                "version": "runtime",
                "source": "flyto-core",
                "domain": str(item.get("category", "core")),
                "description": str(item.get("description", "")),
                "label": str(item.get("label", "")),
                "aliases": [],
                "tags": [str(item.get("category", ""))],
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
    if not 1 <= core_limit <= 100:
        raise CapabilityRoutingError("core_limit must be between 1 and 100")
    active_goal_frame = (
        normalize_goal_frame(goal_frame) if goal_frame is not None else None
    )
    discovery_query = (
        _semantic_query(active_goal_frame) if active_goal_frame is not None else goal
    )
    active_context: dict[str, object] = dict(context or {})
    if "allowed_sources" not in active_context:
        sources = {_manifest_source(item) for item in manifests}
        active_context["allowed_sources"] = sorted(
            sources or {"external"}
        )

    if blueprint_search is None:
        try:
            from flyto_blueprint import get_engine

            blueprint_search = get_engine().search
        except Exception:
            blueprint_search = _empty_blueprint_search
    try:
        blueprint_candidates = list(blueprint_search(discovery_query))
    except Exception:
        blueprint_candidates = []

    if core_dispatch is None:
        from flyto_ai.tools.core_tools import dispatch_core_tool

        core_dispatch = dispatch_core_tool
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
    combined = [dict(item) for item in manifests]
    combined.extend(_core_manifests(core_result))
    route = route_capabilities(
        goal,
        combined,
        goal_frame=active_goal_frame,
        context=active_context,
        limit=limit,
        blueprint_candidates=blueprint_candidates,
    )
    trusted_hints = _trusted_blueprint_module_hints(blueprint_candidates)
    return {
        "contract_version": ROUTING_DECISION_VERSION,
        "route": route,
        "discovery_evidence": {
            "blueprint": {
                "query_mode": (
                    "semantic_frame" if active_goal_frame is not None else "raw_goal"
                ),
                "candidate_count": len(blueprint_candidates),
                "trusted_module_hints": sorted(trusted_hints),
            },
            "core": {
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
                "candidate_count": len(_core_manifests(core_result)),
                "used_bridge": "flyto_ai.tools.core_tools.dispatch_core_tool",
            },
        },
    }


async def prepare_planner_request(
    request: Mapping[str, Any],
    *,
    context: Mapping[str, object] | None = None,
    limit: int = 8,
    require_goal_frame: bool = False,
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

    decision = await route_with_flyto(
        goal,
        manifests,
        goal_frame=goal_frame,
        context=context,
        limit=limit,
        core_dispatch=core_dispatch,
        blueprint_search=blueprint_search,
    )
    route = decision["route"]
    selected_names = {
        str(candidate["runtime_name"]) for candidate in route["candidates"]
    }
    prepared = dict(request)
    prepared["capabilities"] = [
        manifest
        for manifest in manifests
        if _runtime_name(manifest) in selected_names
    ]
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
    "CAPABILITY_ROUTE_VERSION",
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
