from __future__ import annotations

import sys
import hashlib
import json
from dataclasses import replace

import pytest

from flyto_ai.capability_router import (
    BLUEPRINT_CANDIDATE_LIMIT,
    CAPABILITY_GROUP_LIMIT,
    CORE_RUNTIME_CONTRACT,
    DISCOVERY_STATUSES,
    EMITTED_PROVIDER_ROW_LIMIT,
    CapabilityRoutingError,
    CapabilityRetrievalAuthority,
    _selected_manifests,
    capability_routing_bounds,
    goal_frame_request,
    prepare_planner_request,
    route_capabilities,
    route_with_flyto,
    validate_capability_retrieval,
)


def _manifest(
    name: str,
    *,
    canonical_id: str | None = None,
    aliases: list[str] | None = None,
    control_class: str = "motion",
    observations: list[str] | None = None,
    intent_ids: list[str] | None = None,
    affordances: list[str] | None = None,
    effects: list[str] | None = None,
    handled_events: list[str] | None = None,
) -> dict[str, object]:
    return {
        "manifest_contract": "flyto.capability-manifest.v1",
        "canonical_id": canonical_id or f"robotics.test.{name}@1",
        "runtime_name": name,
        "name": name,
        "version": "1.0.0",
        "domain": "robotics",
        "description": f"Atomic {name} capability.",
        "tags": [],
        "aliases": aliases or [],
        "control_class": control_class,
        "required_observations": observations or [],
        "required_resources": [],
        "required_permissions": [],
        "compatible_robots": ["*"],
        "positive_examples": [],
        "negative_examples": [],
        "intent_ids": intent_ids or [],
        "affordances": affordances or [],
        "effects": effects or [],
        "handled_events": handled_events or [],
    }


def _route_frame() -> dict[str, object]:
    return {
        "contract_version": "flyto.goal-frame.v1",
        "intent_ids": ["route.follow.sequence"],
        "required_affordances": [
            "motion.follow.visual_line",
            "safety.wait_until_clear",
        ],
        "desired_effects": [
            "robot.motion.stopped",
            "route.sequence.completed",
        ],
        "trigger_events": ["human.detected"],
        "constraints": [
            {
                "key": "route.sequence",
                "operator": "ordered",
                "value": ["blue", "yellow", "purple"],
            }
        ],
    }


def _digest(value: object) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _retrieval(
    manifests: list[dict[str, object]],
    scores: list[int],
    *,
    goal: str = "neutral",
    context: dict[str, object] | None = None,
    goal_frame: dict[str, object] | None = None,
    requirements: dict[str, list[str]] | None = None,
) -> tuple[dict[str, object], CapabilityRetrievalAuthority]:
    fixed = {
        "tenant_id": "tenant-a",
        "workspace_id": "workspace-a",
        "space_id": "space-a",
    }
    model = {
        "model_id": "embedding-v1",
        "model_version": "2026-08-13",
        "dimensions": 3,
    }
    model["model_digest"] = _digest(model)
    index_digest, snapshot_digest = _digest({"index": 1}), _digest({"snapshot": 7})
    top_k = max(1, len(manifests))
    request = {
        "request_version": "flyto.capability-search-query.v1",
        "query": "accepted upstream query",
        "top_k": top_k,
        "page_size": top_k,
        "hard_filters": {
            "tenant_id": "tenant-a",
            "space_id": "space-a",
            "status": "active",
            "acl_principals": ["principal-a"],
            "acl_scopes": ["capability.read"],
            "risk_classification": "high",
            "resource_ids": [],
            "capability_ids": [],
        },
        "prefilter_required": True,
        "retrieval_order": ["hard_filter", "lexical", "ann", "fuse"],
        "model": model,
        "index_digest": index_digest,
        "snapshot_digest": snapshot_digest,
        "weights": {"lexical": 50, "vector": 50},
        "cursor": None,
    }
    request["request_digest"] = _digest(
        {key: value for key, value in request.items() if key != "cursor"}
    )
    candidates = []
    for manifest, score in zip(manifests, scores, strict=True):
        manifest["source_projection_digest"] = _digest(
            {"source": manifest["runtime_name"]}
        )
        manifest["upstream_content_digest"] = _digest(
            {"upstream": manifest["runtime_name"]}
        )
        manifest["document_digest"] = _digest({"document": manifest["runtime_name"]})
        candidate = {
            "tenant_id": "tenant-a",
            "space_id": "space-a",
            "capability_id": manifest["canonical_id"],
            "status": "active",
            "acl_principals": ["principal-a"],
            "acl_scopes": ["capability.read"],
            "risk_classification": "low",
            "resource_ids": sorted(manifest["required_resources"]),
            "score": score,
            "source_projection_digest": manifest["source_projection_digest"],
            "upstream_content_digest": manifest["upstream_content_digest"],
            "document_digest": manifest["document_digest"],
            "model_digest": model["model_digest"],
            "index_digest": index_digest,
            "snapshot_digest": snapshot_digest,
            "candidate_only": True,
            "execution_authority": False,
        }
        candidate["candidate_digest"] = _digest(candidate)
        candidates.append(candidate)
    candidates.sort(key=lambda item: (-item["score"], item["capability_id"]))
    requirement_map = requirements or {}
    candidate_resources = {
        capability_id: next(
            item["resource_ids"]
            for item in candidates
            if item["capability_id"] == capability_id
        )
        for capability_id in requirement_map
    }
    requirements_value = {
        "requirements_version": "flyto.cloud.capability-resource-requirements.v1",
        **fixed,
        "request_digest": request["request_digest"],
        "capability_resources": requirement_map,
    }
    requirements_value["requirements_digest"] = _digest(requirements_value)
    feasibility = {
        "result_version": "flyto.cloud.capability-resource-feasibility.v1",
        "requirements_digest": requirements_value["requirements_digest"],
        "candidate_resources": dict(sorted(candidate_resources.items())),
        "feasible": True,
        "candidate_only": True,
        "execution_authority": False,
    }
    feasibility["result_digest"] = _digest(feasibility)
    page = {
        "page_version": "flyto.capability-search-page.v1",
        "request_digest": request["request_digest"],
        "candidates": candidates,
        "next_cursor": None,
        "candidate_only": True,
        "execution_authority": False,
    }
    query_context_digest = _digest({"accepted_cloud_context": 7})
    result = {
        "result_version": "flyto.cloud.capability-index-result.v1",
        "query_context_digest": query_context_digest,
        "cloud_next_continuation": None,
        "page": page,
        "feasibility": feasibility,
        "candidate_only": True,
        "execution_authority": False,
    }
    result["result_digest"] = _digest(result)
    handoff = {
        "contract_version": "flyto.ai.capability-retrieval-handoff.v2",
        **fixed,
        "goal_digest": _digest(
            {"digest_version": "flyto.ai.goal-digest.v1", "goal": goal}
        ),
        "routing_context_digest": _digest(
            {
                "digest_version": "flyto.ai.routing-context-digest.v1",
                "routing_context": context or {},
            }
        ),
        "goal_frame_digest": _digest(
            {
                "digest_version": "flyto.ai.goal-frame-digest.v1",
                "goal_frame": goal_frame,
            }
        ),
        "request": request,
        "result": result,
        "candidate_only": True,
        "execution_authority": False,
    }
    handoff["handoff_digest"] = _digest(handoff)
    authority = CapabilityRetrievalAuthority(
        **fixed,
        request_digest=request["request_digest"],
        model_digest=model["model_digest"],
        index_digest=index_digest,
        snapshot_digest=snapshot_digest,
        query_context_digest=query_context_digest,
        requirements_digest=requirements_value["requirements_digest"],
        result_digest=result["result_digest"],
        goal_digest=handoff["goal_digest"],
        routing_context_digest=handoff["routing_context_digest"],
        goal_frame_digest=handoff["goal_frame_digest"],
        handoff_digest=handoff["handoff_digest"],
        host_verified=True,
    )
    return handoff, authority


def _rebind_retrieval_digests(
    handoff: dict[str, object], authority: CapabilityRetrievalAuthority
) -> CapabilityRetrievalAuthority:
    feasibility = handoff["result"]["feasibility"]
    feasibility["result_digest"] = _digest(
        {key: value for key, value in feasibility.items() if key != "result_digest"}
    )
    result = handoff["result"]
    result["result_digest"] = _digest(
        {key: value for key, value in result.items() if key != "result_digest"}
    )
    handoff["handoff_digest"] = _digest(
        {key: value for key, value in handoff.items() if key != "handoff_digest"}
    )
    return replace(
        authority,
        result_digest=result["result_digest"],
        handoff_digest=handoff["handoff_digest"],
    )


def test_terminal_retrieval_narrows_installed_catalog_and_emits_evidence() -> None:
    manifests = [
        _manifest(
            "software", canonical_id="software.api.fetch@1", intent_ids=["task.fetch"]
        ),
        _manifest(
            "hardware", canonical_id="hardware.sensor.read@1", intent_ids=["task.read"]
        ),
        _manifest(
            "robotics", canonical_id="robotics.motion.move@1", intent_ids=["task.move"]
        ),
    ]
    frame = {
        **_route_frame(),
        "intent_ids": ["task.fetch"],
        "required_affordances": [],
        "desired_effects": [],
        "trigger_events": [],
        "constraints": [],
    }
    handoff, authority = _retrieval([manifests[0]], [900_000_000], goal_frame=frame)

    first = route_capabilities(
        "neutral",
        manifests,
        goal_frame=frame,
        retrieval_handoff=handoff,
        retrieval_authority=authority,
    )
    second = route_capabilities(
        "neutral",
        manifests,
        goal_frame=frame,
        retrieval_handoff=handoff,
        retrieval_authority=authority,
    )

    assert first == second
    assert [item["runtime_name"] for item in first["candidates"]] == ["software"]
    assert first["retrieval_evidence"]["candidate_only"] is True
    assert first["retrieval_evidence"]["execution_authority"] is False
    assert first["retrieval_evidence"]["planning_required"] is True


def test_capability_candidate_expands_all_distinct_installed_coproviders() -> None:
    first = _manifest(
        "provider_a", canonical_id="task.shared@1", intent_ids=["intent.shared"]
    )
    second = _manifest(
        "provider_b", canonical_id="task.shared@1", intent_ids=["intent.shared"]
    )
    first["plugin"], second["plugin"] = "plugin-a", "plugin-b"
    frame = {
        **_route_frame(),
        "intent_ids": ["intent.shared"],
        "required_affordances": [],
        "desired_effects": [],
        "trigger_events": [],
        "constraints": [],
    }
    handoff, authority = _retrieval([first], [900_000_000], goal_frame=frame)
    for field in (
        "source_projection_digest",
        "upstream_content_digest",
        "document_digest",
    ):
        second[field] = first[field]

    route = route_capabilities(
        "neutral",
        [second, first],
        limit=1,
        goal_frame=frame,
        retrieval_handoff=handoff,
        retrieval_authority=authority,
    )

    assert [(item["runtime_name"], item["plugin"]) for item in route["candidates"]] == [
        ("provider_a", "plugin-a"),
        ("provider_b", "plugin-b"),
    ]


def test_capability_group_ties_use_deterministic_group_and_provider_order() -> None:
    manifests = [
        _manifest("z_provider", canonical_id="task.beta@1"),
        _manifest("b_provider", canonical_id="task.alpha@1"),
        _manifest("a_provider", canonical_id="task.alpha@1"),
    ]

    route = route_capabilities("neutral", manifests, limit=1)

    assert [item["runtime_name"] for item in route["candidates"]] == [
        "a_provider",
        "b_provider",
    ]


def test_coprovider_multiplicity_and_order_do_not_create_capability_ambiguity() -> None:
    first = _manifest("shared", canonical_id="task.shared@1")
    second = _manifest("shared_alternative", canonical_id="task.shared@1")
    first["plugin"], second["plugin"] = "plugin-a", "plugin-b"

    single = route_capabilities("task shared", [first], limit=1)
    forward = route_capabilities("task shared", [first, second], limit=1)
    reverse = route_capabilities("task shared", [second, first], limit=1)

    assert single["needs_clarification"] is False
    assert forward["needs_clarification"] is False
    assert reverse["needs_clarification"] is False
    assert forward["confidence"] == reverse["confidence"] == single["confidence"]
    assert [item["runtime_name"] for item in forward["candidates"]] == [
        item["runtime_name"] for item in reverse["candidates"]
    ] == ["shared", "shared_alternative"]


def test_two_distinct_tied_capability_groups_still_require_clarification() -> None:
    first = _manifest("alpha_provider", canonical_id="task.alpha@1", aliases=["shared"])
    second = _manifest("beta_provider", canonical_id="task.beta@1", aliases=["shared"])

    route = route_capabilities("shared", [second, first], limit=2)

    assert route["needs_clarification"] is True
    assert [item["canonical_id"] for item in route["candidates"]] == [
        "task.alpha@1",
        "task.beta@1",
    ]


def test_provider_group_overflow_fails_without_partial_output() -> None:
    manifests = [
        _manifest(f"provider_{index:02d}", canonical_id="task.shared@1")
        for index in range(33)
    ]

    with pytest.raises(
        CapabilityRoutingError,
        match=(
            "^selected capability provider groups exceed the 32 emitted provider "
            "row limit$"
        ),
    ):
        route_capabilities("neutral", manifests, limit=1)


def test_public_limit_counts_capability_groups_not_provider_rows() -> None:
    assert capability_routing_bounds() == {
        "capability_groups": CAPABILITY_GROUP_LIMIT,
        "emitted_provider_rows": EMITTED_PROVIDER_ROW_LIMIT,
    }
    manifests = [
        _manifest(f"provider_{index:02d}", canonical_id=f"task.group-{index:02d}@1")
        for index in range(CAPABILITY_GROUP_LIMIT)
    ]

    route = route_capabilities("neutral", manifests, limit=CAPABILITY_GROUP_LIMIT)

    assert len(route["candidates"]) == CAPABILITY_GROUP_LIMIT
    with pytest.raises(
        CapabilityRoutingError,
        match=(
            f"^limit must be between 1 and {CAPABILITY_GROUP_LIMIT} capability "
            "groups$"
        ),
    ):
        route_capabilities("neutral", manifests, limit=CAPABILITY_GROUP_LIMIT + 1)


def test_provider_row_ceiling_is_independent_and_atomic_at_exact_boundary() -> None:
    exact_group = [
        _manifest(f"provider_{index:02d}", canonical_id="task.shared@1")
        for index in range(EMITTED_PROVIDER_ROW_LIMIT)
    ]
    overflow_group = [
        *exact_group,
        _manifest("provider_overflow", canonical_id="task.shared@1"),
    ]

    exact = route_capabilities("neutral", exact_group, limit=1)
    assert len(exact["candidates"]) == EMITTED_PROVIDER_ROW_LIMIT
    assert {item["canonical_id"] for item in exact["candidates"]} == {
        "task.shared@1"
    }
    with pytest.raises(CapabilityRoutingError) as caught:
        route_capabilities("neutral", overflow_group, limit=1)
    assert str(caught.value) == (
        "selected capability provider groups exceed the "
        f"{EMITTED_PROVIDER_ROW_LIMIT} emitted provider row limit"
    )


def test_exact_blueprint_model_dialect_and_request_digest_are_accepted() -> None:
    manifest = _manifest("software", canonical_id="software.fetch@1")
    handoff, authority = _retrieval([manifest], [900_000_000])
    request = handoff["request"]

    assert set(request["model"]) == {
        "model_id",
        "model_version",
        "dimensions",
        "model_digest",
    }
    assert request["request_digest"] == _digest(
        {
            key: value
            for key, value in request.items()
            if key not in {"cursor", "request_digest"}
        }
    )
    assert validate_capability_retrieval(handoff, authority) == handoff


def test_producer_open_discovery_and_slash_identifier_fixture_routes() -> None:
    """Fixture mirrors Blueprint build+validate output without runtime imports."""
    manifest = _manifest(
        "software_fetch",
        canonical_id="software/fetch@1",
        intent_ids=["software.fetch"],
    )
    frame = {
        **_route_frame(),
        "intent_ids": ["software.fetch"],
        "required_affordances": [],
        "desired_effects": [],
        "trigger_events": [],
        "constraints": [],
    }
    handoff, authority = _retrieval([manifest], [900_000_000], goal_frame=frame)
    request = handoff["request"]
    request["hard_filters"].update(
        tenant_id="tenant/a",
        space_id="space/a",
        acl_principals=["principal/a"],
        acl_scopes=["capability/read"],
        resource_ids=["resource/a"],
        capability_ids=[],
    )
    candidate = handoff["result"]["page"]["candidates"][0]
    candidate.update(
        tenant_id="tenant/a",
        space_id="space/a",
        acl_principals=["principal/a"],
        acl_scopes=["capability/read"],
        resource_ids=["resource/a"],
    )
    candidate["candidate_digest"] = _digest(
        {key: value for key, value in candidate.items() if key != "candidate_digest"}
    )
    request["request_digest"] = _digest(
        {
            key: value
            for key, value in request.items()
            if key not in {"cursor", "request_digest"}
        }
    )
    page = handoff["result"]["page"]
    page["request_digest"] = request["request_digest"]
    result = handoff["result"]
    result["result_digest"] = _digest(
        {key: value for key, value in result.items() if key != "result_digest"}
    )
    handoff.update(tenant_id="tenant/a", space_id="space/a")
    handoff["handoff_digest"] = _digest(
        {key: value for key, value in handoff.items() if key != "handoff_digest"}
    )
    authority = replace(
        authority,
        tenant_id="tenant/a",
        space_id="space/a",
        request_digest=request["request_digest"],
        result_digest=result["result_digest"],
        handoff_digest=handoff["handoff_digest"],
    )

    route = route_capabilities(
        "neutral",
        [manifest],
        goal_frame=frame,
        retrieval_handoff=handoff,
        retrieval_authority=authority,
    )
    assert [item["canonical_id"] for item in route["candidates"]] == [
        "software/fetch@1"
    ]


def test_nonempty_upstream_capability_filter_is_enforced() -> None:
    manifest = _manifest("software", canonical_id="software/fetch@1")
    handoff, authority = _retrieval([manifest], [900_000_000])
    request = handoff["request"]
    request["hard_filters"]["capability_ids"] = ["software/other@1"]
    request["request_digest"] = _digest(
        {
            key: value
            for key, value in request.items()
            if key not in {"cursor", "request_digest"}
        }
    )
    handoff["result"]["page"]["request_digest"] = request["request_digest"]
    result = handoff["result"]
    result["result_digest"] = _digest(
        {key: value for key, value in result.items() if key != "result_digest"}
    )
    handoff["handoff_digest"] = _digest(
        {key: value for key, value in handoff.items() if key != "handoff_digest"}
    )
    authority = replace(
        authority,
        request_digest=request["request_digest"],
        result_digest=result["result_digest"],
        handoff_digest=handoff["handoff_digest"],
    )
    with pytest.raises(
        CapabilityRoutingError, match="^invalid capability retrieval handoff$"
    ):
        validate_capability_retrieval(handoff, authority)


@pytest.mark.parametrize(
    "context",
    [
        {"hostile": object()},
        {"integer": 2**63},
        {"nested": [[[[[[[[["too-deep"]]]]]]]]]},
    ],
)
def test_retrieval_binding_rejects_unbounded_context_content_free(
    context: dict[str, object],
) -> None:
    manifest = _manifest("software", canonical_id="software.fetch@1")
    handoff, authority = _retrieval([manifest], [900_000_000])
    with pytest.raises(CapabilityRoutingError) as caught:
        route_capabilities(
            "neutral",
            [manifest],
            context=context,
            retrieval_handoff=handoff,
            retrieval_authority=authority,
        )
    assert str(caught.value) == "invalid capability retrieval handoff"


def test_retrieval_binding_rejects_deep_goal_frame_content_free() -> None:
    manifest = _manifest("software", canonical_id="software.fetch@1")
    handoff, authority = _retrieval([manifest], [900_000_000])
    deep: object = "leaf"
    for _ in range(20):
        deep = [deep]
    frame = {
        **_route_frame(),
        "constraints": [{"key": "x", "operator": "eq", "value": deep}],
    }
    with pytest.raises(CapabilityRoutingError) as caught:
        route_capabilities(
            "neutral",
            [manifest],
            goal_frame=frame,
            retrieval_handoff=handoff,
            retrieval_authority=authority,
        )
    assert str(caught.value) == "invalid capability retrieval handoff"


@pytest.mark.parametrize(("length", "accepted"), [(128, True), (129, False)])
def test_upstream_model_identifier_uses_exact_128_character_bound(
    length: int, accepted: bool
) -> None:
    manifest = _manifest("software", canonical_id="software.fetch@1")
    handoff, authority = _retrieval([manifest], [900_000_000])
    request = handoff["request"]
    model = request["model"]
    model["model_id"] = "m" * length
    model["model_version"] = "v" * length
    model["model_digest"] = _digest(
        {key: value for key, value in model.items() if key != "model_digest"}
    )
    request["request_digest"] = _digest(
        {
            key: value
            for key, value in request.items()
            if key not in {"cursor", "request_digest"}
        }
    )
    candidate = handoff["result"]["page"]["candidates"][0]
    candidate["model_digest"] = model["model_digest"]
    candidate["candidate_digest"] = _digest(
        {key: value for key, value in candidate.items() if key != "candidate_digest"}
    )
    handoff["result"]["page"]["request_digest"] = request["request_digest"]
    result = handoff["result"]
    result["result_digest"] = _digest(
        {key: value for key, value in result.items() if key != "result_digest"}
    )
    handoff["handoff_digest"] = _digest(
        {key: value for key, value in handoff.items() if key != "handoff_digest"}
    )
    authority = replace(
        authority,
        request_digest=request["request_digest"],
        model_digest=model["model_digest"],
        result_digest=result["result_digest"],
        handoff_digest=handoff["handoff_digest"],
    )

    if accepted:
        assert validate_capability_retrieval(handoff, authority) == handoff
    else:
        with pytest.raises(CapabilityRoutingError) as caught:
            validate_capability_retrieval(handoff, authority)
        assert str(caught.value) == "invalid capability retrieval handoff"


@pytest.mark.parametrize(
    ("filter_field", "filter_value", "candidate_field", "candidate_value"),
    [
        ("status", "retired", None, None),
        ("acl_principals", ["principal-a"], "acl_principals", ["attacker"]),
        ("acl_scopes", ["capability.read"], "acl_scopes", ["capability.write"]),
        ("risk_classification", "low", "risk_classification", "critical"),
        ("resource_ids", ["mac-camera"], "resource_ids", ["pi-motor"]),
        ("capability_ids", ["other.capability@1"], None, None),
    ],
)
def test_recomputed_upstream_hard_filter_drift_still_fails_closed(
    filter_field: str,
    filter_value: object,
    candidate_field: str | None,
    candidate_value: object,
) -> None:
    manifest = _manifest("vision", canonical_id="robotics.vision@1")
    handoff, authority = _retrieval([manifest], [900_000_000])
    request, result = handoff["request"], handoff["result"]
    request["hard_filters"][filter_field] = filter_value
    request["request_digest"] = _digest(
        {
            key: value
            for key, value in request.items()
            if key not in {"cursor", "request_digest"}
        }
    )
    result["page"]["request_digest"] = request["request_digest"]
    if candidate_field is not None:
        candidate = result["page"]["candidates"][0]
        candidate[candidate_field] = candidate_value
        candidate["candidate_digest"] = _digest(
            {
                key: value
                for key, value in candidate.items()
                if key != "candidate_digest"
            }
        )
    result["result_digest"] = _digest(
        {key: value for key, value in result.items() if key != "result_digest"}
    )
    handoff["handoff_digest"] = _digest(
        {key: value for key, value in handoff.items() if key != "handoff_digest"}
    )
    authority = replace(
        authority,
        request_digest=request["request_digest"],
        result_digest=result["result_digest"],
        handoff_digest=handoff["handoff_digest"],
    )

    with pytest.raises(CapabilityRoutingError) as caught:
        validate_capability_retrieval(handoff, authority)
    assert str(caught.value) == "invalid capability retrieval handoff"


def test_retrieval_score_cannot_invert_goal_frame_semantics_or_merge_providers() -> (
    None
):
    preferred = _manifest(
        "preferred", canonical_id="task.preferred@1", intent_ids=["intent.required"]
    )
    other = _manifest("other", canonical_id="task.other@1", intent_ids=["intent.other"])
    preferred["plugin"] = "plugin-a"
    other["plugin"] = "plugin-b"
    frame = {
        **_route_frame(),
        "intent_ids": ["intent.required"],
        "required_affordances": [],
        "desired_effects": [],
        "trigger_events": [],
        "constraints": [],
    }
    handoff, authority = _retrieval(
        [other, preferred], [1_000_000_000, 0], goal_frame=frame
    )

    route = route_capabilities(
        "neutral",
        [preferred, other],
        goal_frame=frame,
        retrieval_handoff=handoff,
        retrieval_authority=authority,
    )

    assert [(item["runtime_name"], item["plugin"]) for item in route["candidates"]] == [
        ("preferred", "plugin-a")
    ]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("tenant_id", "tenant-b"),
        ("workspace_id", "workspace-b"),
        ("space_id", "space-b"),
        ("goal_digest", "sha256:" + "0" * 64),
        ("candidate_only", False),
        ("execution_authority", True),
    ],
)
def test_retrieval_scope_binding_and_pages_fail_with_stable_error(
    field: str, value: object
) -> None:
    manifest = _manifest("one")
    handoff, authority = _retrieval([manifest], [500_000_000])
    handoff[field] = value
    with pytest.raises(
        CapabilityRoutingError, match="^invalid capability retrieval handoff$"
    ):
        validate_capability_retrieval(handoff, authority)


def test_retrieval_empty_is_non_routable_and_hostile_or_stale_data_fails_closed() -> (
    None
):
    manifest = _manifest("one", intent_ids=["intent.one"])
    frame = {
        **_route_frame(),
        "intent_ids": ["intent.one"],
        "required_affordances": [],
        "desired_effects": [],
        "trigger_events": [],
        "constraints": [],
    }
    empty, empty_authority = _retrieval([], [], goal_frame=frame)
    route = route_capabilities(
        "neutral",
        [manifest],
        goal_frame=frame,
        retrieval_handoff=empty,
        retrieval_authority=empty_authority,
    )
    assert route["candidates"] == []
    assert route["needs_clarification"] is True

    handoff, authority = _retrieval([manifest], [500_000_000])
    with pytest.raises(
        CapabilityRoutingError, match="^invalid capability retrieval handoff$"
    ):
        validate_capability_retrieval({**handoff, "unknown": "secret-value"}, authority)
    with pytest.raises(
        CapabilityRoutingError, match="^invalid capability retrieval handoff$"
    ):
        route_capabilities(
            "neutral", [], retrieval_handoff=handoff, retrieval_authority=authority
        )

    class Hostile(dict):
        pass

    with pytest.raises(
        CapabilityRoutingError, match="^invalid capability retrieval handoff$"
    ):
        validate_capability_retrieval(Hostile(handoff), authority)


def test_real_terminal_shapes_support_resource_free_and_distinct_resources() -> None:
    software = _manifest("software", canonical_id="software.fetch@1")
    vision = _manifest("vision", canonical_id="robotics.vision@1")
    movement = _manifest("movement", canonical_id="robotics.move@1")
    vision["required_resources"] = ["mac-camera"]
    movement["required_resources"] = ["pi-motor"]
    requirements = {
        "robotics.vision@1": ["mac-camera"],
        "robotics.move@1": ["pi-motor"],
    }
    handoff, authority = _retrieval(
        [software, vision, movement],
        [900_000_000, 800_000_000, 700_000_000],
        requirements=requirements,
    )

    validated = validate_capability_retrieval(handoff, authority)
    assert set(validated["request"]) == {
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
    }
    assert validated["result"]["page"]["candidates"][0]["resource_ids"] == []
    assert validated["result"]["feasibility"]["candidate_resources"] == requirements


def test_feasibility_may_cover_required_capability_not_returned_on_page() -> None:
    software = _manifest("software", canonical_id="software.fetch@1")
    handoff, authority = _retrieval([software], [900_000_000])
    feasibility = handoff["result"]["feasibility"]
    feasibility["candidate_resources"] = {
        "robotics.move@1": ["pi-motor"],
        "robotics.vision@1": ["mac-camera"],
    }
    feasibility["result_digest"] = _digest(
        {key: value for key, value in feasibility.items() if key != "result_digest"}
    )
    result = handoff["result"]
    result["result_digest"] = _digest(
        {key: value for key, value in result.items() if key != "result_digest"}
    )
    handoff["handoff_digest"] = _digest(
        {key: value for key, value in handoff.items() if key != "handoff_digest"}
    )
    authority = replace(
        authority,
        result_digest=result["result_digest"],
        handoff_digest=handoff["handoff_digest"],
    )

    assert validate_capability_retrieval(handoff, authority) == handoff


@pytest.mark.parametrize("count,accepted", [(128, True), (129, False)])
def test_cloud_feasibility_has_exact_128_capability_key_bound(
    count: int, accepted: bool
) -> None:
    manifest = _manifest("software", canonical_id="software.fetch@1")
    handoff, authority = _retrieval([manifest], [900_000_000])
    handoff["result"]["feasibility"]["candidate_resources"] = {
        f"capability.model/{index:03d}@1": [f"resource-{index:03d}"]
        for index in range(count)
    }
    authority = _rebind_retrieval_digests(handoff, authority)

    if accepted:
        assert validate_capability_retrieval(handoff, authority) == handoff
    else:
        with pytest.raises(CapabilityRoutingError) as caught:
            validate_capability_retrieval(handoff, authority)
        assert str(caught.value) == "invalid capability retrieval handoff"


@pytest.mark.parametrize(
    "mutation",
    ["infeasible", "forged_resources", "bool_score", "nan_score", "partial", "unknown"],
)
def test_retrieval_tamper_variants_share_one_content_free_error(mutation: str) -> None:
    manifest = _manifest("vision", canonical_id="robotics.vision@1")
    manifest["required_resources"] = ["mac-camera"]
    handoff, authority = _retrieval(
        [manifest], [900_000_000], requirements={"robotics.vision@1": ["mac-camera"]}
    )
    if mutation == "infeasible":
        handoff["result"]["feasibility"]["feasible"] = False
    elif mutation == "forged_resources":
        handoff["result"]["feasibility"]["candidate_resources"] = {
            "robotics.vision@1": ["pi-motor"]
        }
    elif mutation == "bool_score":
        handoff["result"]["page"]["candidates"][0]["score"] = True
    elif mutation == "nan_score":
        handoff["result"]["page"]["candidates"][0]["score"] = float("nan")
    elif mutation == "partial":
        handoff["result"]["page"]["next_cursor"] = "continued"
    else:
        handoff["result"]["page"]["unknown"] = "never exposed"
    with pytest.raises(CapabilityRoutingError) as caught:
        validate_capability_retrieval(handoff, authority)
    assert str(caught.value) == "invalid capability retrieval handoff"


def test_router_is_bounded_deterministic_and_multilingual() -> None:
    manifests = [
        _manifest(
            "follow_line",
            canonical_id="robotics.vision.follow_line@1",
            aliases=["循線", "藍線", "黃線", "紫線"],
            observations=["camera.line_scene"],
        ),
        _manifest(
            "safe_stop",
            canonical_id="robotics.safety.safe_stop@1",
            aliases=["安全停止", "停下來"],
            control_class="safety",
        ),
        *[
            _manifest(
                f"noise_{index}",
                control_class="timed",
                aliases=[f"unrelated-{index}"],
            )
            for index in range(100)
        ],
    ]

    first = route_capabilities(
        "先走藍線，再走黃線，最後走紫線並安全停止",
        manifests,
        limit=6,
    )
    second = route_capabilities(
        "先走藍線，再走黃線，最後走紫線並安全停止",
        manifests,
        limit=6,
    )

    assert first == second
    assert len(first["candidates"]) == 6
    names = {candidate["runtime_name"] for candidate in first["candidates"]}
    assert {"follow_line", "safe_stop"}.issubset(names)
    assert first["needs_clarification"] is False
    assert first["registry_snapshot"].startswith("sha256:")


def test_goal_frame_makes_selection_independent_of_input_language() -> None:
    manifests = [
        _manifest(
            "follow_line",
            intent_ids=["route.follow.sequence"],
            affordances=["motion.follow.visual_line"],
            effects=["route.sequence.completed"],
        ),
        _manifest(
            "wait_until_clear",
            control_class="safety",
            affordances=["safety.wait_until_clear"],
            handled_events=["human.detected"],
        ),
        _manifest(
            "safe_stop",
            control_class="safety",
            affordances=["safety.stop.motion"],
            effects=["robot.motion.stopped"],
        ),
        _manifest(
            "navigate",
            intent_ids=["route.navigate.pose"],
            affordances=["motion.navigate.pose"],
            effects=["robot.pose.reached"],
        ),
    ]

    chinese = route_capabilities(
        "先走藍線，再走黃線，最後走紫線；遇到人就等候。",
        manifests,
        goal_frame=_route_frame(),
        limit=4,
    )
    arabic = route_capabilities(
        "اتبع المسارات بالترتيب وانتظر عند وجود شخص.",
        manifests,
        goal_frame=_route_frame(),
        limit=4,
    )
    japanese = route_capabilities(
        "青、黄、紫の順で進み、人がいたら待機する。",
        manifests,
        goal_frame=_route_frame(),
        limit=4,
    )

    assert chinese == arabic == japanese
    assert [item["runtime_name"] for item in chinese["candidates"]] == [
        "follow_line",
        "wait_until_clear",
        "safe_stop",
    ]
    assert chinese["selection_method"] == ("hard_filter_then_semantic_frame_rank_v1")
    assert chinese["semantic_coverage"]["ratio"] == 1.0
    assert chinese["semantic_coverage"]["missing"] == []
    assert chinese["needs_clarification"] is False


def test_semantic_frame_excludes_zero_score_and_blueprint_only_candidates() -> None:
    manifests = [
        _manifest(
            "navigate_to_location",
            canonical_id="robotics.motion.navigate_to_location@1",
            intent_ids=["route.navigate.location"],
            affordances=["motion.navigate.semantic_location"],
            effects=["robot.location.reached"],
        ),
        _manifest(
            "safe_stop",
            canonical_id="robotics.safety.safe_stop@1",
            control_class="safety",
            effects=["robot.motion.stopped"],
        ),
        _manifest(
            "navigate",
            canonical_id="robotics.motion.navigate@1",
            intent_ids=["route.navigate.pose"],
            affordances=["motion.navigate.pose"],
            effects=["robot.pose.reached"],
        ),
        _manifest("dwell", control_class="timed"),
    ]
    manifests[-1]["discovery_score"] = 100.0
    frame = {
        "contract_version": "flyto.goal-frame.v1",
        "intent_ids": ["route.navigate.location"],
        "required_affordances": ["motion.navigate.semantic_location"],
        "desired_effects": ["robot.location.reached", "robot.motion.stopped"],
        "trigger_events": [],
        "constraints": [
            {
                "key": "target.location_id",
                "operator": "equals",
                "value": "hospital.nurse_station.1",
            }
        ],
    }

    route = route_capabilities(
        "input language is irrelevant",
        manifests,
        goal_frame=frame,
        limit=4,
        blueprint_candidates=[
            {
                "id": "official.unrelated",
                "trust_tier": "official",
                "module_ids": ["navigate"],
            }
        ],
    )

    assert [item["runtime_name"] for item in route["candidates"]] == [
        "navigate_to_location",
        "safe_stop",
    ]
    assert route["semantic_coverage"]["ratio"] == 1.0
    assert route["needs_clarification"] is False


def test_goal_frame_request_exposes_ontology_not_language_specific_aliases() -> None:
    manifests = [
        _manifest(
            "follow_line",
            aliases=["循線", "follow line"],
            intent_ids=["route.follow.sequence"],
            affordances=["motion.follow.visual_line"],
            effects=["route.sequence.completed"],
            handled_events=["line.detected"],
        )
    ]

    request = goal_frame_request("任何語言的目標", manifests)

    assert request["contract_version"] == "flyto.goal-frame-request.v1"
    assert request["vocabulary"] == {
        "intent_ids": ["route.follow.sequence"],
        "affordances": ["motion.follow.visual_line"],
        "effects": ["route.sequence.completed"],
        "events": ["line.detected"],
    }
    assert "aliases" not in request["vocabulary"]


def test_runtime_constraints_are_hard_filters_not_prompt_hints() -> None:
    manifests = [
        _manifest(
            "follow_line",
            aliases=["藍線"],
            observations=["camera.line_scene", "minimum_range"],
        ),
        _manifest("navigate", aliases=["導航"], observations=["odometry"]),
    ]

    route = route_capabilities(
        "沿藍線前進",
        manifests,
        context={"available_observations": ["odometry"]},
    )

    assert [candidate["runtime_name"] for candidate in route["candidates"]] == [
        "navigate"
    ]
    assert route["excluded"] == [
        {"runtime_name": "follow_line", "reasons": ["missing_observation"]}
    ]


def test_only_trusted_blueprint_experience_can_boost_a_candidate() -> None:
    manifests = [
        _manifest("alpha", control_class="timed"),
        _manifest("beta", control_class="timed"),
    ]
    community = {
        "id": "community",
        "trust_tier": "community",
        "steps": [{"module": "beta"}],
    }
    official = {
        "id": "official",
        "trust_tier": "official",
        "module_ids": ["beta"],
    }
    lower_ranked_official = {
        "id": "official.second",
        "trust_tier": "official",
        "module_ids": ["alpha"],
    }

    untrusted_route = route_capabilities(
        "perform task",
        manifests,
        blueprint_candidates=[community],
    )
    trusted_route = route_capabilities(
        "perform task",
        manifests,
        blueprint_candidates=[official, lower_ranked_official],
    )

    assert untrusted_route["candidates"][0]["runtime_name"] == "alpha"
    assert trusted_route["candidates"][0]["runtime_name"] == "beta"
    assert "trusted_blueprint_hint" in trusted_route["candidates"][0]["reasons"]


@pytest.mark.asyncio
async def test_flyto_integration_uses_blueprint_and_core_bridges_but_keeps_scope() -> (
    None
):
    calls: list[str] = []

    async def fake_core_dispatch(
        name: str,
        _arguments: dict[str, object],
    ) -> dict[str, object]:
        calls.append(name)
        if name == "search_modules":
            return {
                "total": 1,
                "results": [
                    {
                        "module_id": "browser.robots",
                        "label": "Check robots.txt",
                        "description": "Unrelated web robot rule checker.",
                        "category": "browser",
                        "score": 99.0,
                    }
                ],
            }
        return {
            "contract_version": "flyto-core-mcp.v1",
            "core_version": "2.test",
            "tool_fingerprint": "sha256:core-test",
        }

    robot_manifests = [
        _manifest("follow_line", aliases=["藍線", "循線"]),
        _manifest(
            "safe_stop",
            aliases=["安全停止"],
            control_class="safety",
        ),
    ]
    decision = await route_with_flyto(
        "沿藍線前進並安全停止",
        robot_manifests,
        core_dispatch=fake_core_dispatch,
        blueprint_search=lambda _goal: [
            {
                "id": "official.robot.route",
                "trust_tier": "official",
                "module_ids": ["follow_line"],
            }
        ],
    )

    assert set(calls) == {"search_modules", "get_core_capability_manifest"}
    names = {candidate["runtime_name"] for candidate in decision["route"]["candidates"]}
    assert names == {"follow_line", "safe_stop"}
    # An ordinary Core module declares no capability, so it is never projected
    # into a manifest at all and cannot appear as a candidate or an exclusion.
    assert all(
        item["runtime_name"] != "browser.robots"
        for item in [
            *decision["route"]["candidates"],
            *decision["route"]["excluded"],
        ]
    )
    assert decision["discovery_evidence"]["core"]["candidate_count"] == 0
    assert decision["discovery_evidence"]["blueprint"]["trusted_module_hints"] == [
        "follow_line"
    ]
    assert decision["discovery_evidence"]["core"]["used_bridge"].endswith(
        "dispatch_core_tool"
    )


@pytest.mark.asyncio
async def test_prepare_planner_request_replaces_catalog_with_verified_shortlist() -> (
    None
):
    async def fake_core_dispatch(
        name: str,
        _arguments: dict[str, object],
    ) -> dict[str, object]:
        if name == "search_modules":
            return {"total": 0, "results": []}
        return {"contract_version": "flyto-core-mcp.v1"}

    manifests = [
        _manifest(
            "follow_line",
            aliases=["藍線"],
            intent_ids=["route.follow.sequence"],
            affordances=["motion.follow.visual_line"],
            effects=["route.sequence.completed"],
        ),
        _manifest(
            "safe_stop",
            aliases=["停止"],
            control_class="safety",
            effects=["robot.motion.stopped"],
        ),
        *[_manifest(f"noise_{index}", control_class="timed") for index in range(20)],
    ]
    prepared = await prepare_planner_request(
        {
            "planner_contract": "flyto.robotics.planner-request.v1",
            "goal": "沿藍線前進後停止",
            "goal_frame": {
                "contract_version": "flyto.goal-frame.v1",
                "intent_ids": ["route.follow.sequence"],
                "required_affordances": ["motion.follow.visual_line"],
                "desired_effects": [
                    "robot.motion.stopped",
                    "route.sequence.completed",
                ],
                "trigger_events": [],
                "constraints": [],
            },
            "robot_id": "robot.test",
            "capabilities": manifests,
            "observations": {},
        },
        limit=4,
        core_dispatch=fake_core_dispatch,
        blueprint_search=lambda _goal: [],
    )

    assert [capability["runtime_name"] for capability in prepared["capabilities"]] == [
        "follow_line",
        "safe_stop",
    ]
    assert prepared["capability_route"]["contract_version"] == (
        "flyto.capability-route.v1"
    )
    assert prepared["flyto_routing_decision"]["contract_version"] == (
        "flyto.capability-routing-decision.v1"
    )


def test_invalid_policy_fails_closed() -> None:
    with pytest.raises(CapabilityRoutingError, match="limit"):
        route_capabilities("task", [_manifest("one")], limit=0)

    with pytest.raises(CapabilityRoutingError, match="contract_version"):
        route_capabilities(
            "task",
            [_manifest("one")],
            goal_frame={"contract_version": "unknown"},
        )


@pytest.mark.asyncio
async def test_production_policy_can_require_language_neutral_goal_frame() -> None:
    with pytest.raises(CapabilityRoutingError, match="requires flyto.goal-frame.v1"):
        await prepare_planner_request(
            {
                "planner_contract": "flyto.robotics.planner-request.v1",
                "goal": "任意語言",
                "robot_id": "robot.test",
                "capabilities": [_manifest("one")],
            },
            require_goal_frame=True,
        )


def _core_dispatch(
    *,
    search: dict[str, object] | None = None,
    manifest: dict[str, object] | None = None,
    raises: BaseException | None = None,
):
    search_result = {"total": 0, "results": []} if search is None else search
    manifest_result = (
        {
            "ok": True,
            "contract_version": CORE_RUNTIME_CONTRACT,
            "core_version": "2.test",
        }
        if manifest is None
        else manifest
    )

    async def dispatch(name: str, _arguments: dict[str, object]) -> object:
        if name == "search_modules":
            if raises is not None:
                raise raises
            return search_result
        return manifest_result

    return dispatch


def _planner_request() -> dict[str, object]:
    return {
        "planner_contract": "flyto.robotics.planner-request.v1",
        "goal": "沿藍線前進後停止",
        "robot_id": "robot.test",
        "capabilities": [
            _manifest("follow_line", aliases=["藍線"]),
            _manifest("safe_stop", aliases=["停止"], control_class="safety"),
        ],
        "observations": {},
    }


def test_discovery_status_vocabulary_is_exactly_four_bounded_values() -> None:
    assert DISCOVERY_STATUSES == (
        "applied",
        "not_applicable",
        "unavailable",
        "failed",
    )


@pytest.mark.asyncio
async def test_discovery_evidence_marks_both_lanes_applied_on_real_matches() -> None:
    decision = await route_with_flyto(
        "沿藍線前進並安全停止",
        [_manifest("follow_line", aliases=["藍線"])],
        core_dispatch=_core_dispatch(
            search={
                "total": 1,
                "results": [
                    {
                        "module_id": "browser.robots",
                        "category": "browser",
                        "provides_capability": "web.robots_txt.inspect@1",
                        "plugin": "flyto-browser",
                        "score": 12.0,
                    }
                ],
            }
        ),
        blueprint_search=lambda _goal: [
            {
                "id": "official.robot.route",
                "trust_tier": "official",
                "module_ids": ["follow_line"],
            }
        ],
    )

    evidence = decision["discovery_evidence"]
    assert evidence["blueprint"]["status"] == "applied"
    assert evidence["blueprint"]["status_reason"] == "search_matched"
    assert evidence["blueprint"]["candidate_count"] == 1
    assert evidence["core"]["status"] == "applied"
    assert evidence["core"]["status_reason"] == "discovery_matched"
    assert evidence["core"]["candidate_count"] == 1


@pytest.mark.asyncio
async def test_clean_zero_result_discovery_is_not_applicable_not_a_failure() -> None:
    decision = await route_with_flyto(
        "沿藍線前進",
        [_manifest("follow_line", aliases=["藍線"])],
        core_dispatch=_core_dispatch(),
        blueprint_search=lambda _goal: [],
    )

    evidence = decision["discovery_evidence"]
    assert evidence["blueprint"]["status"] == "not_applicable"
    assert evidence["blueprint"]["status_reason"] == "search_empty"
    assert evidence["core"]["status"] == "not_applicable"
    assert evidence["core"]["status_reason"] == "discovery_empty"
    # A legitimate no-match must still produce a usable route.
    assert decision["route"]["candidates"]


@pytest.mark.asyncio
async def test_absent_blueprint_package_is_unavailable_not_an_empty_no_match(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "flyto_blueprint", None)

    decision = await route_with_flyto(
        "沿藍線前進",
        [_manifest("follow_line", aliases=["藍線"])],
        core_dispatch=_core_dispatch(),
    )

    blueprint = decision["discovery_evidence"]["blueprint"]
    assert blueprint["status"] == "unavailable"
    assert blueprint["status_reason"] == "engine_unavailable"
    assert blueprint["candidate_count"] == 0
    assert blueprint["trusted_module_hints"] == []


@pytest.mark.asyncio
async def test_blueprint_search_exception_is_failed_and_never_leaks_raw_text() -> None:
    def exploding_search(_goal: str):
        raise RuntimeError("blueprint-internal-secret-path")

    decision = await route_with_flyto(
        "沿藍線前進",
        [_manifest("follow_line", aliases=["藍線"])],
        core_dispatch=_core_dispatch(),
        blueprint_search=exploding_search,
    )

    blueprint = decision["discovery_evidence"]["blueprint"]
    assert blueprint["status"] == "failed"
    assert blueprint["status_reason"] == "search_failed"
    assert "blueprint-internal-secret-path" not in repr(decision)


@pytest.mark.asyncio
async def test_invalid_blueprint_result_shape_is_failed() -> None:
    decision = await route_with_flyto(
        "沿藍線前進",
        [_manifest("follow_line", aliases=["藍線"])],
        core_dispatch=_core_dispatch(),
        blueprint_search=lambda _goal: ["not-a-blueprint-object"],
    )

    blueprint = decision["discovery_evidence"]["blueprint"]
    assert blueprint["status"] == "failed"
    assert blueprint["status_reason"] == "invalid_result"
    assert blueprint["candidate_count"] == 0


@pytest.mark.asyncio
async def test_over_bound_blueprint_result_is_failed_not_truncated_evidence() -> None:
    oversized = [
        {"id": f"official.{index}", "trust_tier": "official", "module_ids": []}
        for index in range(BLUEPRINT_CANDIDATE_LIMIT + 1)
    ]

    decision = await route_with_flyto(
        "沿藍線前進",
        [_manifest("follow_line", aliases=["藍線"])],
        core_dispatch=_core_dispatch(),
        blueprint_search=lambda _goal: oversized,
    )

    blueprint = decision["discovery_evidence"]["blueprint"]
    assert blueprint["status"] == "failed"
    assert blueprint["status_reason"] == "result_over_bound"
    assert blueprint["candidate_count"] == 0
    assert blueprint["trusted_module_hints"] == []


@pytest.mark.asyncio
async def test_blueprint_result_exactly_at_the_documented_bound_is_applied() -> None:
    at_bound = [
        {"id": f"official.{index}", "trust_tier": "official", "module_ids": []}
        for index in range(BLUEPRINT_CANDIDATE_LIMIT)
    ]

    decision = await route_with_flyto(
        "沿藍線前進",
        [_manifest("follow_line", aliases=["藍線"])],
        core_dispatch=_core_dispatch(),
        blueprint_search=lambda _goal: at_bound,
    )

    blueprint = decision["discovery_evidence"]["blueprint"]
    assert blueprint["status"] == "applied"
    assert blueprint["candidate_count"] == BLUEPRINT_CANDIDATE_LIMIT


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("manifest_result", "expected_status", "expected_reason"),
    [
        (
            {"ok": False, "contract_version": CORE_RUNTIME_CONTRACT},
            "unavailable",
            "runtime_unavailable",
        ),
        ({}, "unavailable", "runtime_missing"),
        (
            {"contract_version": CORE_RUNTIME_CONTRACT},
            "unavailable",
            "runtime_missing",
        ),
        (
            {"ok": True, "contract_version": "flyto-core-mcp.v2"},
            "unavailable",
            "runtime_contract_mismatch",
        ),
        (
            {"ok": True, "contract_version": ""},
            "unavailable",
            "runtime_contract_mismatch",
        ),
        (
            {"ok": "true", "contract_version": CORE_RUNTIME_CONTRACT},
            "failed",
            "runtime_malformed",
        ),
        (
            {"ok": 1, "contract_version": CORE_RUNTIME_CONTRACT},
            "failed",
            "runtime_malformed",
        ),
        ({"ok": True, "contract_version": 1}, "failed", "runtime_malformed"),
    ],
)
async def test_core_runtime_evidence_requires_exact_contract_and_explicit_ok(
    manifest_result: dict[str, object],
    expected_status: str,
    expected_reason: str,
) -> None:
    decision = await route_with_flyto(
        "沿藍線前進",
        [_manifest("follow_line", aliases=["藍線"])],
        core_dispatch=_core_dispatch(manifest=manifest_result),
        blueprint_search=lambda _goal: [],
    )

    core = decision["discovery_evidence"]["core"]
    assert core["status"] == expected_status
    assert core["status_reason"] == expected_reason
    assert core["used_bridge"].endswith("dispatch_core_tool")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "search_result",
    [
        {"total": 0, "results": "not-a-list"},
        {"ok": False, "results": []},
        {"ok": "yes", "results": []},
        {"total": 0},
        # A malformed entry must fail the lane, not be filtered into no-match.
        {"total": 1, "results": ["not-an-object"]},
        {"total": 1, "results": [{"label": "no module id"}]},
        {"total": 1, "results": [{"module_id": ""}]},
        {"total": 1, "results": [{"module_id": "unsafe id!"}]},
        {"total": 1, "results": [{"module_id": 42}]},
        {
            "total": 2,
            "results": [{"module_id": "browser.robots"}, {"module_id": None}],
        },
    ],
)
async def test_malformed_core_search_response_is_failed(
    search_result: dict[str, object],
) -> None:
    decision = await route_with_flyto(
        "沿藍線前進",
        [_manifest("follow_line", aliases=["藍線"])],
        core_dispatch=_core_dispatch(search=search_result),
        blueprint_search=lambda _goal: [],
    )

    core = decision["discovery_evidence"]["core"]
    assert core["status"] == "failed"
    assert core["status_reason"] == "search_malformed"


@pytest.mark.asyncio
async def test_core_bridge_exception_becomes_bounded_typed_routing_error() -> None:
    with pytest.raises(CapabilityRoutingError) as excinfo:
        await route_with_flyto(
            "沿藍線前進",
            [_manifest("follow_line", aliases=["藍線"])],
            core_dispatch=_core_dispatch(
                raises=RuntimeError("core-internal-secret-token")
            ),
            blueprint_search=lambda _goal: [],
        )

    assert "core-internal-secret-token" not in str(excinfo.value)
    assert str(excinfo.value) == "flyto-core discovery bridge call failed"


@pytest.mark.asyncio
async def test_strict_discovery_rejects_unresolved_lane_before_provider_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "flyto_blueprint", None)

    with pytest.raises(CapabilityRoutingError) as excinfo:
        await prepare_planner_request(
            _planner_request(),
            limit=4,
            require_discovery=True,
            core_dispatch=_core_dispatch(),
        )

    message = str(excinfo.value)
    assert "requires resolved Flyto discovery" in message
    assert "blueprint=unavailable" in message
    assert "core=not_applicable" in message


@pytest.mark.asyncio
async def test_strict_discovery_accepts_applied_and_not_applicable_lanes() -> None:
    prepared = await prepare_planner_request(
        _planner_request(),
        limit=4,
        require_discovery=True,
        core_dispatch=_core_dispatch(),
        blueprint_search=lambda _goal: [],
    )

    evidence = prepared["flyto_routing_decision"]["discovery_evidence"]
    assert evidence["blueprint"]["status"] == "not_applicable"
    assert evidence["core"]["status"] == "not_applicable"
    assert [item["runtime_name"] for item in prepared["capabilities"]] == [
        "follow_line",
        "safe_stop",
    ]


def _core_module(
    module_id: str,
    *,
    provides_capability: object = "",
    plugin: object = "",
    category: str = "vision",
    score: float = 10.0,
) -> dict[str, object]:
    """Build one `search_modules` result exactly as flyto-core now returns it."""
    return {
        "module_id": module_id,
        "label": module_id,
        "description": f"Registry module {module_id}.",
        "category": category,
        "provides_capability": provides_capability,
        "plugin": plugin,
        "score": score,
    }


@pytest.mark.asyncio
async def test_core_capability_id_is_the_registry_value_and_is_never_derived() -> None:
    decision = await route_with_flyto(
        "detect objects and follow the line",
        [_manifest("follow_line")],
        core_dispatch=_core_dispatch(
            search={
                "total": 1,
                "results": [
                    _core_module(
                        "vision.detect_objects",
                        provides_capability="robotics.vision.detect_objects@2",
                        plugin="flyto-robotics-vision",
                    )
                ],
            }
        ),
        blueprint_search=lambda _goal: [],
    )

    candidates = decision["route"]["candidates"]
    provider = next(
        item for item in candidates if item["runtime_name"] == "vision.detect_objects"
    )
    # Exactly the declared registry ID: not derived from module_id, category,
    # a prefix, or the plugin name.
    assert provider["canonical_id"] == "robotics.vision.detect_objects@2"
    assert provider["plugin"] == "flyto-robotics-vision"
    assert provider["source"] == "flyto-core"
    assert "core.module." not in repr(decision)
    assert "capability.vision.detect_objects@1" not in repr(decision)
    assert decision["discovery_evidence"]["core"]["status"] == "applied"
    assert decision["discovery_evidence"]["core"]["candidate_count"] == 1


@pytest.mark.asyncio
async def test_ordinary_core_modules_are_not_projected_as_capabilities() -> None:
    decision = await route_with_flyto(
        "inspect something",
        [_manifest("follow_line")],
        core_dispatch=_core_dispatch(
            search={
                "total": 2,
                "results": [
                    _core_module("string.trim", category="string"),
                    _core_module("http.request", category="http"),
                ],
            }
        ),
        blueprint_search=lambda _goal: [],
    )

    core = decision["discovery_evidence"]["core"]
    # Well-formed results that declare no capability resolve the lane cleanly.
    assert core["status"] == "not_applicable"
    assert core["status_reason"] == "no_capability_providers"
    assert core["candidate_count"] == 0
    assert [item["runtime_name"] for item in decision["route"]["candidates"]] == [
        "follow_line"
    ]


def test_core_manifest_without_declared_capability_id_is_invalid() -> None:
    route = route_capabilities(
        "inspect something",
        [
            {
                "manifest_contract": "flyto.capability-manifest.v1",
                "runtime_name": "string.trim",
                "source": "flyto-core",
                "domain": "string",
            },
            _manifest("follow_line"),
        ],
    )

    assert route["excluded"] == [
        {"runtime_name": "string.trim", "reasons": ["invalid_or_duplicate_identity"]}
    ]
    assert [item["runtime_name"] for item in route["candidates"]] == ["follow_line"]
    assert "core.module." not in repr(route)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "provides_capability",
    ["unsafe id!", "robotics/vision@1", "-leading.dash@1", "x" * 200],
)
async def test_unsafe_declared_capability_id_is_excluded_never_repaired(
    provides_capability: str,
) -> None:
    decision = await route_with_flyto(
        "detect objects",
        [_manifest("follow_line")],
        core_dispatch=_core_dispatch(
            search={
                "total": 1,
                "results": [
                    _core_module(
                        "vision.detect_objects",
                        provides_capability=provides_capability,
                    )
                ],
            }
        ),
        blueprint_search=lambda _goal: [],
    )

    core = decision["discovery_evidence"]["core"]
    assert core["candidate_count"] == 0
    assert core["status"] == "not_applicable"
    assert all(
        item["runtime_name"] != "vision.detect_objects"
        for item in decision["route"]["candidates"]
    )
    assert provides_capability not in repr(decision["route"]["candidates"])


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "result_item",
    [
        {"module_id": "vision.detect", "provides_capability": 42},
        {"module_id": "vision.detect", "provides_capability": None},
        {"module_id": "vision.detect", "provides_capability": ["a"]},
        {
            "module_id": "vision.detect",
            "provides_capability": "v.detect@1",
            "plugin": 7,
        },
        {"module_id": "vision.detect", "plugin": None},
    ],
)
async def test_malformed_identity_field_types_fail_the_search_contract(
    result_item: dict[str, object],
) -> None:
    decision = await route_with_flyto(
        "detect objects",
        [_manifest("follow_line")],
        core_dispatch=_core_dispatch(search={"total": 1, "results": [result_item]}),
        blueprint_search=lambda _goal: [],
    )

    core = decision["discovery_evidence"]["core"]
    assert core["status"] == "failed"
    assert core["status_reason"] == "search_malformed"
    assert core["candidate_count"] == 0


@pytest.mark.asyncio
async def test_two_modules_may_provide_one_capability_and_stay_distinguishable() -> (
    None
):
    decision = await route_with_flyto(
        "detect objects",
        [_manifest("follow_line")],
        core_dispatch=_core_dispatch(
            search={
                "total": 2,
                "results": [
                    _core_module(
                        "vision.detect_objects_yolo",
                        provides_capability="robotics.vision.detect_objects@2",
                        plugin="flyto-vision-yolo",
                    ),
                    _core_module(
                        "vision.detect_objects_depth",
                        provides_capability="robotics.vision.detect_objects@2",
                        plugin="flyto-vision-depth",
                    ),
                ],
            }
        ),
        blueprint_search=lambda _goal: [],
    )

    core = decision["discovery_evidence"]["core"]
    assert core["candidate_count"] == 2
    providers = [
        (item["canonical_id"], item["runtime_name"], item["plugin"])
        for item in decision["route"]["candidates"]
        if item["canonical_id"] == "robotics.vision.detect_objects@2"
    ]
    # One capability, two provider identities: neither is rejected as a
    # duplicate, and the order is a deterministic identity tiebreak.
    assert providers == [
        (
            "robotics.vision.detect_objects@2",
            "vision.detect_objects_depth",
            "flyto-vision-depth",
        ),
        (
            "robotics.vision.detect_objects@2",
            "vision.detect_objects_yolo",
            "flyto-vision-yolo",
        ),
    ]
    assert decision["route"]["excluded"] == []


def test_identical_provider_identity_is_still_rejected_as_a_duplicate() -> None:
    provider = {
        "manifest_contract": "flyto.capability-manifest.v1",
        "canonical_id": "robotics.vision.detect_objects@2",
        "runtime_name": "vision.detect_objects",
        "plugin": "flyto-vision",
        "source": "flyto-core",
        "domain": "vision",
    }

    route = route_capabilities("detect objects", [dict(provider), dict(provider)])

    assert [item["runtime_name"] for item in route["candidates"]] == [
        "vision.detect_objects"
    ]
    assert route["excluded"] == [
        {
            "runtime_name": "vision.detect_objects",
            "reasons": ["invalid_or_duplicate_identity"],
        }
    ]


@pytest.mark.asyncio
async def test_explicit_allowed_sources_stays_a_ceiling_default_scope_admits_providers() -> (
    None
):
    def dispatch():
        return _core_dispatch(
            search={
                "total": 1,
                "results": [
                    _core_module(
                        "vision.detect_objects",
                        provides_capability="robotics.vision.detect_objects@2",
                        plugin="flyto-vision",
                    )
                ],
            }
        )

    default_scope = await route_with_flyto(
        "detect objects",
        [_manifest("follow_line")],
        core_dispatch=dispatch(),
        blueprint_search=lambda _goal: [],
    )
    explicit_scope = await route_with_flyto(
        "detect objects",
        [_manifest("follow_line")],
        context={"allowed_sources": ["flyto-robotics"]},
        core_dispatch=dispatch(),
        blueprint_search=lambda _goal: [],
    )

    assert "flyto-core" in default_scope["route"]["routing_context"]["allowed_sources"]
    assert any(
        item["runtime_name"] == "vision.detect_objects"
        for item in default_scope["route"]["candidates"]
    )
    # An explicit ceiling is preserved exactly and is never widened by discovery.
    assert explicit_scope["route"]["routing_context"]["allowed_sources"] == [
        "flyto-robotics"
    ]
    assert any(
        item["runtime_name"] == "vision.detect_objects"
        and "source_out_of_scope" in item["reasons"]
        for item in explicit_scope["route"]["excluded"]
    )


@pytest.mark.asyncio
async def test_planner_receives_selected_manifests_by_full_provider_identity() -> None:
    request = {
        "planner_contract": "flyto.robotics.planner-request.v1",
        "goal": "detect objects",
        "robot_id": "robot.test",
        "capabilities": [
            # Same runtime name as the discovered provider, but a different
            # capability identity, and hard-filtered out of the route.
            _manifest(
                "vision.detect_objects",
                canonical_id="robotics.legacy.detect@1",
                observations=["camera.depth"],
            )
        ],
        "observations": {},
    }

    prepared = await prepare_planner_request(
        request,
        context={"available_observations": []},
        limit=4,
        core_dispatch=_core_dispatch(
            search={
                "total": 1,
                "results": [
                    _core_module(
                        "vision.detect_objects",
                        provides_capability="robotics.vision.detect_objects@2",
                        plugin="flyto-vision",
                    )
                ],
            }
        ),
        blueprint_search=lambda _goal: [],
    )

    assert len(prepared["capabilities"]) == 1
    selected = prepared["capabilities"][0]
    # The unselected same-named request manifest must not inherit authority.
    assert selected["canonical_id"] == "robotics.vision.detect_objects@2"
    assert selected["provides_capability"] == "robotics.vision.detect_objects@2"
    assert selected["plugin"] == "flyto-vision"
    assert selected["source"] == "flyto-core"
    assert all(
        item["canonical_id"] != "robotics.legacy.detect@1"
        for item in prepared["capability_route"]["candidates"]
    )


@pytest.mark.asyncio
async def test_default_prepare_planner_request_stays_compatible_when_lane_fails() -> (
    None
):
    def exploding_search(_goal: str):
        raise RuntimeError("boom")

    prepared = await prepare_planner_request(
        _planner_request(),
        limit=4,
        core_dispatch=_core_dispatch(manifest={}),
        blueprint_search=exploding_search,
    )

    evidence = prepared["flyto_routing_decision"]["discovery_evidence"]
    assert evidence["blueprint"]["status"] == "failed"
    assert evidence["core"]["status"] == "unavailable"
    # Default stays non-strict for existing library callers.
    assert [item["runtime_name"] for item in prepared["capabilities"]] == [
        "follow_line",
        "safe_stop",
    ]


def _candidate(manifest: dict[str, object]) -> dict[str, object]:
    """Build the route candidate a manifest would produce, as the router does."""
    canonical_id = str(manifest["canonical_id"])
    return {
        "canonical_id": canonical_id,
        "runtime_name": str(manifest["runtime_name"]),
        "plugin": str(manifest.get("plugin", "")),
        "source": str(
            manifest.get(
                "source",
                "flyto-robotics"
                if canonical_id.startswith("robotics.")
                else "external",
            )
        ),
        "score": 1.0,
        "reasons": ["identifier_match"],
        "selected_by": "deterministic_hybrid_v1",
    }


def test_selected_candidate_that_resolves_to_no_manifest_fails_closed() -> None:
    catalog = [_manifest("follow_line")]
    # Same capability ID and runtime name, different plugin: a partial identity
    # match must not be accepted, and the candidate must not be dropped either.
    mismatched = _candidate(_manifest("follow_line"))
    mismatched["plugin"] = "flyto-somewhere-else"

    with pytest.raises(CapabilityRoutingError, match="unresolved=1") as excinfo:
        _selected_manifests([_candidate(_manifest("follow_line")), mismatched], catalog)

    # Bounded message: counts only, no catalog-controlled identity text.
    assert "flyto-somewhere-else" not in str(excinfo.value)
    assert "ambiguous=0" in str(excinfo.value)


def test_selected_candidate_matching_two_manifests_fails_closed() -> None:
    duplicate = _manifest("follow_line")
    # Two catalog entries share one full provider identity, so route order alone
    # would decide which manifest inherits execution authority.
    catalog = [dict(duplicate), {**duplicate, "required_permissions": ["danger.full"]}]

    with pytest.raises(CapabilityRoutingError, match="ambiguous=1"):
        _selected_manifests([_candidate(duplicate)], catalog)


def test_resolved_candidates_keep_route_order_and_exact_manifests() -> None:
    follow_line = _manifest("follow_line")
    safe_stop = _manifest("safe_stop", control_class="safety")
    catalog = [safe_stop, follow_line]

    resolved = _selected_manifests(
        [_candidate(follow_line), _candidate(safe_stop)],
        catalog,
    )

    assert resolved == [follow_line, safe_stop]


@pytest.mark.asyncio
async def test_planner_propagates_two_providers_of_one_capability_in_order() -> None:
    prepared = await prepare_planner_request(
        {
            "planner_contract": "flyto.robotics.planner-request.v1",
            "goal": "detect objects",
            "robot_id": "robot.test",
            "capabilities": [],
            "observations": {},
        },
        limit=4,
        core_dispatch=_core_dispatch(
            search={
                "total": 2,
                "results": [
                    _core_module(
                        "vision.detect_objects_yolo",
                        provides_capability="robotics.vision.detect_objects@2",
                        plugin="flyto-vision-yolo",
                    ),
                    _core_module(
                        "vision.detect_objects_depth",
                        provides_capability="robotics.vision.detect_objects@2",
                        plugin="flyto-vision-depth",
                    ),
                ],
            }
        ),
        blueprint_search=lambda _goal: [],
    )

    # Both co-providers survive planner propagation, stay distinguishable, and
    # keep the deterministic full-identity order of the route candidates.
    assert [
        (item["runtime_name"], item["plugin"]) for item in prepared["capabilities"]
    ] == [
        ("vision.detect_objects_depth", "flyto-vision-depth"),
        ("vision.detect_objects_yolo", "flyto-vision-yolo"),
    ]
    assert [
        item["runtime_name"] for item in prepared["capability_route"]["candidates"]
    ] == [item["runtime_name"] for item in prepared["capabilities"]]
    assert all(
        item["canonical_id"] == "robotics.vision.detect_objects@2"
        for item in prepared["capabilities"]
    )
