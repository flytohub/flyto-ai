from __future__ import annotations

import sys

import pytest

from flyto_ai.capability_router import (
    BLUEPRINT_CANDIDATE_LIMIT,
    CORE_RUNTIME_CONTRACT,
    DISCOVERY_STATUSES,
    CapabilityRoutingError,
    _selected_manifests,
    goal_frame_request,
    prepare_planner_request,
    route_capabilities,
    route_with_flyto,
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
    assert chinese["selection_method"] == (
        "hard_filter_then_semantic_frame_rank_v1"
    )
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
async def test_flyto_integration_uses_blueprint_and_core_bridges_but_keeps_scope() -> None:
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
    names = {
        candidate["runtime_name"]
        for candidate in decision["route"]["candidates"]
    }
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
    assert decision["discovery_evidence"]["blueprint"][
        "trusted_module_hints"
    ] == ["follow_line"]
    assert decision["discovery_evidence"]["core"]["used_bridge"].endswith(
        "dispatch_core_tool"
    )


@pytest.mark.asyncio
async def test_prepare_planner_request_replaces_catalog_with_verified_shortlist() -> None:
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

    assert [
        capability["runtime_name"]
        for capability in prepared["capabilities"]
    ] == ["follow_line", "safe_stop"]
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
    search_result = (
        {"total": 0, "results": []} if search is None else search
    )
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
        {"module_id": "vision.detect", "provides_capability": "v.detect@1", "plugin": 7},
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
async def test_two_modules_may_provide_one_capability_and_stay_distinguishable() -> None:
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
                "flyto-robotics" if canonical_id.startswith("robotics.") else "external",
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
        (item["runtime_name"], item["plugin"])
        for item in prepared["capabilities"]
    ] == [
        ("vision.detect_objects_depth", "flyto-vision-depth"),
        ("vision.detect_objects_yolo", "flyto-vision-yolo"),
    ]
    assert [
        item["runtime_name"] for item in prepared["capability_route"]["candidates"]
    ] == [
        item["runtime_name"] for item in prepared["capabilities"]
    ]
    assert all(
        item["canonical_id"] == "robotics.vision.detect_objects@2"
        for item in prepared["capabilities"]
    )
