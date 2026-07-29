from __future__ import annotations

import pytest

from flyto_ai.capability_router import (
    CapabilityRoutingError,
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
    assert any(
        item["runtime_name"] == "browser.robots"
        and "source_out_of_scope" in item["reasons"]
        for item in decision["route"]["excluded"]
    )
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
