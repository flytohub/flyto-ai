from __future__ import annotations

import copy
import json
from dataclasses import FrozenInstanceError, replace

import pytest

from flyto_ai.execution_session import (
    ExecutionAuthority,
    ExecutionSessionError,
    EXECUTION_SESSION_REQUEST_VERSION,
    EXECUTION_SESSION_RESULT_VERSION,
    _MAX_CATALOG_NODES,
    _MAX_JSON_DEPTH,
    _MAX_TIMESTAMP_MS,
    prepare_execution_session,
)


def _manifest(
    name: str = "save_response",
    *,
    source: str = "catalog.alpha",
    permission: str = "records.write",
) -> dict[str, object]:
    return {
        "manifest_contract": "flyto.capability-manifest.v1",
        "canonical_id": f"data.workflow.{name}@1",
        "runtime_name": name,
        "name": name,
        "version": "1.0.0",
        "source": source,
        "domain": "data.workflow",
        "description": "Save an already supplied response.",
        "control_class": "data",
        "required_permissions": [permission],
        "intent_ids": ["response.save"],
        "affordances": ["record.write"],
        "effects": ["response.persisted"],
    }


def _authority(**changes: object) -> ExecutionAuthority:
    values = {
        "tenant_id": "tenant.a",
        "principal_id": "principal.7",
        "verified": True,
        "allowed_sources": ("catalog.alpha",),
        "allowed_domains": ("data.workflow",),
        "granted_permissions": ("records.write",),
        "enabled_capabilities": ("save_response",),
    }
    values.update(changes)
    return ExecutionAuthority(**values)  # type: ignore[arg-type]


def _request(*, source: str = "typed", wake: object = None) -> dict[str, object]:
    return {
        "contract_version": EXECUTION_SESSION_REQUEST_VERSION,
        "session_id": "session.1",
        "space": {
            "space_id": "space.lobby",
            "display_name": "Flyto Front Desk",
            "wake_words": ["小飛", "Fly To"],
            "active_timeout_ms": 30_000,
        },
        "activation": {
            "source": source,
            "observed_wake_word": wake,
            "activated_at_ms": 10_000,
            "expires_at_ms": 40_000,
        },
        "goal": {
            "text": "Save the supplied response",
            "frame": {
                "contract_version": "flyto.goal-frame.v1",
                "intent_ids": ["response.save"],
                "required_affordances": ["record.write"],
                "desired_effects": ["response.persisted"],
                "trigger_events": [],
                "constraints": [],
            },
        },
    }


def test_valid_session_is_bounded_attested_json_and_detached() -> None:
    request = _request()
    manifest = _manifest()
    manifests = [manifest]
    blueprint = {"id": "trusted.one", "trust_tier": "official", "module_ids": ["save_response"]}
    blueprints = [blueprint]
    authority_sources = ["catalog.alpha"]
    authority = _authority(allowed_sources=authority_sources)
    result = prepare_execution_session(
        request, manifests, authority, 20_000, trusted_blueprints=blueprints
    )

    assert request["contract_version"] == EXECUTION_SESSION_REQUEST_VERSION
    assert result["contract_version"] == EXECUTION_SESSION_RESULT_VERSION
    assert EXECUTION_SESSION_REQUEST_VERSION != EXECUTION_SESSION_RESULT_VERSION
    assert json.loads(json.dumps(result, ensure_ascii=False)) == result
    assert result["planning_input"]["activation"]["observed_wake_word"] is None
    assert [item["runtime_name"] for item in result["capability_route"]["candidates"]] == [
        "save_response"
    ]
    assert result["authority"]["principal_ref"].startswith("sha256:")
    assert "principal_id" not in result["authority"]
    assert set(result["attestations"]) == {"request", "authority", "route"}
    assert result["overall_digest"].startswith("sha256:")
    with pytest.raises(FrozenInstanceError):
        _authority().tenant_id = "changed"  # type: ignore[misc]
    # Mutate every caller-owned input after return; the result must be detached.
    request["goal"]["text"] = "caller mutation"  # type: ignore[index]
    request["goal"]["frame"]["intent_ids"].append("caller.mutation")  # type: ignore[index,union-attr]
    request["space"]["wake_words"].append("mutated")  # type: ignore[index,union-attr]
    request["activation"]["expires_at_ms"] = 20_000  # type: ignore[index]
    manifest["runtime_name"] = "caller_mutation"
    manifest["required_permissions"].append("caller.mutation")  # type: ignore[union-attr]
    manifests.append(_manifest("caller_added"))
    blueprint["module_ids"] = ["caller_mutation"]
    blueprints.append({"id": "caller_added"})
    authority_sources.append("catalog.mutated")
    assert result["planning_input"]["goal"]["text"] == "Save the supplied response"
    assert result["planning_input"]["goal"]["frame"]["intent_ids"] == ["response.save"]
    assert result["planning_input"]["space"]["wake_words"] == ["小飛", "Fly To"]
    assert result["planning_input"]["activation"]["expires_at_ms"] == 40_000
    assert result["capability_route"]["candidates"][0]["runtime_name"] == "save_response"
    assert result["authority"]["allowed_sources"] == ["catalog.alpha"]


@pytest.mark.parametrize("field", ["identity", "permissions", "context", "manifests", "trusted_manifests"])
def test_each_untrusted_authority_injection_is_rejected(field: str) -> None:
    request = _request()
    request[field] = {}
    with pytest.raises(ExecutionSessionError, match="unsupported"):
        prepare_execution_session(request, [_manifest()], _authority(), 20_000)


def test_display_name_is_not_an_implicit_wake_word() -> None:
    request = _request(wake="Flyto Front Desk")
    with pytest.raises(ExecutionSessionError, match="must be null"):
        prepare_execution_session(request, [_manifest()], _authority(), 20_000)


def test_exact_empty_display_name_round_trips() -> None:
    request = _request()
    request["space"]["display_name"] = ""  # type: ignore[index]
    result = prepare_execution_session(request, [_manifest()], _authority(), 20_000)
    assert result["planning_input"]["space"]["display_name"] == ""


@pytest.mark.parametrize("display_name", ["   ", "\u00a0\u2003\u3000"])
def test_whitespace_only_display_name_fails_closed(display_name: str) -> None:
    request = _request()
    request["space"]["display_name"] = display_name  # type: ignore[index]
    with pytest.raises(ExecutionSessionError, match="bounded text"):
        prepare_execution_session(request, [_manifest()], _authority(), 20_000)


def test_non_empty_display_name_round_trips_without_collapsing_whitespace() -> None:
    request = _request()
    display_name = "  Flyto Front Desk  "
    request["space"]["display_name"] = display_name  # type: ignore[index]
    result = prepare_execution_session(request, [_manifest()], _authority(), 20_000)
    assert result["planning_input"]["space"]["display_name"] == display_name


def test_activation_does_not_grant_permission() -> None:
    authority = _authority(granted_permissions=("records.read",))
    result = prepare_execution_session(_request(), [_manifest()], authority, 20_000)
    assert result["capability_route"]["candidates"] == []
    assert result["capability_route"]["excluded"][0]["reasons"] == ["permission_denied"]


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda r: r["goal"].update({"text": "save\u0000override"}), "control"),
        (lambda r: r["activation"].update({"activated_at_ms": True}), "timestamp"),
        (lambda r: r["space"].update({"wake_words": ["Fly To", "ｆｌｙ ｔｏ"]}), "duplicates"),
        (lambda r: r["activation"].update({"activated_at_ms": float("nan")}), "non-finite"),
        (lambda r: r["activation"].update({"activated_at_ms": 20_001}), "future"),
        (lambda r: r["activation"].update({"expires_at_ms": 19_999}), "expired"),
        (lambda r: r["activation"].update({"expires_at_ms": 40_001}), "active window"),
        (lambda r: r["activation"].update({"observed_wake_word": "not-a-wake"}), "must be null"),
    ],
)
def test_untrusted_adversarial_requests_fail_closed(mutate, message: str) -> None:
    request = _request()
    mutate(request)
    with pytest.raises(ExecutionSessionError, match=message):
        prepare_execution_session(request, [_manifest()], _authority(), 20_000)


@pytest.mark.parametrize(
    "source", ["typed", "voice_reviewed", "external_agent", "mission_card"]
)
def test_canonical_activation_source_round_trips_for_voice_disabled_space(
    source: str,
) -> None:
    request = _request(source=source, wake=None)
    request["space"]["display_name"] = ""  # type: ignore[index]
    request["space"]["wake_words"] = []  # type: ignore[index]
    result = prepare_execution_session(
        request, [_manifest()], _authority(), 20_000
    )
    assert result["planning_input"]["activation"]["source"] == source
    assert result["planning_input"]["activation"]["observed_wake_word"] is None
    assert result["planning_input"]["space"]["display_name"] == ""
    assert result["planning_input"]["space"]["wake_words"] == []


@pytest.mark.parametrize(
    "source", ["typed", "voice_reviewed", "external_agent", "mission_card"]
)
def test_each_canonical_activation_source_requires_exact_null_wake_word(
    source: str,
) -> None:
    with pytest.raises(ExecutionSessionError, match="must be null"):
        prepare_execution_session(
            _request(source=source, wake="claimed wake"),
            [_manifest()],
            _authority(),
            20_000,
        )


@pytest.mark.parametrize("source", ["voice", "button", "scheduler"])
def test_raw_voice_button_and_unknown_activation_sources_fail_closed(
    source: str,
) -> None:
    with pytest.raises(ExecutionSessionError, match="not supported"):
        prepare_execution_session(
            _request(source=source, wake=None),
            [_manifest()],
            _authority(),
            20_000,
        )


def test_overall_digest_is_deterministic_and_covers_every_payload_field() -> None:
    first = prepare_execution_session(_request(), [_manifest()], _authority(), 20_000)
    second = prepare_execution_session(_request(), [_manifest()], _authority(), 20_000)
    assert first["overall_digest"] == second["overall_digest"]

    for field in (
        "contract_version",
        "planning_input",
        "capability_route",
        "authority",
        "attestations",
    ):
        tampered = copy.deepcopy(first)
        if field == "contract_version":
            tampered[field] += ".tampered"
        else:
            tampered[field]["tampered"] = True
        supplied = tampered.pop("overall_digest")
        assert supplied != _result_digest(tampered)


def _result_digest(value: object) -> str:
    import hashlib

    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


@pytest.mark.parametrize("unsafe", ["save\u200bresponse", "save\u202eresponse"])
def test_format_bidi_and_zero_width_text_is_rejected(unsafe: str) -> None:
    request = _request()
    request["goal"]["text"] = unsafe  # type: ignore[index]
    with pytest.raises(ExecutionSessionError, match="control"):
        prepare_execution_session(request, [_manifest()], _authority(), 20_000)


@pytest.mark.parametrize(
    ("activated", "expires", "accepted"),
    [(20_000, 40_000, True), (10_000, 20_000, False)],
)
def test_activation_start_is_inclusive_and_expiry_is_exclusive(
    activated: int, expires: int, accepted: bool
) -> None:
    request = _request()
    request["activation"].update(  # type: ignore[union-attr]
        {"activated_at_ms": activated, "expires_at_ms": expires}
    )
    if accepted:
        prepare_execution_session(request, [_manifest()], _authority(), 20_000)
    else:
        with pytest.raises(ExecutionSessionError, match="expired"):
            prepare_execution_session(request, [_manifest()], _authority(), 20_000)


@pytest.mark.parametrize(("window_ms", "accepted"), [(15_000, True), (15_001, False)])
def test_active_timeout_millisecond_boundary(window_ms: int, accepted: bool) -> None:
    request = _request()
    request["space"]["active_timeout_ms"] = 15_000  # type: ignore[index]
    request["activation"]["expires_at_ms"] = 10_000 + window_ms  # type: ignore[index]
    if accepted:
        prepare_execution_session(request, [_manifest()], _authority(), 20_000)
    else:
        with pytest.raises(ExecutionSessionError, match="active window"):
            prepare_execution_session(request, [_manifest()], _authority(), 20_000)


@pytest.mark.parametrize("timeout", [True, "15000", 0, 300_001])
def test_active_timeout_ms_rejects_invalid_type_and_range(timeout: object) -> None:
    request = _request()
    request["space"]["active_timeout_ms"] = timeout  # type: ignore[index]
    with pytest.raises(ExecutionSessionError, match="active_timeout_ms"):
        prepare_execution_session(request, [_manifest()], _authority(), 20_000)


def test_non_json_request_object_is_rejected() -> None:
    with pytest.raises(ExecutionSessionError, match="object"):
        prepare_execution_session(["not", "an", "object"], [_manifest()], _authority(), 20_000)  # type: ignore[arg-type]


def test_nested_non_json_request_value_is_rejected() -> None:
    request = _request()
    request["goal"]["frame"]["constraints"] = {"not-json-array"}  # type: ignore[index]
    with pytest.raises(ExecutionSessionError, match="JSON"):
        prepare_execution_session(request, [_manifest()], _authority(), 20_000)


@pytest.mark.parametrize(
    ("container", "field", "value"),
    [
        (None, "session_id", "unsafe/id"),
        ("space", "space_id", "unsafe id"),
        ("activation", "source", "unsafe/source"),
    ],
)
def test_unsafe_request_identifiers_are_rejected(container, field: str, value: str) -> None:
    request = _request()
    target = request if container is None else request[container]
    target[field] = value  # type: ignore[index]
    with pytest.raises(ExecutionSessionError, match="safe identifier"):
        prepare_execution_session(request, [_manifest()], _authority(), 20_000)


@pytest.mark.parametrize("field", ["activated_at_ms", "expires_at_ms"])
def test_boolean_activation_timestamps_are_rejected(field: str) -> None:
    request = _request()
    request["activation"][field] = True  # type: ignore[index]
    with pytest.raises(ExecutionSessionError, match="timestamp"):
        prepare_execution_session(request, [_manifest()], _authority(), 20_000)


def test_boolean_now_timestamp_is_rejected() -> None:
    with pytest.raises(ExecutionSessionError, match="timestamp"):
        prepare_execution_session(_request(), [_manifest()], _authority(), True)


def test_deep_constraint_fails_as_execution_session_error_before_recursion() -> None:
    request = _request()
    nested: object = "leaf"
    for _ in range(1_500):
        nested = [nested]
    request["goal"]["frame"]["constraints"] = [  # type: ignore[index]
        {"key": "depth", "operator": "equals", "value": nested}
    ]
    with pytest.raises(ExecutionSessionError, match="depth limit"):
        prepare_execution_session(request, [_manifest()], _authority(), 20_000)


@pytest.mark.parametrize(("nesting", "accepted"), [(27, True), (28, False)])
def test_json_depth_exact_boundary(nesting: int, accepted: bool) -> None:
    request = _request()
    nested: object = "leaf"
    for _ in range(nesting):
        nested = [nested]
    request["goal"]["frame"]["constraints"] = [  # type: ignore[index]
        {"key": "depth", "operator": "equals", "value": nested}
    ]
    assert _MAX_JSON_DEPTH == 32
    if accepted:
        prepare_execution_session(request, [_manifest()], _authority(), 20_000)
    else:
        with pytest.raises(ExecutionSessionError, match="depth limit"):
            prepare_execution_session(request, [_manifest()], _authority(), 20_000)


def test_huge_request_integer_fails_before_attestation_encoding() -> None:
    request = _request()
    request["activation"]["expires_at_ms"] = 10**5_000  # type: ignore[index]
    with pytest.raises(ExecutionSessionError, match="out-of-range integer"):
        prepare_execution_session(request, [_manifest()], _authority(), 20_000)


def test_timestamp_exact_ceiling_and_above_ceiling() -> None:
    request = _request()
    request["activation"].update(  # type: ignore[union-attr]
        {"activated_at_ms": _MAX_TIMESTAMP_MS - 1, "expires_at_ms": _MAX_TIMESTAMP_MS}
    )
    prepare_execution_session(
        request, [_manifest()], _authority(), _MAX_TIMESTAMP_MS - 1
    )
    request["activation"]["expires_at_ms"] = _MAX_TIMESTAMP_MS + 1  # type: ignore[index]
    with pytest.raises(ExecutionSessionError, match="integer timestamp"):
        prepare_execution_session(
            request, [_manifest()], _authority(), _MAX_TIMESTAMP_MS - 1
        )


def test_request_byte_ceiling_fails_closed() -> None:
    request = _request()
    request["goal"]["text"] = "x" * 262_145  # type: ignore[index]
    with pytest.raises(ExecutionSessionError, match="262144 byte limit"):
        prepare_execution_session(request, [_manifest()], _authority(), 20_000)


def test_trusted_catalog_node_exact_ceiling_and_over_ceiling() -> None:
    manifest = _manifest()
    baseline_nodes = 2  # catalog list plus manifest object
    stack = list(manifest.values())
    while stack:
        item = stack.pop()
        baseline_nodes += 1
        if isinstance(item, dict):
            stack.extend(item.values())
        elif isinstance(item, list):
            stack.extend(item)
    padding = [None] * (_MAX_CATALOG_NODES - baseline_nodes - 1)
    manifest["catalog_padding"] = padding
    prepare_execution_session(_request(), [manifest], _authority(), 20_000)
    padding.append(None)
    with pytest.raises(ExecutionSessionError, match="node limit"):
        prepare_execution_session(_request(), [manifest], _authority(), 20_000)


def test_trusted_catalog_depth_ceiling_fails_closed() -> None:
    manifest = _manifest()
    nested: object = "leaf"
    for _ in range(_MAX_JSON_DEPTH + 1):
        nested = [nested]
    manifest["catalog_padding"] = nested
    with pytest.raises(ExecutionSessionError, match="depth limit"):
        prepare_execution_session(_request(), [manifest], _authority(), 20_000)


def test_trusted_catalog_integer_ceiling_fails_closed() -> None:
    manifest = _manifest()
    manifest["catalog_padding"] = 10**5_000
    with pytest.raises(ExecutionSessionError, match="out-of-range integer"):
        prepare_execution_session(_request(), [manifest], _authority(), 20_000)


def test_trusted_catalog_byte_ceiling_fails_closed() -> None:
    manifest = _manifest()
    manifest["description"] = "x" * 8_388_609
    with pytest.raises(ExecutionSessionError, match="8388608 byte limit"):
        prepare_execution_session(_request(), [manifest], _authority(), 20_000)


@pytest.mark.parametrize(
    ("limit", "accepted"),
    [(1, True), (32, True), (True, False), ("8", False), (0, False), (33, False)],
)
def test_route_limit_is_a_strict_bounded_integer(limit: object, accepted: bool) -> None:
    if accepted:
        prepare_execution_session(
            _request(), [_manifest()], _authority(), 20_000, limit=limit  # type: ignore[arg-type]
        )
    else:
        with pytest.raises(ExecutionSessionError, match="integer from 1 to 32"):
            prepare_execution_session(
                _request(), [_manifest()], _authority(), 20_000, limit=limit  # type: ignore[arg-type]
            )


@pytest.mark.parametrize("catalog", [[], [object()], [_manifest()] * 10_001])
def test_invalid_empty_or_oversized_manifest_catalog_fails(catalog) -> None:
    with pytest.raises(ExecutionSessionError):
        prepare_execution_session(_request(), catalog, _authority(), 20_000)


@pytest.mark.parametrize(
    "authority",
    [
        lambda: _authority(verified=False),
        lambda: _authority(allowed_sources=()),
        lambda: _authority(allowed_domains=()),
        lambda: _authority(granted_permissions=()),
        lambda: _authority(enabled_capabilities=()),
    ],
)
def test_unverified_or_empty_authority_ceiling_fails(authority) -> None:
    with pytest.raises(ExecutionSessionError):
        authority()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("tenant_id", "unsafe id"),
        ("principal_id", "unsafe/id"),
        ("allowed_sources", ("unsafe source",)),
        ("allowed_domains", tuple(f"domain.{index}" for index in range(257))),
        ("granted_permissions", ("duplicate", "duplicate")),
        ("enabled_capabilities", ("unsafe/capability",)),
    ],
)
def test_authority_identifier_and_collection_bounds_fail(field: str, value: object) -> None:
    with pytest.raises(ExecutionSessionError):
        _authority(**{field: value})


def test_authority_collection_exact_bound_is_accepted() -> None:
    values = tuple(f"source.{index}" for index in range(256))
    assert len(_authority(allowed_sources=values).allowed_sources) == 256


def test_authorities_produce_distinct_routes_and_attestations() -> None:
    manifests = [_manifest(), _manifest("archive_response", source="catalog.beta")]
    first = prepare_execution_session(_request(), manifests, _authority(), 20_000)
    second_authority = replace(
        _authority(),
        principal_id="principal.8",
        allowed_sources=("catalog.beta",),
        enabled_capabilities=("archive_response",),
    )
    second = prepare_execution_session(_request(), manifests, second_authority, 20_000)
    assert first["capability_route"] != second["capability_route"]
    assert first["attestations"]["authority"] != second["attestations"]["authority"]
    assert first["attestations"]["route"] != second["attestations"]["route"]


@pytest.mark.parametrize(
    "failure",
    [
        TypeError("malformed route"),
        ValueError("malformed route"),
        RecursionError("malformed route"),
        OverflowError("malformed route"),
    ],
)
def test_route_boundary_errors_are_converted(
    monkeypatch: pytest.MonkeyPatch, failure: Exception
) -> None:
    def fail_route(*args, **kwargs):
        raise failure

    monkeypatch.setattr("flyto_ai.execution_session.route_capabilities", fail_route)
    with pytest.raises(ExecutionSessionError, match="malformed route"):
        prepare_execution_session(_request(), [_manifest()], _authority(), 20_000)


def test_malformed_non_json_route_result_is_converted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "flyto_ai.execution_session.route_capabilities",
        lambda *args, **kwargs: {"candidate": object()},
    )
    with pytest.raises(ExecutionSessionError, match="canonical JSON"):
        prepare_execution_session(_request(), [_manifest()], _authority(), 20_000)
