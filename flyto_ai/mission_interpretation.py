# Copyright 2024 Flyto2
# Licensed under the Apache License, Version 2.0
"""Bounded LLM interpretation for physical judge-drawn Mission Station cards."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .robotics_planning import StructuredJsonProvider

REQUEST_CONTRACT = "flyto.ai.mission-interpretation-request.v1"
RESPONSE_CONTRACT = "flyto.ai.mission-interpretation-response.v1"
ATTESTATION_CONTRACT = "flyto.ai.mission-interpretation-attestation.v1"
MAX_REQUEST_BYTES = 128 * 1024
MAX_CAPABILITIES = 64
EVIDENCE_KINDS = frozenset(
    {
        "zone.overview",
        "passage.clearance",
        "device.identifier",
        "arrival.pose",
        "handover.confirmation",
        "human.approval",
    }
)
FORBIDDEN_CONTROL_FIELDS = frozenset(
    {
        "angular_z",
        "cmd_vel",
        "left_wheel",
        "linear_x",
        "motor",
        "pwm",
        "right_wheel",
        "ros_topic",
        "shell",
        "topic",
        "velocity",
        "wheel_speed",
    }
)


class MissionInterpretationError(ValueError):
    """Raised when a judge-card request crosses the LLM trust boundary."""


@dataclass(frozen=True)
class ValidatedMissionRequest:
    """Normalized, card-authoritative request presented to the model as data."""

    payload: dict[str, Any]
    task_id: str
    evidence_requirements: tuple[str, ...]
    required_capability_ids: tuple[str, ...]
    approved_capability_ids: tuple[str, ...]


def validate_mission_request(value: object) -> ValidatedMissionRequest:
    """Validate card provenance, evidence, and the approved capability ceiling."""
    if not isinstance(value, Mapping):
        raise MissionInterpretationError("mission interpretation request must be an object")
    try:
        canonical_payload = _canonical(dict(value))
    except (TypeError, ValueError):
        raise MissionInterpretationError(
            "mission interpretation request must be finite JSON-compatible data"
        ) from None
    if len(canonical_payload) > MAX_REQUEST_BYTES:
        raise MissionInterpretationError("mission interpretation request is too large")
    payload: dict[str, Any] = json.loads(canonical_payload)
    _exact_fields(
        payload,
        {
            "contract_version",
            "task_id",
            "space_id",
            "card_source",
            "zone",
            "objective",
            "approved_capabilities",
            "operator_context",
        },
        "mission interpretation request",
    )
    _reject_control_fields(payload)
    if payload["contract_version"] != REQUEST_CONTRACT:
        raise MissionInterpretationError("mission interpretation contract is unsupported")
    if payload["card_source"] != "judge_draw":
        raise MissionInterpretationError("Zone and Objective cards must be drawn by a judge")
    task_id = _identifier(payload["task_id"], "task_id")
    _identifier(payload["space_id"], "space_id")
    _bounded_text(payload["operator_context"], "operator_context", 2000, allow_empty=True)

    zone = _mapping(payload["zone"], "zone")
    _exact_fields(
        zone,
        {"zone_id", "label", "marker_id", "entry_requires_approval"},
        "zone",
    )
    _identifier(zone["zone_id"], "zone.zone_id")
    _identifier(zone["marker_id"], "zone.marker_id")
    _bounded_text(zone["label"], "zone.label", 200)
    if not isinstance(zone["entry_requires_approval"], bool):
        raise MissionInterpretationError("zone.entry_requires_approval must be boolean")

    objective = _mapping(payload["objective"], "objective")
    _exact_fields(
        objective,
        {
            "objective_id",
            "title",
            "goal",
            "evidence_requirements",
            "required_capability_ids",
        },
        "objective",
    )
    _identifier(objective["objective_id"], "objective.objective_id")
    _bounded_text(objective["title"], "objective.title", 200)
    _bounded_text(objective["goal"], "objective.goal", 2000)
    evidence_requirements = _unique_identifiers(
        objective["evidence_requirements"],
        "objective.evidence_requirements",
        maximum=16,
    )
    if any(kind not in EVIDENCE_KINDS for kind in evidence_requirements):
        raise MissionInterpretationError(
            "objective.evidence_requirements contains an unsupported kind"
        )
    required_capability_ids = _unique_identifiers(
        objective["required_capability_ids"],
        "objective.required_capability_ids",
        maximum=32,
    )

    raw_capabilities = payload["approved_capabilities"]
    if not isinstance(raw_capabilities, list) or not 1 <= len(raw_capabilities) <= MAX_CAPABILITIES:
        raise MissionInterpretationError(
            f"approved_capabilities must contain 1 to {MAX_CAPABILITIES} items"
        )
    approved_ids: list[str] = []
    for index, raw_capability in enumerate(raw_capabilities):
        capability = _mapping(raw_capability, f"approved_capabilities[{index}]")
        _exact_fields(
            capability,
            {
                "capability_id",
                "executor_kind",
                "approval_status",
                "requires_safe_stop",
            },
            f"approved_capabilities[{index}]",
        )
        if capability["approval_status"] != "APPROVED":
            raise MissionInterpretationError(
                "approved_capabilities may contain only APPROVED revisions"
            )
        capability_id = _identifier(
            capability["capability_id"],
            f"approved_capabilities[{index}].capability_id",
        )
        _identifier(
            capability["executor_kind"],
            f"approved_capabilities[{index}].executor_kind",
        )
        if not isinstance(capability["requires_safe_stop"], bool):
            raise MissionInterpretationError(
                "approved capability requires_safe_stop must be boolean"
            )
        approved_ids.append(capability_id)
    if len(approved_ids) != len(set(approved_ids)):
        raise MissionInterpretationError("approved_capabilities contains duplicate IDs")
    missing = sorted(set(required_capability_ids) - set(approved_ids))
    if missing:
        raise MissionInterpretationError(
            "judge-card capabilities are not approved: " + ", ".join(missing)
        )
    return ValidatedMissionRequest(
        payload=payload,
        task_id=task_id,
        evidence_requirements=evidence_requirements,
        required_capability_ids=required_capability_ids,
        approved_capability_ids=tuple(approved_ids),
    )


def build_interpretation_schema(
    request: ValidatedMissionRequest,
) -> dict[str, Any]:
    """Expose only reading, clarification, and approved capability selection."""
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "reading",
            "selected_capability_ids",
            "needs_clarification",
            "clarification_key",
        ],
        "properties": {
            "reading": {"type": "string", "minLength": 1, "maxLength": 1000},
            "selected_capability_ids": {
                "type": "array",
                "minItems": len(request.required_capability_ids),
                "maxItems": min(len(request.approved_capability_ids), 32),
                "uniqueItems": True,
                "items": {"enum": list(request.approved_capability_ids)},
            },
            "needs_clarification": {"type": "boolean"},
            "clarification_key": {"type": "string", "maxLength": 128},
        },
    }


def validate_interpretation(
    value: object,
    request: ValidatedMissionRequest,
) -> dict[str, Any]:
    """Independently constrain hostile structured model output."""
    result = _mapping(value, "mission interpretation")
    _exact_fields(
        result,
        {
            "reading",
            "selected_capability_ids",
            "needs_clarification",
            "clarification_key",
        },
        "mission interpretation",
    )
    _reject_control_fields(result)
    reading = _bounded_text(result["reading"], "reading", 1000)
    selected = _unique_identifiers(
        result["selected_capability_ids"],
        "selected_capability_ids",
        maximum=32,
    )
    if not set(selected).issubset(request.approved_capability_ids):
        raise MissionInterpretationError(
            "model selected a capability outside the approved registry"
        )
    missing = sorted(set(request.required_capability_ids) - set(selected))
    if missing:
        raise MissionInterpretationError(
            "model omitted judge-card capabilities: " + ", ".join(missing)
        )
    needs_clarification = result["needs_clarification"]
    if not isinstance(needs_clarification, bool):
        raise MissionInterpretationError("needs_clarification must be boolean")
    clarification_key = _bounded_text(
        result["clarification_key"],
        "clarification_key",
        128,
        allow_empty=True,
    )
    if needs_clarification != bool(clarification_key):
        raise MissionInterpretationError(
            "clarification_key must be present exactly when clarification is needed"
        )
    return {
        "reading": reading,
        "selected_capability_ids": list(selected),
        "needs_clarification": needs_clarification,
        "clarification_key": clarification_key,
    }


class MissionInterpretationService:
    """Interpret cards through one model call with a deterministic fallback."""

    def __init__(
        self,
        provider: StructuredJsonProvider | None,
        *,
        provider_name: str,
        model: str,
        timeout_seconds: float = 60.0,
    ) -> None:
        self._provider = provider
        self._provider_name = _bounded_text(provider_name, "provider_name", 128)
        self._model = _bounded_text(model, "model", 256)
        if not 1.0 <= timeout_seconds <= 120.0:
            raise ValueError("timeout_seconds must be between 1 and 120")
        self._timeout_seconds = timeout_seconds

    async def interpret(self, raw_request: object) -> dict[str, Any]:
        """Return card-authoritative interpretation and content-addressed evidence."""
        request = validate_mission_request(raw_request)
        schema = build_interpretation_schema(request)
        mode = "deterministic_fallback"
        fallback_reason = "provider_unavailable"
        provider_response_hash = ""
        interpretation = _fallback_interpretation(request)
        if self._provider is not None:
            try:
                result = await self._provider.complete_json_schema(
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "The physical Zone and Objective cards were drawn by a "
                                "judge and are immutable data. Explain the task briefly, "
                                "select only approved capability IDs, and flag genuine "
                                "ambiguity. Never change evidence requirements, draw cards, "
                                "choose resources, claim task completion, or emit commands."
                            ),
                        },
                        {
                            "role": "user",
                            "content": _canonical(request.payload).decode("utf-8"),
                        },
                    ],
                    schema=schema,
                    timeout_seconds=self._timeout_seconds,
                )
                message = result.get("message")
                content = message.get("content") if isinstance(message, Mapping) else None
                if not isinstance(content, str) or not content:
                    raise MissionInterpretationError(
                        "provider response is missing message.content"
                    )
                provider_response_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
                interpretation = validate_interpretation(json.loads(content), request)
                mode = "live_llm"
                fallback_reason = ""
            except (json.JSONDecodeError, MissionInterpretationError):
                fallback_reason = "invalid_structured_output"
            except Exception:  # provider failures are projected without raw error text
                fallback_reason = "provider_unavailable"

        attestation: dict[str, object] = {
            "contract_version": ATTESTATION_CONTRACT,
            "task_id": request.task_id,
            "mode": mode,
            "fallback_reason": fallback_reason,
            "provider": self._provider_name,
            "model": self._model,
            "request_sha256": _sha256(request.payload),
            "schema_sha256": _sha256(schema),
            "provider_response_sha256": provider_response_hash,
            "interpretation_sha256": _sha256(interpretation),
            "card_source": "judge_draw",
        }
        attestation["snapshot"] = _sha256(attestation)
        return {
            "contract_version": RESPONSE_CONTRACT,
            "task_id": request.task_id,
            "card_source": "judge_draw",
            "authoritative_evidence_requirements": list(
                request.evidence_requirements
            ),
            "interpretation": interpretation,
            "attestation": attestation,
        }


def _fallback_interpretation(
    request: ValidatedMissionRequest,
) -> dict[str, Any]:
    objective = request.payload["objective"]
    assert isinstance(objective, Mapping)
    goal = str(objective["goal"]).strip()[:1000]
    return {
        "reading": _bounded_text(goal, "objective.goal", 1000),
        "selected_capability_ids": list(request.required_capability_ids),
        "needs_clarification": False,
        "clarification_key": "",
    }


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256(value: object) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _mapping(value: object, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise MissionInterpretationError(f"{field} must be an object")
    return value


def _exact_fields(value: Mapping[str, object], expected: set[str], field: str) -> None:
    missing = sorted(expected - set(value))
    unknown = sorted(set(value) - expected)
    if missing or unknown:
        details = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if unknown:
            details.append("unsupported " + ", ".join(unknown))
        raise MissionInterpretationError(f"{field} fields are invalid: {'; '.join(details)}")


def _identifier(value: object, field: str) -> str:
    text = _bounded_text(value, field, 256)
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._:@/-")
    if (
        not text[0].isascii()
        or not text[0].isalnum()
        or any(character not in allowed for character in text)
    ):
        raise MissionInterpretationError(f"{field} must be a safe identifier")
    return text


def _bounded_text(
    value: object,
    field: str,
    maximum: int,
    *,
    allow_empty: bool = False,
) -> str:
    if not isinstance(value, str):
        raise MissionInterpretationError(f"{field} must be text")
    normalized = value.strip()
    if (not normalized and not allow_empty) or len(normalized) > maximum:
        raise MissionInterpretationError(f"{field} must be bounded text")
    return normalized


def _unique_identifiers(
    value: object,
    field: str,
    *,
    maximum: int,
) -> tuple[str, ...]:
    if not isinstance(value, list) or not 1 <= len(value) <= maximum:
        raise MissionInterpretationError(f"{field} must contain 1 to {maximum} items")
    parsed = tuple(_identifier(item, field) for item in value)
    if len(parsed) != len(set(parsed)):
        raise MissionInterpretationError(f"{field} must contain unique identifiers")
    return parsed


def _reject_control_fields(value: object, path: str = "request") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = str(key).casefold()
            if normalized in FORBIDDEN_CONTROL_FIELDS:
                raise MissionInterpretationError(
                    f"{path}.{key} is a forbidden execution control field"
                )
            _reject_control_fields(item, f"{path}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, item in enumerate(value):
            _reject_control_fields(item, f"{path}[{index}]")
