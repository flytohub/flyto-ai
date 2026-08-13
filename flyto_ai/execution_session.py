# Copyright 2024 Flyto2
# Licensed under the Apache License, Version 2.0
"""Host-governed bridge from an activation claim to bounded planning input.

This module validates an activation claim.  It does not detect wake words or
provide identity, authorization, speech, provider, device, or scheduling
runtime authority.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from flyto_ai.capability_router import normalize_goal_frame, route_capabilities

EXECUTION_SESSION_REQUEST_VERSION = "flyto.execution-session-request.v1"
EXECUTION_SESSION_RESULT_VERSION = "flyto.execution-session-result.v1"
_SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:@-]{0,191}$")
_MAX_TEXT = 4_000
_MAX_WAKE_WORDS = 32
_MAX_TIMEOUT_MS = 300_000
_MAX_BLUEPRINTS = 32
_MAX_JSON_DEPTH = 32
_MAX_REQUEST_NODES = 4_096
_MAX_REQUEST_BYTES = 262_144
_MAX_CATALOG_NODES = 500_000
_MAX_MANIFEST_BYTES = 8_388_608
_MAX_BLUEPRINT_BYTES = 1_048_576
_MAX_JSON_INTEGER = 9_223_372_036_854_775_807
_MAX_TIMESTAMP_MS = 253_402_300_799_999  # 9999-12-31T23:59:59.999Z
_ACTIVATION_SOURCES = frozenset(
    {"typed", "voice_reviewed", "external_agent", "mission_card"}
)
_FORBIDDEN_REQUEST_FIELDS = frozenset(
    {
        "authority",
        "identity",
        "tenant",
        "tenant_id",
        "principal",
        "principal_id",
        "permissions",
        "granted_permissions",
        "context",
        "manifests",
        "trusted_manifests",
        "allowed_sources",
        "allowed_domains",
        "enabled_capabilities",
    }
)


class ExecutionSessionError(ValueError):
    """Raised when an execution session cannot be proven safe and bounded."""


def _authority_values(value: object, field: str) -> tuple[str, ...]:
    if not isinstance(value, (tuple, list, set, frozenset)) or not value:
        raise ExecutionSessionError(f"authority.{field} must be a non-empty sequence")
    if len(value) > 256:
        raise ExecutionSessionError(f"authority.{field} exceeds 256 items")
    items: list[str] = []
    for item in value:
        if not isinstance(item, str) or not _SAFE_ID.fullmatch(item):
            raise ExecutionSessionError(f"authority.{field} contains an unsafe value")
        items.append(item)
    normalized = tuple(sorted(set(items)))
    if len(normalized) != len(items):
        raise ExecutionSessionError(f"authority.{field} contains duplicates")
    return normalized


@dataclass(frozen=True, slots=True)
class ExecutionAuthority:
    """Verified host authority; none of these values may come from the request."""

    tenant_id: str
    principal_id: str
    verified: bool
    allowed_sources: tuple[str, ...]
    allowed_domains: tuple[str, ...]
    granted_permissions: tuple[str, ...]
    enabled_capabilities: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.verified is not True:
            raise ExecutionSessionError("authority must be explicitly verified")
        for field in ("tenant_id", "principal_id"):
            value = getattr(self, field)
            if not isinstance(value, str) or not _SAFE_ID.fullmatch(value):
                raise ExecutionSessionError(f"authority.{field} is invalid")
        for field in (
            "allowed_sources",
            "allowed_domains",
            "granted_permissions",
            "enabled_capabilities",
        ):
            object.__setattr__(self, field, _authority_values(getattr(self, field), field))


def _preflight_json(
    value: object,
    path: str,
    *,
    max_nodes: int,
) -> None:
    """Reject unsafe JSON shapes iteratively, before recursion or encoding."""
    nodes = 0
    stack: list[tuple[object, int]] = [(value, 0)]
    while stack:
        item, depth = stack.pop()
        nodes += 1
        if nodes > max_nodes:
            raise ExecutionSessionError(f"{path} exceeds the {max_nodes} node limit")
        if depth > _MAX_JSON_DEPTH:
            raise ExecutionSessionError(
                f"{path} exceeds the {_MAX_JSON_DEPTH} level depth limit"
            )
        if item is None or isinstance(item, (bool, str)):
            continue
        if isinstance(item, int):
            if not -_MAX_JSON_INTEGER <= item <= _MAX_JSON_INTEGER:
                raise ExecutionSessionError(f"{path} contains an out-of-range integer")
            continue
        if isinstance(item, float):
            if not math.isfinite(item):
                raise ExecutionSessionError(f"{path} contains a non-finite number")
            continue
        if isinstance(item, Mapping):
            try:
                entries = list(item.items())
            except (RecursionError, TypeError, ValueError, OverflowError) as exc:
                raise ExecutionSessionError(f"{path} is not a safe JSON object") from exc
            if any(not isinstance(key, str) for key, _child in entries):
                raise ExecutionSessionError(f"{path} contains a non-string key")
            stack.extend((child, depth + 1) for _key, child in entries)
            continue
        if isinstance(item, (list, tuple)):
            stack.extend((child, depth + 1) for child in item)
            continue
        raise ExecutionSessionError(f"{path} must contain only JSON values")


def _plain_json(value: object, path: str = "value") -> Any:
    """Copy JSON data while rejecting coercions, controls, and non-finite numbers."""
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ExecutionSessionError(f"{path} contains a non-finite number")
        return value
    if isinstance(value, str):
        if any(unicodedata.category(char) in {"Cc", "Cf", "Cs"} for char in value):
            raise ExecutionSessionError(f"{path} contains unsafe control text")
        return unicodedata.normalize("NFC", value)
    if isinstance(value, Mapping):
        copied: dict[str, Any] = {}
        if any(not isinstance(key, str) for key in value):
            raise ExecutionSessionError(f"{path} contains a non-string key")
        for key in sorted(value):
            copied[key] = _plain_json(value[key], f"{path}.{key}")
        return copied
    if isinstance(value, (list, tuple)):
        return [_plain_json(item, f"{path}[]") for item in value]
    raise ExecutionSessionError(f"{path} must contain only JSON values")


def _canonical_json(
    value: object,
    path: str,
    *,
    max_nodes: int,
    max_bytes: int,
) -> Any:
    _preflight_json(value, path, max_nodes=max_nodes)
    try:
        copied = _plain_json(value, path)
        encoded = json.dumps(
            copied,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except ExecutionSessionError:
        raise
    except (RecursionError, TypeError, ValueError, OverflowError) as exc:
        raise ExecutionSessionError(f"{path} is not canonical JSON") from exc
    if len(encoded) > max_bytes:
        raise ExecutionSessionError(f"{path} exceeds the {max_bytes} byte limit")
    return copied


def _object(value: object, fields: set[str], path: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ExecutionSessionError(f"{path} must be an object")
    unknown = set(value) - fields
    missing = fields - set(value)
    if unknown or missing:
        detail = sorted(unknown or missing)
        raise ExecutionSessionError(f"{path} has unsupported or missing fields: {detail}")
    return _plain_json(value, path)


def _bounded_text(
    value: object,
    path: str,
    *,
    identifier: bool = False,
    allow_empty: bool = False,
) -> str:
    if (
        not isinstance(value, str)
        or (not value.strip() and (not allow_empty or value != ""))
        or len(value) > _MAX_TEXT
    ):
        qualifier = "bounded text" if allow_empty else "non-empty bounded text"
        raise ExecutionSessionError(f"{path} must be {qualifier}")
    result = _plain_json(value, path)
    if identifier and not _SAFE_ID.fullmatch(result):
        raise ExecutionSessionError(f"{path} must be a safe identifier")
    return result


def _timestamp(value: object, path: str) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or not 0 <= value <= _MAX_TIMESTAMP_MS
    ):
        raise ExecutionSessionError(
            f"{path} must be an integer timestamp from 0 to {_MAX_TIMESTAMP_MS}"
        )
    return value


def _wake_key(value: str) -> str:
    return unicodedata.normalize("NFKC", value).casefold()


def _digest(value: object) -> str:
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except ExecutionSessionError:
        raise
    except (RecursionError, TypeError, ValueError, OverflowError) as exc:
        raise ExecutionSessionError("attestation input is not canonical JSON") from exc
    return "sha256:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def prepare_execution_session(
    untrusted_request: Mapping[str, Any],
    trusted_manifests: Sequence[Mapping[str, Any]],
    authority: ExecutionAuthority,
    now_ms: int,
    *,
    trusted_blueprints: Sequence[Mapping[str, Any]] = (),
    limit: int = 8,
) -> Mapping[str, Any]:
    """Validate and attest one host-governed active execution session."""
    if not isinstance(authority, ExecutionAuthority):
        raise ExecutionSessionError("authority must be ExecutionAuthority")
    if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 32:
        raise ExecutionSessionError("limit must be an integer from 1 to 32")
    now = _timestamp(now_ms, "now_ms")
    canonical_request = _canonical_json(
        untrusted_request,
        "request",
        max_nodes=_MAX_REQUEST_NODES,
        max_bytes=_MAX_REQUEST_BYTES,
    )
    request = _object(
        canonical_request,
        {"contract_version", "session_id", "space", "activation", "goal"},
        "request",
    )
    forbidden = _FORBIDDEN_REQUEST_FIELDS.intersection(request)
    if forbidden:
        raise ExecutionSessionError("request attempts to supply host authority")
    if request["contract_version"] != EXECUTION_SESSION_REQUEST_VERSION:
        raise ExecutionSessionError("unsupported execution session contract_version")
    session_id = _bounded_text(request["session_id"], "session_id", identifier=True)

    space = _object(
        request["space"],
        {"space_id", "display_name", "wake_words", "active_timeout_ms"},
        "space",
    )
    space_id = _bounded_text(space["space_id"], "space.space_id", identifier=True)
    display_name = _bounded_text(
        space["display_name"], "space.display_name", allow_empty=True
    )
    wake_words = space["wake_words"]
    if not isinstance(wake_words, list) or len(wake_words) > _MAX_WAKE_WORDS:
        raise ExecutionSessionError("space.wake_words must contain 0 to 32 items")
    clean_wakes = tuple(_bounded_text(item, "space.wake_words[]") for item in wake_words)
    wake_keys = tuple(_wake_key(item) for item in clean_wakes)
    if len(set(wake_keys)) != len(wake_keys):
        raise ExecutionSessionError("space.wake_words contains normalized duplicates")
    timeout = space["active_timeout_ms"]
    if (
        isinstance(timeout, bool)
        or not isinstance(timeout, int)
        or not 1 <= timeout <= _MAX_TIMEOUT_MS
    ):
        raise ExecutionSessionError("space.active_timeout_ms is invalid")

    activation = _object(
        request["activation"],
        {"source", "observed_wake_word", "activated_at_ms", "expires_at_ms"},
        "activation",
    )
    source = _bounded_text(activation["source"], "activation.source", identifier=True)
    if source not in _ACTIVATION_SOURCES:
        raise ExecutionSessionError("activation.source is not supported")
    activated = _timestamp(activation["activated_at_ms"], "activation.activated_at_ms")
    expires = _timestamp(activation["expires_at_ms"], "activation.expires_at_ms")
    if activated > now or expires <= now or expires <= activated:
        raise ExecutionSessionError("activation is future, expired, or empty")
    if expires - activated > timeout:
        raise ExecutionSessionError("activation exceeds the configured active window")
    observed = activation["observed_wake_word"]
    if observed is not None:
        raise ExecutionSessionError(
            "activation.observed_wake_word must be null for every supported source"
        )

    goal = _object(request["goal"], {"text", "frame"}, "goal")
    goal_text = _bounded_text(goal["text"], "goal.text")
    try:
        frame = normalize_goal_frame(goal["frame"])
    except (TypeError, ValueError, RecursionError, OverflowError) as exc:
        raise ExecutionSessionError(str(exc)) from exc
    if not isinstance(trusted_manifests, (list, tuple)) or not 1 <= len(trusted_manifests) <= 10_000:
        raise ExecutionSessionError("trusted manifest catalog must contain 1 to 10000 items")
    if not isinstance(trusted_blueprints, (list, tuple)) or len(trusted_blueprints) > _MAX_BLUEPRINTS:
        raise ExecutionSessionError("trusted blueprint catalog exceeds 32 items")
    manifests = _canonical_json(
        trusted_manifests,
        "trusted_manifests",
        max_nodes=_MAX_CATALOG_NODES,
        max_bytes=_MAX_MANIFEST_BYTES,
    )
    blueprints = _canonical_json(
        trusted_blueprints,
        "trusted_blueprints",
        max_nodes=_MAX_CATALOG_NODES,
        max_bytes=_MAX_BLUEPRINT_BYTES,
    )
    context = {
        "allowed_sources": list(authority.allowed_sources),
        "allowed_domains": list(authority.allowed_domains),
        "granted_permissions": list(authority.granted_permissions),
        "enabled_capabilities": list(authority.enabled_capabilities),
    }
    try:
        route = route_capabilities(
            goal_text,
            manifests,
            goal_frame=frame,
            context=context,
            limit=limit,
            blueprint_candidates=blueprints,
        )
    except (TypeError, ValueError, RecursionError, OverflowError) as exc:
        raise ExecutionSessionError(str(exc)) from exc

    authority_projection = {
        "tenant_id": authority.tenant_id,
        "principal_ref": _digest({"tenant_id": authority.tenant_id, "principal_id": authority.principal_id}),
        **context,
    }
    planning_input = {
        "session_id": session_id,
        "space": {
            "space_id": space_id,
            "display_name": display_name,
            "wake_words": list(clean_wakes),
            "active_timeout_ms": timeout,
        },
        "activation": {
            "source": source,
            "observed_wake_word": observed,
            "activated_at_ms": activated,
            "expires_at_ms": expires,
        },
        "goal": {"text": goal_text, "frame": frame},
    }
    result: dict[str, Any] = {
        "contract_version": EXECUTION_SESSION_RESULT_VERSION,
        "planning_input": planning_input,
        "capability_route": route,
        "authority": authority_projection,
        "attestations": {
            "request": _digest(request),
            "authority": _digest(authority_projection),
            "route": _digest(route),
        },
    }
    # The overall digest covers the complete governed payload and contract
    # version.  Its own field is deliberately absent from the digest input.
    result["overall_digest"] = _digest(result)
    # Every caller-owned input was copied before validation.  Returning ordinary
    # JSON containers keeps the wire result serializable while remaining fully
    # detached from request, catalog, Blueprint, and authority constructor data.
    return _plain_json(result, "result")
