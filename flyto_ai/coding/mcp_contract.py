# Copyright 2024 Flyto2
# Licensed under the Apache License, Version 2.0
"""Closed JSON-RPC contracts shared by the coding MCP supervisor."""
from __future__ import annotations

import json
import math
from typing import Any, Dict, Mapping, Optional


class CodingMCPWorkerUnavailable(RuntimeError):
    """The replaceable worker could not serve a request deterministically."""


def _reject_nonfinite(value: str) -> None:
    raise ValueError("non-finite JSON number: {}".format(value))


def _finite_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError("non-finite JSON number: {}".format(value))
    return parsed


def _unique_object(pairs: list[tuple[str, Any]]) -> Dict[str, Any]:
    value: Dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate JSON object key")
        value[key] = item
    return value


def _decode_json(raw: bytes, contract: str) -> Mapping[str, Any]:
    """Decode strict JSON while rejecting duplicate keys and non-finite values."""

    try:
        value = json.loads(
            raw,
            parse_constant=_reject_nonfinite,
            parse_float=_finite_float,
            object_pairs_hook=_unique_object,
        )
    except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise CodingMCPWorkerUnavailable(
            "worker {} was malformed".format(contract),
        ) from exc
    if not isinstance(value, Mapping):
        raise CodingMCPWorkerUnavailable("worker {} failed".format(contract))
    return value


def json_contract_equal(left: Any, right: Any) -> bool:
    """Compare JSON values without Python's bool/int equality collapse."""

    try:
        options = {"sort_keys": True, "separators": (",", ":"), "allow_nan": False}
        return json.dumps(left, **options) == json.dumps(right, **options)
    except (TypeError, ValueError):
        return False


def validated_response(
    raw: bytes,
    request: Mapping[str, Any],
    contract: str,
) -> Mapping[str, Any]:
    """Validate one exact JSON-RPC response envelope, including errors."""

    value = _decode_json(raw, contract)
    if value.get("jsonrpc") != "2.0":
        raise CodingMCPWorkerUnavailable("worker {} failed".format(contract))
    if (
        "id" not in value
        or "id" not in request
        or not json_contract_equal(value.get("id"), request.get("id"))
    ):
        raise CodingMCPWorkerUnavailable("worker {} failed".format(contract))
    has_result = "result" in value
    has_error = "error" in value
    if has_result == has_error:
        raise CodingMCPWorkerUnavailable("worker {} failed".format(contract))
    if has_error:
        error = value.get("error")
        if (
            not isinstance(error, Mapping)
            or isinstance(error.get("code"), bool)
            or not isinstance(error.get("code"), int)
            or not isinstance(error.get("message"), str)
        ):
            raise CodingMCPWorkerUnavailable("worker {} failed".format(contract))
        return value
    if not isinstance(value.get("result"), Mapping):
        raise CodingMCPWorkerUnavailable("worker {} failed".format(contract))
    return value


def validated_initialize_result(
    raw: bytes,
    request: Mapping[str, Any],
) -> Optional[Dict[str, Any]]:
    """Validate one exact initialize response before trusting its worker."""

    value = validated_response(raw, request, "initialization")
    if "error" in value:
        return None
    result = dict(value["result"])
    params = request.get("params")
    requested_protocol = (
        params.get("protocolVersion") if isinstance(params, Mapping) else None
    )
    server_info = result.get("serverInfo")
    capabilities = result.get("capabilities")
    if (
        not isinstance(requested_protocol, str)
        or result.get("protocolVersion") != requested_protocol
        or not isinstance(server_info, Mapping)
        or not isinstance(server_info.get("name"), str)
        or not isinstance(server_info.get("version"), str)
        or not isinstance(capabilities, Mapping)
        or not isinstance(result.get("instructions"), str)
    ):
        raise CodingMCPWorkerUnavailable("worker initialization failed")
    return result


def validated_tools_list_result(
    raw: bytes,
    request: Mapping[str, Any],
) -> Optional[Dict[str, Any]]:
    """Validate one tool-list envelope before caching its exact result."""

    value = validated_response(raw, request, "tool contract")
    if "error" in value:
        return None
    result = dict(value["result"])
    tools = result.get("tools")
    if (
        not isinstance(tools, list)
        or "nextCursor" in result
        or not all(
            isinstance(tool, Mapping)
            and isinstance(tool.get("name"), str)
            and isinstance(tool.get("inputSchema"), Mapping)
            for tool in tools
        )
    ):
        raise CodingMCPWorkerUnavailable("worker tool contract failed")
    return result


def protocol_error(request_id: Any, code: int, message: str) -> bytes:
    """Encode one bounded JSON-RPC protocol error."""

    return json.dumps({
        "jsonrpc": "2.0",
        "id": request_id,
        "error": {"code": code, "message": message},
    }, separators=(",", ":")).encode() + b"\n"


def tool_error(request_id: Any, code: str) -> bytes:
    """Encode one structured MCP tool-domain refusal."""

    payload = {"ok": False, "error": code}
    return json.dumps({
        "jsonrpc": "2.0",
        "id": request_id,
        "result": {
            "content": [{"type": "text", "text": json.dumps(payload)}],
            "isError": True,
            "structuredContent": payload,
        },
    }, separators=(",", ":")).encode() + b"\n"
