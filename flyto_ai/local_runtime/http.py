"""Cancellable loopback HTTP inference with JSON and observed-image contracts."""

import asyncio
import base64
import binascii

import httpx
from jsonschema import Draft202012Validator, exceptions
from referencing.exceptions import Unresolvable

from flyto_ai.cli_runtime.contracts import (
    MAX_IMAGES, MAX_IMAGE_BYTES, decode_json, encode_json, CliRuntimeError,
)
from .contracts import LocalModelConfig, LocalModelError


def _schema(value):
    encode_json(value, limit=32 * 1024)
    if not isinstance(value, dict) or value.get("type") != "object":
        raise LocalModelError("local_model_invalid_schema")
    def inspect(node, depth=0):
        if depth > 32:
            raise LocalModelError("local_model_invalid_schema")
        if isinstance(node, dict):
            for key, child in node.items():
                if key in {"$ref", "$dynamicRef"} and (not isinstance(child, str) or not child.startswith("#")):
                    raise LocalModelError("local_model_external_schema_refused")
                inspect(child, depth + 1)
        elif isinstance(node, list):
            for child in node:
                inspect(child, depth + 1)
    inspect(value)
    try:
        Draft202012Validator.check_schema(value)
    except exceptions.SchemaError as exc:
        raise LocalModelError("local_model_invalid_schema") from exc


def _images(images):
    if not isinstance(images, (tuple, list)) or len(images) > MAX_IMAGES:
        raise LocalModelError("local_model_invalid_image")
    result = []
    for item in images:
        if not isinstance(item, dict) or item.get("media_type") not in {"image/png", "image/jpeg", "image/webp", "image/gif"}:
            raise LocalModelError("local_model_invalid_image")
        text = item.get("base64")
        try:
            if not isinstance(text, str) or len(text) > MAX_IMAGE_BYTES * 4 // 3 + 4:
                raise ValueError
            raw = base64.b64decode(text, validate=True)
            if not 1 <= len(raw) <= MAX_IMAGE_BYTES:
                raise ValueError
        except (ValueError, binascii.Error) as exc:
            raise LocalModelError("local_model_invalid_image") from exc
        result.append({"media_type": item["media_type"], "base64": text})
    return result


def _payload(local, prompt, schema, system_prompt, images):
    if not isinstance(prompt, str) or not 1 <= len(prompt.encode()) <= 256 * 1024:
        raise LocalModelError("local_model_invalid_input")
    if not isinstance(system_prompt, str) or len(system_prompt.encode()) > 16 * 1024:
        raise LocalModelError("local_model_invalid_input")
    _schema(schema)
    images = _images(images)
    messages = [{"role": "system", "content": system_prompt}]
    user = {"role": "user", "content": prompt}
    body = {"model": local.model, "stream": False, "messages": messages}
    if local.provider == "ollama":
        if images:
            user["images"] = [item["base64"] for item in images]
        body["format"] = schema
    else:
        if images:
            user["content"] = [{"type": "text", "text": prompt}] + [
                {"type": "image_url", "image_url": {"url": "data:" + item["media_type"] + ";base64," + item["base64"]}}
                for item in images
            ]
        body["response_format"] = {"type": "json_schema", "json_schema": {"name": "flyto_result", "strict": True, "schema": schema}}
    messages.append(user)
    return body


def _content(local, value, schema):
    if not isinstance(value, dict) or value.get("error"):
        raise LocalModelError("local_model_provider_failed")
    if value.get("model") and value["model"] != local.model:
        raise LocalModelError("local_model_changed")
    if local.provider == "ollama":
        message = value.get("message")
        if value.get("done") is not True or value.get("done_reason") not in (None, "stop"):
            raise LocalModelError("local_model_incomplete_output")
    else:
        choices = value.get("choices")
        if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict) or choices[0].get("finish_reason") != "stop":
            raise LocalModelError("local_model_incomplete_output")
        message = choices[0].get("message")
    if not isinstance(message, dict) or message.get("tool_calls") or message.get("function_call") or message.get("refusal"):
        raise LocalModelError("local_model_native_action_refused")
    text = message.get("content")
    if not isinstance(text, str) or len(text.encode()) > 64 * 1024:
        raise LocalModelError("local_model_invalid_output")
    parsed = decode_json(text)
    pending = [(parsed, 0)]
    while pending:
        node, depth = pending.pop()
        if depth > 32:
            raise LocalModelError("local_model_output_too_deep")
        if isinstance(node, dict):
            pending.extend((child, depth + 1) for child in node.values())
        elif isinstance(node, list):
            pending.extend((child, depth + 1) for child in node)
    try:
        Draft202012Validator(schema).validate(parsed)
    except (exceptions.ValidationError, exceptions.SchemaError, Unresolvable, RecursionError) as exc:
        raise LocalModelError("local_model_schema_mismatch") from exc
    return encode_json(parsed, limit=64 * 1024)


async def complete_local_json(local, *, prompt, schema, system_prompt="", images=()):
    """One bounded request. Cancellation closes its socket; never retries work."""
    try:
        if not isinstance(local, LocalModelConfig):
            raise LocalModelError("local_model_invalid_config")
        body = _payload(local, prompt, schema, system_prompt, images)
        path = "/api/chat" if local.provider == "ollama" else "/chat/completions"
        async with asyncio.timeout(local.timeout_seconds):
            async with httpx.AsyncClient(trust_env=False, follow_redirects=False, timeout=local.timeout_seconds) as client:
                async with client.stream("POST", local.endpoint + path, json=body) as response:
                    if response.status_code in {401, 403}:
                        raise LocalModelError("local_model_auth_not_supported")
                    if response.status_code == 404:
                        raise LocalModelError("local_model_not_found")
                    if response.status_code in {400, 422}:
                        raise LocalModelError("local_model_request_unsupported")
                    if response.status_code != 200:
                        raise LocalModelError("local_model_http_error")
                    chunks, count = [], 0
                    async for chunk in response.aiter_bytes():
                        count += len(chunk)
                        if count > 2_000_000:
                            raise LocalModelError("local_model_output_too_large")
                        chunks.append(chunk)
        return _content(local, decode_json(b"".join(chunks)), schema)
    except (TimeoutError, httpx.TimeoutException) as exc:
        raise LocalModelError("local_model_timeout") from exc
    except httpx.HTTPError as exc:
        raise LocalModelError("local_model_connection_failed") from exc
    except CliRuntimeError as exc:
        if isinstance(exc, LocalModelError):
            raise
        raise LocalModelError("local_model_invalid_json") from exc
