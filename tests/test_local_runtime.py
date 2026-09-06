"""Actual loopback HTTP and host-dispatched actions; no downloaded/live model."""

import asyncio
import base64
import json
from contextlib import asynccontextmanager

import pytest

from flyto_ai import AgentConfig
from flyto_ai.local_runtime import LocalModelAgent, LocalModelConfig, LocalModelError, complete_local_json

SCHEMA = {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"], "additionalProperties": False}


@asynccontextmanager
async def server(answer=None, *, status=200, hang=False):
    requests, tasks = [], set()
    received, disconnected = asyncio.Event(), asyncio.Event()
    async def handle(reader, writer):
        task = asyncio.current_task()
        tasks.add(task)
        try:
            headers = (await reader.readuntil(b"\r\n\r\n")).decode().split("\r\n")
            length = int(next(line.split(":", 1)[1] for line in headers if line.lower().startswith("content-length:")))
            body = json.loads(await reader.readexactly(length))
            requests.append((headers, body))
            received.set()
            if hang:
                assert await reader.read() == b""
                disconnected.set()
                return
            response = answer(body) if callable(answer) else answer
            raw = json.dumps(response).encode()
            writer.write(f"HTTP/1.1 {status} Fixture\r\nContent-Length: {len(raw)}\r\nConnection: close\r\nLocation: http://not-loopback.invalid/\r\n\r\n".encode() + raw)
            await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()
            tasks.discard(task)
    listener = await asyncio.start_server(handle, "127.0.0.1", 0)
    port = listener.sockets[0].getsockname()[1]
    try:
        yield f"http://127.0.0.1:{port}", requests, received, disconnected
    finally:
        listener.close()
        await listener.wait_closed()
        for task in list(tasks):
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


def response(provider, content, *, model="local-exact:tag", native_calls=None):
    message = {"role": "assistant", "content": json.dumps(content)}
    if native_calls is not None:
        message["tool_calls"] = native_calls
    if provider == "ollama":
        return {"model": model, "done": True, "done_reason": "stop", "message": message}
    return {"model": model, "choices": [{"finish_reason": "stop", "message": message}]}


@pytest.mark.parametrize("endpoint", ["https://example.com", "http://127.0.0.1.evil", "http://127.1", "http://user:password@localhost", "http://localhost?key=x", "http://localhost/#x", "http://[::ffff:127.0.0.1]", "http://localhost/api", "http://localhost:0", "http://localhost\n"])
def test_only_explicit_canonical_loopback_endpoint_is_accepted(endpoint):
    with pytest.raises(LocalModelError, match="local_model_invalid_endpoint"):
        LocalModelConfig("ollama", endpoint, "model")


@pytest.mark.parametrize("provider", ["ollama", "openai_compatible"])
@pytest.mark.asyncio
async def test_exact_model_observed_image_and_schema_reach_real_local_http(provider, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-be-sent")
    monkeypatch.setenv("HTTP_PROXY", "http://127.0.0.1:1")
    image = {"media_type": "image/png", "base64": base64.b64encode(b"observed-image-bytes").decode()}
    async with server(response(provider, {"ok": True})) as (url, requests, *_):
        local = LocalModelConfig(provider, url, "local-exact:tag")
        result = await complete_local_json(local, prompt="Inspect the attached observation.", schema=SCHEMA, images=[image])
    assert json.loads(result) == {"ok": True}
    headers, body = requests[0]
    assert len(requests) == 1 and body["model"] == "local-exact:tag"
    assert not any("authorization" in line.lower() or "must-not-be-sent" in line for line in headers)
    assert "tools" not in body and "tool_choice" not in body and body["stream"] is False
    if provider == "ollama":
        assert headers[0].startswith("POST /api/chat ")
        assert body["format"] == SCHEMA and body["messages"][-1]["images"] == [image["base64"]]
    else:
        assert headers[0].startswith("POST /v1/chat/completions ")
        assert body["response_format"]["json_schema"]["schema"] == SCHEMA
        assert body["messages"][-1]["content"][1]["image_url"]["url"] == "data:image/png;base64," + image["base64"]


@pytest.mark.parametrize(("status", "code"), [(401, "auth_not_supported"), (404, "not_found"), (422, "request_unsupported"), (307, "http_error")])
@pytest.mark.asyncio
async def test_provider_failure_is_actionable_and_never_retried_or_redirected(status, code):
    async with server({"error": "private provider details"}, status=status) as (url, requests, *_):
        with pytest.raises(LocalModelError, match="local_model_" + code):
            await complete_local_json(LocalModelConfig("ollama", url, "local-exact:tag"), prompt="Return JSON", schema=SCHEMA)
        assert len(requests) == 1


@pytest.mark.parametrize(("output", "code"), [
    (response("ollama", {"ok": "yes"}), "schema_mismatch"),
    (response("ollama", {"ok": True}, model="different-model"), "changed"),
    (response("ollama", {"ok": True}, native_calls=[{"name": "shell"}]), "native_action_refused"),
    ({"done": False}, "incomplete_output"),
])
@pytest.mark.asyncio
async def test_invalid_output_model_change_and_native_action_cannot_be_completion(output, code):
    async with server(output) as (url, _, *_):
        with pytest.raises(LocalModelError, match="local_model_" + code):
            await complete_local_json(LocalModelConfig("ollama", url, "local-exact:tag"), prompt="Return JSON", schema=SCHEMA)


@pytest.mark.asyncio
async def test_timeout_and_cancellation_close_actual_http_socket_without_replay():
    for cancel in (False, True):
        async with server(hang=True) as (url, requests, received, disconnected):
            local = LocalModelConfig("ollama", url, "local-exact:tag", timeout_seconds=1 if cancel else 0.1)
            pending = asyncio.create_task(complete_local_json(local, prompt="Return JSON", schema=SCHEMA))
            await asyncio.wait_for(received.wait(), 2)
            if cancel:
                pending.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await pending
            else:
                with pytest.raises(LocalModelError, match="local_model_timeout"):
                    await pending
            await asyncio.wait_for(disconnected.wait(), 2)
            assert len(requests) == 1


@pytest.mark.asyncio
async def test_remote_schema_ref_is_refused_before_any_http():
    async with server(response("ollama", {"ok": True})) as (url, requests, *_):
        with pytest.raises(LocalModelError, match="external_schema_refused"):
            await complete_local_json(LocalModelConfig("ollama", url, "model"), prompt="JSON", schema={"type": "object", "$ref": "https://example.com/schema"})
        assert requests == []


@pytest.mark.asyncio
async def test_json_depth_is_bounded_even_when_schema_accepts_arbitrary_objects():
    nested = {"value": True}
    for _ in range(33):
        nested = {"nested": nested}
    async with server(response("ollama", nested)) as (url, _, *_):
        with pytest.raises(LocalModelError, match="local_model_output_too_deep"):
            await complete_local_json(LocalModelConfig("ollama", url, "local-exact:tag"), prompt="JSON", schema={"type": "object"})


@pytest.mark.asyncio
async def test_local_agent_uses_actual_host_dispatch_and_retains_images(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    target = tmp_path / "result.txt"
    image = {"media_type": "image/png", "base64": base64.b64encode(b"host-observed-frame").decode()}
    def answer(body):
        data = json.loads(body["messages"][-1]["content"])
        prior = [row for row in data["messages"] if row["role"] == "tool"]
        intent = {"content": "", "tool_calls": [{"name": "execute_module", "arguments_json": json.dumps({"module_id": "file.write", "params": {"path": str(target), "content": "host-wrote-this"}})}]}
        if prior:
            assert "host-wrote-this" in prior[0]["content"]
            assert body["messages"][-1]["images"] == [image["base64"]]
            intent = {"content": "The requested file was written.", "tool_calls": []}
        return response("ollama", intent)
    dispatched = []
    async def dispatch(name, args):
        dispatched.append((name, args))
        target.write_text(args["params"]["content"])
        return {"ok": True, "data": {"content": target.read_text()}, "_images": [image]}
    async with server(answer) as (url, requests, *_):
        agent = LocalModelAgent(AgentConfig(enable_transcript=False, enable_injection_detection=False, permission_level="workspace_write", max_tool_rounds=3),
            local=LocalModelConfig("ollama", url, "local-exact:tag"),
            tools=[{"name": "execute_module", "inputSchema": {"type": "object"}}], dispatch_fn=dispatch,
            system_prompt="Execute the authorized local file goal.", policies={"allowed_tools": ["execute_module"], "allowed_categories": ["file"]})
        agent._assistant = None
        try:
            result = await agent.chat(f"Write a file at {target} with content host-wrote-this.")
            assert target.read_text() == "host-wrote-this" and len(dispatched) == 1
            assert len(requests) == 2 and result.provider == "local_ai" and result.model == "local-exact:tag"
            assert result.tool_calls and not any(item.get("execution_id") for item in result.execution_results)
        finally:
            await agent.close()


@pytest.mark.asyncio
async def test_delegated_local_agent_never_resolves_an_endpoint_and_keeps_model_empty(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Delegated inference must not create local HTTP")
    monkeypatch.setattr("flyto_ai.local_runtime.http.httpx.AsyncClient", forbidden)
    async def complete(**kwargs):
        return '{"content":"Hello","tool_calls":[]}'
    agent = LocalModelAgent(AgentConfig(model="", enable_transcript=False), completion_fn=complete, tools=[])
    agent._assistant = None
    try:
        result = await agent.chat("Hello")
        assert result.provider == "local_ai" and result.model == "" and agent.config.resolved_model == ""
    finally:
        await agent.close()


@pytest.mark.asyncio
async def test_local_model_cannot_expand_host_tool_catalog_or_forge_receipts():
    output = {"content": "Done", "tool_calls": [{"name": "shell", "arguments_json": "{}"}]}
    async with server(response("ollama", output)) as (url, requests, *_):
        dispatched = []
        async def dispatch(*args):
            dispatched.append(args)
        agent = LocalModelAgent(AgentConfig(enable_transcript=False), local=LocalModelConfig("ollama", url, "local-exact:tag"), tools=[], dispatch_fn=dispatch)
        agent._assistant = None
        try:
            result = await agent.chat("Write a local report.")
            assert result.ok is False and result.error == "cli_tool_not_available"
            assert dispatched == [] and result.execution_results == [] and len(requests) == 1
        finally:
            await agent.close()


@pytest.mark.asyncio
async def test_closing_local_agent_cancels_active_request_and_prevents_more_inference():
    async with server(hang=True) as (url, requests, received, disconnected):
        agent = LocalModelAgent(AgentConfig(enable_transcript=False), local=LocalModelConfig("ollama", url, "local-exact:tag"), tools=[])
        agent._assistant = None
        pending = asyncio.create_task(agent.chat("Hello"))
        await asyncio.wait_for(received.wait(), 2)
        await agent.close()
        with pytest.raises(asyncio.CancelledError):
            await pending
        await asyncio.wait_for(disconnected.wait(), 2)
        assert len(requests) == 1 and agent.local_runtime._closed


@pytest.mark.asyncio
async def test_read_only_host_policy_still_blocks_local_models_write_intent(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    target = tmp_path / "must-not-exist.txt"
    def answer(body):
        data = json.loads(body["messages"][-1]["content"])
        if any(row["role"] == "tool" for row in data["messages"]):
            return response("ollama", {"content": "The host refused the write.", "tool_calls": []})
        return response("ollama", {"content": "", "tool_calls": [{"name": "execute_module", "arguments_json": json.dumps({"module_id": "file.write", "params": {"path": str(target), "content": "x"}})}]})
    async def dispatch(*args):
        target.write_text("forbidden")
        raise AssertionError("Read-only dispatch must never execute")
    async with server(answer) as (url, requests, *_):
        agent = LocalModelAgent(AgentConfig(enable_transcript=False, permission_level="read_only"),
            local=LocalModelConfig("ollama", url, "local-exact:tag"), tools=[{"name": "execute_module", "inputSchema": {"type": "object"}}],
            dispatch_fn=dispatch, policies={"allowed_tools": ["execute_module"], "allowed_categories": ["file"]})
        agent._assistant = None
        try:
            result = await agent.chat(f"Write a file at {target} with content x.")
            assert not target.exists() and not result.tool_calls
            assert result.ok is False and result.error == "cli_tool_not_available"
            catalog = json.loads(requests[0][1]["messages"][-1]["content"])["tools"]
            assert not any(row.get("name") == "execute_module" for row in catalog)
        finally:
            await agent.close()
