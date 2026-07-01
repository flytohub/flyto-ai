# Copyright 2024 Flyto
# Licensed under the Apache License, Version 2.0
"""Deterministic mock utilities for testing the agent loop.

Inspired by claw-code's ``MockAnthropicService`` with scenario-driven
responses.  Implements the ``ApiClient`` and ``ToolExecutor`` protocols
from :mod:`flyto_ai.protocols`.

Usage::

    from flyto_ai.testing import MockApiClient, MockToolExecutor, MockResponse

    client = MockApiClient(responses=[
        MockResponse(message="I'll search for modules.", tool_calls=[
            {"function": "search_modules", "arguments": {"query": "auth"}},
        ]),
        MockResponse(message="Found 3 auth modules."),
    ])

    executor = MockToolExecutor(responses={
        "search_modules": {"ok": True, "results": [...]},
    })

    agent = Agent(config=cfg, api_client=client, tool_executor=executor)
    response = await agent.chat("find auth modules")

    assert executor.calls[0] == ("search_modules", {"query": "auth"})
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class MockResponse:
    """A canned LLM response for the mock client."""
    message: str = ""
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    rounds_used: int = 1
    usage: Dict[str, int] = field(default_factory=lambda: {
        "prompt_tokens": 100,
        "completion_tokens": 50,
        "total_tokens": 150,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
    })


class MockApiClient:
    """Deterministic mock that satisfies the ``ApiClient`` protocol.

    Pops responses from a queue. When the queue is empty, returns a
    default "no more responses" message.

    The ``chat`` method:
    1. Dispatches any tool_calls in the current response via ``dispatch_fn``
    2. Returns the response message + tool call log
    """

    def __init__(
        self,
        responses: Optional[List[MockResponse]] = None,
        default_response: Optional[MockResponse] = None,
    ) -> None:
        self._responses = list(responses or [])
        self._default = default_response or MockResponse(message="(mock: no more responses)")
        self._call_count = 0
        self.messages_received: List[List[Dict]] = []

    @property
    def call_count(self) -> int:
        return self._call_count

    async def chat(
        self,
        messages: List[Dict[str, Any]],
        system_prompt: str,
        tools: List[Dict],
        dispatch_fn,
        max_rounds: int = 30,
        on_stream=None,
    ) -> Tuple[str, List[Dict[str, Any]], int, Dict[str, int]]:
        """Simulate a chat turn with canned responses."""
        self._call_count += 1
        self.messages_received.append(list(messages))

        resp = self._responses.pop(0) if self._responses else self._default

        tool_call_log: List[Dict[str, Any]] = []
        for tc in resp.tool_calls:
            func_name = tc.get("function", tc.get("name", ""))
            func_args = tc.get("arguments", tc.get("args", {}))
            if dispatch_fn:
                result = await dispatch_fn(func_name, func_args)
            else:
                result = {"ok": True}
            tool_call_log.append({
                "function": func_name,
                "arguments": func_args,
                "result_preview": str(result)[:500],
                "ok": result.get("ok", True) if isinstance(result, dict) else True,
            })

        return resp.message, tool_call_log, resp.rounds_used, dict(resp.usage)


class MockToolExecutor:
    """Records tool calls and returns canned responses.

    Satisfies the ``ToolExecutor`` protocol.

    Usage::

        executor = MockToolExecutor(
            tool_defs=[{"name": "search_modules", "description": "...", "inputSchema": {}}],
            responses={"search_modules": {"ok": True, "results": []}},
        )

        result = await executor.dispatch("search_modules", {"query": "auth"})
        assert executor.calls == [("search_modules", {"query": "auth"})]
    """

    def __init__(
        self,
        tool_defs: Optional[List[Dict]] = None,
        responses: Optional[Dict[str, Any]] = None,
        default_response: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._tools = list(tool_defs or [])
        self._responses = dict(responses or {})
        self._default = default_response or {"ok": True}
        self.calls: List[Tuple[str, Dict[str, Any]]] = []

    @property
    def tools(self) -> List[Dict]:
        return list(self._tools)

    async def dispatch(self, name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Record the call and return the canned response."""
        self.calls.append((name, dict(arguments)))
        response = self._responses.get(name, self._default)
        if callable(response):
            return response(name, arguments)
        return dict(response)

    def set_response(self, tool_name: str, response: Any) -> None:
        """Update the canned response for a tool."""
        self._responses[tool_name] = response
