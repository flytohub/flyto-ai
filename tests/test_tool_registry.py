# Copyright 2024 Flyto
# Licensed under the Apache License, Version 2.0
"""Tests for ToolRegistry — format conversion, dispatch."""
import pytest

from flyto_ai.tools.registry import ToolRegistry


SAMPLE_TOOL = {
    "name": "test_tool",
    "description": "A test tool",
    "inputSchema": {
        "type": "object",
        "properties": {"x": {"type": "string"}},
    },
}


class TestToolRegistry:

    def test_register_and_list(self):
        registry = ToolRegistry()

        async def handler(name, args):
            return {"ok": True}

        registry.register(SAMPLE_TOOL, handler)
        assert len(registry.tools) == 1
        assert registry.tools[0]["name"] == "test_tool"

    @pytest.mark.asyncio
    async def test_dispatch(self):
        registry = ToolRegistry()

        async def handler(name, args):
            return {"ok": True, "name": name, "x": args.get("x")}

        registry.register(SAMPLE_TOOL, handler)
        result = await registry.dispatch("test_tool", {"x": "hello"})
        assert result["ok"] is True
        assert result["x"] == "hello"

    @pytest.mark.asyncio
    async def test_dispatch_unknown(self):
        registry = ToolRegistry()
        result = await registry.dispatch("nonexistent", {})
        assert result["ok"] is False
        assert "Unknown tool" in result["error"]

    @pytest.mark.asyncio
    async def test_dispatch_error_handling(self):
        registry = ToolRegistry()

        async def bad_handler(name, args):
            raise ValueError("boom")

        registry.register(SAMPLE_TOOL, bad_handler)
        result = await registry.dispatch("test_tool", {})
        assert result["ok"] is False
        assert "boom" in result["error"]

    def test_to_openai_format(self):
        registry = ToolRegistry()

        async def handler(name, args):
            return {}

        registry.register(SAMPLE_TOOL, handler)
        tools = registry.to_openai_format()
        assert len(tools) == 1
        assert tools[0]["type"] == "function"
        assert tools[0]["function"]["name"] == "test_tool"
        assert "parameters" in tools[0]["function"]

    def test_to_anthropic_format(self):
        registry = ToolRegistry()

        async def handler(name, args):
            return {}

        registry.register(SAMPLE_TOOL, handler)
        tools = registry.to_anthropic_format()
        assert len(tools) == 1
        assert tools[0]["name"] == "test_tool"
        assert "input_schema" in tools[0]

    def test_register_many(self):
        registry = ToolRegistry()
        tool2 = {**SAMPLE_TOOL, "name": "tool2"}

        async def handler(name, args):
            return {"ok": True}

        registry.register_many([SAMPLE_TOOL, tool2], handler)
        assert len(registry.tools) == 2
