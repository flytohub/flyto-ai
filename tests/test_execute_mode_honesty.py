# Copyright 2024 Flyto2
# Licensed under the Apache License, Version 2.0
"""In execute mode the assistant may not say it is running a workflow it never called.

What the owner saw (AI Space chat, provider openai / gpt-4o): the user typed
幫我登入kintone; the audit recorded tool_calls_count=0, execution_count=0, and
the reply was 「我將執行 "kintone" 工作流程來幫助您登入。請稍候。執行中...」.
Nothing ran until the user typed 執行啊 and the next turn called the tool.
"""
import json
import logging

import pytest

from flyto_ai import Agent, AgentConfig

KINTONE_TOOL = {
    "name": "kintone",
    "description": "Run workflow: kintone login",
    "inputSchema": {"type": "object", "properties": {}},
    "_meta": {"source": {"type": "workflow", "id": "wf-1", "name": "kintone"}},
}

NARRATION = '我將執行 "kintone" 工作流程來幫助您登入。請稍候。執行中...'


class _NarratingProvider:
    """Talks about running the workflow; only calls it when forced to."""

    supports_forced_tool_choice = True

    def __init__(self, calls_tool_when_forced: bool) -> None:
        self._calls_tool_when_forced = calls_tool_when_forced
        self.tool_choices = []

    async def chat(self, messages, system_prompt, tools, dispatch_fn,
                   max_rounds=30, on_stream=None, tool_choice=None):
        self.tool_choices.append(tool_choice)
        if tool_choice == "required" and self._calls_tool_when_forced:
            result = await dispatch_fn("kintone", {})
            log = [{"function": "kintone", "arguments": {}, "result_preview": json.dumps(result)}]
            return "已登入 kintone。", log, 2, {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        return NARRATION, [], 1, {"prompt_tokens": 10, "completion_tokens": 28, "total_tokens": 38}


def _make_agent(provider):
    config = AgentConfig(
        provider="openai",
        api_key="test",
        enable_deterministic=False,
        enable_memory=False,
    )
    agent = Agent(config=config, system_prompt="You are the Space assistant.")
    agent._tools = [KINTONE_TOOL]
    dispatched = []

    async def _dispatch(name, args):
        dispatched.append((name, dict(args)))
        return {"ok": True, "status": "success"}

    agent._dispatch_fn = _dispatch
    agent._provider = provider
    return agent, dispatched


def _last_audit(caplog):
    records = [r for r in caplog.records if r.name == "flyto_ai.audit"]
    assert records, "no audit entry was emitted"
    return json.loads(records[-1].message)


@pytest.mark.asyncio
async def test_narrated_execution_is_retried_with_forced_tool_choice(caplog):
    """First completion narrates 執行中 without a call; the forced retry runs kintone once."""
    provider = _NarratingProvider(calls_tool_when_forced=True)
    agent, dispatched = _make_agent(provider)

    with caplog.at_level(logging.INFO, logger="flyto_ai.audit"):
        result = await agent.chat("幫我登入kintone", mode="execute")

    assert result.ok
    assert dispatched == [("kintone", {})]
    assert [tc["function"] for tc in result.tool_calls] == ["kintone"]
    assert "required" in provider.tool_choices
    assert "執行中" not in result.message
    assert "請稍候" not in result.message
    assert _last_audit(caplog)["tool_calls_count"] == 1


@pytest.mark.asyncio
async def test_narrated_execution_twice_is_replaced_by_honest_reply(caplog):
    """Model narrates even when forced: reply says nothing ran and names the workflow."""
    provider = _NarratingProvider(calls_tool_when_forced=False)
    agent, dispatched = _make_agent(provider)

    with caplog.at_level(logging.INFO, logger="flyto_ai.audit"):
        result = await agent.chat("幫我登入kintone", mode="execute")

    assert result.ok
    assert dispatched == []
    assert result.tool_calls == []
    assert result.execution_results == []
    assert "執行中" not in result.message
    assert "請稍候" not in result.message
    assert "kintone" in result.message
    audit = _last_audit(caplog)
    assert audit["tool_calls_count"] == 0
    assert audit["execution_count"] == 0


@pytest.mark.asyncio
async def test_provider_without_forced_tool_choice_gets_honest_reply():
    """A provider that cannot force a call is not retried; the user still hears the truth."""

    class _PlainProvider:
        supports_forced_tool_choice = False
        calls = 0

        async def chat(self, messages, system_prompt, tools, dispatch_fn,
                       max_rounds=30, on_stream=None, tool_choice=None):
            self.calls += 1
            return NARRATION, [], 1, {}

    provider = _PlainProvider()
    agent, dispatched = _make_agent(provider)

    result = await agent.chat("幫我登入kintone", mode="execute")

    assert result.ok
    assert dispatched == []
    assert "執行中" not in result.message
    assert "kintone" in result.message


@pytest.mark.asyncio
async def test_plain_answer_without_commitment_is_left_alone():
    """A reply that does not claim to be running anything is passed through untouched."""

    class _AnsweringProvider:
        supports_forced_tool_choice = True
        calls = 0

        async def chat(self, messages, system_prompt, tools, dispatch_fn,
                       max_rounds=30, on_stream=None, tool_choice=None):
            self.calls += 1
            return "kintone 是一個雲端資料庫平台。", [], 1, {}

    provider = _AnsweringProvider()
    agent, _ = _make_agent(provider)

    result = await agent.chat("kintone 是什麼", mode="execute")

    assert result.ok
    assert result.message == "kintone 是一個雲端資料庫平台。"
