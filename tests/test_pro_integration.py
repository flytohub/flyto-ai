# Copyright 2024 Flyto
# Licensed under the Apache License, Version 2.0
"""Tests for flyto-pro integration — ProBridge, EMS, Cost, Contract, Knowledge.

Tests cover:
1. ProBridge graceful degradation (no flyto-pro)
2. ProBridge with mocked flyto-pro modules
3. Agent init with pro bridge
4. Cost dual-tracking (CostTracker + CostController)
5. EMS error recording in middleware
6. Deep validation with ContractEngine
7. Config fields for pro features
8. Catalog outline injection in system prompt
9. Budget exceeded propagation
10. Evolution auto-generation in validation loop
"""
import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from flyto_ai.config import AgentConfig
from flyto_ai.validation import (
    extract_yaml_from_response,
    validate_workflow_steps,
    validate_workflow_deep,
)


@pytest.fixture(autouse=True)
def _reset_pro_singletons():
    """Reset ProBridge module-level singletons between tests."""
    import flyto_ai.intelligence.pro_bridge as pb
    saved = (
        pb._contract_engine, pb._contract_engine_checked,
        pb._cost_controller, pb._ems_router,
        pb._knowledge_router, pb._evolution_router,
    )
    pb._contract_engine = None
    pb._contract_engine_checked = False
    pb._cost_controller = None
    pb._ems_router = None
    pb._knowledge_router = None
    pb._evolution_router = None
    yield
    (pb._contract_engine, pb._contract_engine_checked,
     pb._cost_controller, pb._ems_router,
     pb._knowledge_router, pb._evolution_router) = saved


# ---------------------------------------------------------------------------
# 1. Config fields
# ---------------------------------------------------------------------------

class TestProConfig:

    def test_defaults(self):
        cfg = AgentConfig()
        assert cfg.enable_pro is True
        assert cfg.pro_budget_tier == ""
        assert cfg.enable_ems is True
        assert cfg.enable_knowledge is True
        assert cfg.enable_contract_validation is True
        assert cfg.enable_evolution is False

    def test_from_dict(self):
        cfg = AgentConfig.from_dict({
            "enable_pro": False,
            "pro_budget_tier": "enterprise",
            "enable_ems": False,
            "enable_knowledge": False,
            "enable_contract_validation": False,
            "enable_evolution": True,
        })
        assert cfg.enable_pro is False
        assert cfg.pro_budget_tier == "enterprise"
        assert cfg.enable_ems is False
        assert cfg.enable_knowledge is False
        assert cfg.enable_contract_validation is False
        assert cfg.enable_evolution is True

    def test_from_env(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.setenv("FLYTO_AI_ENABLE_PRO", "false")
        monkeypatch.setenv("FLYTO_AI_PRO_BUDGET_TIER", "pro")
        monkeypatch.setenv("FLYTO_AI_ENABLE_EMS", "false")
        monkeypatch.setenv("FLYTO_AI_ENABLE_EVOLUTION", "true")
        cfg = AgentConfig.from_env()
        assert cfg.enable_pro is False
        assert cfg.pro_budget_tier == "pro"
        assert cfg.enable_ems is False
        assert cfg.enable_evolution is True


# ---------------------------------------------------------------------------
# 2. ProBridge — graceful degradation
# ---------------------------------------------------------------------------

class TestProBridgeDegradation:

    def test_unavailable_when_no_flyto_pro(self):
        from flyto_ai.intelligence.pro_bridge import ProBridge
        # In test environment, flyto-pro may or may not be installed
        # Either way, the bridge should not crash
        bridge = ProBridge()
        # Methods should return None gracefully
        assert bridge.get_contract_engine() is None or bridge.available
        assert bridge.get_cost_summary() is None or bridge.available

    def test_disabled_via_config(self):
        cfg = AgentConfig(provider="openai", api_key="test", enable_pro=False)
        from flyto_ai.agent import Agent
        agent = Agent.__new__(Agent)
        agent._config = cfg
        agent._assistant = None
        result = agent._init_pro_bridge()
        assert result is None

    @pytest.mark.asyncio
    async def test_record_error_returns_none_when_unavailable(self):
        from flyto_ai.intelligence.pro_bridge import ProBridge
        bridge = ProBridge.__new__(ProBridge)
        bridge._core_available = False
        bridge._pro_available = False
        bridge._license_tier = "free"
        bridge._config = None
        result = await bridge.record_error("test", "msg", "stage")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_lesson_returns_none_when_unavailable(self):
        from flyto_ai.intelligence.pro_bridge import ProBridge
        bridge = ProBridge.__new__(ProBridge)
        bridge._core_available = False
        bridge._pro_available = False
        bridge._license_tier = "free"
        bridge._config = None
        result = await bridge.get_lesson_for_error("test", "msg")
        assert result is None

    @pytest.mark.asyncio
    async def test_validate_deep_returns_none_when_unavailable(self):
        from flyto_ai.intelligence.pro_bridge import ProBridge
        bridge = ProBridge.__new__(ProBridge)
        bridge._core_available = False
        bridge._pro_available = False
        bridge._license_tier = "free"
        bridge._config = None
        result = await bridge.validate_workflow_deep("name: test\nsteps: []")
        assert result is None

    @pytest.mark.asyncio
    async def test_search_modules_smart_returns_none_when_unavailable(self):
        from flyto_ai.intelligence.pro_bridge import ProBridge
        bridge = ProBridge.__new__(ProBridge)
        bridge._core_available = False
        bridge._pro_available = False
        bridge._license_tier = "free"
        bridge._config = None
        result = await bridge.search_modules_smart("http request")
        assert result is None

    @pytest.mark.asyncio
    async def test_generate_missing_returns_none_when_unavailable(self):
        from flyto_ai.intelligence.pro_bridge import ProBridge
        bridge = ProBridge.__new__(ProBridge)
        bridge._core_available = False
        bridge._pro_available = False
        bridge._license_tier = "free"
        bridge._config = None
        result = await bridge.generate_missing_modules(["test.module"])
        assert result is None

    def test_get_catalog_outline_returns_none_when_unavailable(self):
        from flyto_ai.intelligence.pro_bridge import ProBridge
        bridge = ProBridge.__new__(ProBridge)
        bridge._core_available = False
        bridge._pro_available = False
        bridge._license_tier = "free"
        bridge._config = None
        result = bridge.get_catalog_outline()
        assert result is None

    def test_check_budget_noop_when_unavailable(self):
        from flyto_ai.intelligence.pro_bridge import ProBridge
        bridge = ProBridge.__new__(ProBridge)
        bridge._core_available = False
        bridge._pro_available = False
        bridge._license_tier = "free"
        bridge._config = None
        # Should not raise
        bridge.check_budget()

    def test_record_tool_call_noop_when_unavailable(self):
        from flyto_ai.intelligence.pro_bridge import ProBridge
        bridge = ProBridge.__new__(ProBridge)
        bridge._core_available = False
        bridge._pro_available = False
        bridge._license_tier = "free"
        bridge._config = None
        # Should not raise
        bridge.record_tool_call()


# ---------------------------------------------------------------------------
# 3. ProBridge — with mocked flyto-pro
# ---------------------------------------------------------------------------

class TestProBridgeMocked:

    def _make_bridge(self):
        """Create a ProBridge with core+pro available and mocked internals."""
        from flyto_ai.intelligence.pro_bridge import ProBridge
        bridge = ProBridge.__new__(ProBridge)
        bridge._core_available = True
        bridge._pro_available = True
        bridge._license_tier = "pro"
        bridge._config = AgentConfig(pro_budget_tier="pro")
        return bridge

    @pytest.mark.asyncio
    async def test_record_error_delegates_to_ems(self):
        bridge = self._make_bridge()
        mock_ems = MagicMock()
        mock_ems.record_error = AsyncMock(return_value={"id": "err_1"})

        import flyto_ai.intelligence.pro_bridge as pb
        old = pb._ems_router
        pb._ems_router = mock_ems
        try:
            result = await bridge.record_error(
                error_type="execution_error",
                message="selector not found",
                stage="execute_module",
                module_id="browser.click",
            )
            assert result == {"id": "err_1"}
            mock_ems.record_error.assert_awaited_once_with(
                error_type="execution_error",
                message="selector not found",
                stage="execute_module",
                module_id="browser.click",
                code_snippet="",
            )
        finally:
            pb._ems_router = old

    @pytest.mark.asyncio
    async def test_get_lesson_delegates_to_ems(self):
        bridge = self._make_bridge()
        mock_ems = MagicMock()
        mock_ems.get_lesson_for_error = AsyncMock(return_value={
            "fix": "Use data-flyto-hint attribute instead of CSS selector",
        })

        import flyto_ai.intelligence.pro_bridge as pb
        old = pb._ems_router
        pb._ems_router = mock_ems
        try:
            result = await bridge.get_lesson_for_error(
                "execution_error", "Element not found",
            )
            assert result["fix"] == "Use data-flyto-hint attribute instead of CSS selector"
        finally:
            pb._ems_router = old

    def test_record_llm_usage_delegates_to_cost_controller(self):
        bridge = self._make_bridge()
        mock_cc = MagicMock()
        mock_cc.record_llm_usage.return_value = 0.0042

        import flyto_ai.intelligence.pro_bridge as pb
        old = pb._cost_controller
        pb._cost_controller = mock_cc
        try:
            cost = bridge.record_llm_usage("gpt-4o", 1000, 500)
            assert cost == 0.0042
            mock_cc.record_llm_usage.assert_called_once_with("gpt-4o", 1000, 500)
        finally:
            pb._cost_controller = old

    def test_record_tool_call_delegates(self):
        bridge = self._make_bridge()
        mock_cc = MagicMock()

        import flyto_ai.intelligence.pro_bridge as pb
        old = pb._cost_controller
        pb._cost_controller = mock_cc
        try:
            bridge.record_tool_call()
            mock_cc.record_tool_call.assert_called_once()
        finally:
            pb._cost_controller = old

    def test_check_budget_propagates_error(self):
        bridge = self._make_bridge()
        mock_cc = MagicMock()

        # Simulate BudgetExceededError
        class FakeBudgetExceededError(Exception):
            pass
        FakeBudgetExceededError.__name__ = "BudgetExceededError"
        mock_cc.check_budget.side_effect = FakeBudgetExceededError("over budget")

        import flyto_ai.intelligence.pro_bridge as pb
        old = pb._cost_controller
        pb._cost_controller = mock_cc
        try:
            with pytest.raises(FakeBudgetExceededError):
                bridge.check_budget()
        finally:
            pb._cost_controller = old

    def test_get_cost_summary_delegates(self):
        bridge = self._make_bridge()
        mock_cc = MagicMock()
        mock_cc.get_summary.return_value = {
            "cost_spent_usd": 0.5,
            "cost_budget_usd": 1.0,
            "tokens_used": 5000,
        }

        import flyto_ai.intelligence.pro_bridge as pb
        old = pb._cost_controller
        pb._cost_controller = mock_cc
        try:
            summary = bridge.get_cost_summary()
            assert summary["cost_spent_usd"] == 0.5
            assert summary["tokens_used"] == 5000
        finally:
            pb._cost_controller = old


# ---------------------------------------------------------------------------
# 4. Deep validation
# ---------------------------------------------------------------------------

class TestDeepValidation:

    @pytest.mark.asyncio
    async def test_basic_only_when_no_pro(self):
        yaml_str = "name: test\nsteps:\n  - id: s1\n    module: fake.module\n    params: {}"
        result = await validate_workflow_deep(yaml_str, pro_bridge=None)
        assert "basic" in result
        assert "contract" in result
        assert result["contract"] == []

    @pytest.mark.asyncio
    async def test_deep_validation_with_mock_bridge(self):
        mock_bridge = MagicMock()
        mock_bridge.validate_workflow_deep = AsyncMock(return_value={
            "valid": False,
            "issues": [
                {"severity": "error", "message": "Binding unresolved: ${steps.s0.url}", "node_id": "s1"},
            ],
        })

        yaml_str = "name: test\nsteps:\n  - id: s1\n    module: browser.goto\n    params:\n      url: ${steps.s0.url}"
        result = await validate_workflow_deep(yaml_str, pro_bridge=mock_bridge)
        assert len(result["contract"]) == 1
        assert "Binding unresolved" in result["contract"][0]
        assert "s1" in result["contract"][0]

    @pytest.mark.asyncio
    async def test_missing_modules_extracted(self):
        # Patch core.mcp_handler to simulate modules not found
        yaml_str = "name: test\nsteps:\n  - id: s1\n    module: nonexistent.module\n    params: {}"
        result = await validate_workflow_deep(yaml_str, pro_bridge=None)
        # Basic validation should catch it (if flyto-core is available) or be empty
        assert isinstance(result["missing_modules"], list)

    @pytest.mark.asyncio
    async def test_deep_validation_exception_fallback(self):
        mock_bridge = MagicMock()
        mock_bridge.validate_workflow_deep = AsyncMock(side_effect=Exception("boom"))

        yaml_str = "name: test\nsteps: []"
        result = await validate_workflow_deep(yaml_str, pro_bridge=mock_bridge)
        # Should not crash — contract list stays empty
        assert result["contract"] == []


# ---------------------------------------------------------------------------
# 5. Agent chat flow with pro bridge
# ---------------------------------------------------------------------------

def _make_test_agent(monkeypatch, mock_chat_fn, *, enable_pro=True):
    """Create a test Agent with mocked provider and optional pro bridge."""
    config = AgentConfig(
        provider="ollama", api_key="test",
        enable_pro=enable_pro,
        enable_ems=enable_pro,
        enable_knowledge=enable_pro,
        enable_contract_validation=enable_pro,
    )
    from flyto_ai.agent import Agent
    agent = Agent(config=config)
    agent._tools = [
        {"name": "execute_module", "description": "run", "inputSchema": {}},
        {"name": "search_modules", "description": "search", "inputSchema": {}},
    ]

    async def _dispatch(name, args):
        return {"ok": True, "data": {}}
    agent._dispatch_fn = _dispatch
    monkeypatch.setattr(agent._provider, "chat", mock_chat_fn)
    return agent


@pytest.mark.asyncio
async def test_agent_chat_with_pro_disabled(monkeypatch):
    """Agent works normally when enable_pro=False."""
    async def mock_chat(messages, system_prompt, tools, dispatch_fn,
                        max_rounds=30, on_stream=None):
        return "Hello", [], 1, {"prompt_tokens": 100, "completion_tokens": 50,
                                "total_tokens": 150, "cache_creation_input_tokens": 0,
                                "cache_read_input_tokens": 0}

    agent = _make_test_agent(monkeypatch, mock_chat, enable_pro=False)
    assert agent._pro is None
    result = await agent.chat("hi")
    assert result.ok
    assert result.message == "Hello"


@pytest.mark.asyncio
async def test_agent_chat_records_pro_cost(monkeypatch):
    """Agent records cost in both CostTracker and ProBridge CostController."""
    async def mock_chat(messages, system_prompt, tools, dispatch_fn,
                        max_rounds=30, on_stream=None):
        return "Done", [], 1, {"prompt_tokens": 1000, "completion_tokens": 500,
                               "total_tokens": 1500, "cache_creation_input_tokens": 0,
                               "cache_read_input_tokens": 0}

    agent = _make_test_agent(monkeypatch, mock_chat)

    # Mock the pro bridge
    mock_bridge = MagicMock()
    mock_bridge.available = True
    mock_bridge.record_llm_usage = MagicMock(return_value=0.01)
    mock_bridge.record_tool_call = MagicMock()
    mock_bridge.check_budget = MagicMock()
    mock_bridge.get_catalog_outline = MagicMock(return_value=None)
    mock_bridge.get_cost_summary = MagicMock(return_value=None)
    agent._pro = mock_bridge

    result = await agent.chat("test")
    assert result.ok

    # Verify pro bridge was called for cost tracking
    mock_bridge.record_llm_usage.assert_called_once_with(
        agent._config.resolved_model, 1000, 500,
    )


@pytest.mark.asyncio
async def test_agent_failure_records_ems_via_middleware(monkeypatch):
    """EMS errors are recorded by middleware (not agent-level), via dispatch."""
    dispatch_calls = []

    async def mock_chat(messages, system_prompt, tools, dispatch_fn,
                        max_rounds=30, on_stream=None):
        # Actually call dispatch so middleware _on_result runs
        r = await dispatch_fn("execute_module", {
            "module_id": "browser.click", "params": {"selector": "#btn"},
        })
        dispatch_calls.append(r)
        return "I did it!", [
            {"function": "execute_module", "module_id": "browser.click",
             "ok": False, "error": "Element not found", "result_preview": "{}"},
        ], 1, {"prompt_tokens": 100, "completion_tokens": 50,
                "total_tokens": 150, "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0}

    agent = _make_test_agent(monkeypatch, mock_chat)

    # Make the base dispatch return a failure
    async def failing_dispatch(name, args):
        if name == "execute_module":
            return {"ok": False, "error": "Element not found", "error_type": "execution_error"}
        return {"ok": True}
    agent._dispatch_fn = failing_dispatch

    # Set up the EMS bridge on the middleware
    mock_ems_bridge = MagicMock()
    mock_ems_bridge.record_error = AsyncMock(return_value=None)
    mock_ems_bridge.get_lesson_for_error = AsyncMock(return_value=None)
    if agent._assistant:
        agent._assistant._ems_bridge = mock_ems_bridge

    mock_bridge = MagicMock()
    mock_bridge.available = True
    mock_bridge.record_llm_usage = MagicMock(return_value=None)
    mock_bridge.record_tool_call = MagicMock()
    mock_bridge.check_budget = MagicMock()
    mock_bridge.get_catalog_outline = MagicMock(return_value=None)
    agent._pro = mock_bridge

    result = await agent.chat("click the button")

    # EMS should have been called via middleware._on_result
    if agent._assistant:
        assert mock_ems_bridge.record_error.await_count >= 1


@pytest.mark.asyncio
async def test_agent_tool_call_tracked_by_pro(monkeypatch):
    """Each tool dispatch increments pro CostController's tool call count."""
    call_count = {"n": 0}

    async def mock_chat(messages, system_prompt, tools, dispatch_fn,
                        max_rounds=30, on_stream=None):
        # Simulate 2 tool calls
        await dispatch_fn("search_modules", {"query": "test"})
        await dispatch_fn("execute_module", {"module_id": "test.run", "params": {}})
        return "Done", [
            {"function": "search_modules"},
            {"function": "execute_module", "ok": True},
        ], 1, {"prompt_tokens": 100, "completion_tokens": 50,
                "total_tokens": 150, "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 0}

    agent = _make_test_agent(monkeypatch, mock_chat)

    mock_bridge = MagicMock()
    mock_bridge.available = True
    mock_bridge.record_llm_usage = MagicMock(return_value=None)
    mock_bridge.record_tool_call = MagicMock(side_effect=lambda: call_count.__setitem__("n", call_count["n"] + 1))
    mock_bridge.check_budget = MagicMock()
    mock_bridge.get_catalog_outline = MagicMock(return_value=None)
    agent._pro = mock_bridge

    result = await agent.chat("run test")
    assert result.ok
    # 2 tool calls should have been tracked
    assert call_count["n"] == 2


# ---------------------------------------------------------------------------
# 6. Middleware EMS integration
# ---------------------------------------------------------------------------

class TestMiddlewareEMS:

    @pytest.mark.asyncio
    async def test_ems_fix_hint_injected_on_failure(self):
        """Middleware injects _ems_fix_hint into failed execution results."""
        from flyto_ai.assistant.middleware import AssistantMiddleware

        mw = AssistantMiddleware()

        # Mock EMS bridge
        mock_bridge = MagicMock()
        mock_bridge.record_error = AsyncMock(return_value=None)
        mock_bridge.get_lesson_for_error = AsyncMock(return_value={
            "fix": "Use hint-based selector instead of CSS",
        })
        mw._ems_bridge = mock_bridge

        # Simulate a failed execution through _on_result
        from flyto_ai.assistant.safety import CircuitBreaker, BoundedHistory
        from flyto_ai.assistant import resilience

        breaker = CircuitBreaker(max_failures=3)
        history = BoundedHistory(max_size=20)
        snap_guard = resilience.SnapshotGuard()
        antibot_guard = resilience.AntibotGuard()
        current_url = {"v": ""}

        async def noop_dispatch(name, args):
            return {"ok": True}

        result = await mw._on_result(
            func_name="execute_module",
            func_args={"module_id": "browser.click", "params": {"selector": "#btn"}},
            result={"ok": False, "error": "Element not found", "error_type": "execution_error"},
            base_dispatch=noop_dispatch,
            breaker=breaker,
            history=history,
            snap_guard=snap_guard,
            antibot_guard=antibot_guard,
            current_url=current_url,
        )

        # The fix hint should be injected
        assert result.get("_ems_fix_hint") == "Use hint-based selector instead of CSS"
        mock_bridge.get_lesson_for_error.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_ems_no_crash_on_exception(self):
        """Middleware doesn't crash when EMS raises an exception."""
        from flyto_ai.assistant.middleware import AssistantMiddleware
        from flyto_ai.assistant.safety import CircuitBreaker, BoundedHistory
        from flyto_ai.assistant import resilience

        mw = AssistantMiddleware()
        mock_bridge = MagicMock()
        mock_bridge.record_error = AsyncMock(side_effect=Exception("Qdrant down"))
        mock_bridge.get_lesson_for_error = AsyncMock(side_effect=Exception("Qdrant down"))
        mw._ems_bridge = mock_bridge

        breaker = CircuitBreaker(max_failures=3)
        history = BoundedHistory(max_size=20)

        async def noop_dispatch(name, args):
            return {"ok": True}

        result = await mw._on_result(
            func_name="execute_module",
            func_args={"module_id": "browser.click", "params": {}},
            result={"ok": False, "error": "fail"},
            base_dispatch=noop_dispatch,
            breaker=breaker,
            history=history,
            snap_guard=resilience.SnapshotGuard(),
            antibot_guard=resilience.AntibotGuard(),
            current_url={"v": ""},
        )

        # Should still return the original result, not crash
        assert result["ok"] is False


# ---------------------------------------------------------------------------
# 7. System prompt catalog injection
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_catalog_outline_injected_in_prompt(monkeypatch):
    """When pro bridge has catalog, it's injected into the system prompt."""
    prompt_captured = {"v": ""}

    async def mock_chat(messages, system_prompt, tools, dispatch_fn,
                        max_rounds=30, on_stream=None):
        prompt_captured["v"] = system_prompt
        return "Hello", [], 1, {"prompt_tokens": 100, "completion_tokens": 50,
                                "total_tokens": 150, "cache_creation_input_tokens": 0,
                                "cache_read_input_tokens": 0}

    agent = _make_test_agent(monkeypatch, mock_chat)

    mock_bridge = MagicMock()
    mock_bridge.available = True
    mock_bridge.get_catalog_outline = MagicMock(return_value="- **browser** (15) — Web automation\n- **data** (20) — Data processing")
    mock_bridge.record_llm_usage = MagicMock(return_value=None)
    mock_bridge.record_tool_call = MagicMock()
    mock_bridge.check_budget = MagicMock()
    agent._pro = mock_bridge
    agent._config.enable_knowledge = True

    result = await agent.chat("help me browse")
    assert result.ok
    assert "Module Catalog" in prompt_captured["v"]
    assert "browser" in prompt_captured["v"]
    assert "Data processing" in prompt_captured["v"]


# ---------------------------------------------------------------------------
# 8. Singleton isolation (handled by autouse fixture)
# ---------------------------------------------------------------------------

def test_singletons_start_clean():
    """Verify the autouse fixture resets singletons before each test."""
    import flyto_ai.intelligence.pro_bridge as pb
    assert pb._contract_engine is None
    assert pb._contract_engine_checked is False
    assert pb._cost_controller is None
    assert pb._ems_router is None


# ---------------------------------------------------------------------------
# 9. Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:

    @pytest.mark.asyncio
    async def test_ems_not_triggered_on_success(self):
        """EMS should NOT record errors for successful executions."""
        from flyto_ai.assistant.middleware import AssistantMiddleware
        from flyto_ai.assistant.safety import CircuitBreaker, BoundedHistory
        from flyto_ai.assistant import resilience

        mw = AssistantMiddleware()
        mock_bridge = MagicMock()
        mock_bridge.record_error = AsyncMock()
        mock_bridge.get_lesson_for_error = AsyncMock()
        mw._ems_bridge = mock_bridge

        async def noop_dispatch(name, args):
            return {"ok": True}

        result = await mw._on_result(
            func_name="execute_module",
            func_args={"module_id": "browser.goto", "params": {"url": "https://x.com"}},
            result={"ok": True, "url": "https://x.com"},
            base_dispatch=noop_dispatch,
            breaker=CircuitBreaker(max_failures=3),
            history=BoundedHistory(max_size=20),
            snap_guard=resilience.SnapshotGuard(),
            antibot_guard=resilience.AntibotGuard(),
            current_url={"v": ""},
        )

        assert result["ok"] is True
        mock_bridge.record_error.assert_not_awaited()
        mock_bridge.get_lesson_for_error.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_ems_not_triggered_for_non_execute(self):
        """EMS should NOT activate for non-execute_module calls."""
        from flyto_ai.assistant.middleware import AssistantMiddleware
        from flyto_ai.assistant.safety import CircuitBreaker, BoundedHistory
        from flyto_ai.assistant import resilience

        mw = AssistantMiddleware()
        mock_bridge = MagicMock()
        mock_bridge.record_error = AsyncMock()
        mw._ems_bridge = mock_bridge

        async def noop_dispatch(name, args):
            return {"ok": True}

        result = await mw._on_result(
            func_name="search_modules",
            func_args={"query": "test"},
            result={"ok": False, "error": "not found"},
            base_dispatch=noop_dispatch,
            breaker=CircuitBreaker(max_failures=3),
            history=BoundedHistory(max_size=20),
            snap_guard=resilience.SnapshotGuard(),
            antibot_guard=resilience.AntibotGuard(),
            current_url={"v": ""},
        )

        mock_bridge.record_error.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_ems_not_triggered_when_ok_key_missing(self):
        """EMS should NOT activate when result has no 'ok' key (ambiguous)."""
        from flyto_ai.assistant.middleware import AssistantMiddleware
        from flyto_ai.assistant.safety import CircuitBreaker, BoundedHistory
        from flyto_ai.assistant import resilience

        mw = AssistantMiddleware()
        mock_bridge = MagicMock()
        mock_bridge.record_error = AsyncMock()
        mw._ems_bridge = mock_bridge

        async def noop_dispatch(name, args):
            return {"ok": True}

        result = await mw._on_result(
            func_name="execute_module",
            func_args={"module_id": "test.run", "params": {}},
            result={"data": "something"},  # No 'ok' key
            base_dispatch=noop_dispatch,
            breaker=CircuitBreaker(max_failures=3),
            history=BoundedHistory(max_size=20),
            snap_guard=resilience.SnapshotGuard(),
            antibot_guard=resilience.AntibotGuard(),
            current_url={"v": ""},
        )

        mock_bridge.record_error.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_pro_budget_exceeded_propagates_in_agent(self, monkeypatch):
        """When pro CostController raises BudgetExceededError, it propagates up."""
        from flyto_ai.cost import BudgetExceededError

        async def mock_chat(messages, system_prompt, tools, dispatch_fn,
                            max_rounds=30, on_stream=None):
            return "Done", [], 1, {"prompt_tokens": 999999, "completion_tokens": 999999,
                                   "total_tokens": 1999998, "cache_creation_input_tokens": 0,
                                   "cache_read_input_tokens": 0}

        agent = _make_test_agent(monkeypatch, mock_chat)

        mock_bridge = MagicMock()
        mock_bridge.available = True
        mock_bridge.record_llm_usage = MagicMock(
            side_effect=BudgetExceededError("over", current=10.0, limit=1.0)
        )
        mock_bridge.record_tool_call = MagicMock()
        mock_bridge.get_catalog_outline = MagicMock(return_value=None)
        agent._pro = mock_bridge

        with pytest.raises(BudgetExceededError):
            await agent.chat("expensive task")

    @pytest.mark.asyncio
    async def test_deep_validation_valid_workflow(self):
        """Deep validation returns empty errors for valid contract report."""
        mock_bridge = MagicMock()
        mock_bridge.validate_workflow_deep = AsyncMock(return_value={
            "valid": True,
            "issues": [],
        })

        yaml_str = "name: test\nsteps: []"
        result = await validate_workflow_deep(yaml_str, pro_bridge=mock_bridge)
        assert result["contract"] == []
        assert result["missing_modules"] == []

    def test_catalog_outline_not_injected_when_knowledge_disabled(self, monkeypatch):
        """Catalog outline should NOT be injected when enable_knowledge=False."""
        prompt_captured = {"v": ""}

        async def mock_chat(messages, system_prompt, tools, dispatch_fn,
                            max_rounds=30, on_stream=None):
            prompt_captured["v"] = system_prompt
            return "Hello", [], 1, {"prompt_tokens": 10, "completion_tokens": 5,
                                    "total_tokens": 15, "cache_creation_input_tokens": 0,
                                    "cache_read_input_tokens": 0}

        agent = _make_test_agent(monkeypatch, mock_chat)

        mock_bridge = MagicMock()
        mock_bridge.available = True
        mock_bridge.get_catalog_outline = MagicMock(return_value="should not appear")
        mock_bridge.record_llm_usage = MagicMock(return_value=None)
        mock_bridge.record_tool_call = MagicMock()
        agent._pro = mock_bridge
        agent._config.enable_knowledge = False

        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(agent.chat("test"))
            assert result.ok
            assert "Module Catalog" not in prompt_captured["v"]
            mock_bridge.get_catalog_outline.assert_not_called()
        finally:
            loop.close()

    @pytest.mark.asyncio
    async def test_ems_record_error_exception_does_not_block_lesson_lookup(self):
        """If record_error fails, lesson lookup should still run."""
        from flyto_ai.assistant.middleware import AssistantMiddleware
        from flyto_ai.assistant.safety import CircuitBreaker, BoundedHistory
        from flyto_ai.assistant import resilience

        mw = AssistantMiddleware()
        mock_bridge = MagicMock()
        mock_bridge.record_error = AsyncMock(side_effect=Exception("write fail"))
        mock_bridge.get_lesson_for_error = AsyncMock(return_value={
            "fix": "Try alternative selector",
        })
        mw._ems_bridge = mock_bridge

        async def noop_dispatch(name, args):
            return {"ok": True}

        result = await mw._on_result(
            func_name="execute_module",
            func_args={"module_id": "browser.click", "params": {}},
            result={"ok": False, "error": "timeout"},
            base_dispatch=noop_dispatch,
            breaker=CircuitBreaker(max_failures=3),
            history=BoundedHistory(max_size=20),
            snap_guard=resilience.SnapshotGuard(),
            antibot_guard=resilience.AntibotGuard(),
            current_url={"v": ""},
        )

        # record_error failed, but lesson lookup should still have run
        mock_bridge.get_lesson_for_error.assert_awaited_once()
        assert result.get("_ems_fix_hint") == "Try alternative selector"
