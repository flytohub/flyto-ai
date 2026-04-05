# Copyright 2024 Flyto
# Licensed under the Apache License, Version 2.0
"""Comprehensive test suite for the claw-code architecture upgrade.

Tests all phases:
  Phase 1A: Protocols (ApiClient, ToolExecutor)
  Phase 1B: Permissions (PermissionLevel, PermissionEnforcer)
  Phase 2A: Hook Pipeline (ShellHookRunner, HookDecision, HookRegistry)
  Phase 2B: Auto-Compact (auto_compact_from_usage, transcript rotation)
  Phase 3A: Provider Auto-Detection (detect_provider, MODEL_PREFIX_MAP)
  Phase 3B: Prompt Cache (PromptCache, fnv1a_64, CacheStats)
  Phase 3C: MCP Client Lifecycle (McpConnectionState, McpClientManager)
  Phase 4A: Telemetry (TelemetrySink, SessionTracer, sinks)
  Phase 4B: Mock Testing (MockApiClient, MockToolExecutor)

These tests are self-contained and require only stdlib — no pydantic,
no openai, no external deps.
"""
import asyncio
import importlib
import importlib.util
import json
import os
import shutil
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

# Ensure flyto_ai directory is on the path
_BASE = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, _BASE)

# Direct module loader — bypasses flyto_ai/__init__.py to avoid pydantic dependency
_MODULE_CACHE = {}


def _load_module(dotted_name: str, file_path: str):
    """Load a single module by file path, bypassing package __init__.py."""
    if dotted_name in _MODULE_CACHE:
        return _MODULE_CACHE[dotted_name]
    fpath = os.path.join(_BASE, file_path)
    spec = importlib.util.spec_from_file_location(dotted_name, fpath)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[dotted_name] = mod  # register so child imports resolve
    spec.loader.exec_module(mod)
    _MODULE_CACHE[dotted_name] = mod
    return mod


# Pre-load the new modules that have no pydantic deps
# Order matters — load leaf modules first
# Try normal imports first (works when pydantic is installed).
# Fall back to direct module loading when running without a venv.
try:
    import pydantic as _pydantic_check  # noqa: F401
    _HAS_PYDANTIC = True
except ImportError:
    _HAS_PYDANTIC = False

if not _HAS_PYDANTIC:
    # Register stub packages so submodule imports resolve
    import types
    for _pkg in ("flyto_ai", "flyto_ai.extensions", "flyto_ai.memory"):
        if _pkg not in sys.modules:
            sys.modules[_pkg] = types.ModuleType(_pkg)

    _load_module("flyto_ai.extensions.base", os.path.join("flyto_ai", "extensions", "base.py"))
    _load_module("flyto_ai.extensions.shell_hook", os.path.join("flyto_ai", "extensions", "shell_hook.py"))
    _load_module("flyto_ai.extensions.hooks", os.path.join("flyto_ai", "extensions", "hooks.py"))
    _load_module("flyto_ai.permissions", os.path.join("flyto_ai", "permissions.py"))
    _load_module("flyto_ai.cache", os.path.join("flyto_ai", "cache.py"))
    _load_module("flyto_ai.telemetry", os.path.join("flyto_ai", "telemetry.py"))
    _load_module("flyto_ai.testing", os.path.join("flyto_ai", "testing.py"))
    _load_module("flyto_ai.mcp_client", os.path.join("flyto_ai", "mcp_client.py"))
    _load_module("flyto_ai.memory.compaction", os.path.join("flyto_ai", "memory", "compaction.py"))
    _load_module("flyto_ai.transcript", os.path.join("flyto_ai", "transcript.py"))
    _load_module("flyto_ai.protocols", os.path.join("flyto_ai", "protocols.py"))


# ═══════════════════════════════════════════════════════════════════
# Phase 1A: Protocols
# ═══════════════════════════════════════════════════════════════════

class TestProtocols(unittest.TestCase):
    """Test ApiClient and ToolExecutor protocol definitions."""

    def test_protocol_classes_exist(self):
        from flyto_ai.protocols import ApiClient, ToolExecutor, DispatchFn
        self.assertTrue(hasattr(ApiClient, 'chat'))
        self.assertTrue(hasattr(ToolExecutor, 'tools'))
        self.assertTrue(hasattr(ToolExecutor, 'dispatch'))

    def test_runtime_checkable(self):
        from flyto_ai.protocols import ApiClient, ToolExecutor

        class FakeClient:
            async def chat(self, messages, system_prompt, tools, dispatch_fn,
                           max_rounds=30, on_stream=None):
                return ("hi", [], 1, {})

        class FakeExecutor:
            @property
            def tools(self):
                return []
            async def dispatch(self, name, arguments):
                return {}

        self.assertIsInstance(FakeClient(), ApiClient)
        self.assertIsInstance(FakeExecutor(), ToolExecutor)

    def test_non_conforming_rejected(self):
        from flyto_ai.protocols import ApiClient, ToolExecutor

        class NotAClient:
            pass

        class NotAnExecutor:
            pass

        self.assertNotIsInstance(NotAClient(), ApiClient)
        self.assertNotIsInstance(NotAnExecutor(), ToolExecutor)


# ═══════════════════════════════════════════════════════════════════
# Phase 1B: Permissions
# ═══════════════════════════════════════════════════════════════════

class TestPermissions(unittest.TestCase):
    """Test three-tier permission model."""

    def test_permission_level_ordering(self):
        from flyto_ai.permissions import PermissionLevel
        self.assertTrue(PermissionLevel.READ_ONLY < PermissionLevel.WORKSPACE_WRITE)
        self.assertTrue(PermissionLevel.WORKSPACE_WRITE < PermissionLevel.DANGER_FULL)

    def test_permission_level_from_string(self):
        from flyto_ai.permissions import PermissionLevel
        self.assertEqual(PermissionLevel["READ_ONLY"], PermissionLevel.READ_ONLY)
        self.assertEqual(PermissionLevel["WORKSPACE_WRITE"], PermissionLevel.WORKSPACE_WRITE)
        self.assertEqual(PermissionLevel["DANGER_FULL"], PermissionLevel.DANGER_FULL)

    def test_read_only_allows_discovery(self):
        from flyto_ai.permissions import PermissionEnforcer, PermissionLevel
        e = PermissionEnforcer(PermissionLevel.READ_ONLY)
        for tool in ["list_modules", "search_modules", "get_module_info", "ask_user"]:
            d = e.check(tool)
            self.assertTrue(d.allowed, f"{tool} should be allowed at READ_ONLY")

    def test_read_only_blocks_execution(self):
        from flyto_ai.permissions import PermissionEnforcer, PermissionLevel
        e = PermissionEnforcer(PermissionLevel.READ_ONLY)
        d = e.check("execute_module", {"module_id": "browser.click"})
        self.assertFalse(d.allowed)
        self.assertIn("WORKSPACE_WRITE", d.reason)

    def test_workspace_allows_safe_modules(self):
        from flyto_ai.permissions import PermissionEnforcer, PermissionLevel
        e = PermissionEnforcer(PermissionLevel.WORKSPACE_WRITE)
        d = e.check("execute_module", {"module_id": "browser.click"})
        self.assertTrue(d.allowed)

    def test_workspace_blocks_dangerous_modules(self):
        from flyto_ai.permissions import PermissionEnforcer, PermissionLevel
        e = PermissionEnforcer(PermissionLevel.WORKSPACE_WRITE)
        for mod in ["shell.run", "docker.exec", "k8s.apply", "ssh.connect", "file.write"]:
            d = e.check("execute_module", {"module_id": mod})
            self.assertFalse(d.allowed, f"{mod} should be blocked at WORKSPACE_WRITE")
            self.assertIn("DANGER_FULL", d.reason)

    def test_danger_full_allows_everything(self):
        from flyto_ai.permissions import PermissionEnforcer, PermissionLevel
        e = PermissionEnforcer(PermissionLevel.DANGER_FULL)
        for tool, args in [
            ("list_modules", {}),
            ("execute_module", {"module_id": "shell.run"}),
            ("execute_module", {"module_id": "docker.exec"}),
        ]:
            d = e.check(tool, args)
            self.assertTrue(d.allowed, f"{tool} should be allowed at DANGER_FULL")

    def test_overrides(self):
        from flyto_ai.permissions import PermissionEnforcer, PermissionLevel
        e = PermissionEnforcer(
            PermissionLevel.READ_ONLY,
            overrides={"execute_module": PermissionLevel.READ_ONLY},
        )
        # Override makes execute_module allowed at READ_ONLY
        # (but module category check still applies)
        d = e.check("execute_module", {"module_id": "browser.click"})
        # Module category requires WORKSPACE_WRITE, which is > READ_ONLY
        self.assertFalse(d.allowed)

    def test_unknown_tool_defaults_to_workspace(self):
        from flyto_ai.permissions import PermissionEnforcer, PermissionLevel
        e = PermissionEnforcer(PermissionLevel.READ_ONLY)
        d = e.check("some_unknown_tool", {})
        self.assertFalse(d.allowed)
        e2 = PermissionEnforcer(PermissionLevel.WORKSPACE_WRITE)
        d2 = e2.check("some_unknown_tool", {})
        self.assertTrue(d2.allowed)


# ═══════════════════════════════════════════════════════════════════
# Phase 2A: Hook Pipeline
# ═══════════════════════════════════════════════════════════════════

class TestShellHookRunner(unittest.TestCase):
    """Test shell hook runner with deny semantics."""

    def test_hook_decision_dataclass(self):
        from flyto_ai.extensions.shell_hook import HookDecision
        d = HookDecision(allowed=True)
        self.assertTrue(d.allowed)
        self.assertEqual(d.reason, "")
        self.assertIsNone(d.modified_arguments)

        d2 = HookDecision(allowed=False, reason="budget exceeded")
        self.assertFalse(d2.allowed)
        self.assertEqual(d2.reason, "budget exceeded")

    def test_no_hooks_allows(self):
        from flyto_ai.extensions.shell_hook import ShellHookRunner
        runner = ShellHookRunner([])
        result = asyncio.get_event_loop().run_until_complete(
            runner.run_before_tool_call("test", {})
        )
        self.assertTrue(result.allowed)

    def test_allow_hook(self):
        from flyto_ai.extensions.shell_hook import ShellHookRunner, ShellHookConfig
        runner = ShellHookRunner([
            ShellHookConfig(command="exit 0", event="before_tool_call", name="allow"),
        ])
        result = asyncio.get_event_loop().run_until_complete(
            runner.run_before_tool_call("test_tool", {"arg": "val"})
        )
        self.assertTrue(result.allowed)

    def test_deny_hook(self):
        from flyto_ai.extensions.shell_hook import ShellHookRunner, ShellHookConfig
        runner = ShellHookRunner([
            ShellHookConfig(
                command='echo "budget exceeded" >&2; exit 2',
                event="before_tool_call",
                name="deny",
            ),
        ])
        result = asyncio.get_event_loop().run_until_complete(
            runner.run_before_tool_call("execute_module", {"module_id": "shell.run"})
        )
        self.assertFalse(result.allowed)
        self.assertIn("budget exceeded", result.reason)

    def test_short_circuit_on_deny(self):
        from flyto_ai.extensions.shell_hook import ShellHookRunner, ShellHookConfig
        # Second hook should never run
        runner = ShellHookRunner([
            ShellHookConfig(command='exit 2', event="before_tool_call", name="deny"),
            ShellHookConfig(command='echo "should not run"', event="before_tool_call", name="allow"),
        ])
        result = asyncio.get_event_loop().run_until_complete(
            runner.run_before_tool_call("test", {})
        )
        self.assertFalse(result.allowed)

    def test_error_hook_continues(self):
        from flyto_ai.extensions.shell_hook import ShellHookRunner, ShellHookConfig
        # Exit code 1 = error, pipeline continues
        runner = ShellHookRunner([
            ShellHookConfig(command='exit 1', event="before_tool_call", name="error"),
        ])
        result = asyncio.get_event_loop().run_until_complete(
            runner.run_before_tool_call("test", {})
        )
        self.assertTrue(result.allowed)

    def test_hooks_for_event_filtering(self):
        from flyto_ai.extensions.shell_hook import ShellHookRunner, ShellHookConfig
        runner = ShellHookRunner([
            ShellHookConfig(command='exit 0', event="before_tool_call"),
            ShellHookConfig(command='exit 0', event="after_tool_call"),
            ShellHookConfig(command='exit 0', event="before_tool_call"),
        ])
        self.assertEqual(len(runner.hooks_for_event("before_tool_call")), 2)
        self.assertEqual(len(runner.hooks_for_event("after_tool_call")), 1)
        self.assertEqual(len(runner.hooks_for_event("on_error")), 0)


class TestHookRegistryDecision(unittest.TestCase):
    """Test enhanced HookRegistry with HookDecision return type."""

    def test_no_extensions_allows(self):
        from flyto_ai.extensions.hooks import HookRegistry
        registry = HookRegistry()
        result = asyncio.get_event_loop().run_until_complete(
            registry.invoke_before_tool_call("test", {"arg": 1})
        )
        self.assertTrue(result.allowed)

    def test_blocking_extension_returns_decision(self):
        from flyto_ai.extensions.hooks import HookRegistry
        from flyto_ai.extensions.base import ExtensionBase, ExtensionManifest

        class BlockingExt(ExtensionBase):
            async def before_tool_call(self, tool_name, arguments):
                return {"_block": True, "_reason": "test block reason"}

        ext = BlockingExt(ExtensionManifest(
            name="blocker", version="1.0", hooks=["before_tool_call"],
        ))
        registry = HookRegistry()
        registry.register(ext)

        result = asyncio.get_event_loop().run_until_complete(
            registry.invoke_before_tool_call("execute_module", {"module_id": "shell.run"})
        )
        self.assertFalse(result.allowed)
        self.assertIn("test block reason", result.reason)


# ═══════════════════════════════════════════════════════════════════
# Phase 2B: Auto-Compact
# ═══════════════════════════════════════════════════════════════════

class TestAutoCompact(unittest.TestCase):
    """Test auto-compact with real token counts."""

    def test_auto_compact_below_threshold(self):
        from flyto_ai.memory.compaction import ContextCompactor
        c = ContextCompactor(soft_threshold=80000, hard_threshold=120000)
        msgs = [{"role": "user", "content": "hello"}]
        result, compacted = c.auto_compact_from_usage(msgs, prompt_tokens=5000)
        self.assertFalse(compacted)
        self.assertEqual(len(result), 1)

    def test_auto_compact_soft_threshold(self):
        from flyto_ai.memory.compaction import ContextCompactor
        c = ContextCompactor(soft_threshold=80000, hard_threshold=120000, keep_recent=2)
        msgs = [{"role": "user", "content": f"msg {i}"} for i in range(20)]
        result, compacted = c.auto_compact_from_usage(msgs, prompt_tokens=90000)
        self.assertTrue(compacted)
        # Should keep recent messages + summary
        self.assertTrue(len(result) < len(msgs))

    def test_auto_compact_hard_threshold(self):
        from flyto_ai.memory.compaction import ContextCompactor
        c = ContextCompactor(soft_threshold=80000, hard_threshold=120000, keep_recent=4)
        msgs = [{"role": "user", "content": f"msg {i}"} for i in range(30)]
        result, compacted = c.auto_compact_from_usage(msgs, prompt_tokens=130000)
        self.assertTrue(compacted)
        # Hard compact keeps max(3, keep_recent//2) messages + optional summary
        self.assertTrue(len(result) <= 4)  # keep + summary
        self.assertTrue(len(result) < len(msgs))

    def test_auto_compact_zero_tokens_fallback(self):
        from flyto_ai.memory.compaction import ContextCompactor
        c = ContextCompactor(soft_threshold=10, hard_threshold=20, keep_recent=2)
        msgs = [{"role": "user", "content": "x" * 100} for _ in range(10)]
        # 0 tokens → falls back to heuristic maybe_compact
        result, compacted = c.auto_compact_from_usage(msgs, prompt_tokens=0)
        self.assertTrue(compacted)


class TestTranscriptRotation(unittest.TestCase):
    """Test transcript file rotation."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_rotation_on_size_exceeded(self):
        from flyto_ai.transcript import TranscriptWriter, MAX_FILE_SIZE_BYTES

        tw = TranscriptWriter("test-session", transcript_dir=self.tmpdir)
        # Write enough data to exceed MAX_FILE_SIZE_BYTES
        big_data = "x" * (MAX_FILE_SIZE_BYTES + 100)
        tw.record_meta({"payload": big_data})
        # File should exist now
        self.assertTrue(tw.path.exists() or (tw.path.parent / "test-session.1.jsonl").exists())

        # Write again — should trigger rotation
        tw.record_meta({"event": "after_rotation"})
        rotated = tw.path.parent / "test-session.1.jsonl"
        # At least one file should exist
        files = list(Path(self.tmpdir).glob("test-session*.jsonl"))
        self.assertGreaterEqual(len(files), 1)

    def test_max_rotated_files(self):
        from flyto_ai.transcript import TranscriptWriter, MAX_FILE_SIZE_BYTES, MAX_ROTATED_FILES

        tw = TranscriptWriter("rot-test", transcript_dir=self.tmpdir)
        big = "y" * (MAX_FILE_SIZE_BYTES + 100)

        # Force multiple rotations
        for i in range(MAX_ROTATED_FILES + 2):
            tw.record_meta({"payload": big, "iteration": i})

        files = list(Path(self.tmpdir).glob("rot-test*.jsonl"))
        # Should not exceed MAX_ROTATED_FILES + 1 (current + rotated)
        self.assertLessEqual(len(files), MAX_ROTATED_FILES + 1)


# ═══════════════════════════════════════════════════════════════════
# Phase 3A: Provider Auto-Detection
# ═══════════════════════════════════════════════════════════════════

def _get_provider_detection():
    """Extract detect_provider and MODEL_PREFIX_MAP without importing pydantic.

    The providers/__init__.py imports LLMProvider (→ pydantic), but
    detect_provider only uses stdlib. We extract just the functions we need.
    """
    fpath = os.path.join(_BASE, "flyto_ai", "providers", "__init__.py")
    with open(fpath) as f:
        src = f.read()
    # Execute only the parts after "from flyto_ai.providers.base import LLMProvider"
    # by replacing that import with a no-op
    src = src.replace("from flyto_ai.providers.base import LLMProvider", "LLMProvider = None")
    ns = {"__name__": "flyto_ai.providers", "__file__": fpath}
    exec(compile(src, fpath, "exec"), ns)
    return ns["detect_provider"], ns["MODEL_PREFIX_MAP"]


_detect_provider, _MODEL_PREFIX_MAP = _get_provider_detection()


class TestProviderDetection(unittest.TestCase):
    """Test model-name-based provider auto-detection."""

    def test_anthropic_models(self):
        self.assertEqual(_detect_provider("claude-sonnet-4-5-20250929"), "anthropic")
        self.assertEqual(_detect_provider("claude-haiku-4-5-20251001"), "anthropic")
        self.assertEqual(_detect_provider("claude-opus-4-6"), "anthropic")

    def test_openai_models(self):
        self.assertEqual(_detect_provider("gpt-4o"), "openai")
        self.assertEqual(_detect_provider("gpt-4o-mini"), "openai")
        self.assertEqual(_detect_provider("o1-preview"), "openai")
        self.assertEqual(_detect_provider("o3-mini"), "openai")

    def test_ollama_models(self):
        self.assertEqual(_detect_provider("llama3.2"), "ollama")
        self.assertEqual(_detect_provider("qwen2.5:7b"), "ollama")
        self.assertEqual(_detect_provider("mistral"), "ollama")
        self.assertEqual(_detect_provider("phi3"), "ollama")
        self.assertEqual(_detect_provider("gemma2"), "ollama")

    def test_deepseek_with_key(self):
        self.assertEqual(_detect_provider("deepseek-r1", api_key="sk-xxx"), "openai")

    def test_deepseek_without_key(self):
        with patch.dict(os.environ, {}, clear=True):
            result = _detect_provider("deepseek-r1", api_key="")
            self.assertEqual(result, "ollama")

    def test_case_insensitive(self):
        self.assertEqual(_detect_provider("Claude-Sonnet-4-5"), "anthropic")
        self.assertEqual(_detect_provider("GPT-4o"), "openai")

    def test_unknown_model_with_key(self):
        self.assertEqual(_detect_provider("custom-model", api_key="sk-xxx"), "openai")

    def test_env_var_fallback(self):
        with patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-xxx"}, clear=True):
            self.assertEqual(_detect_provider("custom-model"), "anthropic")

    def test_model_prefix_map_complete(self):
        self.assertIn("claude-", _MODEL_PREFIX_MAP)
        self.assertIn("gpt-", _MODEL_PREFIX_MAP)
        self.assertIn("llama", _MODEL_PREFIX_MAP)
        self.assertIn("deepseek", _MODEL_PREFIX_MAP)


# ═══════════════════════════════════════════════════════════════════
# Phase 3B: Prompt Cache
# ═══════════════════════════════════════════════════════════════════

class TestPromptCache(unittest.TestCase):
    """Test prompt cache fingerprinting and tracking."""

    def test_fnv1a_hash(self):
        from flyto_ai.cache import fnv1a_64
        h1 = fnv1a_64("hello")
        h2 = fnv1a_64("hello")
        h3 = fnv1a_64("world")
        self.assertEqual(h1, h2)
        self.assertNotEqual(h1, h3)
        self.assertIsInstance(h1, int)

    def test_fingerprint_stable(self):
        from flyto_ai.cache import PromptCache
        cache = PromptCache()
        fp1 = cache.update_fingerprint("system prompt", [{"name": "tool_a"}, {"name": "tool_b"}])
        fp2 = cache.update_fingerprint("system prompt", [{"name": "tool_b"}, {"name": "tool_a"}])
        # Tool names sorted → same fingerprint regardless of order
        self.assertEqual(fp1, fp2)

    def test_fingerprint_changes_on_prompt_change(self):
        from flyto_ai.cache import PromptCache
        cache = PromptCache()
        fp1 = cache.update_fingerprint("prompt v1", [])
        fp2 = cache.update_fingerprint("prompt v2", [])
        self.assertNotEqual(fp1, fp2)

    def test_expect_cache_hit_within_ttl(self):
        from flyto_ai.cache import PromptCache
        cache = PromptCache(ttl_seconds=60.0)
        cache.update_fingerprint("test", [])
        self.assertTrue(cache.should_expect_cache_hit())

    def test_no_expect_hit_without_fingerprint(self):
        from flyto_ai.cache import PromptCache
        cache = PromptCache()
        self.assertFalse(cache.should_expect_cache_hit())

    def test_record_usage_tracks_hits(self):
        from flyto_ai.cache import PromptCache
        cache = PromptCache(ttl_seconds=60.0)
        cache.update_fingerprint("test", [])

        cache.record_usage(cache_creation_tokens=0, cache_read_tokens=500)
        self.assertEqual(cache.stats.actual_hits, 1)
        self.assertEqual(cache.stats.total_cache_read_tokens, 500)

    def test_unexpected_cache_break(self):
        from flyto_ai.cache import PromptCache
        cache = PromptCache(ttl_seconds=60.0)
        cache.update_fingerprint("test", [])

        # Expected hit but got creation instead of read
        cache.record_usage(cache_creation_tokens=1000, cache_read_tokens=0)
        self.assertEqual(cache.stats.unexpected_breaks, 1)

    def test_cache_stats_summary(self):
        from flyto_ai.cache import CacheStats
        stats = CacheStats(
            total_calls=10, expected_hits=8, actual_hits=6,
            unexpected_breaks=2, total_cache_creation_tokens=1000,
            total_cache_read_tokens=5000,
        )
        summary = stats.summary()
        self.assertEqual(summary["hit_rate"], 0.6)
        self.assertEqual(summary["total_calls"], 10)

    def test_reset(self):
        from flyto_ai.cache import PromptCache
        cache = PromptCache()
        cache.update_fingerprint("test", [])
        cache.record_usage(cache_read_tokens=100)
        cache.reset()
        self.assertEqual(cache.fingerprint, 0)
        self.assertEqual(cache.stats.total_calls, 0)


# ═══════════════════════════════════════════════════════════════════
# Phase 3C: MCP Client Lifecycle
# ═══════════════════════════════════════════════════════════════════

class TestMcpClientLifecycle(unittest.TestCase):
    """Test MCP client state machine."""

    def test_initial_state(self):
        from flyto_ai.mcp_client import McpClientManager, McpConnectionState
        mgr = McpClientManager(["echo", "test"], name="test-server")
        self.assertEqual(mgr.state, McpConnectionState.DISCONNECTED)
        self.assertFalse(mgr.is_available)
        self.assertEqual(mgr.tools, [])

    def test_state_enum_values(self):
        from flyto_ai.mcp_client import McpConnectionState
        self.assertEqual(McpConnectionState.INITIALIZING.value, "initializing")
        self.assertEqual(McpConnectionState.READY.value, "ready")
        self.assertEqual(McpConnectionState.DEGRADED.value, "degraded")
        self.assertEqual(McpConnectionState.DISCONNECTED.value, "disconnected")
        self.assertEqual(McpConnectionState.RECONNECTING.value, "reconnecting")

    def test_is_available_states(self):
        from flyto_ai.mcp_client import McpClientManager, McpConnectionState
        mgr = McpClientManager(["echo"])

        mgr._state = McpConnectionState.READY
        self.assertTrue(mgr.is_available)

        mgr._state = McpConnectionState.DEGRADED
        self.assertTrue(mgr.is_available)

        mgr._state = McpConnectionState.DISCONNECTED
        self.assertFalse(mgr.is_available)

        mgr._state = McpConnectionState.RECONNECTING
        self.assertFalse(mgr.is_available)

    def test_call_tool_when_disconnected(self):
        from flyto_ai.mcp_client import McpClientManager
        mgr = McpClientManager(["echo"])
        result = asyncio.get_event_loop().run_until_complete(
            mgr.call_tool("search", {"query": "test"})
        )
        self.assertFalse(result.get("ok", True))
        self.assertIn("disconnected", result.get("error", ""))

    def test_tool_info_dataclass(self):
        from flyto_ai.mcp_client import McpToolInfo
        tool = McpToolInfo(name="search", description="Search code")
        self.assertEqual(tool.name, "search")
        self.assertEqual(tool.description, "Search code")
        self.assertEqual(tool.input_schema, {})


# ═══════════════════════════════════════════════════════════════════
# Phase 4A: Telemetry
# ═══════════════════════════════════════════════════════════════════

class TestTelemetry(unittest.TestCase):
    """Test unified telemetry infrastructure."""

    def test_memory_sink_collects_events(self):
        from flyto_ai.telemetry import MemoryTelemetrySink, TelemetryEvent, TelemetryEventType
        sink = MemoryTelemetrySink()
        event = TelemetryEvent(
            type=TelemetryEventType.TOOL_CALL,
            session_id="test",
            data={"tool_name": "search"},
        )
        sink.emit(event)
        self.assertEqual(len(sink.events), 1)
        self.assertEqual(sink.events[0].type, TelemetryEventType.TOOL_CALL)

    def test_memory_sink_events_of_type(self):
        from flyto_ai.telemetry import MemoryTelemetrySink, TelemetryEvent, TelemetryEventType
        sink = MemoryTelemetrySink()
        sink.emit(TelemetryEvent(type=TelemetryEventType.TOOL_CALL, data={}))
        sink.emit(TelemetryEvent(type=TelemetryEventType.LLM_RESPONSE, data={}))
        sink.emit(TelemetryEvent(type=TelemetryEventType.TOOL_CALL, data={}))
        self.assertEqual(len(sink.events_of_type(TelemetryEventType.TOOL_CALL)), 2)
        self.assertEqual(len(sink.events_of_type(TelemetryEventType.LLM_RESPONSE)), 1)

    def test_memory_sink_clear(self):
        from flyto_ai.telemetry import MemoryTelemetrySink, TelemetryEvent, TelemetryEventType
        sink = MemoryTelemetrySink()
        sink.emit(TelemetryEvent(type=TelemetryEventType.TOOL_CALL, data={}))
        sink.clear()
        self.assertEqual(len(sink.events), 0)

    def test_jsonl_sink_writes_file(self):
        from flyto_ai.telemetry import JsonlTelemetrySink, TelemetryEvent, TelemetryEventType
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            path = f.name
        try:
            sink = JsonlTelemetrySink(path)
            sink.emit(TelemetryEvent(
                type=TelemetryEventType.SESSION_START,
                session_id="abc",
                data={"provider": "openai"},
            ))
            sink.close()

            with open(path) as f:
                lines = f.readlines()
            self.assertEqual(len(lines), 1)
            record = json.loads(lines[0])
            self.assertEqual(record["type"], "session_start")
            self.assertEqual(record["session_id"], "abc")
        finally:
            os.unlink(path)

    def test_composite_sink_fans_out(self):
        from flyto_ai.telemetry import CompositeTelemetrySink, MemoryTelemetrySink, TelemetryEvent, TelemetryEventType
        sink1 = MemoryTelemetrySink()
        sink2 = MemoryTelemetrySink()
        composite = CompositeTelemetrySink([sink1, sink2])
        composite.emit(TelemetryEvent(type=TelemetryEventType.TOOL_CALL, data={}))
        self.assertEqual(len(sink1.events), 1)
        self.assertEqual(len(sink2.events), 1)

    def test_session_tracer_sequence(self):
        from flyto_ai.telemetry import SessionTracer, MemoryTelemetrySink, TelemetryEventType
        sink = MemoryTelemetrySink()
        tracer = SessionTracer("sess-123", sinks=[sink])

        tracer.trace_session_start(provider="openai", model="gpt-4o")
        tracer.trace_tool_call("search_modules", {"query": "auth"}, {"ok": True}, 42)
        tracer.trace_llm_call("gpt-4o", prompt_tokens=500, completion_tokens=200)

        # session_start(1) + tool_call(2) + tool_result(3) + llm_response(4) = 4
        self.assertEqual(len(sink.events), 4)
        self.assertEqual(sink.events[0].sequence, 1)
        self.assertEqual(sink.events[1].sequence, 2)
        self.assertEqual(sink.events[0].session_id, "sess-123")

    def test_session_tracer_permission_check(self):
        from flyto_ai.telemetry import SessionTracer, MemoryTelemetrySink, TelemetryEventType
        sink = MemoryTelemetrySink()
        tracer = SessionTracer("sess", sinks=[sink])
        tracer.trace_permission_check("shell.run", "READ_ONLY", False, "denied")
        events = sink.events_of_type(TelemetryEventType.PERMISSION_CHECK)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].data["allowed"], False)

    def test_session_tracer_cost_update(self):
        from flyto_ai.telemetry import SessionTracer, MemoryTelemetrySink, TelemetryEventType
        sink = MemoryTelemetrySink()
        tracer = SessionTracer("sess", sinks=[sink])
        tracer.trace_cost_update("gpt-4o", 0.005, 0.025)
        events = sink.events_of_type(TelemetryEventType.COST_UPDATE)
        self.assertEqual(len(events), 1)
        self.assertAlmostEqual(events[0].data["cost_usd"], 0.005)

    def test_telemetry_sink_protocol(self):
        from flyto_ai.telemetry import TelemetrySink, MemoryTelemetrySink
        self.assertIsInstance(MemoryTelemetrySink(), TelemetrySink)


# ═══════════════════════════════════════════════════════════════════
# Phase 4B: Mock Testing
# ═══════════════════════════════════════════════════════════════════

class TestMockApiClient(unittest.TestCase):
    """Test deterministic mock API client."""

    def test_basic_response(self):
        from flyto_ai.testing import MockApiClient, MockResponse
        client = MockApiClient(responses=[
            MockResponse(message="Hello, I found 3 results."),
        ])
        msg, tc, rounds, usage = asyncio.get_event_loop().run_until_complete(
            client.chat([{"role": "user", "content": "find auth"}], "system", [], None)
        )
        self.assertEqual(msg, "Hello, I found 3 results.")
        self.assertEqual(tc, [])
        self.assertEqual(rounds, 1)
        self.assertEqual(client.call_count, 1)

    def test_tool_calls_dispatched(self):
        from flyto_ai.testing import MockApiClient, MockResponse, MockToolExecutor
        executor = MockToolExecutor(
            responses={"search_modules": {"ok": True, "results": ["auth.login"]}},
        )
        client = MockApiClient(responses=[
            MockResponse(
                message="Found modules.",
                tool_calls=[{"function": "search_modules", "arguments": {"query": "auth"}}],
            ),
        ])
        msg, tc, rounds, usage = asyncio.get_event_loop().run_until_complete(
            client.chat([], "system", [], executor.dispatch)
        )
        self.assertEqual(len(tc), 1)
        self.assertEqual(tc[0]["function"], "search_modules")
        self.assertEqual(executor.calls, [("search_modules", {"query": "auth"})])

    def test_default_response_when_exhausted(self):
        from flyto_ai.testing import MockApiClient, MockResponse
        client = MockApiClient(responses=[])
        msg, _, _, _ = asyncio.get_event_loop().run_until_complete(
            client.chat([], "", [], None)
        )
        self.assertIn("no more responses", msg)

    def test_messages_recorded(self):
        from flyto_ai.testing import MockApiClient, MockResponse
        client = MockApiClient(responses=[MockResponse()])
        msgs = [{"role": "user", "content": "test"}]
        asyncio.get_event_loop().run_until_complete(client.chat(msgs, "", [], None))
        self.assertEqual(len(client.messages_received), 1)
        self.assertEqual(client.messages_received[0][0]["content"], "test")


class TestMockToolExecutor(unittest.TestCase):
    """Test deterministic mock tool executor."""

    def test_basic_dispatch(self):
        from flyto_ai.testing import MockToolExecutor
        executor = MockToolExecutor(
            tool_defs=[{"name": "test_tool", "description": "test", "inputSchema": {}}],
            responses={"test_tool": {"ok": True, "data": "hello"}},
        )
        result = asyncio.get_event_loop().run_until_complete(
            executor.dispatch("test_tool", {"arg": "val"})
        )
        self.assertEqual(result["ok"], True)
        self.assertEqual(result["data"], "hello")
        self.assertEqual(executor.calls, [("test_tool", {"arg": "val"})])

    def test_default_response(self):
        from flyto_ai.testing import MockToolExecutor
        executor = MockToolExecutor()
        result = asyncio.get_event_loop().run_until_complete(
            executor.dispatch("unknown", {})
        )
        self.assertEqual(result, {"ok": True})

    def test_callable_response(self):
        from flyto_ai.testing import MockToolExecutor
        executor = MockToolExecutor(
            responses={"echo": lambda name, args: {"echoed": args.get("msg", "")}},
        )
        result = asyncio.get_event_loop().run_until_complete(
            executor.dispatch("echo", {"msg": "hi"})
        )
        self.assertEqual(result["echoed"], "hi")

    def test_set_response(self):
        from flyto_ai.testing import MockToolExecutor
        executor = MockToolExecutor()
        executor.set_response("tool_a", {"ok": True, "value": 42})
        result = asyncio.get_event_loop().run_until_complete(
            executor.dispatch("tool_a", {})
        )
        self.assertEqual(result["value"], 42)

    def test_tools_property(self):
        from flyto_ai.testing import MockToolExecutor
        defs = [{"name": "a", "description": "A", "inputSchema": {}}]
        executor = MockToolExecutor(tool_defs=defs)
        self.assertEqual(len(executor.tools), 1)
        self.assertEqual(executor.tools[0]["name"], "a")
        # Verify it's a copy
        executor.tools.append({"name": "b"})
        self.assertEqual(len(executor.tools), 1)

    def test_protocol_conformance(self):
        from flyto_ai.protocols import ToolExecutor
        from flyto_ai.testing import MockToolExecutor
        executor = MockToolExecutor()
        self.assertIsInstance(executor, ToolExecutor)


# ═══════════════════════════════════════════════════════════════════
# Integration: Cross-phase tests
# ═══════════════════════════════════════════════════════════════════

class TestCrossPhaseIntegration(unittest.TestCase):
    """Integration tests spanning multiple phases."""

    def test_permission_enforcer_with_telemetry(self):
        """Permission denials should be traceable via telemetry."""
        from flyto_ai.permissions import PermissionEnforcer, PermissionLevel
        from flyto_ai.telemetry import SessionTracer, MemoryTelemetrySink, TelemetryEventType

        sink = MemoryTelemetrySink()
        tracer = SessionTracer("test", sinks=[sink])
        enforcer = PermissionEnforcer(PermissionLevel.READ_ONLY)

        decision = enforcer.check("execute_module", {"module_id": "shell.run"})
        tracer.trace_permission_check("execute_module", "READ_ONLY", decision.allowed, decision.reason)

        events = sink.events_of_type(TelemetryEventType.PERMISSION_CHECK)
        self.assertEqual(len(events), 1)
        self.assertFalse(events[0].data["allowed"])

    def test_cache_stats_in_telemetry(self):
        """Cache statistics should flow through telemetry."""
        from flyto_ai.cache import PromptCache
        from flyto_ai.telemetry import SessionTracer, MemoryTelemetrySink, TelemetryEventType

        sink = MemoryTelemetrySink()
        tracer = SessionTracer("test", sinks=[sink])
        cache = PromptCache(ttl_seconds=60.0)

        cache.update_fingerprint("system prompt", [{"name": "tool"}])
        cache.record_usage(cache_creation_tokens=0, cache_read_tokens=1000)

        tracer.trace_llm_call(
            "gpt-4o",
            cache_creation_tokens=0,
            cache_read_tokens=1000,
        )

        events = sink.events_of_type(TelemetryEventType.LLM_RESPONSE)
        self.assertEqual(events[0].data["cache_read_tokens"], 1000)

    def test_mock_executor_protocol_with_permissions(self):
        """MockToolExecutor should work with PermissionEnforcer."""
        from flyto_ai.testing import MockToolExecutor
        from flyto_ai.permissions import PermissionEnforcer, PermissionLevel

        executor = MockToolExecutor(
            tool_defs=[{"name": "execute_module", "description": "exec", "inputSchema": {}}],
            responses={"execute_module": {"ok": True}},
        )
        enforcer = PermissionEnforcer(PermissionLevel.WORKSPACE_WRITE)

        # Safe module — allowed
        d = enforcer.check("execute_module", {"module_id": "browser.click"})
        self.assertTrue(d.allowed)

        # Dangerous module — denied
        d = enforcer.check("execute_module", {"module_id": "shell.run"})
        self.assertFalse(d.allowed)


# ═══════════════════════════════════════════════════════════════════
# File integrity checks
# ═══════════════════════════════════════════════════════════════════

class TestSecurityFixes(unittest.TestCase):
    """Tests for audit-identified security issues."""

    def test_shell_hook_safe_env(self):
        """Hook scripts should NOT receive API keys from environment."""
        from flyto_ai.extensions.shell_hook import _safe_hook_env
        with patch.dict(os.environ, {
            "OPENAI_API_KEY": "sk-secret",
            "ANTHROPIC_API_KEY": "sk-ant-secret",
            "PATH": "/usr/bin",
            "HOME": "/home/user",
            "AWS_SECRET_ACCESS_KEY": "aws-secret",
        }, clear=True):
            env = _safe_hook_env({"HOOK_EVENT": "test"})
            self.assertIn("PATH", env)
            self.assertIn("HOME", env)
            self.assertIn("HOOK_EVENT", env)
            self.assertNotIn("OPENAI_API_KEY", env)
            self.assertNotIn("ANTHROPIC_API_KEY", env)
            self.assertNotIn("AWS_SECRET_ACCESS_KEY", env)

    def test_transcript_path_traversal_sanitized(self):
        """Session IDs with path traversal chars should be sanitized."""
        from flyto_ai.transcript import TranscriptWriter
        tmpdir = tempfile.mkdtemp()
        try:
            tw = TranscriptWriter("../../evil/session", transcript_dir=tmpdir)
            # Should have been sanitized — no path components
            self.assertNotIn("..", tw._session_id)
            self.assertNotIn("/", tw._session_id)
            # File should be within tmpdir
            self.assertTrue(str(tw.path).startswith(tmpdir))
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_transcript_safe_session_id_unchanged(self):
        """Normal session IDs should pass through unchanged."""
        from flyto_ai.transcript import TranscriptWriter
        tmpdir = tempfile.mkdtemp()
        try:
            tw = TranscriptWriter("abc123-def_456", transcript_dir=tmpdir)
            self.assertEqual(tw._session_id, "abc123-def_456")
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    def test_cache_ttl_resets_on_same_fingerprint(self):
        """TTL should reset even when fingerprint doesn't change."""
        from flyto_ai.cache import PromptCache
        cache = PromptCache(ttl_seconds=0.1)
        cache.update_fingerprint("prompt", [])
        time.sleep(0.15)  # Exceed TTL
        # Without fix: should_expect_cache_hit() = False
        # With fix: update_fingerprint resets timer
        cache.update_fingerprint("prompt", [])  # Same prompt
        self.assertTrue(cache.should_expect_cache_hit())


class TestFileIntegrity(unittest.TestCase):
    """Verify all new/modified files parse correctly."""

    def test_all_files_parse(self):
        import ast
        base = os.path.join(os.path.dirname(__file__), "..", "flyto_ai")
        files = [
            "protocols.py",
            "permissions.py",
            "cache.py",
            "telemetry.py",
            "testing.py",
            "mcp_client.py",
            "extensions/shell_hook.py",
            "extensions/hooks.py",
            "memory/compaction.py",
            "transcript.py",
            "providers/__init__.py",
            "agent.py",
            "config.py",
            "__init__.py",
        ]
        for fname in files:
            fpath = os.path.join(base, fname)
            with open(fpath) as f:
                try:
                    ast.parse(f.read())
                except SyntaxError as e:
                    self.fail(f"Syntax error in {fname}: {e}")

    def test_agent_imports_protocols(self):
        with open(os.path.join(os.path.dirname(__file__), "..", "flyto_ai", "agent.py")) as f:
            src = f.read()
        self.assertIn("from flyto_ai.protocols import ApiClient, ToolExecutor", src)
        self.assertIn("from flyto_ai.permissions import", src)
        self.assertIn("api_client: Optional[ApiClient]", src)
        self.assertIn("tool_executor: Optional[ToolExecutor]", src)
        self.assertIn("self._permission_enforcer", src)

    def test_config_has_permission_level(self):
        with open(os.path.join(os.path.dirname(__file__), "..", "flyto_ai", "config.py")) as f:
            src = f.read()
        self.assertIn("permission_level", src)
        self.assertIn("FLYTO_AI_PERMISSION_LEVEL", src)

    def test_init_exports(self):
        with open(os.path.join(os.path.dirname(__file__), "..", "flyto_ai", "__init__.py")) as f:
            src = f.read()
        for name in ["ApiClient", "ToolExecutor", "PermissionLevel", "PermissionEnforcer"]:
            self.assertIn(name, src, f"{name} not exported from __init__.py")


if __name__ == "__main__":
    unittest.main()
