# Copyright 2024 Flyto2
# Licensed under the Apache License, Version 2.0
"""Shell-based hook runner — claw-code style hook pipeline.

Hooks are shell commands that receive JSON on stdin and communicate
decisions via exit codes:
  - exit 0: allow (continue pipeline)
  - exit 2: deny (block tool call, stderr = reason)
  - other:  error (logged, pipeline continues)

Environment variables set for each hook invocation:
  - HOOK_EVENT: the hook event name (e.g., "before_tool_call")
  - HOOK_TOOL_NAME: the tool being called
  - HOOK_TOOL_INPUT: JSON-encoded tool arguments
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Exit code semantics (matching claw-code)
EXIT_ALLOW = 0
EXIT_DENY = 2

# Environment variables safe to pass to hook scripts
_SAFE_ENV_KEYS = frozenset({
    "PATH", "HOME", "USER", "LANG", "LC_ALL", "TERM", "SHELL",
    "TMPDIR", "TMP", "TEMP",
})


def _safe_hook_env(hook_vars: Dict[str, str]) -> Dict[str, str]:
    """Build a minimal environment for hook subprocesses.

    Only inherits safe system variables (PATH, HOME, LANG, etc.)
    to prevent leaking API keys and secrets to hook scripts.
    """
    env = {k: v for k, v in os.environ.items() if k in _SAFE_ENV_KEYS}
    env.update(hook_vars)
    return env


@dataclass
class HookDecision:
    """Result of a hook pipeline evaluation."""
    allowed: bool
    reason: str = ""
    modified_arguments: Optional[Dict[str, Any]] = None


@dataclass
class ShellHookConfig:
    """Configuration for a single shell hook."""
    command: str
    event: str  # "before_tool_call", "after_tool_call", etc.
    name: str = ""
    timeout_seconds: float = 10.0


class ShellHookRunner:
    """Runs shell command hooks with claw-code exit-code semantics.

    Usage::

        runner = ShellHookRunner([
            ShellHookConfig(command="./hooks/check_budget.sh", event="before_tool_call"),
            ShellHookConfig(command="./hooks/audit_log.sh", event="after_tool_call"),
        ])

        decision = await runner.run_before_tool_call("execute_module", {"module_id": "shell.run"})
        if not decision.allowed:
            return {"ok": False, "error": decision.reason}
    """

    def __init__(self, hooks: Optional[List[ShellHookConfig]] = None) -> None:
        self._hooks = hooks or []

    def add_hook(self, hook: ShellHookConfig) -> None:
        self._hooks.append(hook)

    def hooks_for_event(self, event: str) -> List[ShellHookConfig]:
        return [h for h in self._hooks if h.event == event]

    async def run_before_tool_call(
        self, tool_name: str, arguments: Dict[str, Any],
    ) -> HookDecision:
        """Run all before_tool_call hooks sequentially. Short-circuits on deny."""
        hooks = self.hooks_for_event("before_tool_call")
        if not hooks:
            return HookDecision(allowed=True)

        payload = json.dumps({
            "event": "before_tool_call",
            "tool_name": tool_name,
            "tool_input": arguments,
        }, ensure_ascii=False, default=str)

        env = _safe_hook_env({
            "HOOK_EVENT": "before_tool_call",
            "HOOK_TOOL_NAME": tool_name,
            "HOOK_TOOL_INPUT": json.dumps(arguments, ensure_ascii=False, default=str),
        })

        for hook in hooks:
            result = await self._exec_hook(hook, payload, env)
            if not result.allowed:
                return result  # Short-circuit on first deny

        return HookDecision(allowed=True)

    async def run_after_tool_call(
        self, tool_name: str, arguments: Dict[str, Any], result: Any,
    ) -> None:
        """Run all after_tool_call hooks (fire-and-forget, no deny semantics)."""
        hooks = self.hooks_for_event("after_tool_call")
        if not hooks:
            return

        result_str = json.dumps(result, ensure_ascii=False, default=str)[:8000]
        payload = json.dumps({
            "event": "after_tool_call",
            "tool_name": tool_name,
            "tool_input": arguments,
            "tool_output": result_str,
        }, ensure_ascii=False, default=str)

        env = _safe_hook_env({
            "HOOK_EVENT": "after_tool_call",
            "HOOK_TOOL_NAME": tool_name,
            "HOOK_TOOL_INPUT": json.dumps(arguments, ensure_ascii=False, default=str),
            "HOOK_TOOL_OUTPUT": result_str,
        })

        for hook in hooks:
            await self._exec_hook(hook, payload, env)

    async def _exec_hook(
        self, hook: ShellHookConfig, stdin_data: str, env: Dict[str, str],
    ) -> HookDecision:
        """Execute a single shell hook and interpret exit code."""
        hook_label = hook.name or hook.command[:40]
        try:
            proc = await asyncio.create_subprocess_shell(
                hook.command,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(stdin_data.encode()),
                timeout=hook.timeout_seconds,
            )

            if proc.returncode == EXIT_ALLOW:
                # Check if stdout contains modified arguments
                modified = None
                if stdout.strip():
                    try:
                        modified = json.loads(stdout)
                    except json.JSONDecodeError:
                        pass
                return HookDecision(allowed=True, modified_arguments=modified)

            if proc.returncode == EXIT_DENY:
                reason = stderr.decode(errors="replace").strip() or "Denied by hook: {}".format(hook_label)
                logger.info("Hook '%s' denied tool call: %s", hook_label, reason)
                return HookDecision(allowed=False, reason=reason)

            # Other exit code — treat as error, log and continue
            logger.warning(
                "Hook '%s' exited with code %d: %s",
                hook_label, proc.returncode, stderr.decode(errors="replace")[:200],
            )
            return HookDecision(allowed=True)

        except asyncio.TimeoutError:
            logger.warning("Hook '%s' timed out after %.1fs", hook_label, hook.timeout_seconds)
            return HookDecision(allowed=True)
        except Exception as e:
            logger.warning("Hook '%s' execution failed: %s", hook_label, e)
            return HookDecision(allowed=True)
