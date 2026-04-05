# Copyright 2024 Flyto
# Licensed under the Apache License, Version 2.0
"""Hook registry — manages extension hook points."""
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

from flyto_ai.extensions.base import ExtensionBase
from flyto_ai.extensions.shell_hook import HookDecision

logger = logging.getLogger(__name__)


class HookRegistry:
    """Registry for extension hook points.

    Manages the lifecycle of hooks: register, invoke, and unregister.
    Hooks are invoked in registration order.

    Usage::

        registry = HookRegistry()
        registry.register(my_extension)

        # Invoke hooks
        modified = await registry.invoke_before_chat("hello", {})
        await registry.invoke_after_tool_call("search_modules", args, result)
    """

    def __init__(self) -> None:
        self._extensions: List[ExtensionBase] = []

    @property
    def extension_count(self) -> int:
        return len(self._extensions)

    @property
    def extension_names(self) -> List[str]:
        return [e.name for e in self._extensions]

    def register(self, extension: ExtensionBase) -> None:
        """Register an extension's hooks."""
        self._extensions.append(extension)
        logger.info("Extension registered: %s (hooks: %s)",
                     extension.name, ", ".join(extension.manifest.hooks))

    def unregister(self, name: str) -> bool:
        """Unregister an extension by name. Returns True if found."""
        for i, ext in enumerate(self._extensions):
            if ext.name == name:
                self._extensions.pop(i)
                logger.info("Extension unregistered: %s", name)
                return True
        return False

    async def invoke_before_chat(self, message: str, metadata: Dict[str, Any]) -> str:
        """Invoke all before_chat hooks. Returns potentially modified message."""
        result = message
        for ext in self._extensions:
            if "before_chat" in ext.manifest.hooks:
                try:
                    modified = await ext.before_chat(result, metadata)
                    if modified is not None:
                        result = modified
                except Exception as e:
                    logger.warning("Extension %s before_chat failed: %s", ext.name, e)
        return result

    async def invoke_after_chat(self, response: str, metadata: Dict[str, Any]) -> str:
        """Invoke all after_chat hooks. Returns potentially modified response."""
        result = response
        for ext in self._extensions:
            if "after_chat" in ext.manifest.hooks:
                try:
                    modified = await ext.after_chat(result, metadata)
                    if modified is not None:
                        result = modified
                except Exception as e:
                    logger.warning("Extension %s after_chat failed: %s", ext.name, e)
        return result

    async def invoke_before_tool_call(
        self, tool_name: str, arguments: Dict
    ) -> HookDecision:
        """Invoke all before_tool_call hooks sequentially.

        Returns a ``HookDecision``. Short-circuits on first deny.
        """
        result = dict(arguments)
        for ext in self._extensions:
            if "before_tool_call" in ext.manifest.hooks:
                try:
                    modified = await ext.before_tool_call(tool_name, result)
                    if modified is not None:
                        if modified.get("_block"):
                            reason = modified.get("_reason", "Blocked by extension: {}".format(ext.name))
                            logger.info("Extension %s blocked tool call: %s (%s)", ext.name, tool_name, reason)
                            return HookDecision(allowed=False, reason=reason)
                        result = modified
                except Exception as e:
                    logger.warning("Extension %s before_tool_call failed: %s", ext.name, e)
        return HookDecision(allowed=True, modified_arguments=result)

    async def invoke_after_tool_call(
        self, tool_name: str, arguments: Dict, result: Any
    ) -> None:
        """Invoke all after_tool_call hooks."""
        for ext in self._extensions:
            if "after_tool_call" in ext.manifest.hooks:
                try:
                    await ext.after_tool_call(tool_name, arguments, result)
                except Exception as e:
                    logger.warning("Extension %s after_tool_call failed: %s", ext.name, e)

    async def invoke_on_error(self, error: Exception, context: str) -> None:
        """Invoke all on_error hooks."""
        for ext in self._extensions:
            if "on_error" in ext.manifest.hooks:
                try:
                    await ext.on_error(error, context)
                except Exception as e:
                    logger.warning("Extension %s on_error failed: %s", ext.name, e)

    async def invoke_on_load(self) -> None:
        """Invoke on_load for all extensions."""
        for ext in self._extensions:
            if "on_load" in ext.manifest.hooks:
                try:
                    await ext.on_load()
                except Exception as e:
                    logger.warning("Extension %s on_load failed: %s", ext.name, e)

    async def invoke_on_unload(self) -> None:
        """Invoke on_unload for all extensions."""
        for ext in self._extensions:
            if "on_unload" in ext.manifest.hooks:
                try:
                    await ext.on_unload()
                except Exception as e:
                    logger.warning("Extension %s on_unload failed: %s", ext.name, e)
