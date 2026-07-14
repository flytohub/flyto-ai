# Copyright 2024 Flyto2
# Licensed under the Apache License, Version 2.0
"""Extension base class and manifest definition.

Security-first design (unlike OpenClaw's marketplace where 36% have vulnerabilities):
- Every extension requires a manifest.json with capability declarations
- Extensions declare what permissions they need
- The loader validates manifest before loading
"""
from abc import ABC
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class ExtensionManifest:
    """Extension manifest — declares capabilities and requirements.

    Every extension MUST have a manifest.json in its directory.
    """
    name: str
    version: str
    description: str = ""
    author: str = ""
    # Capabilities this extension requires
    capabilities: List[str] = field(default_factory=list)
    # Hook points this extension uses
    hooks: List[str] = field(default_factory=list)
    # Minimum flyto-ai version
    min_version: str = ""

    # Valid capabilities
    VALID_CAPABILITIES = frozenset({
        "read_messages",     # can read user messages
        "modify_messages",   # can modify messages before sending
        "read_tool_results", # can read tool call results
        "call_tools",        # can trigger tool calls
        "read_config",       # can read agent config
        "network_access",    # can make HTTP requests
        "file_access",       # can read/write files
    })

    # Valid hook points
    VALID_HOOKS = frozenset({
        "before_chat",       # before processing user message
        "after_chat",        # after generating response
        "before_tool_call",  # before dispatching a tool
        "after_tool_call",   # after tool returns
        "on_error",          # when an error occurs
        "on_load",           # when extension is loaded
        "on_unload",         # when extension is unloaded
    })

    def validate(self) -> List[str]:
        """Validate the manifest. Returns list of error messages."""
        errors = []
        if not self.name:
            errors.append("Extension name is required")
        if not self.version:
            errors.append("Extension version is required")
        for cap in self.capabilities:
            if cap not in self.VALID_CAPABILITIES:
                errors.append("Unknown capability: {}".format(cap))
        for hook in self.hooks:
            if hook not in self.VALID_HOOKS:
                errors.append("Unknown hook: {}".format(hook))
        return errors

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExtensionManifest":
        """Create manifest from a dict (parsed from manifest.json)."""
        return cls(
            name=data.get("name", ""),
            version=data.get("version", ""),
            description=data.get("description", ""),
            author=data.get("author", ""),
            capabilities=data.get("capabilities", []),
            hooks=data.get("hooks", []),
            min_version=data.get("min_version", ""),
        )


class ExtensionBase(ABC):
    """Base class for flyto-ai extensions.

    Subclass this and implement the hook methods you need.
    Place in ~/.flyto/extensions/<name>/ with a manifest.json.
    """

    def __init__(self, manifest: ExtensionManifest) -> None:
        self._manifest = manifest

    @property
    def name(self) -> str:
        return self._manifest.name

    @property
    def manifest(self) -> ExtensionManifest:
        return self._manifest

    async def on_load(self) -> None:
        """Called when the extension is loaded."""

    async def on_unload(self) -> None:
        """Called when the extension is unloaded."""

    async def before_chat(self, message: str, metadata: Dict[str, Any]) -> Optional[str]:
        """Called before processing a user message.

        Return modified message, or None to keep original.
        """
        return None

    async def after_chat(self, response: str, metadata: Dict[str, Any]) -> Optional[str]:
        """Called after generating a response.

        Return modified response, or None to keep original.
        """
        return None

    async def before_tool_call(self, tool_name: str, arguments: Dict) -> Optional[Dict]:
        """Called before dispatching a tool.

        Return modified arguments, or None to keep original.
        Return {"_block": True} to block the tool call.
        """
        return None

    async def after_tool_call(self, tool_name: str, arguments: Dict, result: Any) -> None:
        """Called after a tool returns."""

    async def on_error(self, error: Exception, context: str) -> None:
        """Called when an error occurs."""
