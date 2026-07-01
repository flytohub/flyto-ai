# Copyright 2024 Flyto
# Licensed under the Apache License, Version 2.0
"""Tests for plugin / extension system."""
import json
import os
import tempfile
import shutil

import pytest

from flyto_ai.extensions.base import ExtensionBase, ExtensionManifest
from flyto_ai.extensions.hooks import HookRegistry
from flyto_ai.extensions.loader import ExtensionLoader


# --- Manifest tests ---

def test_manifest_valid():
    m = ExtensionManifest(
        name="test", version="1.0",
        capabilities=["read_messages"],
        hooks=["before_chat"],
    )
    assert m.validate() == []


def test_manifest_empty_name():
    m = ExtensionManifest(name="", version="1.0")
    errors = m.validate()
    assert any("name" in e.lower() for e in errors)


def test_manifest_invalid_capability():
    m = ExtensionManifest(
        name="test", version="1.0",
        capabilities=["hack_system"],
    )
    errors = m.validate()
    assert any("capability" in e.lower() for e in errors)


def test_manifest_invalid_hook():
    m = ExtensionManifest(
        name="test", version="1.0",
        hooks=["nonexistent_hook"],
    )
    errors = m.validate()
    assert any("hook" in e.lower() for e in errors)


def test_manifest_from_dict():
    data = {
        "name": "my-ext",
        "version": "2.0",
        "description": "Test extension",
        "capabilities": ["read_messages", "read_tool_results"],
        "hooks": ["before_chat", "after_chat"],
    }
    m = ExtensionManifest.from_dict(data)
    assert m.name == "my-ext"
    assert m.version == "2.0"
    assert len(m.capabilities) == 2


# --- HookRegistry tests ---

class MockExtension(ExtensionBase):
    def __init__(self):
        super().__init__(ExtensionManifest(
            name="mock",
            version="1.0",
            hooks=["before_chat", "after_chat", "before_tool_call", "after_tool_call", "on_error"],
        ))
        self.calls = []

    async def before_chat(self, message, metadata):
        self.calls.append(("before_chat", message))
        return message.upper()

    async def after_chat(self, response, metadata):
        self.calls.append(("after_chat", response))
        return None  # no modification

    async def before_tool_call(self, tool_name, arguments):
        self.calls.append(("before_tool_call", tool_name))
        return None

    async def after_tool_call(self, tool_name, arguments, result):
        self.calls.append(("after_tool_call", tool_name))

    async def on_error(self, error, context):
        self.calls.append(("on_error", str(error)))


@pytest.mark.asyncio
async def test_hook_before_chat():
    reg = HookRegistry()
    ext = MockExtension()
    reg.register(ext)

    result = await reg.invoke_before_chat("hello", {})
    assert result == "HELLO"
    assert ("before_chat", "hello") in ext.calls


@pytest.mark.asyncio
async def test_hook_after_chat():
    reg = HookRegistry()
    ext = MockExtension()
    reg.register(ext)

    result = await reg.invoke_after_chat("response", {})
    assert result == "response"  # ext returns None = no change


@pytest.mark.asyncio
async def test_hook_before_tool_call():
    reg = HookRegistry()
    ext = MockExtension()
    reg.register(ext)

    result = await reg.invoke_before_tool_call("search_modules", {"query": "email"})
    assert result.allowed is True
    assert result.modified_arguments == {"query": "email"}


@pytest.mark.asyncio
async def test_hook_block_tool_call():
    class BlockingExt(ExtensionBase):
        def __init__(self):
            super().__init__(ExtensionManifest(
                name="blocker", version="1.0", hooks=["before_tool_call"],
            ))

        async def before_tool_call(self, tool_name, arguments):
            if tool_name == "dangerous_tool":
                return {"_block": True}
            return None

    reg = HookRegistry()
    reg.register(BlockingExt())

    result = await reg.invoke_before_tool_call("dangerous_tool", {})
    assert result.allowed is False

    result = await reg.invoke_before_tool_call("safe_tool", {})
    assert result.allowed is True


@pytest.mark.asyncio
async def test_hook_on_error():
    reg = HookRegistry()
    ext = MockExtension()
    reg.register(ext)

    await reg.invoke_on_error(ValueError("test error"), "test")
    assert ("on_error", "test error") in ext.calls


def test_hook_registry_count():
    reg = HookRegistry()
    assert reg.extension_count == 0
    reg.register(MockExtension())
    assert reg.extension_count == 1
    assert "mock" in reg.extension_names


def test_hook_registry_unregister():
    reg = HookRegistry()
    reg.register(MockExtension())
    assert reg.unregister("mock") is True
    assert reg.extension_count == 0
    assert reg.unregister("nonexistent") is False


# --- Loader tests ---

@pytest.fixture
def ext_dir():
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d, ignore_errors=True)


def _create_test_extension(base_dir, name="test-ext"):
    """Create a minimal extension directory."""
    ext_path = os.path.join(base_dir, name)
    os.makedirs(ext_path, exist_ok=True)

    manifest = {
        "name": name,
        "version": "1.0",
        "description": "Test extension",
        "capabilities": ["read_messages"],
        "hooks": ["before_chat"],
    }
    with open(os.path.join(ext_path, "manifest.json"), "w") as f:
        json.dump(manifest, f)

    code = '''
from flyto_ai.extensions.base import ExtensionBase

class TestExtension(ExtensionBase):
    async def before_chat(self, message, metadata):
        return "[EXT] " + message
'''
    with open(os.path.join(ext_path, "extension.py"), "w") as f:
        f.write(code)

    return ext_path


def test_loader_discover(ext_dir):
    _create_test_extension(ext_dir, "ext-a")
    _create_test_extension(ext_dir, "ext-b")

    loader = ExtensionLoader(ext_dir)
    dirs = loader.discover()
    assert len(dirs) == 2


def test_loader_discover_empty(ext_dir):
    loader = ExtensionLoader(ext_dir)
    assert loader.discover() == []


def test_loader_load_manifest(ext_dir):
    from pathlib import Path
    ext_path = _create_test_extension(ext_dir)
    loader = ExtensionLoader(ext_dir)
    manifest = loader.load_manifest(Path(ext_path))
    assert manifest is not None
    assert manifest.name == "test-ext"


def test_loader_load_all(ext_dir):
    _create_test_extension(ext_dir)
    loader = ExtensionLoader(ext_dir)
    registry = loader.load_all()
    assert registry.extension_count == 1


def test_loader_capability_filter(ext_dir):
    _create_test_extension(ext_dir)
    loader = ExtensionLoader(ext_dir)

    # Allow read_messages → should load
    reg = loader.load_all(allowed_capabilities={"read_messages"})
    assert reg.extension_count == 1

    # Only allow network_access → should reject (ext needs read_messages)
    reg = loader.load_all(allowed_capabilities={"network_access"})
    assert reg.extension_count == 0


@pytest.mark.asyncio
async def test_loaded_extension_works(ext_dir):
    _create_test_extension(ext_dir)
    loader = ExtensionLoader(ext_dir)
    registry = loader.load_all()

    result = await registry.invoke_before_chat("hello", {})
    assert result == "[EXT] hello"
