# Copyright 2024 Flyto
# Licensed under the Apache License, Version 2.0
"""Tests for YAML/JSON config file support."""
import json
import os
import tempfile
import shutil

import pytest

from flyto_ai.config_file import ConfigFile, load_global_config, _merge_dicts


@pytest.fixture
def tmpdir():
    d = tempfile.mkdtemp()
    yield d
    shutil.rmtree(d, ignore_errors=True)


def test_load_json(tmpdir):
    path = os.path.join(tmpdir, "config.json")
    with open(path, "w") as f:
        json.dump({"provider": "anthropic", "model": "claude-sonnet-4-5"}, f)

    cfg = ConfigFile(path)
    assert cfg.load() is True
    assert cfg.get("provider") == "anthropic"
    assert cfg.get("model") == "claude-sonnet-4-5"


def test_load_yaml(tmpdir):
    path = os.path.join(tmpdir, "config.yaml")
    with open(path, "w") as f:
        f.write("provider: openai\nmodel: gpt-4o-mini\n")

    cfg = ConfigFile(path)
    loaded = cfg.load()
    # YAML needs pyyaml
    try:
        import yaml
        assert loaded is True
        assert cfg.get("provider") == "openai"
    except ImportError:
        pass  # OK if pyyaml not installed


def test_load_nonexistent(tmpdir):
    cfg = ConfigFile(os.path.join(tmpdir, "nonexistent.json"))
    assert cfg.load() is False


def test_get_dot_path(tmpdir):
    path = os.path.join(tmpdir, "config.json")
    with open(path, "w") as f:
        json.dump({"memory": {"db_path": "/custom/path.db", "enable": True}}, f)

    cfg = ConfigFile(path)
    cfg.load()
    assert cfg.get("memory.db_path") == "/custom/path.db"
    assert cfg.get("memory.enable") is True
    assert cfg.get("memory.nonexistent", "default") == "default"


def test_get_missing_key(tmpdir):
    path = os.path.join(tmpdir, "config.json")
    with open(path, "w") as f:
        json.dump({}, f)

    cfg = ConfigFile(path)
    cfg.load()
    assert cfg.get("missing") is None
    assert cfg.get("missing", "fallback") == "fallback"


def test_merge_dicts():
    base = {"a": 1, "b": {"c": 2, "d": 3}}
    override = {"b": {"c": 99, "e": 4}, "f": 5}
    result = _merge_dicts(base, override)
    assert result == {"a": 1, "b": {"c": 99, "d": 3, "e": 4}, "f": 5}


def test_load_with_overrides(tmpdir):
    base_path = os.path.join(tmpdir, "base.json")
    override_path = os.path.join(tmpdir, "override.json")

    with open(base_path, "w") as f:
        json.dump({"provider": "openai", "model": "gpt-4o-mini"}, f)
    with open(override_path, "w") as f:
        json.dump({"model": "gpt-4o"}, f)

    cfg = ConfigFile(base_path)
    cfg.load_with_overrides(override_path)
    assert cfg.get("provider") == "openai"
    assert cfg.get("model") == "gpt-4o"  # overridden


def test_data_property(tmpdir):
    path = os.path.join(tmpdir, "config.json")
    with open(path, "w") as f:
        json.dump({"key": "value"}, f)

    cfg = ConfigFile(path)
    cfg.load()
    data = cfg.data
    assert data == {"key": "value"}
    # Should be a copy
    data["key"] = "modified"
    assert cfg.get("key") == "value"


def test_load_global_config(tmpdir):
    path = os.path.join(tmpdir, "global.json")
    agent_path = os.path.join(tmpdir, "agent.json")

    with open(path, "w") as f:
        json.dump({"provider": "anthropic", "temperature": 0.5}, f)
    with open(agent_path, "w") as f:
        json.dump({"temperature": 0.9, "model": "claude-haiku"}, f)

    data = load_global_config(config_path=path, agent_config_path=agent_path)
    assert data["provider"] == "anthropic"
    assert data["temperature"] == 0.9  # agent override
    assert data["model"] == "claude-haiku"


def test_config_file_hot_reload(tmpdir):
    path = os.path.join(tmpdir, "config.json")
    with open(path, "w") as f:
        json.dump({"provider": "openai"}, f)

    reload_calls = []

    def on_reload(data):
        reload_calls.append(data)

    cfg = ConfigFile(path, on_reload=on_reload)
    cfg.load()

    # Modify file
    with open(path, "w") as f:
        json.dump({"provider": "anthropic"}, f)

    # Manually trigger check
    reloaded = cfg._check_reload()
    assert reloaded is True
    assert cfg.get("provider") == "anthropic"
    assert len(reload_calls) == 1


def test_empty_config_file(tmpdir):
    path = os.path.join(tmpdir, "empty.json")
    with open(path, "w") as f:
        f.write("")

    cfg = ConfigFile(path)
    assert cfg.load() is False
