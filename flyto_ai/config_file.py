# Copyright 2024 Flyto2
# Licensed under the Apache License, Version 2.0
"""YAML/JSON config file support with hot-reload.

Layered resolution: global (~/.flyto/config.yaml) → agent → session.
"""
import json
import logging
import os
import threading
from pathlib import Path
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)

_DEFAULT_CONFIG_PATH = "~/.flyto/config.yaml"


def _load_yaml_or_json(path: Path) -> Dict[str, Any]:
    """Load a YAML or JSON config file."""
    if not path.exists():
        return {}

    content = path.read_text(encoding="utf-8")
    if not content.strip():
        return {}

    suffix = path.suffix.lower()
    if suffix in (".yaml", ".yml"):
        try:
            import yaml
            return yaml.safe_load(content) or {}
        except ImportError:
            logger.warning("PyYAML not installed — cannot load %s", path)
            return {}
    elif suffix == ".json":
        return json.loads(content)
    else:
        # Try JSON first, then YAML
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            try:
                import yaml
                return yaml.safe_load(content) or {}
            except ImportError:
                return {}


def _merge_dicts(base: Dict, override: Dict) -> Dict:
    """Deep merge override into base (override wins)."""
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _merge_dicts(result[key], value)
        else:
            result[key] = value
    return result


class ConfigFile:
    """Manages a layered config file with optional hot-reload.

    Usage::

        cfg = ConfigFile("~/.flyto/config.yaml")
        cfg.load()
        value = cfg.get("provider", default="openai")

        # Enable hot-reload
        cfg.start_watching()

        # Convert to AgentConfig
        from flyto_ai.config import AgentConfig
        agent_config = AgentConfig.from_dict(cfg.data)
    """

    def __init__(
        self,
        path: Optional[str] = None,
        on_reload: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> None:
        self._path = Path(os.path.expanduser(path or _DEFAULT_CONFIG_PATH))
        self._data: Dict[str, Any] = {}
        self._last_mtime: float = 0.0
        self._on_reload = on_reload
        self._watcher_thread: Optional[threading.Thread] = None
        self._watcher_stop = threading.Event()

    @property
    def path(self) -> Path:
        return self._path

    @property
    def data(self) -> Dict[str, Any]:
        """Current config data."""
        return dict(self._data)

    def get(self, key: str, default: Any = None) -> Any:
        """Get a config value by dot-separated key path.

        Example: cfg.get("memory.db_path", default="~/.flyto/memory.db")
        """
        keys = key.split(".")
        value = self._data
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default
            if value is None:
                return default
        return value

    def load(self) -> bool:
        """Load config from file. Returns True if loaded."""
        try:
            self._data = _load_yaml_or_json(self._path)
            if self._path.exists():
                self._last_mtime = self._path.stat().st_mtime
            logger.info("Config loaded from %s (%d keys)", self._path, len(self._data))
            return bool(self._data)
        except Exception as e:
            logger.warning("Config load failed: %s", e)
            return False

    def load_with_overrides(self, *override_paths: str) -> None:
        """Load base config and merge overrides (layered resolution)."""
        self.load()
        for op in override_paths:
            p = Path(os.path.expanduser(op))
            override = _load_yaml_or_json(p)
            if override:
                self._data = _merge_dicts(self._data, override)
                logger.debug("Config override merged: %s", p)

    def _check_reload(self) -> bool:
        """Check if the file changed and reload if so."""
        try:
            if not self._path.exists():
                return False
            mtime = self._path.stat().st_mtime
            if mtime > self._last_mtime:
                old_data = dict(self._data)
                self.load()
                if self._data != old_data:
                    logger.info("Config hot-reloaded: %s", self._path)
                    if self._on_reload:
                        try:
                            self._on_reload(self._data)
                        except Exception as e:
                            logger.warning("Config reload callback failed: %s", e)
                    return True
        except Exception as e:
            logger.debug("Config reload check failed: %s", e)
        return False

    def start_watching(self, interval: float = 2.0) -> None:
        """Start a background thread to watch for config file changes."""
        if self._watcher_thread and self._watcher_thread.is_alive():
            return

        self._watcher_stop.clear()

        def _watch():
            while not self._watcher_stop.is_set():
                self._check_reload()
                self._watcher_stop.wait(interval)

        self._watcher_thread = threading.Thread(target=_watch, daemon=True, name="flyto-config-watcher")
        self._watcher_thread.start()
        logger.info("Config watcher started: %s (interval=%.1fs)", self._path, interval)

    def stop_watching(self) -> None:
        """Stop the config file watcher."""
        self._watcher_stop.set()
        if self._watcher_thread:
            self._watcher_thread.join(timeout=5.0)
            self._watcher_thread = None
        logger.debug("Config watcher stopped")


def load_global_config(
    config_path: Optional[str] = None,
    agent_config_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Load layered config: global → agent-specific.

    Returns merged config dict suitable for AgentConfig.from_dict().
    """
    cfg = ConfigFile(config_path or _DEFAULT_CONFIG_PATH)
    cfg.load()

    if agent_config_path:
        agent_data = _load_yaml_or_json(Path(os.path.expanduser(agent_config_path)))
        if agent_data:
            cfg._data = _merge_dicts(cfg._data, agent_data)

    return cfg.data
