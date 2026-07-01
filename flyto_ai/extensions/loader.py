# Copyright 2024 Flyto
# Licensed under the Apache License, Version 2.0
"""Extension loader — discovers and loads extensions from disk.

Security-first: validates manifest, checks capabilities, rejects invalid extensions.
"""
import importlib
import importlib.util
import json
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional, Set

from flyto_ai.extensions.base import ExtensionBase, ExtensionManifest
from flyto_ai.extensions.hooks import HookRegistry

logger = logging.getLogger(__name__)

_DEFAULT_EXTENSIONS_DIR = "~/.flyto/extensions"


class ExtensionLoader:
    """Discovers, validates, and loads extensions from a directory.

    Each extension is a directory under ~/.flyto/extensions/ containing:
    - manifest.json (required)
    - extension.py (required, must define a class extending ExtensionBase)

    Usage::

        loader = ExtensionLoader()
        registry = loader.load_all()
        # or
        registry = loader.load_all(allowed_capabilities={"read_messages", "read_tool_results"})
    """

    def __init__(self, extensions_dir: Optional[str] = None) -> None:
        self._dir = Path(os.path.expanduser(extensions_dir or _DEFAULT_EXTENSIONS_DIR))

    @property
    def extensions_dir(self) -> Path:
        return self._dir

    def discover(self) -> List[Path]:
        """Discover extension directories (each must have manifest.json)."""
        if not self._dir.exists():
            return []

        dirs = []
        for entry in sorted(self._dir.iterdir()):
            if entry.is_dir() and (entry / "manifest.json").exists():
                dirs.append(entry)
        return dirs

    def load_manifest(self, ext_dir: Path) -> Optional[ExtensionManifest]:
        """Load and validate a manifest from an extension directory."""
        manifest_path = ext_dir / "manifest.json"
        try:
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest = ExtensionManifest.from_dict(data)
            errors = manifest.validate()
            if errors:
                logger.warning(
                    "Extension %s: manifest validation failed: %s",
                    ext_dir.name, "; ".join(errors),
                )
                return None
            return manifest
        except Exception as e:
            logger.warning("Extension %s: manifest load failed: %s", ext_dir.name, e)
            return None

    def load_extension(
        self,
        ext_dir: Path,
        manifest: ExtensionManifest,
    ) -> Optional[ExtensionBase]:
        """Load an extension module and instantiate its class."""
        ext_file = ext_dir / "extension.py"
        if not ext_file.exists():
            logger.warning("Extension %s: missing extension.py", manifest.name)
            return None

        try:
            module_name = "flyto_ext_{}".format(manifest.name.replace("-", "_"))
            spec = importlib.util.spec_from_file_location(module_name, str(ext_file))
            if spec is None or spec.loader is None:
                return None

            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)

            # Find the ExtensionBase subclass
            ext_class = None
            for attr_name in dir(module):
                attr = getattr(module, attr_name)
                if (
                    isinstance(attr, type)
                    and issubclass(attr, ExtensionBase)
                    and attr is not ExtensionBase
                ):
                    ext_class = attr
                    break

            if ext_class is None:
                logger.warning("Extension %s: no ExtensionBase subclass found", manifest.name)
                return None

            instance = ext_class(manifest)
            logger.info("Extension loaded: %s v%s", manifest.name, manifest.version)
            return instance

        except Exception as e:
            logger.warning("Extension %s: load failed: %s", manifest.name, e)
            return None

    def load_all(
        self,
        allowed_capabilities: Optional[Set[str]] = None,
    ) -> HookRegistry:
        """Discover, validate, and load all extensions.

        Parameters
        ----------
        allowed_capabilities : set, optional
            If set, only load extensions that require a subset of these.
            Extensions requesting disallowed capabilities are rejected.

        Returns
        -------
        HookRegistry
            Registry with all loaded extensions registered.
        """
        registry = HookRegistry()
        dirs = self.discover()

        if not dirs:
            logger.debug("No extensions found in %s", self._dir)
            return registry

        for ext_dir in dirs:
            manifest = self.load_manifest(ext_dir)
            if not manifest:
                continue

            # Capability check
            if allowed_capabilities is not None:
                disallowed = set(manifest.capabilities) - allowed_capabilities
                if disallowed:
                    logger.warning(
                        "Extension %s rejected: requires disallowed capabilities: %s",
                        manifest.name, ", ".join(disallowed),
                    )
                    continue

            extension = self.load_extension(ext_dir, manifest)
            if extension:
                registry.register(extension)

        logger.info(
            "Extensions loaded: %d/%d (dir: %s)",
            registry.extension_count, len(dirs), self._dir,
        )
        return registry
