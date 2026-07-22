"""Single-source package and runtime capability metadata."""

from __future__ import annotations

import re
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Optional


def package_version() -> str:
    """Return the source-tree version during development, then installed metadata."""
    project = Path(__file__).resolve().parents[1] / "pyproject.toml"
    if project.exists():
        match = re.search(r'^version\s*=\s*"([^"]+)"', project.read_text(encoding="utf-8"), re.MULTILINE)
        if match:
            return match.group(1)
    try:
        return version("flyto-ai")
    except PackageNotFoundError:
        pass
    return "0+unknown"


def runtime_module_count() -> Optional[int]:
    """Return the installed flyto-core registry size when it can be discovered."""
    try:
        from core.modules.registry import ModuleRegistry

        return len(ModuleRegistry.get_all_metadata())
    except Exception:
        return None
