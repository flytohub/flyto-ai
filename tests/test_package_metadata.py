"""Regression tests for package and runtime capability metadata."""

from pathlib import Path
import re

import flyto_ai
from flyto_ai.package_metadata import package_version, runtime_module_count


def test_source_package_version_matches_project_metadata():
    project = Path(__file__).resolve().parents[1] / "pyproject.toml"
    expected = re.search(
        r'^version\s*=\s*"([^"]+)"',
        project.read_text(encoding="utf-8"),
        re.MULTILINE,
    )

    assert expected is not None
    assert package_version() == expected.group(1)
    assert flyto_ai.__version__ == expected.group(1)


def test_runtime_module_count_is_discovered_or_unavailable():
    count = runtime_module_count()

    assert count is None or count > 0
