# Copyright 2024 Flyto2
# Licensed under the Apache License, Version 2.0
"""One source of truth for local and GitHub dependency revisions."""
import json
from pathlib import Path

import pytest

from scripts.stack_lock import DEPENDENCIES, load_stack_lock


def test_repository_stack_lock_is_closed_and_fully_pinned() -> None:
    root = Path(__file__).resolve().parents[1]
    dependencies = load_stack_lock(root / "stack-lock.json")
    assert tuple(dependencies) == DEPENDENCIES
    assert all(len(item["revision"]) == 40 for item in dependencies.values())
    workflow = (root / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")
    for name in DEPENDENCIES:
        key = name.replace("-", "_")
        for field in ("repository", "path", "revision"):
            assert "steps.stack_lock.outputs.{}_{}".format(key, field) in workflow
        assert dependencies[name]["revision"] not in workflow
        assert dependencies[name]["repository"] not in workflow
    coding = (root / ".flyto" / "coding.yaml").read_text(encoding="utf-8")
    assert "scripts/stack_lock.py" in coding


def test_stack_lock_rejects_a_floating_revision(tmp_path: Path) -> None:
    source = Path(__file__).resolve().parents[1] / "stack-lock.json"
    value = json.loads(source.read_text(encoding="utf-8"))
    value["dependencies"]["flyto-core"]["revision"] = "main"
    target = tmp_path / "stack-lock.json"
    target.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(ValueError, match="full Git SHA-1"):
        load_stack_lock(target)
