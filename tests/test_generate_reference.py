"""Tests for the source-backed reference generator."""

import importlib.util
from pathlib import Path


def _load_generator():
    script = Path(__file__).parents[1] / "scripts" / "generate_reference.py"
    spec = importlib.util.spec_from_file_location("flyto_generate_reference", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_generated_markdown_has_no_trailing_whitespace():
    generator = _load_generator()

    for path, content in generator.outputs().items():
        bad_lines = [
            index
            for index, line in enumerate(content.splitlines(), start=1)
            if line != line.rstrip()
        ]
        assert not bad_lines, "{} has trailing whitespace on lines {}".format(
            path,
            bad_lines,
        )
