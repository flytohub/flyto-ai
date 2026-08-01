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


def test_generated_paths_never_collide_with_agent_instruction_files():
    generator = _load_generator()

    assert generator.group_filename("agents") == "agents-package.md"
    assert generator.group_filename("providers") == "providers.md"
    reserved = generator.RESERVED_INSTRUCTION_FILENAMES
    generated_names = {
        path.name.casefold()
        for path in generator.outputs()
        if path.parent == generator.PYTHON_REFERENCE
    }
    assert generated_names.isdisjoint(reserved)
