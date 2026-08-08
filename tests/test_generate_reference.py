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


def _cli_rows():
    """Parse the generated CLI table into (owner, item) pairs."""
    generator = _load_generator()
    rows = []
    for line in generator.cli_reference().splitlines():
        if not line.startswith("| `"):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        rows.append((cells[0].strip("`"), cells[1].strip("`")))
    return rows


def test_cli_reference_lists_both_public_coding_service_commands():
    rows = _cli_rows()
    commands = {owner for owner, item in rows if item == "command"}
    assert "code-mcp" in commands
    assert "code-serve" in commands
    # The subcommand name is retained, not the local parser variable.
    owners = {owner for owner, _item in rows}
    assert "code_mcp_p" not in owners
    assert "code_serve_p" not in owners


def test_cli_reference_attributes_shared_helper_options_to_each_command():
    rows = _cli_rows()
    for command in ("code-mcp", "code-serve"):
        options = {item for owner, item in rows if owner == command}
        # Association must be per command, not a global substring match.
        assert "--implementation-backend" in options, command
        assert "--max-rework-rounds" in options, command
        # Other helper-shared options must land on both commands too.
        assert "--tenant" in options, command
        assert "--workspace-root" in options, command
    serve_only = {item for owner, item in rows if owner == "code-serve"}
    mcp_only = {item for owner, item in rows if owner == "code-mcp"}
    assert "--host" in serve_only and "--host" not in mcp_only


def _cli_purposes():
    """Parse the generated CLI table into (owner, item) -> purpose."""
    generator = _load_generator()
    purposes = {}
    for line in generator.cli_reference().splitlines():
        if not line.startswith("| `"):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        purposes[(cells[0].strip("`"), cells[1].strip("`"))] = cells[2]
    return purposes


def test_static_format_arguments_are_resolved_in_help_text():
    purposes = _cli_purposes()
    for command in ("code-mcp", "code-serve"):
        purpose = purposes[(command, "--implementation-backend")]
        assert "FLYTO_AI_CODING_BACKEND" in purpose, command
        assert "{}" not in purpose, command
        assert "{" not in purpose and "}" not in purpose, command


def test_static_text_resolves_only_provable_expressions():
    generator = _load_generator()
    import ast

    constants = {"NAME": "FLYTO_AI_CODING_BACKEND"}

    def render(expression):
        return generator.static_text(ast.parse(expression, mode="eval").body, constants)

    assert render("'plain'") == "plain"
    assert render("NAME") == "FLYTO_AI_CODING_BACKEND"
    assert render("'a ' + NAME") == "a FLYTO_AI_CODING_BACKEND"
    assert render("'use {}'.format(NAME)") == "use FLYTO_AI_CODING_BACKEND"
    assert render("'use {0} and {0}'.format(NAME)") == (
        "use FLYTO_AI_CODING_BACKEND and FLYTO_AI_CODING_BACKEND"
    )
    assert render("'use {key}'.format(key=NAME)") == "use FLYTO_AI_CODING_BACKEND"
    assert render("f'use {NAME}'") == "use FLYTO_AI_CODING_BACKEND"
    # Nothing dynamic is ever guessed or executed.
    assert render("'use {}'.format(unknown_name)") is None
    assert render("'use {}'.format(compute())") is None
    assert render("'use {}'.format()") is None
    assert render("'use {missing}'.format(other=NAME)") is None
    assert render("compute()") is None
    assert render("'{0.__class__}'.format(NAME)") is None


def _environment_rows():
    generator = _load_generator()
    rows = {}
    for line in generator.environment_reference().splitlines():
        if not line.startswith("| `"):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        rows[cells[0].strip("`")] = cells[2]
    return rows


def test_environment_reference_resolves_module_constant_reads():
    rows = _environment_rows()
    # `CODING_BACKEND_ENV = "FLYTO_AI_CODING_BACKEND"` then
    # `_os.environ.get(CODING_BACKEND_ENV, "")`.
    assert "FLYTO_AI_CODING_BACKEND" in rows
    assert "flyto_ai/cli.py" in rows["FLYTO_AI_CODING_BACKEND"]


def test_environment_reference_resolves_static_helper_reads():
    rows = _environment_rows()
    for name in (
        "FLYTO_AI_CC_MAX_TURNS",
        "FLYTO_AI_CC_MAX_BUDGET",
        "FLYTO_AI_CC_MAX_FIX_ATTEMPTS",
    ):
        assert name in rows, name
        assert "flyto_ai/config.py" in rows[name], name


def test_environment_extraction_stays_bounded_to_real_reads():
    generator = _load_generator()
    import ast

    tree = ast.parse(
        "OTHER = 'NOT_AN_ENV_VAR'\n"
        "READS = 'REAL_ENV_VAR'\n"
        "def _reader(name, fallback):\n"
        "    return os.getenv(name, fallback)\n"
        "def _not_a_reader(name):\n"
        "    return name.upper()\n"
    )
    assert generator.environment_constants(tree) == {
        "OTHER": "NOT_AN_ENV_VAR", "READS": "REAL_ENV_VAR",
    }
    helpers = generator.environment_helpers(tree)
    assert helpers == {"_reader": (0, 1)}
    assert "_not_a_reader" not in helpers


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
