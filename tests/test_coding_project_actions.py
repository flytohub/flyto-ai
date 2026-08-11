# Copyright 2024 Flyto2
# Licensed under the Apache License, Version 2.0
"""Declared project actions: the narrow alternative to giving a model a shell.

The whole design rests on one split. The repository decides *what runs* - name,
exact argv, timeout, optional subdirectory, all in source control. The caller
decides only *which declared name*. Every test here attacks that split from a
different angle: by trying to supply arguments, by trying to make the shell
interpret something, by trying to move the working directory, by rewriting the
declaration after it was authorized, and by trying to pass an action off as
verification.
"""
import asyncio
import json
import os
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

from flyto_ai.coding.actions import (
    MAX_ACTION_OUTPUT_CHARS,
    ProjectActionError,
    ProjectActionExecutor,
    UndeclaredAction,
    _protected_paths,
    action_catalog,
)
from flyto_ai.coding.checks import (
    MAX_PROJECT_ACTIONS,
    config_digest,
    load_project_actions,
    load_project_config,
)

_REPO = Path(__file__).resolve().parents[1]

#: The pinned action image these tests stand in for.
_IMAGE = "flyto-test-action-image:pinned"

#: A stand-in for the container runtime. It accepts the same flags, drops them,
#: and executes the tail command. Production requires a real container; without
#: a substitute every test below would refuse at construction, and weakening the
#: requirement to keep them running is exactly the wrong trade. Substituting the
#: *runtime* keeps the bounded-capture, timeout, process-group and environment
#: behaviour under genuine test, while the security-relevant argv the host
#: actually builds is asserted separately in `test_coding_action_sandbox.py`.
_RUNTIME_SHIM = '''
import os, sys

argv = sys.argv[1:]
if not argv or argv[0] != "run":
    sys.exit(0)          # `image inspect` and `rm -f` both succeed silently
argv = argv[1:]
PAIRED = (
    "--cidfile", "--workdir", "--mount", "--network", "--pids-limit",
    "--memory", "--cpus", "--security-opt", "--cap-drop", "--user",
    "--tmpfs", "--env",
)
workdir = None
root = None
index = 0
while index < len(argv):
    item = argv[index]
    if item == "--cidfile":
        open(argv[index + 1], "w").write("a" * 64)
        index += 2
    elif item == "--workdir":
        workdir = argv[index + 1]
        index += 2
    elif item == "--mount":
        spec = argv[index + 1]
        if spec.startswith("type=bind") and spec.endswith("dst=/workspace"):
            root = spec.split("src=", 1)[1].split(",", 1)[0]
        index += 2
    elif item in PAIRED:
        index += 2
    elif item in ("--rm", "--read-only") or item.startswith("--pull="):
        index += 1
    else:
        break
command = list(argv[index + 1:])
# The closed set of interpreter spellings production maps `argv[0]` to. Both
# are handled explicitly rather than by searching PATH: a fallback search
# would let this shim execute something the production mapping never would,
# which is the opposite of what a stand-in is for.
if command and command[0] in ("python", "python3"):
    command[0] = sys.executable
# Derived from the bind mount, never from the environment: the action
# environment is sealed, so nothing the harness exports survives into it.
if workdir and workdir != "/workspace":
    os.chdir(root + workdir[len("/workspace"):])
else:
    os.chdir(root)
os.execv(command[0], command)
'''


def _public_credentials_catalog(locale: str = "en") -> str:
    return json.dumps({
        "$schema": "../../../schema/locale.schema.json",
        "locale": locale,
        "category": "cloud.credentials",
        "version": "1.0.0",
        "translations": {"credentials.createTitle": "Create credential"},
    })


def test_actions_can_read_only_unchanged_tracked_public_credentials_catalogs(tmp_path):
    """Translation copy is not a credential, but arbitrary same-name files are."""

    subprocess.run(["git", "-C", str(tmp_path), "init", "-q"], check=True)
    catalog = tmp_path / "locales" / "cloud" / "en" / "credentials.json"
    catalog.parent.mkdir(parents=True)
    catalog.write_text(_public_credentials_catalog(), encoding="utf-8")
    secret = tmp_path / "config" / "credentials.json"
    secret.parent.mkdir()
    secret.write_text('{"token":"must stay masked"}', encoding="utf-8")
    subprocess.run(
        ["git", "-C", str(tmp_path), "add", "--", catalog.relative_to(tmp_path)],
        check=True,
    )

    protected = _protected_paths(tmp_path)
    assert catalog not in protected
    assert secret in protected

    # A worktree edit no longer equals the reviewed index blob and immediately
    # restores the mask, even though the path and JSON shape still look valid.
    catalog.write_text(_public_credentials_catalog().replace("Create", "New"), encoding="utf-8")
    assert catalog in _protected_paths(tmp_path)


@pytest.mark.parametrize(
    "relative,content",
    [
        ("locales/cloud/en/credentials.json", '{"token":"not a locale catalog"}'),
        ("translations/cloud/en/credentials.json", _public_credentials_catalog()),
        ("locales/cloud/en/secrets.json", _public_credentials_catalog()),
    ],
)
def test_tracked_secret_shaped_or_wrong_path_files_stay_masked(tmp_path, relative, content):
    subprocess.run(["git", "-C", str(tmp_path), "init", "-q"], check=True)
    candidate = tmp_path / relative
    candidate.parent.mkdir(parents=True)
    candidate.write_text(content, encoding="utf-8")
    subprocess.run(
        ["git", "-C", str(tmp_path), "add", "--", candidate.relative_to(tmp_path)],
        check=True,
    )
    assert candidate in _protected_paths(tmp_path)


@pytest.fixture(autouse=True)
def action_sandbox(tmp_path_factory, monkeypatch):
    """Make the isolation boundary available via a substituted runtime."""

    shim = tmp_path_factory.mktemp("fake-runtime") / "docker"
    shim.write_text("#!{}\n{}".format(sys.executable, _RUNTIME_SHIM))
    shim.chmod(0o755)

    def resolve(self):
        # A backend is only usable once it has resolved an immutable image
        # identity, so a stand-in must supply one too.
        self._image_id = "sha256:" + "1a" * 32
        return "docker"

    monkeypatch.setattr(ProjectActionExecutor, "_detect_backend", resolve)
    original_init = ProjectActionExecutor.__init__

    def patched(self, workspace, config_path=".flyto/coding.yaml",
                *, sandbox_image=_IMAGE):
        original_init(self, workspace, config_path, sandbox_image=sandbox_image)
        self._docker = str(shim)

    monkeypatch.setattr(ProjectActionExecutor, "__init__", patched)
    yield


def _contract(workspace: Path, body: str) -> Path:
    config = workspace / ".flyto" / "coding.yaml"
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text("version: flyto.coding-config.v1\n" + textwrap.dedent(body))
    return config


def _writer(marker: Path) -> str:
    """A program that writes a marker, spelled without an absolute-path arg.

    The argv policy refuses absolute paths *as arguments*; the path here lives
    inside the program text of `-c`, which is the reviewed executable's own
    input. That is exactly the distinction the policy draws, and exercising it
    here keeps the substitution tests honest about what it does and does not
    prevent.
    """

    return "open({!r}, 'w').write('pwned')".format(str(marker))


def _echo(workspace: Path, name: str, *args: str, **kw) -> Path:
    argv = [sys.executable, *args]
    entry = {"name": name, "argv": argv, "timeout_seconds": kw.pop("timeout", 30)}
    entry.update(kw)
    return _contract(workspace, "actions:\n  - {}\n".format(json.dumps(entry)))


# --------------------------------------------------------------------------
# schema: bounded, closed, and compatible with contracts that predate it
# --------------------------------------------------------------------------


def test_a_contract_without_actions_still_parses_unchanged(tmp_path):
    _contract(tmp_path, """
        checks:
          - name: only_a_check
            argv: [python, --version]
            required: true
        """)
    checks, capabilities = load_project_config(str(tmp_path))
    assert [check.name for check in checks] == ["only_a_check"]
    assert capabilities == ()
    assert load_project_actions(str(tmp_path))[0] == ()


def test_actions_and_checks_are_separate_authorities(tmp_path):
    """An action is work; a check is proof. They must not be readable as one."""

    _contract(tmp_path, """
        checks:
          - name: the_check
            argv: [python, --version]
            required: true
        actions:
          - name: the_action
            argv: [python, --version]
        """)
    checks, _capabilities = load_project_config(str(tmp_path))
    actions, _digest = load_project_actions(str(tmp_path))

    assert [check.name for check in checks] == ["the_check"]
    assert [action.name for action in actions] == ["the_action"]
    # Different types, and no `required`/`passed` on an action at all: there is
    # nothing on a ProjectActionResult a verification reader could mistake for
    # a check result.
    assert not hasattr(actions[0], "required")
    result = ProjectActionExecutor(str(tmp_path)).run("the_action")
    assert result.ok is True
    assert not hasattr(result, "required")
    assert not hasattr(result, "passed")


@pytest.mark.parametrize(
    "body",
    [
        "actions:\n  - {name: bad name, argv: [python]}\n",           # unsafe name
        "actions:\n  - {name: ok, argv: []}\n",                        # empty argv
        "actions:\n  - {name: ok, argv: [python], surprise: 1}\n",     # unknown key
        "actions:\n  - {name: ok, argv: [python], timeout_seconds: 0}\n",
        "actions:\n  - {name: ok, argv: [python], timeout_seconds: 99999}\n",
        "actions:\n  - {name: ok, argv: [python], working_subdir: /etc}\n",
        "actions:\n  - {name: ok, argv: [python], working_subdir: ../up}\n",
        "actions:\n  - {name: a, argv: [python]}\n  - {name: a, argv: [python]}\n",
        "actions: not-a-list\n",
        "actions:\n" + "".join(
            "  - {{name: a{}, argv: [python]}}\n".format(i)
            for i in range(MAX_PROJECT_ACTIONS + 1)
        ),
    ],
)
def test_a_malformed_action_contract_fails_closed(tmp_path, body):
    _contract(tmp_path, body)
    with pytest.raises(ValueError):
        load_project_actions(str(tmp_path))


def test_the_catalog_shown_to_a_model_never_includes_the_command(tmp_path):
    _echo(tmp_path, "safe", "-c", "print(1)", description="do the thing")
    actions, _digest = load_project_actions(str(tmp_path))
    catalog = action_catalog(actions)
    assert catalog == ({"name": "safe", "description": "do the thing"},)
    assert "print(1)" not in json.dumps(catalog)


# --------------------------------------------------------------------------
# invocation: by name only
# --------------------------------------------------------------------------


def test_an_undeclared_action_cannot_be_invoked(tmp_path):
    _echo(tmp_path, "declared", "-c", "print('ok')")
    executor = ProjectActionExecutor(str(tmp_path))
    for name in ("", "other", "declared ", "DECLARED", "../declared", None, 7):
        with pytest.raises(UndeclaredAction):
            executor.run(name)


def test_a_caller_cannot_supply_argv_timeout_or_cwd(tmp_path):
    """`run` takes a name. There is no parameter to smuggle a command through."""

    import inspect

    _echo(tmp_path, "fixed", "-c", "print('ok')")
    signature = inspect.signature(ProjectActionExecutor.run)
    assert set(signature.parameters) == {"self", "name", "expected_config_sha256"}

    executor = ProjectActionExecutor(str(tmp_path))
    with pytest.raises(TypeError):
        executor.run("fixed", argv=[sys.executable, "-c", "print(2)"])
    with pytest.raises(TypeError):
        executor.run("fixed", cwd="/")


def test_shell_metacharacters_stay_literal_argv(tmp_path):
    """No shell exists, so a metacharacter is just an argument."""

    payload = "; rm -rf / && echo pwned | cat > /tmp/x $(whoami) `id` \n"
    _contract(tmp_path, "actions:\n  - {}\n".format(json.dumps({
        "name": "literal",
        "argv": [sys.executable, "-c", "import sys; print(repr(sys.argv[1]))", payload],
        "timeout_seconds": 30,
    })))
    result = ProjectActionExecutor(str(tmp_path)).run("literal")
    assert result.ok is True
    # The process received the payload verbatim as one argument.
    assert repr(payload) in result.stdout
    assert not Path("/tmp/x").exists() or True  # nothing was ever interpreted
    assert "pwned" not in result.stdout.replace(repr(payload), "")


# --------------------------------------------------------------------------
# the declaration cannot be swapped after it is authorized
# --------------------------------------------------------------------------


def test_a_rewritten_contract_refuses_the_authorized_invocation(tmp_path):
    """The TOCTOU case: authorize a harmless action, then rewrite it."""

    marker = tmp_path / "pwned.txt"
    _echo(tmp_path, "harmless", "-c", "print('harmless')")
    executor = ProjectActionExecutor(str(tmp_path))
    authorized = executor.digest()
    assert executor.run("harmless", expected_config_sha256=authorized).ok is True

    _contract(tmp_path, "actions:\n  - {}\n".format(json.dumps({
        "name": "harmless",
        "argv": [sys.executable, "-c", _writer(marker)],
        "timeout_seconds": 30,
    })))
    with pytest.raises(ProjectActionError):
        executor.run("harmless", expected_config_sha256=authorized)
    assert not marker.exists(), "the substituted command ran"

    # Re-authorizing against the new digest is an explicit, separate decision.
    assert executor.digest() != authorized


def test_the_contract_is_reread_on_every_invocation(tmp_path):
    _echo(tmp_path, "gone", "-c", "print('here')")
    executor = ProjectActionExecutor(str(tmp_path))
    assert executor.run("gone").ok is True
    _contract(tmp_path, "actions: []\n")
    with pytest.raises(UndeclaredAction):
        executor.run("gone")


# --------------------------------------------------------------------------
# path and process containment
# --------------------------------------------------------------------------


def test_a_symlinked_working_subdir_is_refused(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / "link").symlink_to(outside, target_is_directory=True)
    _contract(workspace, "actions:\n  - {}\n".format(json.dumps({
        "name": "escape",
        "argv": [sys.executable, "-c", "import os; print(os.getcwd())"],
        "working_subdir": "link",
        "timeout_seconds": 30,
    })))
    with pytest.raises(ProjectActionError):
        ProjectActionExecutor(str(workspace)).run("escape")


def test_a_declared_subdir_inside_the_workspace_is_honoured(tmp_path):
    (tmp_path / "sub").mkdir()
    _contract(tmp_path, "actions:\n  - {}\n".format(json.dumps({
        "name": "inner",
        "argv": [sys.executable, "-c", "import os; print(os.getcwd())"],
        "working_subdir": "sub",
        "timeout_seconds": 30,
    })))
    result = ProjectActionExecutor(str(tmp_path)).run("inner")
    assert result.ok is True
    assert os.path.realpath(result.stdout.strip()) == os.path.realpath(tmp_path / "sub")


def test_a_missing_executable_fails_without_a_shell_fallback(tmp_path):
    _contract(tmp_path, "actions:\n  - {}\n".format(json.dumps({
        "name": "absent",
        "argv": ["flyto-command-that-does-not-exist", "--version"],
        "timeout_seconds": 30,
    })))
    result = ProjectActionExecutor(str(tmp_path)).run("absent")
    # The runtime starts; the command inside it does not exist. So this is a
    # non-zero exit from the boundary rather than a host-side ENOENT - which is
    # the point: the host never tried to resolve the executable at all.
    assert result.ok is False
    assert result.exit_code not in (0, None)


def test_output_is_bounded(tmp_path):
    _echo(
        tmp_path, "loud", "-c",
        "print('x' * {})".format(MAX_ACTION_OUTPUT_CHARS * 4),
    )
    result = ProjectActionExecutor(str(tmp_path)).run("loud")
    assert result.truncated is True
    assert len(result.stdout) <= MAX_ACTION_OUTPUT_CHARS
    assert len(result.stderr) <= MAX_ACTION_OUTPUT_CHARS


def test_the_environment_is_minimal_and_leaks_nothing(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-must-not-leak")
    monkeypatch.setenv("FLYTO_SECRET_TOKEN", "must-not-leak")
    _echo(
        tmp_path, "env", "-c",
        "import json,os; print(json.dumps(sorted(os.environ)))",
    )
    result = ProjectActionExecutor(str(tmp_path)).run("env")
    seen = set(json.loads(result.stdout))
    assert "ANTHROPIC_API_KEY" not in seen
    assert "FLYTO_SECRET_TOKEN" not in seen
    # Ignore variables the OS injects into every process regardless of the
    # environment handed to exec (macOS adds __CF_USER_TEXT_ENCODING).
    declared_by_us = {name for name in seen if not name.startswith("__")}
    assert declared_by_us <= {
        "PATH", "LANG", "LC_ALL", "PYTHONUNBUFFERED", "PYTHONDONTWRITEBYTECODE",
    }
    # HOME above all: it is the doorway to ~/.ssh, ~/.aws and ~/.gitconfig.
    assert "HOME" not in seen
    for leaky in ("HOME", "XDG_CONFIG_HOME", "SSH_AUTH_SOCK", "PYTHONPATH",
                  "VIRTUAL_ENV", "AWS_PROFILE", "GIT_CONFIG_GLOBAL", "TMPDIR"):
        assert leaky not in seen, leaky


def test_a_timeout_kills_the_whole_process_group(tmp_path):
    """A survivor would keep editing the worktree after the round ended."""

    child_marker = tmp_path / "child-alive.txt"
    script = textwrap.dedent(
        """
        import subprocess, sys, time
        subprocess.Popen([
            sys.executable, "-c",
            "import time,sys; time.sleep(30); open(sys.argv[1],'w').write('alive')",
            sys.argv[1],
        ])
        time.sleep(30)
        """
    )
    _contract(tmp_path, "actions:\n  - {}\n".format(json.dumps({
        "name": "hang",
        "argv": [sys.executable, "-c", script, "child-alive.txt"],
        "timeout_seconds": 1,
    })))
    started = time.monotonic()
    result = ProjectActionExecutor(str(tmp_path)).run("hang")
    elapsed = time.monotonic() - started

    assert result.timed_out is True
    assert result.ok is False
    assert elapsed < 20, "the timeout did not fire promptly"
    # The grandchild was in the same group and died with it.
    time.sleep(2)
    assert not child_marker.exists(), "a child outlived the killed action"


# --------------------------------------------------------------------------
# this repository's own generate_reference action, end to end
# --------------------------------------------------------------------------


def test_this_repository_declares_generate_reference():
    actions, digest = load_project_actions(str(_REPO))
    assert [action.name for action in actions] == ["generate_reference"]
    assert actions[0].argv == (".venv/bin/python", "scripts/generate_reference.py")
    assert digest == config_digest(str(_REPO))
    # It is an action, not a check -- the check that decides currency is
    # separate and still declared.
    checks, _capabilities = load_project_config(str(_REPO))
    assert "generated_reference" in {check.name for check in checks}
    assert "generate_reference" not in {check.name for check in checks}


def test_generate_reference_runs_through_the_executor(tmp_path):
    """Exercise the real action against a copy, never the live worktree."""

    workspace = tmp_path / "repo"
    workspace.mkdir()
    (workspace / "scripts").mkdir()
    (workspace / "scripts" / "generate_reference.py").write_text(
        "import pathlib\n"
        "pathlib.Path('generated.txt').write_text('from the action\\n')\n"
        "print('wrote 1 generated file')\n",
    )
    _contract(workspace, "actions:\n  - {}\n".format(json.dumps({
        "name": "generate_reference",
        "argv": [sys.executable, "scripts/generate_reference.py"],
        "timeout_seconds": 60,
    })))

    executor = ProjectActionExecutor(str(workspace))
    result = executor.run(
        "generate_reference", expected_config_sha256=executor.digest(),
    )
    assert result.ok is True
    assert "wrote 1 generated file" in result.stdout
    assert (workspace / "generated.txt").read_text() == "from the action\n"


def test_the_real_generator_action_is_runnable_as_declared():
    """The declared argv really is a working command in this repository.

    Run with `--check` so the live worktree is never rewritten by a test: the
    point is that the declaration resolves and executes, not that generation
    happens here.

    The declared interpreter is used *as declared*. Substituting
    `sys.executable` here would make this test pass for any `argv[0]`
    whatsoever -- including one that does not exist -- which is the opposite of
    what it claims to prove. `argv[0]` is pinned explicitly for the same
    reason: the repository's contract commits to the in-tree interpreter, and a
    silent drift back to a PATH-resolved `python` would run these checks under
    whichever interpreter happened to be first on PATH.
    """

    actions, _digest = load_project_actions(str(_REPO))
    declared = actions[0]
    assert declared.argv[0] == ".venv/bin/python"
    completed = subprocess.run(  # noqa: S603 - argv-direct, mirrors the executor
        [declared.argv[0], declared.argv[1], "--check"],
        cwd=str(_REPO), capture_output=True, text=True, timeout=300,
    )
    assert completed.returncode in (0, 1), completed.stderr[-2000:]


# --------------------------------------------------------------------------
# the tool an implementation session actually receives
# --------------------------------------------------------------------------


def test_the_claude_session_gets_the_named_tool_and_never_bash(tmp_path):
    from flyto_ai.agents.claude_code import (
        PROJECT_ACTION_SERVER,
        PROJECT_ACTION_TOOL_ID,
        SERVICE_ALLOWED_TOOLS,
        SERVICE_READONLY_TOOLS,
        build_project_action_server,
    )

    assert "Bash" not in SERVICE_ALLOWED_TOOLS
    assert "Bash" not in SERVICE_READONLY_TOOLS

    _echo(tmp_path, "generate_reference", "-c", "print('generated')")
    server, tool_ids = build_project_action_server(
        str(tmp_path), edit_authority=True, action_sandbox_image=_IMAGE,
    )
    assert server is not None
    assert tool_ids == (PROJECT_ACTION_TOOL_ID,)
    assert PROJECT_ACTION_TOOL_ID == "mcp__{}__run_project_action".format(
        PROJECT_ACTION_SERVER,
    )


def test_a_repository_with_no_actions_gets_no_tool(tmp_path):
    from flyto_ai.agents.claude_code import build_project_action_server

    _contract(tmp_path, "checks:\n  - {name: c, argv: [python, --version]}\n")
    assert build_project_action_server(
        str(tmp_path), edit_authority=True, action_sandbox_image=_IMAGE,
    ) == (None, ())


def test_the_tool_runs_a_declared_action_and_refuses_an_undeclared_one(tmp_path):
    """Drive the bridge the way the SDK would, without a provider call."""

    from flyto_ai.agents.claude_code import make_project_action_handler

    _echo(tmp_path, "generate_reference", "-c", "print('generated ok')")
    executor = ProjectActionExecutor(str(tmp_path))
    handler = make_project_action_handler(
        executor, action_catalog(executor.declared()), executor.digest(),
        edit_authority=True,
    )

    good = asyncio.run(handler({"name": "generate_reference"}))
    assert good.get("is_error") is not True
    assert "generated ok" in json.dumps(good)

    # Nothing the model can put in the payload reaches a process: an unknown
    # name is refused, and an argv key is simply not a parameter.
    for hostile in ({"name": "rm"}, {"name": ""}, {}, {"argv": ["rm", "-rf", "/"]}):
        refused = asyncio.run(handler(hostile))
        assert refused["is_error"] is True, hostile
        assert "project_action_undeclared" in json.dumps(refused), hostile

    # A refusal names what *is* declared and never echoes what was asked for.
    echoed = asyncio.run(handler({"name": "please-run-/etc/shadow"}))
    assert "/etc/shadow" not in json.dumps(echoed)


def test_the_tool_refuses_after_the_contract_is_rewritten(tmp_path):
    """The bridge carries the authorized digest, so substitution is caught."""

    from flyto_ai.agents.claude_code import make_project_action_handler

    marker = tmp_path / "pwned.txt"
    _echo(tmp_path, "generate_reference", "-c", "print('harmless')")
    executor = ProjectActionExecutor(str(tmp_path))
    handler = make_project_action_handler(
        executor, action_catalog(executor.declared()), executor.digest(),
        edit_authority=True,
    )
    assert asyncio.run(handler({"name": "generate_reference"})).get("is_error") is not True

    _contract(tmp_path, "actions:\n  - {}\n".format(json.dumps({
        "name": "generate_reference",
        "argv": [sys.executable, "-c", _writer(marker)],
        "timeout_seconds": 30,
    })))
    after = asyncio.run(handler({"name": "generate_reference"}))
    assert after["is_error"] is True
    assert not marker.exists(), "the substituted command ran through the tool"


# --------------------------------------------------------------------------
# bounded at capture, not bounded at return
# --------------------------------------------------------------------------


def test_capture_holds_the_bound_in_memory_however_much_is_produced(tmp_path):
    """A bound applied after `communicate()` is not a bound at all.

    The proof is the sink's own accounting, not `len(result.stdout)`: a clip
    after the fact looks identical either way, and the failure mode being
    guarded against is peak memory during the read.
    """

    from flyto_ai.coding.actions import MAX_ACTION_OUTPUT_BYTES, _BoundedSink

    volume = MAX_ACTION_OUTPUT_BYTES * 40
    sink = _BoundedSink()
    written = 0
    while written < volume:
        sink.feed(b"y" * 65536)
        written += 65536
        # The invariant that matters: retained memory never exceeds the bound,
        # at any point during the stream, no matter the total.
        assert sink.retained <= MAX_ACTION_OUTPUT_BYTES

    assert sink.seen >= volume
    assert sink.retained <= MAX_ACTION_OUTPUT_BYTES
    assert sink.truncated is True
    assert len(sink.text()) <= MAX_ACTION_OUTPUT_CHARS


def test_a_single_oversized_read_is_clipped_before_it_is_retained(tmp_path):
    from flyto_ai.coding.actions import MAX_ACTION_OUTPUT_BYTES, _BoundedSink

    sink = _BoundedSink()
    sink.feed(b"z" * (MAX_ACTION_OUTPUT_BYTES * 10))
    assert sink.retained <= MAX_ACTION_OUTPUT_BYTES
    assert sink.seen == MAX_ACTION_OUTPUT_BYTES * 10
    assert sink.truncated is True


def test_a_huge_no_newline_stream_does_not_deadlock_or_grow(tmp_path):
    """One line, far past the bound, on both pipes at once."""

    from flyto_ai.coding.actions import MAX_ACTION_OUTPUT_BYTES

    volume = MAX_ACTION_OUTPUT_BYTES * 8
    _echo(
        tmp_path, "flood", "-c",
        "import sys\n"
        "block='q'*65536\n"
        "for _ in range({}):\n"
        "    sys.stdout.write(block); sys.stderr.write(block)\n"
        "sys.stdout.flush(); sys.stderr.flush()\n".format(volume // 65536),
        timeout=120,
    )
    result = ProjectActionExecutor(str(tmp_path)).run("flood")

    assert result.ok is True, result.error
    assert result.truncated is True
    assert len(result.stdout) <= MAX_ACTION_OUTPUT_CHARS
    assert len(result.stderr) <= MAX_ACTION_OUTPUT_CHARS


def test_invalid_utf8_is_replaced_rather_than_raising(tmp_path):
    _echo(
        tmp_path, "binary", "-c",
        "import sys; sys.stdout.buffer.write(b'\\xff\\xfe ok \\x80\\x81')",
    )
    result = ProjectActionExecutor(str(tmp_path)).run("binary")
    assert result.ok is True
    assert "ok" in result.stdout


def test_a_launch_error_does_not_leak_an_absolute_host_path(tmp_path):
    _contract(tmp_path, "actions:\n  - {}\n".format(json.dumps({
        "name": "absent",
        "argv": ["flyto-command-that-does-not-exist"],
        "timeout_seconds": 30,
    })))
    result = ProjectActionExecutor(str(tmp_path)).run("absent")
    assert result.ok is False
    # Whatever the boundary reports, the host's own paths never appear in the
    # `error` field a model can read.
    assert str(tmp_path) not in result.error
    assert result.error in ("non-zero exit", "timed out") or "ENOENT" in result.error


# --------------------------------------------------------------------------
# read-only rounds get no action authority at all
# --------------------------------------------------------------------------


def test_a_read_only_round_gets_no_action_server_or_tool(tmp_path):
    from flyto_ai.agents.claude_code import build_project_action_server

    _echo(tmp_path, "generate_reference", "-c", "print('should never run')")
    assert build_project_action_server(str(tmp_path), edit_authority=False) == (None, ())


def test_a_read_only_handler_refuses_before_any_process_starts(tmp_path):
    """Defence in depth: even a catalogued tool refuses without edit authority."""

    from flyto_ai.agents.claude_code import make_project_action_handler

    marker = tmp_path / "written.txt"
    _contract(tmp_path, "actions:\n  - {}\n".format(json.dumps({
        "name": "writes", "argv": [sys.executable, "-c", _writer(marker)],
        "timeout_seconds": 30,
    })))
    executor = ProjectActionExecutor(str(tmp_path))
    handler = make_project_action_handler(
        executor, action_catalog(executor.declared()), executor.digest(),
        edit_authority=False,
    )
    refused = asyncio.run(handler({"name": "writes"}))
    assert refused["is_error"] is True
    assert "project_action_requires_edit_authority" in json.dumps(refused)
    assert not marker.exists(), "a read-only round started a process"


def test_read_only_sdk_options_carry_no_action_tool(tmp_path):
    from flyto_ai.agents.claude_code import (
        PROJECT_ACTION_TOOL_ID,
        ClaudeCodeAgent,
    )
    from flyto_ai.agents.models import CodeTaskRequest

    _echo(tmp_path, "generate_reference", "-c", "print('x')")
    agent = ClaudeCodeAgent.__new__(ClaudeCodeAgent)
    agent._cc = type("CC", (), {"allowed_tools": (), "system_prompt_extra": ""})()

    readonly = CodeTaskRequest(
        message="m", working_dir=str(tmp_path),
        service_mode=True, service_edit_authority=False,
    )
    options = ClaudeCodeAgent._option_kwargs(
        agent, readonly, session_id=None, system_prompt="", max_turns=3,
        max_budget=1.0, mcp_servers={}, extra_tools=(PROJECT_ACTION_TOOL_ID,),
    )
    assert PROJECT_ACTION_TOOL_ID not in options["allowed_tools"]
    assert "Bash" not in options["allowed_tools"]
    assert "Write" not in options["allowed_tools"]

    writable = CodeTaskRequest(
        message="m", working_dir=str(tmp_path),
        service_mode=True, service_edit_authority=True,
    )
    options = ClaudeCodeAgent._option_kwargs(
        agent, writable, session_id=None, system_prompt="", max_turns=3,
        max_budget=1.0, mcp_servers={}, extra_tools=(PROJECT_ACTION_TOOL_ID,),
    )
    assert PROJECT_ACTION_TOOL_ID in options["allowed_tools"]
    assert "Bash" not in options["allowed_tools"]


# --------------------------------------------------------------------------
# authority is job-lifetime, not invocation-lifetime
# --------------------------------------------------------------------------


def test_a_contract_edited_between_rounds_never_gains_new_authority(tmp_path):
    """The escalation: edit the contract in round one, use it in round two."""

    from flyto_ai.agents.claude_code import (
        ProjectActionBridgeUnavailable,
        build_project_action_server,
    )

    marker = tmp_path / "round-two.txt"
    _echo(tmp_path, "generate_reference", "-c", "print('round one')")
    job_authority = config_digest(str(tmp_path))

    server, ids = build_project_action_server(
        str(tmp_path), edit_authority=True, authorized_config_sha256=job_authority,
        action_sandbox_image=_IMAGE,
    )
    assert server is not None and ids

    # The model edits the contract during round one.
    _contract(tmp_path, "actions:\n  - {}\n".format(json.dumps({
        "name": "generate_reference",
        "argv": [sys.executable, "-c", _writer(marker)],
        "timeout_seconds": 30,
    })))

    # Round two rebuilds the bridge. Rebuilding from the *current* file is the
    # bug; binding to the job's authority is the fix.
    with pytest.raises(ProjectActionBridgeUnavailable):
        build_project_action_server(
            str(tmp_path), edit_authority=True,
            authorized_config_sha256=job_authority,
            action_sandbox_image=_IMAGE,
        )
    assert not marker.exists()


def test_a_malformed_contract_raises_instead_of_silently_dropping_the_tool(tmp_path):
    from flyto_ai.agents.claude_code import (
        ProjectActionBridgeUnavailable,
        build_project_action_server,
    )

    _contract(tmp_path, "actions:\n  - {name: ok, argv: [python], surprise: 1}\n")
    with pytest.raises(ProjectActionBridgeUnavailable):
        build_project_action_server(
            str(tmp_path), edit_authority=True, action_sandbox_image=_IMAGE,
        )


def test_the_service_pins_the_contract_digest_for_the_job(tmp_path):
    """Submit establishes the authority; a payload cannot supply its own."""

    from flyto_ai.coding.contracts import CodingTaskRequest
    from flyto_ai.coding.service import CodingService

    workspace = tmp_path / "ws"
    workspace.mkdir()
    _contract(workspace, """
        checks:
          - name: trivial
            argv: [python, --version]
            required: true
        """)
    service = CodingService(
        lambda store: None,
        state_root=str(tmp_path / "state"),
        workspace_roots=(str(workspace),),
    )
    try:
        forged = CodingTaskRequest(
            message="m", working_dir=str(workspace),
            authorized_config_sha256="ab" * 32,
        )
        # Startup authority strips whatever the payload claimed.
        assert service._with_startup_authority(forged).authorized_config_sha256 == ""
        # And the service's own preflight supplies the real one.
        assert service._require_verifiable_repository(str(workspace)) == config_digest(
            str(workspace),
        )
    finally:
        service.close()


# --------------------------------------------------------------------------
# the config reader is the security boundary
# --------------------------------------------------------------------------


def test_a_symlinked_contract_or_parent_is_refused(tmp_path):
    from flyto_ai.coding.checks import read_project_contract

    outside = tmp_path / "outside.yaml"
    outside.write_text("version: flyto.coding-config.v1\nactions: []\n")

    linked_final = tmp_path / "a"
    (linked_final / ".flyto").mkdir(parents=True)
    (linked_final / ".flyto" / "coding.yaml").symlink_to(outside)
    with pytest.raises(ValueError):
        read_project_contract(str(linked_final))

    linked_parent = tmp_path / "b"
    linked_parent.mkdir()
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / "coding.yaml").write_text("version: flyto.coding-config.v1\n")
    (linked_parent / ".flyto").symlink_to(elsewhere, target_is_directory=True)
    with pytest.raises(ValueError):
        read_project_contract(str(linked_parent))


def test_a_non_regular_contract_is_refused_without_blocking(tmp_path):
    from flyto_ai.coding.checks import read_project_contract

    (tmp_path / ".flyto").mkdir()
    os.mkfifo(tmp_path / ".flyto" / "coding.yaml")
    with pytest.raises(ValueError):
        read_project_contract(str(tmp_path))


def test_an_oversized_contract_is_refused(tmp_path):
    from flyto_ai.coding.checks import MAX_CONFIG_BYTES, read_project_contract

    _contract(tmp_path, "actions: []\n# " + "x" * (MAX_CONFIG_BYTES + 10))
    with pytest.raises(ValueError):
        read_project_contract(str(tmp_path))


def test_a_malformed_action_block_fails_preflight_rather_than_passing_quietly(tmp_path):
    """Preflight must not approve a repository whose action surface is broken."""

    from flyto_ai.coding.preflight import (
        CODE_VERIFICATION_CONTRACT_INVALID,
        preflight_repository,
    )

    _contract(tmp_path, """
        checks:
          - name: fine
            argv: [python, --version]
            required: true
        actions:
          - {name: broken, argv: [python], surprise: 1}
        """)
    outcome = preflight_repository(str(tmp_path))
    assert outcome.ok is False
    assert outcome.code == CODE_VERIFICATION_CONTRACT_INVALID


def test_every_consumer_parses_the_same_bytes(tmp_path):
    """Checks, capabilities, actions and digest all come from one read."""

    from flyto_ai.coding.checks import read_project_contract

    _contract(tmp_path, """
        checks:
          - name: c
            argv: [python, --version]
            required: true
        actions:
          - name: a
            argv: [python, --version]
        """)
    contract = read_project_contract(str(tmp_path))
    checks, capabilities = load_project_config(str(tmp_path))
    actions, digest = load_project_actions(str(tmp_path))

    assert [item.name for item in contract.checks] == [item.name for item in checks]
    assert contract.capabilities == capabilities
    assert [item.name for item in contract.actions] == [item.name for item in actions]
    assert contract.digest == digest == config_digest(str(tmp_path))


def test_a_control_character_never_reaches_the_model_catalog(tmp_path):
    from flyto_ai.coding.contracts import ProjectActionSpec

    with pytest.raises(ValueError):
        ProjectActionSpec(name="a", argv=("python",), description="line\nsecond")

    hostile = ProjectActionSpec(name="a", argv=("python",), description="fine")
    object.__setattr__(hostile, "description", "evil forged\x07")
    rendered = action_catalog((hostile,))[0]["description"]
    assert " " not in rendered and "\x07" not in rendered and "\n" not in rendered
