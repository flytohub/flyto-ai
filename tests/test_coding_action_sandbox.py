# Copyright 2024 Flyto2
# Licensed under the Apache License, Version 2.0
"""A declared action must never execute repository code on the host.

Pinning the declaration was not enough, and the reason is worth stating plainly:
`python scripts/generate.py` is a pinned *instruction to run whatever that
script says at the moment it runs*. The implementer holds edit authority over
that script. So a digest-bound argv, a clean environment and a fixed PATH still
composed into arbitrary host code execution - the pin constrained the sentence,
not the program.

Isolation is what closes it. These tests assert on the launch argv rather than
on a running container, because the property being protected is *what the host
was asked to start*: a container with no network, no Docker socket, no host home
or credentials, a read-only root, dropped capabilities, and exactly one writable
bind. A test that needed a live daemon would silently skip on the machines that
most need the guarantee checked.
"""
import json
import os
import shutil as shutil_module
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

import pytest

from flyto_ai.coding.actions import (
    ACTION_SANDBOX_UNAVAILABLE,
    ActionSandboxUnavailable,
    ProjectActionExecutor,
)
from flyto_ai.coding.checks import config_digest
from flyto_ai.coding.workspace import CONTAINER_WORKSPACE

_IMAGE = "flyto-action-image:pinned"
#: The immutable identity the probe is taken to have resolved `_IMAGE` to. The
#: tag locates the image; only this ever reaches `docker run`.
_IMAGE_ID_X = "sha256:" + "1a" * 32
_IMAGE_ID_Y = "sha256:" + "2b" * 32


def _contract(workspace: Path, body: str) -> Path:
    config = workspace / ".flyto" / "coding.yaml"
    config.parent.mkdir(parents=True, exist_ok=True)
    config.write_text("version: flyto.coding-config.v1\n" + textwrap.dedent(body))
    return config


def _declare(workspace: Path, *args: str, name: str = "regenerate") -> Path:
    return _contract(workspace, "actions:\n  - {}\n".format(json.dumps({
        "name": name,
        "argv": [sys.executable, *args],
        "timeout_seconds": 30,
    })))


@pytest.fixture()
def sandboxed(monkeypatch):
    """An executor whose backend probe succeeds without needing a daemon."""

    def resolve(self):
        # Stands in for `docker image inspect --format {{.Id}}`: a backend is
        # only usable once it has produced an immutable identity.
        self._image_id = _IMAGE_ID_X
        return "docker"

    monkeypatch.setattr(ProjectActionExecutor, "_detect_backend", resolve)

    def build(workspace: Path) -> ProjectActionExecutor:
        executor = ProjectActionExecutor(str(workspace), sandbox_image=_IMAGE)
        # A deterministic client path so argv assertions do not depend on where
        # docker happens to be installed.
        executor._docker = "/usr/bin/docker"
        return executor

    return build


def _argv(executor: ProjectActionExecutor, workspace: Path):
    action = executor.declared()[0]
    argv, _cidfile = executor._sandboxed_argv(action, workspace)
    return argv


def _flag_values(argv, flag):
    return [argv[index + 1] for index, item in enumerate(argv) if item == flag]


# --------------------------------------------------------------------------
# the finding: repository code never runs on the host
# --------------------------------------------------------------------------


def test_a_workspace_script_edited_after_authorization_is_never_run_on_the_host(
    tmp_path, sandboxed, monkeypatch,
):
    """The exact escape: edit the script the pinned argv executes, then invoke.

    The declaration is untouched, so its digest still matches - which is
    precisely why the digest could never have prevented this.
    """

    workspace = tmp_path / "ws"
    (workspace / "scripts").mkdir(parents=True)
    marker = tmp_path / "host-was-reached.txt"
    (workspace / "scripts" / "generate.py").write_text("print('harmless')\n")
    _declare(workspace, "scripts/generate.py")

    executor = sandboxed(workspace)
    authorized = config_digest(str(workspace))

    # The implementer rewrites the *script*, not the contract.
    (workspace / "scripts" / "generate.py").write_text(
        "open({!r}, 'w').write('pwned')\n".format(str(marker)),
    )
    assert config_digest(str(workspace)) == authorized, (
        "the contract itself must be unchanged for this test to mean anything"
    )

    launched = []

    class Fake:
        returncode = 0
        stdout = stderr = None

        def __init__(self, argv, **kwargs):
            launched.append(argv)

        def wait(self, timeout=None):
            return 0

    monkeypatch.setattr(subprocess, "Popen", Fake)
    executor.run("regenerate", expected_config_sha256=authorized)

    assert launched, "nothing was launched at all"
    argv = launched[0]
    # The host was asked to start a container, not the interpreter.
    assert argv[0].endswith("docker") and argv[1] == "run", argv[:3]
    assert sys.executable not in argv
    assert not marker.exists(), "repository code executed on the host"


def test_there_is_no_host_execution_fallback_anywhere(tmp_path, monkeypatch):
    """No sandbox means no action - never a direct launch "just this once"."""

    workspace = tmp_path / "ws"
    workspace.mkdir()
    _declare(workspace, "-c", "print('x')")

    monkeypatch.setattr(ProjectActionExecutor, "_detect_backend", lambda self: "")
    launched = []
    monkeypatch.setattr(
        subprocess, "Popen",
        lambda *a, **k: launched.append(a) or pytest.fail("a process was started"),
    )

    with pytest.raises(ActionSandboxUnavailable) as refused:
        ProjectActionExecutor(str(workspace), sandbox_image=_IMAGE)

    assert refused.value.code == ACTION_SANDBOX_UNAVAILABLE
    assert launched == []

    # And an empty pinned image is equally refused: no image, no root
    # filesystem, no boundary.
    monkeypatch.setattr(
        ProjectActionExecutor, "_detect_backend", lambda self: "docker",
    )
    with pytest.raises(ActionSandboxUnavailable):
        ProjectActionExecutor(str(workspace), sandbox_image="")


def test_the_module_contains_no_unsandboxed_launch_path():
    """A second `Popen` would be a second, unreviewed boundary."""

    import flyto_ai.coding.actions as actions

    source = Path(actions.__file__).read_text(encoding="utf-8")
    # One call site; the other occurrence is a type annotation on the reaper.
    assert source.count("subprocess.Popen(") == 1
    # The single launch passes the *wrapped* argv. `action.argv` reaches a
    # process only through `_sandboxed_argv`, which puts the container in front
    # of it; a direct `Popen(list(action.argv)` would be the regression.
    assert "Popen(list(action.argv" not in source
    assert "Popen(action.argv" not in source
    lines = source.splitlines()
    index = next(i for i, line in enumerate(lines) if "subprocess.Popen(" in line)
    assert lines[index + 1].strip() == "argv,", lines[index + 1]


# --------------------------------------------------------------------------
# what the boundary actually asks for
# --------------------------------------------------------------------------


def test_the_container_has_no_network_socket_or_host_identity(tmp_path, sandboxed):
    workspace = tmp_path / "ws"
    workspace.mkdir()
    _declare(workspace, "-c", "print('x')")
    argv = _argv(sandboxed(workspace), workspace)

    assert _flag_values(argv, "--network") == ["none"]
    assert "--read-only" in argv
    assert "--cap-drop" in argv and _flag_values(argv, "--cap-drop") == ["ALL"]
    assert _flag_values(argv, "--security-opt") == ["no-new-privileges"]
    assert _flag_values(argv, "--pids-limit") == ["128"]
    assert _flag_values(argv, "--memory") == ["1g"]
    assert _flag_values(argv, "--cpus") == ["2"]

    rendered = " ".join(argv)
    # The Docker socket would make every other flag decorative.
    assert "docker.sock" not in rendered
    assert "/var/run" not in rendered
    # No host home, credential store or provider configuration is bound.  The
    # trusted check runner gives pytest an empty synthetic HOME and places its
    # tmp directory below it, so the claimed test workspace may legitimately
    # contain the synthetic HOME string.  What must never happen is binding
    # HOME itself into the container.
    bind_sources = {
        item.split("src=", 1)[1].split(",", 1)[0]
        for item in _flag_values(argv, "--mount")
        if item.startswith("type=bind,") and "src=" in item
    }
    assert str(Path.home()) not in bind_sources
    for leaked in (".ssh", ".aws", ".gnupg", ".netrc", ".gitconfig",
                   ".config", "ANTHROPIC"):
        assert leaked not in rendered, leaked
    # HOME inside points at container-local scratch only.
    assert "HOME=/tmp/home" in argv


def test_only_the_claimed_workspace_is_writable(tmp_path, sandboxed):
    workspace = tmp_path / "ws"
    workspace.mkdir()
    _declare(workspace, "-c", "print('x')")
    argv = _argv(sandboxed(workspace), workspace)

    mounts = _flag_values(argv, "--mount")
    binds = [item for item in mounts if item.startswith("type=bind")]
    writable = [item for item in binds if "readonly" not in item]

    assert len(writable) == 1, writable
    assert writable[0] == "type=bind,src={},dst={}".format(
        workspace.resolve(), CONTAINER_WORKSPACE,
    )
    # Writes must be attributable to the job, so the container runs as the host
    # identity that owns the tree rather than as `nobody`.
    if os.name == "posix":
        assert _flag_values(argv, "--user") == [
            "{}:{}".format(os.getuid(), os.getgid()),
        ]


def test_a_sibling_repository_is_never_mounted(tmp_path, sandboxed):
    """Sharing a parent directory is not a reason to expose another repo."""

    sibling = tmp_path / "other-repo"
    (sibling / "src").mkdir(parents=True)
    (sibling / "src" / "secret.py").write_text("token = 'nope'\n")
    workspace = tmp_path / "ws"
    workspace.mkdir()
    _declare(workspace, "-c", "print('x')")

    argv = _argv(sandboxed(workspace), workspace)
    rendered = " ".join(argv)

    assert str(sibling) not in rendered
    # The shared parent is not mounted either, which is how the sibling would
    # have arrived without ever being named.
    assert "src={},".format(tmp_path) not in rendered
    for mount in _flag_values(argv, "--mount"):
        if mount.startswith("type=bind") and "readonly" not in mount:
            assert mount.split("src=")[1].split(",")[0] == str(workspace.resolve())


def test_protected_paths_are_masked_inside_the_container(tmp_path, sandboxed):
    """A writable mount would otherwise hand `.git` and secrets to the action."""

    workspace = tmp_path / "ws"
    (workspace / ".git").mkdir(parents=True)
    (workspace / ".git" / "config").write_text("[remote]\n")
    (workspace / ".env").write_text("TOKEN=nope\n")
    (workspace / "deploy.pem").write_text("-----BEGIN-----\n")
    _declare(workspace, "-c", "print('x')")

    argv = _argv(sandboxed(workspace), workspace)
    mounts = _flag_values(argv, "--mount")

    masked_dirs = [item for item in mounts if item.startswith("type=tmpfs")]
    assert any("{}/.git".format(CONTAINER_WORKSPACE) in item for item in masked_dirs)
    assert all("tmpfs-mode=000" in item for item in masked_dirs)

    empty_read_only = [
        item for item in mounts
        if "blocked-file" in item and "readonly" in item
    ]
    covered = " ".join(empty_read_only)
    assert "{}/.env".format(CONTAINER_WORKSPACE) in covered
    assert "{}/deploy.pem".format(CONTAINER_WORKSPACE) in covered


def test_generators_receive_only_a_sanitized_read_only_file_manifest(
    tmp_path, sandboxed,
):
    workspace = tmp_path / "ws"
    workspace.mkdir()
    subprocess.run(
        ["git", "init", "-q", str(workspace)], check=True,
        stdin=subprocess.DEVNULL,
    )
    (workspace / "tracked.py").write_text("print('tracked')\n")
    (workspace / "untracked.ts").write_text("export const value = 1\n")
    (workspace / ".env").write_text("TOKEN=nope\n")
    subprocess.run(
        ["git", "-C", str(workspace), "add", "tracked.py"], check=True,
        stdin=subprocess.DEVNULL,
    )
    _declare(workspace, "-c", "print('x')")

    argv = _argv(sandboxed(workspace), workspace)
    manifests = [
        item for item in _flag_values(argv, "--mount")
        if "dst=/run/flyto-action/tracked-files" in item
    ]

    assert len(manifests) == 1
    assert manifests[0].endswith(",readonly")
    source = manifests[0].split("src=", 1)[1].split(",dst=", 1)[0]
    assert Path(source).read_text(encoding="utf-8") == (
        ".flyto/coding.yaml\ntracked.py\nuntracked.ts\n"
    )
    rendered = " ".join(argv)
    assert "{}/.git".format(CONTAINER_WORKSPACE) in rendered
    assert "TOKEN=nope" not in rendered


def test_the_image_comes_from_startup_and_not_from_the_contract(tmp_path, sandboxed):
    workspace = tmp_path / "ws"
    workspace.mkdir()
    # A contract cannot even express an image; assert the pinned one is used
    # and that it sits immediately before the command.
    _declare(workspace, "-c", "print('x')")
    argv = _argv(sandboxed(workspace), workspace)

    # The resolved identity, and the mutable tag nowhere at all.
    assert _IMAGE_ID_X in argv
    assert _IMAGE not in argv
    assert argv[argv.index(_IMAGE_ID_X) + 1:] == ["python3", "-c", "print('x')"]


def test_the_declared_argv_is_carried_verbatim_after_the_image(tmp_path, sandboxed):
    workspace = tmp_path / "ws"
    workspace.mkdir()
    payload = "; rm -rf / && echo pwned"
    _contract(workspace, "actions:\n  - {}\n".format(json.dumps({
        "name": "literal",
        "argv": [sys.executable, "-c", payload],
        "timeout_seconds": 30,
    })))
    argv = _argv(sandboxed(workspace), workspace)

    tail = argv[argv.index(_IMAGE_ID_X) + 1:]
    # Only argv[0] is re-pointed at the image's interpreter; arguments are
    # untouched and there is still no shell to interpret them.
    assert tail == ["python3", "-c", payload]


def test_a_declared_subdir_moves_the_workdir_inside_the_container(tmp_path, sandboxed):
    workspace = tmp_path / "ws"
    (workspace / "sub").mkdir(parents=True)
    _contract(workspace, "actions:\n  - {}\n".format(json.dumps({
        "name": "inner",
        "argv": [sys.executable, "-c", "print('x')"],
        "working_subdir": "sub",
        "timeout_seconds": 30,
    })))
    argv = _argv(sandboxed(workspace), workspace / "sub")

    assert _flag_values(argv, "--workdir")[-1] == "{}/sub".format(CONTAINER_WORKSPACE)
    rendered = " ".join(argv)
    assert str(workspace / "sub") not in rendered.split("src=")[0]


# --------------------------------------------------------------------------
# cleanup and contract authority survive the move into the sandbox
# --------------------------------------------------------------------------


def test_a_timeout_reaps_the_container_not_just_the_client(tmp_path, sandboxed, monkeypatch):
    """Killing `docker run` does not stop what it started."""

    workspace = tmp_path / "ws"
    workspace.mkdir()
    _declare(workspace, "-c", "print('x')")
    executor = sandboxed(workspace)

    container_id = "a" * 64
    removed = []

    class HangingProcess:
        returncode = -9
        stdout = stderr = None
        pid = 4242

        def __init__(self, argv, **kwargs):
            cidfile = argv[argv.index("--cidfile") + 1]
            Path(cidfile).write_text(container_id, encoding="utf-8")

        def wait(self, timeout=None):
            if timeout is not None:
                raise subprocess.TimeoutExpired("docker", timeout)
            return -9

    monkeypatch.setattr(subprocess, "Popen", HangingProcess)
    monkeypatch.setattr(
        ProjectActionExecutor, "_terminate_group", staticmethod(lambda process: None),
    )

    def record_run(argv, **kwargs):
        removed.append(argv)
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(subprocess, "run", record_run)
    result = executor.run("regenerate")

    assert result.timed_out is True
    assert removed, "the container was never removed"
    assert removed[0][1:] == ["rm", "-f", container_id]


def test_a_contract_mutation_still_refuses_before_any_container_starts(
    tmp_path, sandboxed, monkeypatch,
):
    workspace = tmp_path / "ws"
    workspace.mkdir()
    _declare(workspace, "-c", "print('x')")
    executor = sandboxed(workspace)
    authorized = config_digest(str(workspace))

    _declare(workspace, "-c", "print('different')")
    assert config_digest(str(workspace)) != authorized

    launched = []
    monkeypatch.setattr(
        subprocess, "Popen",
        lambda *a, **k: launched.append(a) or pytest.fail("a container was started"),
    )
    from flyto_ai.coding.actions import ProjectActionError

    with pytest.raises(ProjectActionError) as refused:
        executor.run("regenerate", expected_config_sha256=authorized)

    # Still the contract code, not a sandbox or provider code.
    assert refused.value.code == "project_action_failed"
    assert not isinstance(refused.value, ActionSandboxUnavailable)
    assert launched == []


def test_the_read_only_model_boundary_is_unchanged(tmp_path):
    """A read/write action mode must not loosen the existing coding_run sandbox."""

    from flyto_ai.coding.workspace import (
        container_hardening_argv,
        container_workspace_mount_argv,
    )

    shared = container_hardening_argv("/usr/bin/docker", "/tmp/cid")
    assert "--network" in shared and shared[shared.index("--network") + 1] == "none"
    assert "--read-only" in shared

    read_only = container_workspace_mount_argv(Path("/w"), writable=False)
    writable = container_workspace_mount_argv(Path("/w"), writable=True)
    assert read_only[1].endswith(",readonly")
    assert not writable[1].endswith(",readonly")
    # The mount mode is the *only* difference; nothing in the shared hardening
    # is parameterised by it.
    assert "writable" not in " ".join(shared)


# --------------------------------------------------------------------------
# pre-session classification
# --------------------------------------------------------------------------


def test_a_missing_sandbox_stops_the_round_before_the_provider(tmp_path, monkeypatch):
    """Zero provider calls, zero subprocess action calls, one stable code."""

    import asyncio

    from flyto_ai.agents.claude_code import ClaudeCodingAgent
    from flyto_ai.coding.contracts import (
        ACTION_SANDBOX_UNAVAILABLE as CONTRACT_CODE,
        ApprovalPolicy,
        CodingTaskRequest,
        SandboxMode,
    )
    from flyto_ai.coding.store import ThreadStore

    workspace = tmp_path / "ws"
    workspace.mkdir()
    _contract(workspace, """
        checks:
          - name: trivial
            argv: {argv}
            required: true
        actions:
          - name: regenerate
            argv: {argv}
            timeout_seconds: 30
        """.format(argv=json.dumps([sys.executable, "-c", "pass"])))

    provider_calls = []
    subprocess_calls = []

    class NeverCalled:
        async def run(self, request):
            provider_calls.append(request)
            raise AssertionError("the provider must not be reached")

    monkeypatch.setattr(ProjectActionExecutor, "_detect_backend", lambda self: "")
    # Record every launch rather than failing on the first one: `WorkspaceTools`
    # runs its own unrelated backend probe, and asserting on that would test the
    # harness. What must be zero is *action* launches, identified by the pinned
    # image that only the action boundary ever names.
    real_popen = subprocess.Popen

    def record(argv, *rest, **kwargs):
        subprocess_calls.append(list(argv) if isinstance(argv, (list, tuple)) else [argv])
        return real_popen(argv, *rest, **kwargs)

    monkeypatch.setattr(subprocess, "Popen", record)

    agent = ClaudeCodingAgent(
        ThreadStore(str(tmp_path / "threads")), agent=NeverCalled(),
    )
    result = asyncio.run(agent.run(CodingTaskRequest(
        message="regenerate please",
        working_dir=str(workspace),
        approval_policy=ApprovalPolicy.NEVER,
        sandbox_mode=SandboxMode.WORKSPACE_WRITE,
        authorized_config_sha256=config_digest(str(workspace)),
        command_sandbox_image=_IMAGE,
    )))

    assert result.ok is False
    assert result.failure_code == CONTRACT_CODE == ACTION_SANDBOX_UNAVAILABLE
    assert provider_calls == []
    # A capability *probe* (`docker image inspect`) is how the refusal is
    # decided and is expected. What must never happen is `docker run`, or any
    # direct launch of the declared command.
    action_launches = [
        call for call in subprocess_calls
        if "run" in call[1:2] or any(str(item).endswith(".py") for item in call[1:])
    ]
    assert action_launches == [], action_launches


def test_the_sandbox_code_has_typed_non_provider_semantics():
    from flyto_ai.coding.contracts import (
        ACTION_PROVISION_ACTION_SANDBOX,
        ACTION_SANDBOX_UNAVAILABLE as CODE,
        CodingJobReceipt,
        CodingJobState,
    )
    from flyto_ai.coding.service import receipt_to_mapping

    projected = receipt_to_mapping(CodingJobReceipt(
        job_id="job_" + "a1b2c3d4" * 3,
        state=CodingJobState.FAILED,
        submitted_at=1.0,
        updated_at=2.0,
        failure_code=CODE,
    ))
    assert projected["failure_phase"] == "preflight"
    assert projected["failure_phase"] != "provider"
    assert projected["retryable"] is False
    assert projected["required_actions"] == [ACTION_PROVISION_ACTION_SANDBOX]


# --------------------------------------------------------------------------
# the image is content-pinned, not name-pinned
# --------------------------------------------------------------------------


def _resolving(monkeypatch, resolved, returncode=0):
    """Make the probe resolve the startup tag to `resolved`."""

    real_run = subprocess.run

    def fake_run(argv, **kwargs):
        # Only the image probe is simulated. `WorkspaceTools` runs its own
        # unrelated `docker context inspect`, and hijacking that would be
        # testing the harness.
        if list(argv[1:4]) != ["image", "inspect", "--format"]:
            return real_run(argv, **kwargs)
        return subprocess.CompletedProcess(argv, returncode, stdout=resolved)

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(
        shutil_module, "which",
        lambda name, path=None: "/usr/bin/{}".format(name),
    )


def test_the_launch_uses_the_resolved_identity_not_the_retargeted_tag(
    tmp_path, monkeypatch,
):
    """Probe approves X; the tag is retargeted to Y before the launch.

    A mutable tag is not an identity. Had the launch spelled the tag, the probe
    would have approved one root filesystem and the run used another - the same
    authority-substitution shape as the contract digest bug, one layer down.
    """

    workspace = tmp_path / "ws"
    workspace.mkdir()
    _declare(workspace, "-c", "print('x')")

    _resolving(monkeypatch, _IMAGE_ID_X + "\n")
    executor = ProjectActionExecutor(str(workspace), sandbox_image=_IMAGE)
    assert executor._image_id == _IMAGE_ID_X

    _resolving(monkeypatch, _IMAGE_ID_Y + "\n")   # the tag now points elsewhere
    argv = _argv(executor, workspace)

    assert _IMAGE_ID_X in argv
    assert _IMAGE_ID_Y not in argv
    assert _IMAGE not in argv, "the mutable tag reached the launch"
    assert argv[argv.index(_IMAGE_ID_X) + 1] == "python3"


def test_the_launch_forbids_an_implicit_daemon_pull(tmp_path, monkeypatch):
    """Container network isolation says nothing about the daemon's own fetches."""

    workspace = tmp_path / "ws"
    workspace.mkdir()
    _declare(workspace, "-c", "print('x')")
    _resolving(monkeypatch, _IMAGE_ID_X)

    argv = _argv(ProjectActionExecutor(str(workspace), sandbox_image=_IMAGE), workspace)

    assert "--pull=never" in argv
    identity = argv[argv.index(_IMAGE_ID_X)]
    assert identity == _IMAGE_ID_X
    # `sha256:` plus 64 hex characters, and nothing looser.
    assert identity.startswith("sha256:") and len(identity) == 71
    # It is the last element before the command, so nothing sits between the
    # image and the argv it runs.
    assert argv[argv.index(_IMAGE_ID_X) + 1:] == ["python3", "-c", "print('x')"]


@pytest.mark.parametrize(
    "stdout,returncode",
    [
        ("", 0),
        ("\n", 0),
        (_IMAGE_ID_X + "\n" + _IMAGE_ID_Y + "\n", 0),
        ("1a" * 32 + "\n", 0),
        ("sha256:" + "1a" * 10 + "\n", 0),
        ("sha256:" + "zz" * 32 + "\n", 0),
        ("<no value>\n", 0),
        (_IMAGE_ID_X + "\n", 1),
    ],
)
def test_malformed_or_ambiguous_inspect_output_fails_closed(
    tmp_path, monkeypatch, stdout, returncode,
):
    workspace = tmp_path / "ws"
    workspace.mkdir()
    _declare(workspace, "-c", "print('x')")
    _resolving(monkeypatch, stdout, returncode)

    with pytest.raises(ActionSandboxUnavailable) as refused:
        ProjectActionExecutor(str(workspace), sandbox_image=_IMAGE)
    assert refused.value.code == ACTION_SANDBOX_UNAVAILABLE


def test_a_malformed_inspect_stops_the_round_with_zero_docker_runs(
    tmp_path, monkeypatch,
):
    """Pre-session: no provider call, and no `docker run` at all."""

    import asyncio

    from flyto_ai.agents.claude_code import ClaudeCodingAgent
    from flyto_ai.coding.contracts import (
        ACTION_SANDBOX_UNAVAILABLE as CONTRACT_CODE,
        ApprovalPolicy,
        CodingTaskRequest,
        SandboxMode,
    )
    from flyto_ai.coding.store import ThreadStore

    workspace = tmp_path / "ws"
    workspace.mkdir()
    _contract(workspace, """
        checks:
          - name: trivial
            argv: {argv}
            required: true
        actions:
          - name: regenerate
            argv: {argv}
            timeout_seconds: 30
        """.format(argv=json.dumps([sys.executable, "-c", "pass"])))

    provider_calls = []
    runs = []

    class NeverCalled:
        async def run(self, request):
            provider_calls.append(request)
            raise AssertionError("the provider must not be reached")

    _resolving(monkeypatch, "not-an-image-id\n")
    real_popen = subprocess.Popen

    def record(argv, *rest, **kwargs):
        runs.append(list(argv) if isinstance(argv, (list, tuple)) else [argv])
        return real_popen(argv, *rest, **kwargs)

    monkeypatch.setattr(subprocess, "Popen", record)

    result = asyncio.run(ClaudeCodingAgent(
        ThreadStore(str(tmp_path / "threads")), agent=NeverCalled(),
    ).run(CodingTaskRequest(
        message="regenerate please",
        working_dir=str(workspace),
        approval_policy=ApprovalPolicy.NEVER,
        sandbox_mode=SandboxMode.WORKSPACE_WRITE,
        authorized_config_sha256=config_digest(str(workspace)),
        command_sandbox_image=_IMAGE,
    )))

    assert result.ok is False
    assert result.failure_code == CONTRACT_CODE
    assert provider_calls == []
    assert [call for call in runs if "run" in call[1:2]] == []


# --------------------------------------------------------------------------
# cleanup is unconditional once a container exists
# --------------------------------------------------------------------------


_CONTAINER = "c" * 64


class _CidWritingProcess:
    """A launched `docker run` that has already produced a container id."""

    returncode = -9
    stdout = stderr = None
    pid = 5150
    raises = KeyboardInterrupt()

    def __init__(self, argv, **kwargs):
        Path(argv[argv.index("--cidfile") + 1]).write_text(
            _CONTAINER, encoding="utf-8",
        )

    def wait(self, timeout=None):
        raise type(self).raises


@pytest.mark.parametrize(
    "exception",
    [KeyboardInterrupt(), SystemExit(1), BaseException("cancelled")],
)
def test_cancellation_after_launch_kills_the_group_and_removes_the_container(
    tmp_path, sandboxed, monkeypatch, exception,
):
    """A container outliving its cancelled round keeps editing the worktree."""

    workspace = tmp_path / "ws"
    workspace.mkdir()
    _declare(workspace, "-c", "print('x')")
    executor = sandboxed(workspace)

    terminated = []
    removed = []

    class Process(_CidWritingProcess):
        raises = exception

    monkeypatch.setattr(subprocess, "Popen", Process)
    monkeypatch.setattr(
        ProjectActionExecutor, "_terminate_group",
        staticmethod(lambda process: terminated.append(process.pid)),
    )
    monkeypatch.setattr(
        subprocess, "run",
        lambda argv, **kwargs: removed.append(list(argv))
        or subprocess.CompletedProcess(argv, 0),
    )

    with pytest.raises(type(exception)):
        executor.run("regenerate")

    assert terminated == [5150], "the process group was not killed"
    assert removed, "the container was never removed"
    assert removed[0][1:] == ["rm", "-f", _CONTAINER]


def test_a_foreign_or_malformed_cidfile_never_reaches_docker_rm(
    tmp_path, sandboxed, monkeypatch,
):
    """Removing somebody else's container is worse than leaking one of ours."""

    workspace = tmp_path / "ws"
    workspace.mkdir()
    _declare(workspace, "-c", "print('x')")

    removed = []
    monkeypatch.setattr(
        subprocess, "run",
        lambda argv, **kwargs: removed.append(list(argv))
        or subprocess.CompletedProcess(argv, 0),
    )

    for planted in ("", "not-a-container", "../../etc/passwd", "c" * 8,
                    "C" * 64, "$(id)", "c" * 65):
        executor = sandboxed(workspace)
        runtime = Path(tempfile.mkdtemp())
        cidfile = runtime / "container.cid"
        cidfile.write_text(planted, encoding="utf-8")
        executor._cidfile = cidfile
        executor._runtime = runtime
        executor._reap_container()

    assert removed == [], removed


def test_reaping_is_idempotent(tmp_path, sandboxed, monkeypatch):
    workspace = tmp_path / "ws"
    workspace.mkdir()
    _declare(workspace, "-c", "print('x')")
    executor = sandboxed(workspace)

    removed = []
    monkeypatch.setattr(
        subprocess, "run",
        lambda argv, **kwargs: removed.append(list(argv))
        or subprocess.CompletedProcess(argv, 0),
    )

    runtime = Path(tempfile.mkdtemp())
    cidfile = runtime / "container.cid"
    cidfile.write_text(_CONTAINER, encoding="utf-8")
    executor._cidfile = cidfile
    executor._runtime = runtime

    for _ in range(3):
        executor._reap_container()

    assert len(removed) == 1, removed
    assert removed[0][1:] == ["rm", "-f", _CONTAINER]


def test_the_reap_is_inside_an_unconditional_finally():
    """Structural: the guarantee must not depend on which exception was raised."""

    import flyto_ai.coding.actions as actions

    source = Path(actions.__file__).read_text(encoding="utf-8")
    launch = source.split("def _launch(", 1)[1].split("\n    @staticmethod", 1)[0]
    after_popen = launch.split("subprocess.Popen(", 1)[1]
    assert "finally:" in after_popen
    tail = after_popen[after_popen.index("finally:"):]
    assert "self._reap_container()" in tail
