"""Permission checks must distinguish workspace files from unbounded filesystem access."""
import pytest

from flyto_ai.permissions import PermissionEnforcer, PermissionLevel
from test_execution_continuation import make_agent


def call(module_id, path):
    return {"module_id": module_id, "params": {"path": str(path)}}


@pytest.mark.parametrize("module_id", ["file.read", "file.write"])
@pytest.mark.parametrize("absolute", [False, True])
def test_workspace_paths_are_admitted_without_elevating_other_operations(tmp_path, monkeypatch, module_id, absolute):
    monkeypatch.chdir(tmp_path)
    enforcer = PermissionEnforcer(PermissionLevel.WORKSPACE_WRITE)
    path = tmp_path / "output" / "report.txt" if absolute else "output/report.txt"
    assert enforcer.check("execute_module", call(module_id, path)).allowed
    assert enforcer.level == PermissionLevel.WORKSPACE_WRITE
    assert not enforcer.check("execute_module", call("file.delete", path)).allowed
    assert not enforcer.check("execute_module", {"module_id": "env.get", "params": {}}).allowed


@pytest.mark.parametrize("path", ["../outside.txt", "/etc/passwd", "${params.path}", "{{path}}", "${{env.HOME}}/file", "", "\x00"])
def test_unbounded_and_unresolved_paths_stay_restricted(tmp_path, monkeypatch, path):
    monkeypatch.chdir(tmp_path)
    enforcer = PermissionEnforcer()
    assert not enforcer.check("execute_module", call("file.write", path)).allowed


@pytest.mark.parametrize("kind", ["leaf", "parent", "dangling"])
def test_symlink_escape_is_not_a_workspace_operation(tmp_path, monkeypatch, kind):
    workspace, outside = tmp_path / "work", tmp_path / "outside"
    workspace.mkdir(); outside.mkdir()
    monkeypatch.chdir(workspace)
    enforcer = PermissionEnforcer()
    if kind == "leaf":
        (outside / "data.txt").write_text("unchanged")
        (workspace / "link.txt").symlink_to(outside / "data.txt")
        path = workspace / "link.txt"
    else:
        (workspace / "link").symlink_to(outside if kind == "parent" else outside / "missing")
        path = workspace / "link" / "data.txt"
    assert not enforcer.check("execute_module", call("file.write", path)).allowed
    assert not (outside / "missing").exists()


def test_current_working_directory_cannot_expand_a_captured_workspace(tmp_path, monkeypatch):
    workspace, outside = tmp_path / "work", tmp_path / "outside"
    workspace.mkdir(); outside.mkdir()
    monkeypatch.chdir(workspace)
    enforcer = PermissionEnforcer()
    monkeypatch.chdir(outside)
    assert not enforcer.check("execute_module", call("file.write", "report.txt")).allowed
    assert enforcer.check("execute_module", call("file.write", workspace / "report.txt")).allowed


@pytest.mark.parametrize("module_id", ["file.copy", "file.delete", "file.edit", "path.join", "shell.run", "env.get"])
def test_unreviewed_operations_keep_existing_danger_requirement(tmp_path, monkeypatch, module_id):
    monkeypatch.chdir(tmp_path)
    assert PermissionEnforcer().required_level("execute_module", call(module_id, "x")) == PermissionLevel.DANGER_FULL


@pytest.mark.asyncio
async def test_changed_workspace_invalidates_continuation(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    agent, _, _ = make_agent()
    goal = "Read the local file."
    await agent.chat(goal)
    other = tmp_path / "other"
    other.mkdir()
    monkeypatch.chdir(other)
    agent._permission_enforcer = PermissionEnforcer(PermissionLevel.WORKSPACE_WRITE)
    with pytest.raises(PermissionError, match="policy changed"):
        await agent.continue_execution(message="Observe the remaining result.", goal=goal)


@pytest.mark.asyncio
@pytest.mark.parametrize("read_only", [False, True])
async def test_real_core_read_write_reread_still_passes_through_agent_policy(tmp_path, monkeypatch, read_only):
    from flyto_ai.tools.core_tools import dispatch_core_tool
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("FLYTO_SANDBOX_DIR", str(tmp_path))
    source, output = tmp_path / "source.txt", tmp_path / "output.txt"
    source.write_text("status: pending\nMarker: fixture\n")
    agent, _, _ = make_agent()
    if read_only:
        agent._permission_enforcer = PermissionEnforcer(PermissionLevel.READ_ONLY)
    agent._dispatch_fn = dispatch_core_tool
    observed = []

    class Provider:
        async def chat(self, messages, system_prompt, tools, dispatch_fn, max_rounds=30, **kwargs):
            if "execute_module" not in [tool.get("name") for tool in tools]:
                return "No action exposed.", [], 1, {}
            first = await dispatch_fn("execute_module", call("file.read", source))
            observed.append(first)
            if first.get("ok") is False:
                return "Read refused.", [], 1, {}
            content = first["data"]["content"].replace("pending", "reviewed")
            write_args = call("file.write", output)
            write_args["params"]["content"] = content
            observed.append(await dispatch_fn("execute_module", write_args))
            observed.append(await dispatch_fn("execute_module", call("file.read", output)))
            return "Observed files.", [], 3, {}

    agent._provider = Provider()
    await agent.chat("請使用這台電腦的工具，讀取檔案、替換內容、存成新檔，再讀取確認。")
    if read_only:
        assert not observed and not output.exists()
    else:
        assert len(observed) == 3
        assert observed[-1]["data"]["content"] == "status: reviewed\nMarker: fixture\n"
        assert output.read_text() == observed[-1]["data"]["content"]
    assert source.read_text() == "status: pending\nMarker: fixture\n"


@pytest.mark.asyncio
async def test_core_environment_can_narrow_workspace_access(tmp_path, monkeypatch):
    from flyto_ai.tools.core_tools import dispatch_core_tool
    monkeypatch.chdir(tmp_path)
    sandbox = tmp_path / "narrower"
    sandbox.mkdir()
    monkeypatch.setenv("FLYTO_SANDBOX_DIR", str(sandbox))
    enforcer = PermissionEnforcer()
    args = call("file.write", tmp_path / "outside-core.txt")
    args["params"]["content"] = "must not write"
    assert enforcer.check("execute_module", args).allowed
    result = await dispatch_core_tool("execute_module", args)
    assert result.get("ok") is False
    assert not (tmp_path / "outside-core.txt").exists()


@pytest.mark.asyncio
async def test_core_rechecks_a_symlink_changed_after_permission_admission(tmp_path, monkeypatch):
    from flyto_ai.tools.core_tools import dispatch_core_tool
    workspace, outside = tmp_path / "work", tmp_path / "outside"
    workspace.mkdir(); outside.mkdir()
    monkeypatch.chdir(workspace)
    monkeypatch.setenv("FLYTO_SANDBOX_DIR", str(workspace))
    (workspace / "target.txt").write_text("original")
    link = workspace / "link.txt"
    link.symlink_to(workspace / "target.txt")
    args = call("file.write", link)
    args["params"]["content"] = "must not escape"
    assert PermissionEnforcer().check("execute_module", args).allowed
    link.unlink()
    link.symlink_to(outside / "target.txt")
    result = await dispatch_core_tool("execute_module", args)
    assert result.get("ok") is False
    assert not (outside / "target.txt").exists()
    assert (workspace / "target.txt").read_text() == "original"
