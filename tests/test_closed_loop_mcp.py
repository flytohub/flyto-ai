"""Real MCP protocol tests for the token-bounded closed-loop facade."""
from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

import pytest

from flyto_ai.closed_loop_mcp import (
    MCP_CONTRACT_VERSION,
    ClosedLoopMCPServer,
)


class StdioMCPClient:
    """Tiny real STDIO MCP client used to test the packaged server boundary."""

    def __init__(self, process: asyncio.subprocess.Process) -> None:
        self.process = process
        self._request_id = 0

    @classmethod
    async def start(
        cls,
        state_dir: Path,
        *,
        fail_once_module: str = "",
    ) -> "StdioMCPClient":
        project_root = Path(__file__).resolve().parents[1]
        workspace = project_root.parent
        paths = [
            str(project_root),
            str(workspace / "flyto-blueprint"),
            str(workspace / "flyto-core" / "src"),
        ]
        existing_path = os.environ.get("PYTHONPATH")
        if existing_path:
            paths.append(existing_path)
        env = {
            **os.environ,
            "PYTHONPATH": os.pathsep.join(paths),
            "FLYTO_CLOSED_LOOP_STATE_DIR": str(state_dir),
            "FLYTO_CLOSED_LOOP_PERMISSION": "workspace_write",
            "FLYTO_CLOSED_LOOP_MAX_REPAIRS": "1",
        }
        if fail_once_module:
            env["FLYTO_CLOSED_LOOP_MCP_FAIL_ONCE_MODULE"] = fail_once_module
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "flyto_ai.closed_loop_mcp",
            cwd=str(project_root),
            env=env,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        return cls(process)

    async def request(self, method: str, params=None):
        self._request_id += 1
        payload = {
            "jsonrpc": "2.0",
            "id": self._request_id,
            "method": method,
        }
        if params is not None:
            payload["params"] = params
        assert self.process.stdin is not None
        self.process.stdin.write((
            json.dumps(payload, ensure_ascii=False) + "\n"
        ).encode("utf-8"))
        await self.process.stdin.drain()
        assert self.process.stdout is not None
        line = await asyncio.wait_for(
            self.process.stdout.readline(),
            timeout=30,
        )
        if not line:
            assert self.process.stderr is not None
            stderr = (await self.process.stderr.read()).decode(
                "utf-8",
                errors="replace",
            )
            raise AssertionError("MCP server exited without response: {}".format(stderr))
        response = json.loads(line)
        assert response["id"] == self._request_id
        return response

    async def notify(self, method: str, params=None) -> None:
        payload = {"jsonrpc": "2.0", "method": method}
        if params is not None:
            payload["params"] = params
        assert self.process.stdin is not None
        self.process.stdin.write((
            json.dumps(payload, ensure_ascii=False) + "\n"
        ).encode("utf-8"))
        await self.process.stdin.drain()

    async def call(self, name: str, arguments: dict) -> dict:
        response = await self.request(
            "tools/call",
            {"name": name, "arguments": arguments},
        )
        assert "error" not in response
        return response["result"]

    async def close(self) -> None:
        if self.process.stdin is not None:
            self.process.stdin.close()
            await self.process.stdin.wait_closed()
        try:
            await asyncio.wait_for(self.process.wait(), timeout=10)
        except asyncio.TimeoutError:
            self.process.terminate()
            await self.process.wait()


@pytest.mark.asyncio
async def test_protocol_contract_is_compact_and_rejects_invalid_plan(tmp_path):
    server = ClosedLoopMCPServer(str(tmp_path))
    initialized = await server.handle({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {"protocolVersion": "2025-06-18"},
    })

    assert initialized["result"]["serverInfo"]["name"] == "flyto-closed-loop"
    assert len(initialized["result"]["instructions"]) < 512

    listed = await server.handle({
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/list",
    })
    assert [item["name"] for item in listed["result"]["tools"]] == [
        "plan",
        "execute",
        "verify",
        "get_evidence",
    ]

    invalid = await server.call_tool("plan", {
        "message": "invalid forward reference",
        "steps": [{
            "id": "first",
            "module": "string.uppercase",
            "params": {"text": "${later.result}"},
        }],
    })
    assert invalid["isError"] is True
    assert invalid["structuredContent"]["gate"]["pass"] is False


@pytest.mark.asyncio
async def test_verify_distinguishes_missing_unknown_and_unexecuted_plan(tmp_path):
    server = ClosedLoopMCPServer(str(tmp_path))

    missing = await server.call_tool("verify", {})
    assert missing["isError"] is True
    assert (
        missing["structuredContent"]["error"]
        == "execution_id or plan_id is required"
    )

    unknown = await server.call_tool("verify", {"plan_id": "missing-plan"})
    assert unknown["isError"] is True
    assert unknown["structuredContent"]["error"] == "Unknown plan_id"

    planned = await server.call_tool("plan", {
        "message": "valid unexecuted plan",
        "steps": [{
            "id": "upper",
            "module": "string.uppercase",
            "params": {"text": "Flyto"},
        }],
    })
    plan_id = planned["structuredContent"]["plan_id"]
    unexecuted = await server.call_tool("verify", {"plan_id": plan_id})
    assert unexecuted["isError"] is True
    assert (
        unexecuted["structuredContent"]["error"]
        == "Plan has no execution evidence"
    )


@pytest.mark.asyncio
async def test_real_stdio_mcp_executes_verifies_and_saves_tokens(tmp_path):
    client = await StdioMCPClient.start(
        tmp_path,
        fail_once_module="string.reverse",
    )
    try:
        initialized = await client.request(
            "initialize",
            {
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "flyto-test", "version": "1"},
            },
        )
        assert initialized["result"]["protocolVersion"] == "2025-06-18"
        await client.notify("notifications/initialized")

        listed = await client.request("tools/list")
        assert len(listed["result"]["tools"]) == 4

        planned = await client.call("plan", {
            "message": "MCP verified text workflow",
            "steps": [
                {
                    "id": "upper",
                    "module": "string.uppercase",
                    "params": {"text": "Flyto"},
                    "assertions": {
                        "path": "data.result",
                        "op": "equals",
                        "value": "FLYTO",
                    },
                },
                {
                    "id": "lower",
                    "module": "string.lowercase",
                    "params": {"text": "${upper.result}"},
                    "assertions": {
                        "path": "data.result",
                        "op": "equals",
                        "value": "flyto",
                    },
                },
                {
                    "id": "upper_again",
                    "module": "string.uppercase",
                    "params": {"text": "${lower.result}"},
                    "assertions": {
                        "path": "data.result",
                        "op": "equals",
                        "value": "FLYTO",
                    },
                },
            ],
        })
        plan = planned["structuredContent"]
        assert plan["ok"] is True
        assert plan["plan_ir_version"] == "flyto.plan-ir.v1"
        assert plan["model_route"]["mode"] == "deterministic"

        executed = await client.call(
            "execute",
            {"plan_id": plan["plan_id"]},
        )
        execution = executed["structuredContent"]
        assert execution["closed_loop_ok"] is True
        assert execution["passed_steps"] == 3
        assert execution["assertion_passed"] is True
        assert execution["outcome_reported"] is True
        assert execution["checkpoint_cleared"] is True
        profile = execution["token_profile"]
        assert profile["full_evidence_chars"] > profile["compact_chars"]
        assert profile["reduction_percent"] > 0
        assert len(executed["content"][0]["text"]) < profile["full_evidence_chars"]

        verified = await client.call(
            "verify",
            {"execution_id": execution["execution_id"], "min_steps": 3},
        )
        verification = verified["structuredContent"]
        assert verification["verified"] is True
        assert all(verification["checks"].values())
        assert verification["distillation"]["eligible"] is True
        assert verification["distillation"]["score"] == 70
        assert verification["distillation"]["blueprint_id"]

        raw = await client.call("get_evidence", {
            "execution_id": execution["execution_id"],
            "section": "raw",
            "offset": 0,
            "limit": 1000,
        })
        raw_content = raw["structuredContent"]
        assert raw_content["ok"] is True
        assert len(raw_content["chunk"]) <= 1000
        assert raw_content["total_chars"] == profile["full_evidence_chars"]

        resume_plan_result = await client.call("plan", {
            "message": "MCP checkpoint resume workflow",
            "steps": [
                {
                    "id": "resume_upper",
                    "module": "string.uppercase",
                    "params": {"text": "Flyto"},
                },
                {
                    "id": "resume_reverse",
                    "module": "string.reverse",
                    "params": {"text": "${resume_upper.result}"},
                    "assertions": {
                        "path": "data.result",
                        "op": "equals",
                        "value": "OTYLF",
                    },
                },
            ],
        })
        resume_plan = resume_plan_result["structuredContent"]

        interrupted = await client.call("execute", {
            "plan_id": resume_plan["plan_id"],
            "max_repairs": 0,
        })
        interrupted_summary = interrupted["structuredContent"]
        assert interrupted_summary["closed_loop_ok"] is False
        assert interrupted_summary["failed_step_id"] == "resume_reverse"

        resumed = await client.call("execute", {
            "plan_id": resume_plan["plan_id"],
            "max_repairs": 0,
        })
        resumed_summary = resumed["structuredContent"]
        assert resumed_summary["closed_loop_ok"] is True
        assert resumed_summary["resumed_step_ids"] == ["resume_upper"]
        assert resumed_summary["module_call_counts"] == {
            "string.uppercase": 1,
            "string.reverse": 2,
        }
        assert resumed_summary["checkpoint_cleared"] is True

        resumed_evidence = await client.call("get_evidence", {
            "execution_id": resumed_summary["execution_id"],
            "section": "executions",
            "limit": 10,
        })
        executions = resumed_evidence["structuredContent"]["executions"]
        assert executions[0]["step_id"] == "resume_upper"
        assert executions[0]["phase"] == "resume"
        assert executions[0]["executed"] is False

        record_files = list((tmp_path / "records").glob("*.json"))
        assert record_files
        assert all(path.stat().st_mode & 0o777 == 0o600 for path in record_files)
    finally:
        await client.close()


def test_contract_version_is_stable():
    assert MCP_CONTRACT_VERSION == "flyto.closed-loop-mcp.v1"
