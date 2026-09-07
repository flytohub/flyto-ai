"""Local-computer imperatives retain normal admission and permission ceilings."""

import pytest

from flyto_ai.intelligence.planner import classify_tool_intent
from flyto_ai.permissions import PermissionEnforcer, PermissionLevel
from test_execution_continuation import make_agent

FILE_GOAL = (
    "Use this computer to read output/ai-space-acceptance/cli-c48b5c8e7d-input.txt. "
    "Change only Invoice status: pending to Invoice status: reviewed, preserve every other byte, "
    "and save to output/ai-space-acceptance/cli-c48b5c8e7d-result.txt. "
    "Read the saved result again to verify it. Report the Marker you actually read and the output path. "
    "Complete the work with the available local tools."
)


@pytest.mark.parametrize("goal", [
    FILE_GOAL,
    "Please use this computer to read report.txt, modify its status and save the result.",
    "Could you use my computer to write output/report.txt and read it again?",
    "Use the available local tools to read input.txt and write result.txt.",
    "請讀取本機檔案 report.txt，修改 Invoice status 後另存 result.txt，再讀取驗證。",
])
def test_explicit_local_computer_actions_reach_action_admission(goal):
    decision = classify_tool_intent(goal)
    assert decision.mode == "action" and decision.tool_eligible


@pytest.mark.parametrize("goal", [
    "Use this computer to explain how to read and save a file.",
    "How do I use this computer to read a file?",
    "Do not use this computer to write any file.",
    "Use this computer to not write any file.",
    "Use this computer to read a file, but do not use tools.",
    "Log: Use this computer to read input.txt and write result.txt.",
    'The error message says "Use this computer to read a file".',
    'Explain the sentence "Use this computer to read a file".',
    '"Use this computer to read a file"',
    "Use this computer to explain whether writing files is safe.",
    "Use this computer to",
    "請說明如何讀取本機檔案、修改、另存與驗證。",
    "請不要讀取本機檔案，只說明修改與另存步驟。",
])
def test_explanation_quotation_negation_or_preface_alone_never_grants_actions(goal):
    assert not classify_tool_intent(goal).tool_eligible


@pytest.mark.asyncio
@pytest.mark.parametrize("restricted", [False, True])
async def test_admitted_real_wording_never_overrides_permission_ceiling(restricted):
    agent, provider, calls = make_agent()
    if restricted:
        agent._permission_enforcer = PermissionEnforcer(PermissionLevel.READ_ONLY)
    try:
        await agent.chat(FILE_GOAL)
        assert agent._last_routing_decision.mode == "action"
        assert ("execute_module" in provider.visible[0]) is not restricted
        assert bool(calls) is not restricted
    finally:
        await agent.close()
