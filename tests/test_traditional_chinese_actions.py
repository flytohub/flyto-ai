"""Local instructions must expose execution without promoting discussion to actions."""
import pytest

from flyto_ai.intelligence.planner import classify_tool_intent
from flyto_ai.permissions import PermissionEnforcer, PermissionLevel
from test_execution_continuation import make_agent

FILE_GOAL = (
    "請使用這台電腦的工具，讀取 runtime 工作目錄中的 output/source.txt。"
    "將其中的 Invoice status: pending 改成 Invoice status: reviewed，"
    "其餘內容完整保留，存成 output/reviewed.txt。"
    "完成後重新讀取輸出檔確認內容，回報實際讀到的 Marker 和輸出檔案位置。"
    "不要只描述操作步驟，請實際完成。"
)


@pytest.mark.parametrize("goal", [
    FILE_GOAL,
    "請幫我讀取 output/source.txt，修改內容後另存成 output/result.txt。",
    "麻煩你用本機工具讀取 report.txt 並儲存摘要。",
    "請在這台電腦上，讀取 report.txt 後替換指定文字。",
    "可以幫我使用檔案工具，讀取 source.txt 再寫入 result.txt 嗎？",
    "能不能用這台電腦的工具，讀取資料並存成新檔？",
    "透過本機的工具，將報表改寫成文字檔。",
    "請使用 flyto-core 工具，讀取 invoice.txt 並存成 reviewed.txt。",
    "请使用这台电脑的工具，读取 source.txt 并保存结果。",
    "請把 report.txt 的 pending 改成 reviewed。",
    "請另存成 output/report.txt。",
])
def test_explicit_computer_actions_are_admitted(goal):
    decision = classify_tool_intent(goal)
    assert decision.mode == "action"
    assert decision.tool_eligible


@pytest.mark.parametrize("goal", [
    "請使用這台電腦的工具，說明如何讀取與儲存檔案。",
    "請使用工具，解釋讀取檔案的流程。",
    "請使用本機工具，不要修改任何檔案。",
    "請不要使用工具，讀取檔案的原理是什麼？",
    "不要實際讀取，只說明步驟。",
    "請幫我不要寫入任何檔案。",
    "如果我說『請使用工具讀取檔案』，系統應該如何判斷？",
    "『請使用本機工具，讀取 report.txt』這句話是什麼意思？",
    "使用本機工具讀取檔案是否安全？",
    "在這台電腦上，讀取檔案會不會改變內容？",
    "請使用工具，『讀取 source.txt』這句話的意思是什麼？",
    "請使用工具，檔案目前是什麼狀態？",
    "這台電腦可以使用工具讀取檔案。",
    "請使用教學中的工具說明，讀取檔案是什麼意思？",
])
def test_explanations_negations_and_quotations_never_admit_execution(goal):
    assert not classify_tool_intent(goal).tool_eligible


@pytest.mark.asyncio
@pytest.mark.parametrize("restricted", [False, True])
async def test_natural_file_goal_reaches_agent_tool_policy_without_override(restricted):
    agent, provider, calls = make_agent()
    if restricted:
        agent._permission_enforcer = PermissionEnforcer(PermissionLevel.READ_ONLY)
    await agent.chat(FILE_GOAL)
    assert agent._last_routing_decision.mode == "action"
    assert ("execute_module" in provider.visible[0]) is not restricted
    assert bool(calls) is not restricted
