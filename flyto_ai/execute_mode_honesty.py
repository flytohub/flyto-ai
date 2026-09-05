# Copyright 2024 Flyto2
# Licensed under the Apache License, Version 2.0
"""In execute mode the assistant runs what it says it runs.

What the owner met (AI Space chat, provider openai / gpt-4o): the user typed
幫我登入kintone and the reply was 「我將執行 "kintone" 工作流程來幫助您登入。
請稍候。執行中...」. The audit for that turn recorded tool_calls_count=0. The
chat showed progress for a run that never began, and the workflow only ran
after the user typed 執行啊 and the next turn happened to call the tool.

Two things let that through. The provider forces a tool choice only for
messages its browser-task word list recognises, and a Space's own workflow
names are on no list, so tool_choice stayed "auto". And the discovery nudge --
a second completion asking the model to act if the task needs it -- is a
request the model is free to ignore, which it did.

So two passes stand after the first completion, in this order:

* ``nudge_discovery_only_reply`` is the request. A reply that only looked
  things up (search, list) is asked once to act; the retry is kept only if it
  ran something, and a Space's own workflows count as something -- without
  that, a nudge that made the model run ``kintone`` was thrown away after it ran.
* ``guard_narrated_execution`` is the rule. When the turn has a tool it could
  run, called none, and the reply reads as a commitment (執行中, 請稍候, "I'll
  run"), retry once with the tool choice forced where the provider can; if the
  model still narrates, or the provider cannot force, replace the reply with a
  plain statement that nothing ran and what could be run.

These are plain functions rather than ``Agent`` methods. ``agent.py`` is over
its recorded line budget (``tests/test_complexity_budget.py``) and that number
only moves down, and neither pass needs anything of the agent beyond a way to
make one more completion, which arrives as a callable. The agent keeps a
call site of a few lines; the behaviour is pinned by
``tests/test_execute_mode_honesty.py``.
"""
from __future__ import annotations

import logging
from typing import Awaitable, Callable, Dict, List, Optional, Set, Tuple

from flyto_ai.permissions import TOOL_PERMISSION_MAP, PermissionEnforcer, PermissionLevel

logger = logging.getLogger(__name__)

# What one provider call comes back as: reply text, the tool-call log, the
# number of rounds it took, and its token usage.
Completion = Tuple[str, List[Dict], int, Dict[str, int]]

# Calls that count as the model having done something, for the nudge.
_ACTION_TOOLS = frozenset({"execute_module", "use_blueprint", "navigate_website", "ask_user"})
# Calls that count as the model having run something, for keeping the retry.
_EXECUTION_TOOLS = frozenset({"execute_module", "use_blueprint", "ask_user"})

_NUDGE = (
    "If this task requires you to actually DO something (go to a website, "
    "execute a module, automate an action), use the tools available. "
    "If this is just a knowledge question, answer as you did."
)

# Phrases a model uses when it presents a run as under way or about to start.
# Checked only in execute mode, only against a reply that called nothing that
# runs. The owner's turn read 「我將執行 "kintone" 工作流程來幫助您登入。請稍候。
# 執行中...」 with tool_calls_count=0; the chat showed progress for a run that
# never began until the user typed 執行啊.
_COMMITMENT_PHRASES = (
    "執行中", "执行中", "處理中", "处理中", "請稍候", "请稍候", "請稍等", "请稍等",
    "稍等", "我將", "我将", "我會執行", "我会执行", "為您執行", "为您执行",
    "正在執行", "正在执行", "正在為", "正在为", "馬上", "马上", "立即執行",
    "立即执行", "開始執行", "开始执行",
    "実行します", "実行中", "お待ちください",
    "실행하겠습니다", "실행 중", "잠시만",
    "i will ", "i'll ", "i am going to", "i'm going to", "let me ", "executing",
    "is running", "now running", "running the", "please wait", "one moment",
    "hold on", "in progress", "kicking off", "starting the",
)


def reads_as_commitment(text: str) -> bool:
    lower = text.lower()
    return any(phrase in lower for phrase in _COMMITMENT_PHRASES)


def nothing_ran_message(language: str, workflow_names: List[str]) -> str:
    """The reply for a turn that promised a run and made no call.

    Must not itself contain any of the phrases above -- the whole point is
    that no fake progress word reaches the user.
    """
    if language.startswith("Traditional Chinese"):
        if workflow_names:
            return "我沒有執行任何工作流程。這裡可以執行的工作流程：{}。要我現在執行它嗎？".format(
                "、".join(workflow_names),
            )
        return "我沒有執行任何操作，也沒有任何工作流程在跑。請告訴我要執行哪一個。"
    if language.startswith("Simplified Chinese"):
        if workflow_names:
            return "我没有执行任何工作流程。这里可以执行的工作流程：{}。要我现在执行它吗？".format(
                "、".join(workflow_names),
            )
        return "我没有执行任何操作，也没有任何工作流程在跑。请告诉我要执行哪一个。"
    if language.startswith("Japanese"):
        if workflow_names:
            return "ワークフローはまだ実行していません。ここで実行できるワークフロー：{}。今すぐ実行しますか？".format(
                "、".join(workflow_names),
            )
        return "まだ何も実行していません。どのワークフローを実行するか教えてください。"
    if workflow_names:
        return "Nothing was run. The workflow available here is {}. Ask me to run it and it gets called as a tool.".format(
            ", ".join(workflow_names),
        )
    return "Nothing was run: no module or workflow was executed this turn. Tell me exactly what to run."


def runnable_tool_names(
    active_tools: Optional[List[Dict]],
    enforcer: PermissionEnforcer,
    tool_name: Callable[[Dict], str],
) -> Tuple[Set[str], List[str]]:
    """Names this turn could actually run, and which of them are the Space's
    own workflows (registered by the caller, unknown to the static permission
    map). Read-only tools only look things up."""
    runnable: Set[str] = set()
    workflows: List[str] = []
    for tool in active_tools or []:
        name = tool_name(tool)
        if not name:
            continue
        if enforcer.required_level(name, {}) <= PermissionLevel.READ_ONLY:
            continue
        runnable.add(name)
        if name not in TOOL_PERMISSION_MAP:
            workflows.append(name)
    return runnable, workflows


def _add_usage(total_usage: Dict[str, int], retry_usage: Dict[str, int]) -> None:
    for k in total_usage:
        total_usage[k] += retry_usage.get(k, 0)


async def nudge_discovery_only_reply(
    completion: Completion,
    workflow_names: List[str],
    messages: List[Dict],
    complete: Callable[[List[Dict]], Awaitable[Completion]],
) -> Completion:
    """Ask a reply that only searched or listed to act, once.

    The retry is kept only if it actually ran something; a reply that just
    searched again is discarded. A failed retry leaves the turn as it was.
    """
    response_content, tool_calls, total_rounds, total_usage = completion
    if any(tc.get("function") in _ACTION_TOOLS or tc.get("function") in workflow_names
           for tc in tool_calls):
        return completion
    if not response_content or total_rounds > 1:
        return completion
    try:
        nudge_messages = messages + [
            {"role": "assistant", "content": response_content},
            {"role": "user", "content": _NUDGE},
        ]
        retry_content, retry_tc, retry_rounds, retry_usage = await complete(nudge_messages)
        has_execution = any(
            tc.get("function") in _EXECUTION_TOOLS or tc.get("function") in workflow_names
            for tc in retry_tc
        )
        if has_execution:
            logger.info("Nudge accepted: LLM used execution tools")
            _add_usage(total_usage, retry_usage)
            return retry_content, retry_tc, total_rounds + retry_rounds, total_usage
    except Exception:
        pass
    return completion


async def guard_narrated_execution(
    completion: Completion,
    runnable_names: Set[str],
    workflow_names: List[str],
    *,
    can_force: bool,
    force_completion: Callable[[], Awaitable[Completion]],
    language: Callable[[], str],
) -> Completion:
    """Never let a reply present a run that this turn did not call.

    ``can_force`` is the provider's ``supports_forced_tool_choice``;
    ``force_completion`` makes one more completion with the tool choice
    forced; ``language`` names the operator's language, asked only when the
    honest reply is actually needed.
    """
    response_content, tool_calls, total_rounds, total_usage = completion
    if not runnable_names or not response_content:
        return completion
    if any(tc.get("function") in runnable_names for tc in tool_calls):
        return completion
    if not reads_as_commitment(response_content):
        return completion

    if can_force:
        retry_content, retry_tc, retry_rounds, retry_usage = await force_completion()
        total_rounds += retry_rounds
        _add_usage(total_usage, retry_usage)
        if any(tc.get("function") in runnable_names for tc in retry_tc):
            logger.info(
                "Forced tool choice accepted: model called %s after narrating",
                [tc.get("function") for tc in retry_tc],
            )
            return retry_content or "", retry_tc, total_rounds, total_usage
        if retry_tc:
            # Lookups the forced round did make really happened; keep
            # them in the log even though nothing ran.
            tool_calls = retry_tc

    logger.warning(
        "Execute-mode reply narrated a run without calling a tool; "
        "replaced with a statement that nothing ran (runnable: %s)",
        sorted(runnable_names),
    )
    return nothing_ran_message(language(), workflow_names), tool_calls, total_rounds, total_usage
