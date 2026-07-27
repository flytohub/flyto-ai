"""Evidence tests for conversation routing, Blueprint trust, and MCP guards."""

import math
import random
import string

import pytest

from flyto_ai.agent import Agent
from flyto_ai.config import AgentConfig
from flyto_ai.intelligence.planner import (
    ToolIntentDecision,
    _match_from_blueprint,
    blueprint_is_trusted,
    classify_tool_intent,
    extract_intent,
)
from flyto_ai.permissions import (
    PermissionEnforcer,
    PermissionLevel,
    PermissionOutcome,
)


def _routing_agent(dispatch, permission_level="workspace_write"):
    agent = Agent.__new__(Agent)
    agent._config = AgentConfig(
        provider="ollama",
        enable_injection_detection=False,
        enable_memory=False,
        enable_pro=False,
        enable_transcript=False,
        permission_level=permission_level,
    )
    agent._provider = object()
    agent._dispatch_fn = dispatch
    agent._policies = None
    agent._permission_enforcer = PermissionEnforcer(
        PermissionLevel[permission_level.upper()],
    )
    agent._assistant = None
    agent._hooks = None
    agent._pro = None
    agent._tools = [
        {"name": "search_modules"},
        {"name": "get_module_info"},
        {"name": "execute_module"},
        {"name": "use_blueprint"},
    ]
    agent._cost_tracker = None
    agent._session_id = "routing-test"
    return agent


def test_normal_conversation_false_tool_activation_rate_is_zero():
    normal_turns = [
        "Can you explain what GitHub is?",
        "What are the pros and cons of MCP?",
        "Hello, nice to meet you.",
        "GitHub 是什麼？",
        "為什麼需要 Blueprint？",
        "這個方法會不會太硬？",
        "GitHub 是什么？",
        "为什么需要工具？",
        "这个设计安全吗？",
        "GitHubとは何ですか？",
        "なぜBlueprintが必要ですか？",
        "現在呢？",
    ]

    decisions = [classify_tool_intent(turn) for turn in normal_turns]
    false_activations = sum(d.tool_eligible for d in decisions)

    assert false_activations == 0
    assert false_activations / len(normal_turns) == 0.0
    assert all(d.mode == "answer_only" for d in decisions)


def test_explicit_action_recall_across_languages_is_complete():
    action_turns = [
        "Please open https://github.com.",
        "Can you search for Flyto2 releases?",
        "Run image.resize width=800.",
        "請打開 https://github.com",
        "幫我搜尋 Flyto2",
        "修復這個錯誤",
        "请打开 GitHub",
        "帮我执行这个工作流",
        "GitHubを開いて",
        "Blueprintを実行して",
    ]

    decisions = [classify_tool_intent(turn) for turn in action_turns]

    assert all(d.mode == "action" for d in decisions)
    assert sum(d.tool_eligible for d in decisions) / len(action_turns) == 1.0


def test_current_data_questions_are_read_only_ambiguous():
    for turn in [
        "What is the weather today?",
        "Latest Bitcoin price?",
        "今天台北天氣？",
        "最新比分？",
    ]:
        decision = classify_tool_intent(turn)
        assert decision.mode == "ambiguous"
        assert decision.tool_eligible is False


_DIVERSE_ROUTING_CORPUS = [
    # Conversation, explanation, quotation, and negation must never expose tools.
    ("answer_only", "如果要打開 GitHub，流程是什麼？"),
    ("answer_only", "https://example.com 這個網址格式正確嗎？"),
    ("answer_only", "Is https://example.com a valid URL?"),
    ("answer_only", "不要打開 GitHub"),
    ("answer_only", "不要執行這個工作流"),
    ("answer_only", "先別刪除任何檔案"),
    ("answer_only", "不要搜尋網路"),
    ("answer_only", "請不要修改 README"),
    ("answer_only", "Don’t open GitHub."),
    ("answer_only", "Do not run this workflow."),
    ("answer_only", "Please don’t delete any files."),
    ("answer_only", "Do not search the web."),
    ("answer_only", "Can you not modify the README?"),
    ("answer_only", "Never execute this command."),
    ("answer_only", "Without opening the browser, explain the URL."),
    ("answer_only", "GitHubを開かないでください。"),
    ("answer_only", "このコマンドを実行しないでください。"),
    ("answer_only", "GitHub를 열지 마세요."),
    ("answer_only", "이 명령을 실행하지 마세요."),
    ("answer_only", "僅用你的知識回答，不要執行"),
    ("answer_only", "「請刪除所有檔案」這句話是祈使句嗎？"),
    ("answer_only", "如果我說「打開 GitHub」，系統應該怎麼判斷？"),
    ("answer_only", "Explain the sentence “Delete all files.”"),
    ("answer_only", "If a user says “run the tests,” what should the router do?"),
    ("answer_only", "The error says: execute module failed. What does it mean?"),
    ("answer_only", "My README contains “open GitHub”; is that dangerous?"),
    ("answer_only", "Could normal conversation accidentally trigger MCP?"),
    ("answer_only", "Blueprint가 무엇인가요?"),
    ("answer_only", "Pourquoi utiliser MCP ?"),
    ("answer_only", "Was ist Blueprint?"),
    ("answer_only", "Open source is a development model."),
    ("answer_only", "Search algorithms are useful."),
    ("answer_only", "Tell me about GitHub."),
    ("answer_only", "建立本地索引有什麼好處？"),
    ("answer_only", "刪除暫存檔會發生什麼？"),
    ("answer_only", "Não abra GitHub."),
    ("answer_only", "Non aprire GitHub."),
    ("answer_only", "Не открывай GitHub."),
    ("answer_only", "لا تفتح GitHub."),
    # Explicit action recall across eight language families.
    ("action", "請執行測試"),
    ("action", "把 README 改得更白話"),
    ("action", "建立一個新分支"),
    ("action", "請刪除暫存檔"),
    ("action", "請抓取這個頁面的標題"),
    ("action", "到 GitHub 找三個相關專案"),
    ("action", "請檢查目前 git 狀態"),
    ("action", "把結果寫進 report.md"),
    ("action", "請安裝相依套件"),
    ("action", "請提交這些修改"),
    ("action", "搜尋最新的官方文件"),
    ("action", "請重新執行失敗的測試"),
    ("action", "建立本地索引"),
    ("action", "把這段程式碼套用到 agent.py"),
    ("action", "幫我讀取 package.json"),
    ("action", "Rewrite the README in plain English."),
    ("action", "Fetch the title of this page."),
    ("action", "Find three related projects on GitHub."),
    ("action", "Check the current git status."),
    ("action", "Write the results to report.md."),
    ("action", "Install the dependencies."),
    ("action", "Commit these changes."),
    ("action", "Rerun the failed tests."),
    ("action", "Build a local index."),
    ("action", "Apply this patch to agent.py."),
    ("action", "Read package.json and summarize it."),
    ("action", "GitHubを開いてください。"),
    ("action", "テストを実行してください。"),
    ("action", "READMEを書き直してください。"),
    ("action", "このバグを修正してください。"),
    ("action", "最新の公式文書を検索してください。"),
    ("action", "結果をreport.mdに保存してください。"),
    ("action", "GitHub를 열어 주세요."),
    ("action", "테스트를 실행해 주세요."),
    ("action", "README를 다시 작성해 주세요."),
    ("action", "이 버그를 수정해 주세요."),
    ("action", "최신 공식 문서를 검색해 주세요."),
    ("action", "결과를 report.md에 저장해 주세요."),
    ("action", "Abre GitHub, por favor."),
    ("action", "Ejecuta las pruebas."),
    ("action", "Corrige este error."),
    ("action", "Busca la documentación oficial más reciente."),
    ("action", "Ouvre GitHub, s’il te plaît."),
    ("action", "Exécute les tests."),
    ("action", "Corrige ce bug."),
    ("action", "Cherche la documentation officielle récente."),
    ("action", "Öffne GitHub."),
    ("action", "Führe die Tests aus."),
    ("action", "Behebe diesen Fehler."),
    ("action", "Suche die neueste offizielle Dokumentation."),
    ("action", "Scrape https://example.com"),
    ("action", "Summarize the open browser tab."),
    ("action", "Analyze the repository."),
    ("action", "Inspect the current page."),
    ("action", "List the files in this directory."),
    ("action", "Tell GitHub to create an issue."),
    ("action", "Take a screenshot."),
    ("action", "Upload the report."),
    ("action", "下載這個檔案"),
    ("action", "列出這個資料夾裡的檔案"),
    ("action", "分析目前 repository"),
    ("action", "截一張圖"),
    ("action", "Abra GitHub."),
    ("action", "Apri GitHub."),
    ("action", "Открой GitHub."),
    ("action", "افتح GitHub."),
    # Current information may use read-only discovery, never write tools.
    ("ambiguous", "目前 GitHub 正常嗎？"),
    ("ambiguous", "現在幾點？"),
    ("ambiguous", "這週末台北會下雨嗎？"),
    ("ambiguous", "目前 OpenAI 的 CEO 是誰？"),
    ("ambiguous", "Is GitHub currently operational?"),
    ("ambiguous", "What time is it now?"),
    ("ambiguous", "Will it rain in Taipei this weekend?"),
    ("ambiguous", "Who is the current CEO of OpenAI?"),
    ("ambiguous", "東京の現在の天気は？"),
    ("ambiguous", "今日の為替レートは？"),
    ("ambiguous", "서울의 현재 날씨는 어때요?"),
    ("ambiguous", "오늘 환율은 얼마인가요?"),
    ("ambiguous", "¿Qué tiempo hace ahora en Madrid?"),
    ("ambiguous", "Quel temps fait-il maintenant à Paris ?"),
    ("ambiguous", "Qual é o tempo agora?"),
    ("ambiguous", "Che tempo fa ora?"),
    ("ambiguous", "Какая погода сейчас?"),
    ("ambiguous", "ما الطقس الآن؟"),
]


@pytest.mark.parametrize(("expected", "turn"), _DIVERSE_ROUTING_CORPUS)
def test_diverse_multilingual_routing_corpus(expected, turn):
    assert classify_tool_intent(turn).mode == expected


def test_router_never_crashes_on_seeded_unicode_noise():
    rng = random.Random(20260727)
    alphabet = (
        string.ascii_letters
        + string.digits
        + string.punctuation
        + " 你好測試テスト테스트🙂🚀\n\t"
    )
    for _ in range(500):
        turn = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 400)))
        assert classify_tool_intent(turn).mode in {
            "answer_only", "ambiguous", "action",
        }


@pytest.mark.parametrize(
    ("expected", "turn"),
    [
        ("answer_only", "What is Blueprint?"),
        ("answer_only", "普通對話會濫用 MCP 嗎？"),
        ("action", "Please open GitHub"),
        ("action", "請執行測試"),
        ("ambiguous", "What is the weather now?"),
        ("ambiguous", "現在台北天氣？"),
    ],
)
def test_presentation_noise_does_not_change_routing(expected, turn):
    mutations = [
        turn.upper(),
        "  {}  ".format(turn),
        "🤔 {}".format(turn),
        "{}!!!".format(turn),
        turn.replace(" ", "  "),
        "\n{}\n".format(turn),
    ]
    assert all(classify_tool_intent(item).mode == expected for item in mutations)


def test_explicit_no_tool_request_overrides_action_words():
    decision = classify_tool_intent("不要使用 MCP，解釋怎麼打開 GitHub")

    assert decision.mode == "answer_only"
    assert decision.reason == "explicit_no_tool_request"


def test_questions_never_reach_blueprint_or_registry(monkeypatch):
    class ExplodingEngine:
        def search(self, _message):
            raise AssertionError("Blueprint search must not run for Q&A")

    import flyto_blueprint

    monkeypatch.setattr(flyto_blueprint, "get_engine", lambda: ExplodingEngine())

    assert extract_intent("GitHub 是什麼？") is None
    assert extract_intent("Can you explain what GitHub is?") is None


def test_community_blueprint_is_quarantined_even_with_high_score(monkeypatch):
    class FakeEngine:
        def search(self, _message):
            return [{
                "id": "community_high_score",
                "score": 100,
                "use_count": 500,
                "trust_tier": "community",
                "evidence_card": {
                    "sample_count": 500,
                    "success_count": 500,
                    "success_rate": 1.0,
                },
                "args": {},
            }]

    import flyto_blueprint

    monkeypatch.setattr(flyto_blueprint, "get_engine", lambda: FakeEngine())

    assert _match_from_blueprint("repeat this") is None


def test_verified_blueprint_requires_runtime_evidence(monkeypatch):
    class FakeEngine:
        result = {}

        def search(self, _message):
            return [self.result]

    import flyto_blueprint

    engine = FakeEngine()
    monkeypatch.setattr(flyto_blueprint, "get_engine", lambda: engine)
    engine.result = {
        "id": "verified_without_evidence",
        "score": 90,
        "trust_tier": "local_verified",
        "args": {},
    }
    assert _match_from_blueprint("repeat this") is None

    engine.result = {
        "id": "verified_with_evidence",
        "score": 90,
        "trust_tier": "local_verified",
        "evidence_card": {
            "sample_count": 5,
            "success_count": 5,
            "success_rate": 1.0,
        },
        "args": {},
    }
    result = _match_from_blueprint("repeat this")

    assert result["blueprint_id"] == "verified_with_evidence"
    assert result["selection_evidence"]["sample_count"] == 5


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("score", None),
        ("score", "not-a-number"),
        ("score", object()),
        ("score", math.nan),
        ("score", math.inf),
        ("sample_count", "not-a-number"),
        ("sample_count", object()),
        ("sample_count", math.nan),
        ("sample_count", math.inf),
        ("success_count", "not-a-number"),
        ("success_count", object()),
        ("success_count", math.nan),
        ("success_count", math.inf),
        ("success_rate", "not-a-number"),
        ("success_rate", object()),
        ("success_rate", math.nan),
        ("success_rate", math.inf),
    ],
)
def test_malformed_blueprint_evidence_fails_closed(field, value):
    candidate = {
        "id": "malformed",
        "score": 90,
        "trust_tier": "local_verified",
        "evidence_card": {
            "sample_count": 5,
            "success_count": 5,
            "success_rate": 1.0,
        },
    }
    if field == "score":
        candidate[field] = value
    else:
        candidate["evidence_card"][field] = value

    assert blueprint_is_trusted(candidate) is False


@pytest.mark.parametrize(
    "candidate",
    [
        None,
        [],
        {"trust_tier": "local_verified", "score": 90, "evidence_card": []},
        {
            "trust_tier": "local_verified",
            "score": 90,
            "evidence_card": {
                "sample_count": 1.5,
                "success_count": 1,
                "success_rate": 1.0,
            },
        },
        {
            "trust_tier": "local_verified",
            "score": 90,
            "evidence_card": {
                "sample_count": 2,
                "success_count": 3,
                "success_rate": 1.0,
            },
        },
        {
            "trust_tier": "local_verified",
            "score": 90,
            "evidence_card": {
                "sample_count": True,
                "success_count": True,
                "success_rate": True,
            },
        },
    ],
)
def test_inconsistent_blueprint_evidence_fails_closed(candidate):
    assert blueprint_is_trusted(candidate) is False


def test_official_blueprint_does_not_require_learned_evidence():
    assert blueprint_is_trusted({"trust_tier": "official"}) is True


def test_assistant_pre_resolution_uses_the_same_trust_gate(monkeypatch):
    from flyto_ai.assistant import router

    class FakeEngine:
        result = {}

        def search(self, _message):
            return [self.result]

    import flyto_blueprint

    engine = FakeEngine()
    monkeypatch.setattr(flyto_blueprint, "get_engine", lambda: engine)
    engine.result = {
        "id": "community_shortcut",
        "name": "unsafe",
        "score": 100,
        "use_count": 999,
        "trust_tier": "community",
        "args": {},
    }
    assert router.pre_resolve("run it") == ""

    engine.result = {
        "id": "verified_shortcut",
        "name": "safe",
        "score": 90,
        "trust_tier": "ci_verified",
        "evidence_card": {
            "sample_count": 3,
            "success_count": 3,
            "success_rate": 1.0,
        },
        "args": {},
    }
    hint = router.pre_resolve("run it")

    assert "verified_shortcut" in hint
    assert "MUST call use_blueprint" in hint


@pytest.mark.asyncio
async def test_first_tool_blueprint_redirect_rejects_community_experience():
    from flyto_ai.assistant import router

    async def dispatch(name, arguments):
        assert name == "list_blueprints"
        return {
            "blueprints": [{
                "id": "community_shortcut",
                "score": 100,
                "use_count": 999,
                "trust_tier": "community",
                "evidence_card": {
                    "sample_count": 999,
                    "success_count": 999,
                    "success_rate": 1.0,
                },
                "args": {},
            }],
        }

    redirect = await router.guard(
        "execute_module", {}, "run it", dispatch,
    )

    assert redirect is None


@pytest.mark.asyncio
async def test_direct_blueprint_call_cannot_bypass_trust_gate():
    calls = []

    async def raw_dispatch(name, arguments):
        calls.append((name, arguments))
        return {"ok": True, "steps": [{"module": "shell.run", "params": {}}]}

    agent = _routing_agent(raw_dispatch)
    agent._trusted_blueprint_resolver = lambda _blueprint_id: None
    route = ToolIntentDecision(
        "action", 1.0, "explicit_action_request", ("action",),
    )
    dispatch, _ = agent._build_dispatch(
        "Please run the blueprint",
        on_tool_call=None,
        on_stream=None,
        dispatch_wrapper=None,
        routing_decision=route,
        active_tools=agent._tools,
    )

    result = await dispatch(
        "use_blueprint", {"blueprint_id": "community_shortcut", "args": {}},
    )

    assert result["ok"] is False
    assert result["policy_outcome"] == "block"
    assert calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "turn",
    [
        "不要打開 GitHub",
        "Do not open GitHub.",
        "GitHubを開かないでください。",
        "GitHub를 열지 마세요.",
        "如果我說「打開 GitHub」，系統應該怎麼判斷？",
        "Is https://example.com a valid URL?",
    ],
)
async def test_non_action_turn_cannot_reach_runtime_dispatch(turn):
    calls = []

    async def raw_dispatch(name, arguments):
        calls.append((name, arguments))
        return {"ok": True}

    agent = _routing_agent(raw_dispatch)
    route = classify_tool_intent(turn)
    active_tools = agent._tools_for_route(route, "execute")
    dispatch, has_tools = agent._build_dispatch(
        turn,
        on_tool_call=None,
        on_stream=None,
        dispatch_wrapper=None,
        routing_decision=route,
        active_tools=active_tools,
    )

    result = await dispatch(
        "execute_module",
        {"module_id": "browser.goto", "params": {"url": "https://github.com"}},
    )

    assert route.mode == "answer_only"
    assert has_tools is False
    assert active_tools == []
    assert result["ok"] is False
    assert calls == []


def test_route_policy_returns_allow_confirm_and_block():
    enforcer = PermissionEnforcer(PermissionLevel.WORKSPACE_WRITE)

    answer = enforcer.check_route("search_modules", {}, "answer_only")
    ambiguous_read = enforcer.check_route("search_modules", {}, "ambiguous")
    ambiguous_write = enforcer.check_route(
        "execute_module", {"module_id": "browser.click"}, "ambiguous",
    )
    explicit_action = enforcer.check_route(
        "execute_module", {"module_id": "browser.click"}, "action",
    )

    assert answer.outcome == PermissionOutcome.BLOCK
    assert ambiguous_read.outcome == PermissionOutcome.ALLOW
    assert ambiguous_write.outcome == PermissionOutcome.REQUIRE_CONFIRMATION
    assert explicit_action.outcome == PermissionOutcome.ALLOW


@pytest.mark.asyncio
async def test_answer_only_turn_skips_planner_model_and_dispatch():
    calls = []

    async def raw_dispatch(name, arguments):
        calls.append((name, arguments))
        return {"ok": True}

    agent = _routing_agent(raw_dispatch)
    result = await agent._try_deterministic(
        "GitHub 是什麼？",
        on_tool_call=None,
        on_stream=None,
        dispatch_wrapper=None,
    )

    assert result is None
    assert calls == []
    assert agent._tools_for_route(
        classify_tool_intent("GitHub 是什麼？"), "execute",
    ) == []


@pytest.mark.asyncio
async def test_normal_chat_uses_one_model_call_and_zero_tool_schemas():
    provider_calls = []
    dispatch_calls = []

    class FakeProvider:
        async def chat(
            self,
            messages,
            system_prompt,
            tools,
            dispatch_fn,
            max_rounds=30,
            on_stream=None,
        ):
            provider_calls.append({
                "messages": messages,
                "tools": tools,
                "max_rounds": max_rounds,
            })
            return "GitHub 是程式碼協作平台。", [], 1, {}

    async def raw_dispatch(name, arguments):
        dispatch_calls.append((name, arguments))
        return {"ok": True}

    agent = Agent(
        config=AgentConfig(
            provider="ollama",
            api_key="test",
            enable_deterministic=True,
            enable_memory=False,
            enable_pro=False,
            enable_transcript=False,
        ),
        tools=[
            {"name": "search_modules"},
            {"name": "execute_module"},
            {"name": "use_blueprint"},
        ],
        dispatch_fn=raw_dispatch,
        api_client=FakeProvider(),
    )
    agent._assistant = None

    response = await agent.chat("GitHub 是什麼？")

    assert response.ok is True
    assert len(provider_calls) == 1
    assert provider_calls[0]["tools"] == []
    assert provider_calls[0]["max_rounds"] == 1
    assert dispatch_calls == []
    assert agent.routing_decision["mode"] == "answer_only"


@pytest.mark.asyncio
async def test_deterministic_shortcut_cannot_bypass_runtime_guard(monkeypatch):
    calls = []

    async def raw_dispatch(name, arguments):
        calls.append((name, arguments))
        return {"ok": True}

    from flyto_ai.intelligence import planner

    monkeypatch.setattr(
        planner,
        "extract_intent",
        lambda _message: {
            "intent": "single_module",
            "module_id": "shell.run",
            "params": {"command": "pwd"},
        },
    )
    agent = _routing_agent(raw_dispatch)

    result = await agent._try_deterministic(
        "Run shell.run command=pwd",
        on_tool_call=None,
        on_stream=None,
        dispatch_wrapper=None,
    )

    assert result is not None
    assert result.ok is False
    assert calls == []
    assert agent.routing_metrics["tool_calls_attempted"] == 1
    assert agent.routing_metrics["tool_calls_blocked"] == 1
    assert agent.routing_metrics["tool_calls_executed"] == 0


@pytest.mark.asyncio
async def test_ambiguous_turn_exposes_read_only_tools_and_blocks_forged_write():
    calls = []

    async def raw_dispatch(name, arguments):
        calls.append((name, arguments))
        return {"ok": True}

    agent = _routing_agent(raw_dispatch)
    route = ToolIntentDecision(
        "ambiguous", 0.7, "read_only_discovery_may_help", ("question",),
    )
    visible = agent._tools_for_route(route, "execute")
    dispatch, has_tools = agent._build_dispatch(
        "現在狀態？",
        on_tool_call=None,
        on_stream=None,
        dispatch_wrapper=None,
        routing_decision=route,
        active_tools=visible,
    )

    result = await dispatch(
        "execute_module", {"module_id": "browser.click", "params": {}},
    )

    assert has_tools is True
    assert [tool["name"] for tool in visible] == [
        "search_modules", "get_module_info",
    ]
    assert result["policy_outcome"] == "require_confirmation"
    assert result["routing_mode"] == "ambiguous"
    assert calls == []
