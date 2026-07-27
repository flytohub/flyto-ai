# Copyright 2024 Flyto2
# Licensed under the Apache License, Version 2.0
"""Deterministic Planner — maps user intent to executable module sequences.

Replaces LLM freestyle module selection with a deterministic pipeline:
1. Intent extraction (1 LLM call, structured output)
2. Recipe matching (zero LLM — Blueprint + Knowledge)
3. Parameter filling (zero LLM — from intent params + defaults)
4. Contract validation (zero LLM — ContractEngine)
5. Execution (zero LLM — sequential module dispatch)

Falls back to LLM freestyle when no recipe matches.
"""
import logging
import math
import re
from dataclasses import asdict, dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Common intent → module recipes (built-in, no learning needed)
BUILTIN_RECIPES: Dict[str, List[Dict[str, Any]]] = {
    "open_website": [
        {"module": "browser.launch", "params": {"headless": False, "channel": "chrome"}},
        {"module": "browser.goto", "params_from": ["url"]},
    ],
    "search_on_website": [
        {"module": "browser.snapshot", "params": {}},
        {"module": "browser.type", "params_from": ["search_text"], "selector_from_snapshot": True},
        {"module": "browser.click", "params": {}, "selector_from_snapshot": True, "selector_hint": "search button"},
    ],
    "click_element": [
        {"module": "browser.snapshot", "params": {}},
        {"module": "browser.click", "params_from": ["target"], "selector_from_snapshot": True},
    ],
    "extract_page": [
        {"module": "browser.snapshot", "params": {}},
    ],
    "screenshot": [
        {"module": "browser.screenshot", "params_from": ["output_path"]},
    ],
}

_NO_TOOL_RE = re.compile(
    r"(?:do\s+not|don't|dont|without|no)\s+"
    r"(?:use|using|call|calling|run|running).{0,16}(?:tools?|mcp)"
    r"|(?:answer|reply|respond).{0,20}(?:without|no).{0,10}(?:tools?|mcp)"
    r"|(?:use\s+(?:your\s+)?knowledge\s+only|no\s+tool\s+calls?)"
    r"|(?:不要|別|别|不准|禁止).{0,10}(?:使用|呼叫|调用|執行|执行).{0,10}"
    r"(?:工具|tool|mcp)"
    r"|(?:只|僅|仅).{0,8}(?:回答|說明|说明).{0,12}(?:不要|不使用|不用).{0,8}"
    r"(?:工具|tool|mcp|執行|执行)"
    r"|(?:ツール|mcp).{0,8}(?:使わず|使用せず)"
    r"|(?:도구|mcp).{0,8}(?:사용하지\s*말고|없이)"
    r"|(?:sin\s+usar|no\s+uses?).{0,8}(?:herramientas?|mcp)"
    r"|(?:sans\s+utiliser|n'utilise\s+pas).{0,8}(?:outils?|mcp)"
    r"|(?:ohne|nicht).{0,8}(?:werkzeuge?|tools?|mcp).{0,8}(?:benutzen|verwenden)?",
    re.IGNORECASE,
)
_NEGATED_ACTION_RE = re.compile(
    r"^\s*(?:please\s+)?(?:do\s+not|don't|dont|never)\s+"
    r"(?:open|visit|run|execute|delete|remove|modify|update|search|write|"
    r"install|commit|deploy|send|upload|download|create|fix|scrape|inspect)"
    r"|^\s*(?:can|could|would)\s+you\s+not\s+"
    r"(?:open|run|execute|delete|modify|search|write|install|deploy)"
    r"|^\s*without\s+(?:opening|visiting|running|executing|deleting|"
    r"modifying|searching|using)"
    r"|^\s*(?:請|请)?\s*(?:先)?(?:不要|別|别|不准|禁止)\s*"
    r"(?:打開|打开|開啟|开启|執行|执行|運行|运行|刪除|删除|修改|"
    r"搜尋|搜索|查詢|查询|寫入|写入|安裝|安装|提交|部署|下載|下载|"
    r"上傳|上传|建立|创建|修復|修复)"
    r"|(?:開かないで|実行しないで|削除しないで|変更しないで|検索しないで)"
    r"|(?:열지\s*마세요|실행하지\s*마세요|삭제하지\s*마세요|"
    r"수정하지\s*마세요|검색하지\s*마세요)"
    r"|^\s*(?:por\s+favor,\s*)?no\s+"
    r"(?:abras?|ejecutes?|borres?|modifiques?|busques?|instales?|despliegues?)"
    r"|^\s*(?:n['’]|ne\s+).{0,30}\s+pas\b"
    r"|^\s*(?:öffne|führe|lösche|ändere|suche|installiere).{0,30}\bnicht\b"
    r"|^\s*n[aã]o\s+(?:abra|execute|exclua|modifique|procure|instale)"
    r"|^\s*non\s+(?:aprire|eseguire|eliminare|modificare|cercare|installare)"
    r"|^\s*не\s+(?:открывай|запускай|выполняй|удаляй|изменяй|ищи)"
    r"|^\s*لا\s+(?:تفتح|تشغل|تنفذ|تحذف|تعدل|تبحث)",
    re.IGNORECASE,
)
_META_REQUEST_RE = re.compile(
    r"\b(?:explain|analyse|analyze)\s+(?:the\s+)?(?:sentence|phrase|wording)\b"
    r"|\btell\s+me\s+about\b"
    r"|\bif\s+(?:i|a\s+user|the\s+user|someone)\s+(?:say|says|said)\b"
    r"|\bwhat\s+should\s+(?:the\s+)?(?:router|system|agent)\s+do\b"
    r"|\b(?:readme|error|message|text)\s+(?:says|contains)\b"
    r"|(?:這句話|这句话|這段話|这段话).{0,24}(?:意思|祈使|命令|怎麼|怎么)"
    r"|(?:如果|假如).{0,60}(?:流程|怎麼|怎么|如何|系統|系统|判斷|判断)"
    r"|(?:エラー|文).{0,20}(?:意味|説明)"
    r"|(?:문장|오류).{0,20}(?:뜻|설명)",
    re.IGNORECASE,
)
_DECLARATIVE_OR_ACTION_QUESTION_RE = re.compile(
    r"^\s*(?:open\s+source|search\s+(?:algorithms?|engines?)|"
    r"build\s+systems?|list\s+comprehensions?|read\s+consistency|"
    r"write\s+amplification|commit\s+history|installation\s+process)\s+"
    r"(?:is|are|was|were|means?|refers?)\b"
    r"|^\s*(?:建立|创建|刪除|删除|執行|执行|修改|分析|搜尋|搜索).{0,48}"
    r"(?:有什麼|有什么|會發生什麼|会发生什么|會怎樣|会怎样|"
    r"是否|安全嗎|安全吗|優缺點|优缺点|風險|风险|後果|后果).{0,16}[？?]?$",
    re.IGNORECASE,
)
_EXPLANATION_RE = re.compile(
    r"\b(?:what\s+is|what\s+are|why|how\s+does|how\s+do|explain|"
    r"tell\s+me\s+about|pros?\s+and\s+cons?|is\s+it\s+safe|"
    r"is\s+https?://\S+\s+(?:a\s+)?valid)\b"
    r"|(?:是什麼|是什么|為什麼|为什么|怎麼運作|怎么运作|解釋|解释|"
    r"介紹|介绍|聊聊|優缺點|优缺点|安全嗎|安全吗|強嗎|强吗|會不會|会不会)"
    r"|(?:とは|なぜ|説明して)"
    r"|(?:무엇|왜|설명해)"
    r"|(?:qué\s+es|por\s+qué|expl[ií]ca)"
    r"|(?:qu['’]est-ce|pourquoi|explique)"
    r"|(?:was\s+ist|warum|erkläre)",
    re.IGNORECASE,
)
_EN_ACTION_RE = re.compile(
    r"^\s*(?:(?:please|kindly)\s+|(?:can|could|would)\s+you\s+|"
    r"help\s+me(?:\s+to)?\s+)?"
    r"(?:open|visit|go\s+to|search(?:\s+for)?|run|execute|click|download|"
    r"upload|create|update|delete|remove|fix|repair|push|deploy|send|"
    r"take\s+(?:a\s+)?screenshot|repeat|rewrite|fetch|find|check|write|"
    r"install|commit|rerun|build|apply|read|summari[sz]e|analy[sz]e|"
    r"inspect|list|scrape|extract|save|tell)\b",
    re.IGNORECASE,
)
_CJK_ACTION_RE = re.compile(
    r"^\s*(?:請|请|麻煩|麻烦|幫我|帮我|替我|可以幫我|可以帮我)?\s*"
    r"(?:打開|打开|開啟|开启|前往|搜尋|搜索|查詢|查询|執行|执行|運行|运行|"
    r"點擊|点击|下載|下载|上傳|上传|建立|創建|创建|更新|刪除|删除|修復|修复|"
    r"部署|推送|上去|截圖|截图|重複|重复|重新執行|重新执行|修改|改寫|改写|"
    r"重寫|重写|抓取|尋找|查找|找出|檢查|检查|寫入|写入|安裝|安装|提交|"
    r"讀取|读取|分析|列出|套用|儲存|储存|摘要|截)"
    r"|^\s*(?:把|將|将).{1,48}?(?:改|修改|改寫|改写|重寫|重写|寫|写|"
    r"刪除|删除|更新|套用|儲存|储存|提交)"
    r"|^\s*(?:到|去).{1,32}?(?:找|搜尋|搜索|查詢|查询)",
    re.IGNORECASE,
)
_JA_ACTION_RE = re.compile(
    r"^\s*.{0,48}(?:開いて|アクセスして|検索して|実行して|クリックして|"
    r"ダウンロードして|アップロードして|作成して|更新して|削除して|"
    r"修正して|書き直して|保存して|確認して|分析して|一覧にして)"
    r"(?:ください|下さい|くれますか|[。.!]?$)"
)
_KO_ACTION_RE = re.compile(
    r"^\s*.{0,48}(?:열어|실행해|작성해|수정해|검색해|저장해|삭제해|"
    r"다운로드해|업로드해|배포해|확인해|분석해|나열해|스크린샷)"
    r"(?:\s*주세요|\s*주십시오|\s*줘|세요|[.!]?$)"
)
_ES_ACTION_RE = re.compile(
    r"^\s*[¡¿]?\s*(?:por\s+favor[,:]?\s*)?"
    r"(?:abre|ejecuta|corrige|busca|escribe|crea|elimina|descarga|sube|"
    r"despliega|guarda|analiza|lista|inspecciona|instala|toma)\b",
    re.IGNORECASE,
)
_FR_ACTION_RE = re.compile(
    r"^\s*(?:s['’]il\s+te\s+pla[iî]t[,:]?\s*)?"
    r"(?:ouvre|ex[eé]cute|corrige|cherche|[eé]cris|cr[eé]e|supprime|"
    r"t[eé]l[eé]charge|d[eé]ploie|enregistre|analyse|liste|inspecte|installe|prends)\b",
    re.IGNORECASE,
)
_DE_ACTION_RE = re.compile(
    r"^\s*(?:bitte\s+)?(?:öffne|oeffne|behebe|suche|schreibe|erstelle|"
    r"lösche|loesche|speichere|analysiere|liste|prüfe|pruefe|installiere)\b"
    r"|^\s*(?:bitte\s+)?führe\b.{0,36}\baus\b",
    re.IGNORECASE,
)
_PT_IT_ACTION_RE = re.compile(
    r"^\s*(?:por\s+favor[,:]?\s*)?(?:abra|execute|corrija|procure|escreva|"
    r"crie|exclua|salve|analise|instale)\b"
    r"|^\s*(?:per\s+favore[,:]?\s*)?(?:apri|esegui|correggi|cerca|scrivi|"
    r"crea|elimina|salva|analizza|installa)\b",
    re.IGNORECASE,
)
_RU_ACTION_RE = re.compile(
    r"^\s*(?:пожалуйста[,:]?\s*)?(?:открой|запусти|выполни|исправь|"
    r"найди|создай|удали|сохрани|установи|разверни|проверь|прочитай|"
    r"проанализируй)\b",
    re.IGNORECASE,
)
_AR_ACTION_RE = re.compile(
    r"^\s*(?:من\s+فضلك[،,:]?\s*)?(?:افتح|شغ[ّ]?ل|نف[ّ]?ذ|أصلح|اصلح|"
    r"ابحث|أنشئ|انشئ|احذف|احفظ|ثب[ّ]?ت|انشر|افحص|اقرأ|حل[ّ]?ل)\b",
    re.IGNORECASE,
)
_STATUS_QUESTION_RE = re.compile(
    r"\b(?:status|latest|today|weather|price|score|exchange\s+rate)\b"
    r"|\b(?:currently\s+operational|current\s+(?:ceo|version|president)|"
    r"what\s+time\s+is\s+it|this\s+weekend)\b"
    r"|(?:最新|今天|今日|天氣|天气|價格|价格|比分|狀態|状态|匯率|汇率|"
    r"幾點|几点|這週末|这周末|本週末|本周末|目前.{0,20}(?:正常|ceo|誰|谁|版本))"
    r"|(?:現在の天気|今日の為替|最新|現在時刻)"
    r"|(?:현재\s*날씨|오늘\s*환율|최신|현재\s*(?:시간|ceo|상태))"
    r"|(?:tiempo.{0,12}ahora|tipo\s+de\s+cambio.{0,12}hoy|m[aá]s\s+reciente)"
    r"|(?:temps.{0,12}maintenant|taux\s+de\s+change.{0,12}aujourd|plus\s+r[eé]cent)"
    r"|(?:wetter.{0,12}jetzt|wechselkurs.{0,12}heute|neueste|aktuell)"
    r"|(?:tempo.{0,12}agora|c[aâ]mbio.{0,12}hoje|mais\s+recente)"
    r"|(?:(?:meteo|tempo).{0,12}ora|cambio.{0,12}oggi|pi[uù]\s+recente)"
    r"|(?:погода.{0,12}сейчас|курс.{0,12}сегодня|последн(?:яя|ий|ие))"
    r"|(?:الطقس.{0,12}الآن|سعر\s+الصرف.{0,12}اليوم|أحدث|احدث)",
    re.IGNORECASE,
)
_STRUCTURED_MODULE_RE = re.compile(
    r"(?<![/:])\b[a-z][\w-]*\.[a-z][\w.-]*\b",
    re.IGNORECASE,
)
_EXPLICIT_PARAMS_RE = re.compile(
    r"\b[\w-]+\s*=\s*\S+|(?:params?|arguments?)\s*:\s*(?:\{|\S)|\{[^{}]*\}",
    re.IGNORECASE,
)
_LEADING_EMOJI_RE = re.compile(
    r"^\s*(?:[\U0001F300-\U0001FAFF\u2600-\u27BF]\ufe0f?\s*)+",
)


@dataclass(frozen=True)
class ToolIntentDecision:
    """Deterministic, inspectable decision made before exposing any tool."""

    mode: str
    confidence: float
    reason: str
    signals: Tuple[str, ...] = ()

    @property
    def tool_eligible(self) -> bool:
        return self.mode == "action"

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["tool_eligible"] = self.tool_eligible
        return data


def classify_tool_intent(message: str) -> ToolIntentDecision:
    """Classify a turn as answer-only, read-only ambiguous, or explicit action.

    This deliberately avoids an LLM routing call.  It is a safety boundary:
    explanation requests never reach Blueprint, registry matching, or write tools.
    """
    msg = (message or "").strip()
    if not msg:
        return ToolIntentDecision("answer_only", 1.0, "empty_message", ("empty",))
    semantic_msg = _LEADING_EMOJI_RE.sub("", msg).strip()
    if not semantic_msg:
        return ToolIntentDecision("answer_only", 1.0, "empty_message", ("empty",))

    if _NO_TOOL_RE.search(semantic_msg):
        return ToolIntentDecision(
            "answer_only", 1.0, "explicit_no_tool_request", ("no_tool",),
        )

    if _NEGATED_ACTION_RE.search(semantic_msg):
        return ToolIntentDecision(
            "answer_only", 1.0, "negated_action_request", ("negation",),
        )

    if _META_REQUEST_RE.search(semantic_msg):
        return ToolIntentDecision(
            "answer_only", 0.99, "quoted_or_hypothetical", ("meta_request",),
        )

    if _DECLARATIVE_OR_ACTION_QUESTION_RE.search(semantic_msg):
        return ToolIntentDecision(
            "answer_only", 0.98, "declarative_or_action_question", ("question",),
        )

    action_signals = []
    action_patterns = (
        ("action_verb_en", _EN_ACTION_RE),
        ("action_verb_cjk", _CJK_ACTION_RE),
        ("action_verb_ja", _JA_ACTION_RE),
        ("action_verb_ko", _KO_ACTION_RE),
        ("action_verb_es", _ES_ACTION_RE),
        ("action_verb_fr", _FR_ACTION_RE),
        ("action_verb_de", _DE_ACTION_RE),
        ("action_verb_pt_it", _PT_IT_ACTION_RE),
        ("action_verb_ru", _RU_ACTION_RE),
        ("action_verb_ar", _AR_ACTION_RE),
    )
    for signal, pattern in action_patterns:
        if pattern.search(semantic_msg):
            action_signals.append(signal)
    if action_signals:
        return ToolIntentDecision(
            "action", 0.96, "explicit_action_request", tuple(action_signals),
        )

    # Current-data questions may benefit from discovery, but never mutation.
    if _STATUS_QUESTION_RE.search(semantic_msg):
        return ToolIntentDecision(
            "ambiguous", 0.82, "read_only_discovery_may_help", ("current_data",),
        )

    if _EXPLANATION_RE.search(semantic_msg):
        return ToolIntentDecision(
            "answer_only", 0.98, "explanation_or_opinion", ("explanation",),
        )

    # A module id plus explicit parameters is an action even without a natural
    # language verb, e.g. ``image.resize width=800``.
    if (
        _STRUCTURED_MODULE_RE.search(semantic_msg)
        and _EXPLICIT_PARAMS_RE.search(semantic_msg)
    ):
        return ToolIntentDecision(
            "action", 0.92, "structured_module_request", ("module_id", "params"),
        )

    if semantic_msg.endswith(("?", "？")):
        return ToolIntentDecision(
            "answer_only", 0.78, "question_without_action", ("question",),
        )

    return ToolIntentDecision(
        "answer_only", 0.86, "no_action_signal", ("conversation",),
    )



async def extract_intent_llm(message: str, provider) -> Optional[Dict[str, Any]]:
    """Extract structured intent using 1 cheap LLM call. Language-agnostic.

    Returns {"action": "...", "target": "...", "query": "...", "params": {...}}
    or None if the message is a question/conversation (not an action).
    """
    prompt = (
        "Extract the user's ACTION intent as JSON. If it's a question or conversation, return null.\n\n"
        "Possible actions: navigate, search, click, play, screenshot, download, upload, "
        "resize, convert, generate, send, extract, scrape\n\n"
        "For navigate: resolve the site name to a full URL.\n\n"
        "User: {}\n\n"
        "Return ONLY valid JSON (no markdown), example:\n"
        '{{"action":"navigate","url":"https://www.youtube.com","query":""}}\n'
        '{{"action":"search","url":"https://www.youtube.com","query":"周杰倫"}}\n'
        '{{"action":"click","target":"first video"}}\n'
        '{{"action":"resize","target":"image.png","params":{{"width":800,"height":600}}}}\n'
        "null (for questions/conversation)"
    ).format(message)

    try:
        response, _, _, _ = await provider.chat(
            [{"role": "user", "content": prompt}],
            system_prompt="You extract intent as JSON. Return ONLY JSON or null.",
            tools=[], dispatch_fn=None, max_rounds=1,
        )
        if not response or response.strip() == "null":
            return None

        import json
        data = json.loads(response.strip().removeprefix("```json").removesuffix("```").strip())
        if not isinstance(data, dict) or "action" not in data:
            return None
        return data
    except Exception:
        return None


def extract_intent(message: str) -> Optional[Dict[str, Any]]:
    """Synchronous intent extraction — pure data-driven, zero hardcoding.

    Priority:
    1. Explicit URL in message → navigate
    2. Blueprint (learned from past success)
    3. Module registry (395 module schemas)
    4. Returns None → caller uses extract_intent_llm()
    """
    msg = message.strip()
    if not classify_tool_intent(msg).tool_eligible:
        return None

    # 1. Explicit URL → navigate directly
    url_match = re.search(r'(https?://\S+)', msg)
    if url_match:
        return {"intent": "open_website", "url": url_match.group(1), "site": url_match.group(1)}

    # Domain pattern (xxx.com) → navigate
    domain_match = re.search(r'(\w+\.\w{2,}(?:\.\w{2,})?)', msg)
    if domain_match:
        return {"intent": "open_website", "url": "https://" + domain_match.group(1), "site": domain_match.group(1)}

    # 2. Blueprint (learned from past executions)
    blueprint_intent = _match_from_blueprint(msg)
    if blueprint_intent:
        return blueprint_intent

    # 3. Module registry (395 module descriptions)
    registry_intent = _match_from_registry(msg.lower())
    if registry_intent:
        return registry_intent

    # No sync match → caller should try extract_intent_llm()
    return None


def blueprint_is_trusted(
    candidate: Dict[str, Any],
    *,
    min_score: float = 50,
    min_samples: int = 1,
) -> bool:
    """Return whether learned experience is authorized for automatic reuse."""
    if not isinstance(candidate, dict):
        return False

    trust_tier = str(candidate.get("trust_tier", "community"))
    if trust_tier == "official":
        return True
    if trust_tier not in {"local_verified", "ci_verified"}:
        return False

    evidence = candidate.get("evidence_card")
    if not isinstance(evidence, dict):
        return False

    def finite_number(value: Any) -> Optional[float]:
        if isinstance(value, bool):
            return None
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError):
            return None
        return number if math.isfinite(number) else None

    score = finite_number(candidate.get("score"))
    score_floor = finite_number(min_score)
    samples_floor = finite_number(min_samples)
    sample_count = finite_number(evidence.get("sample_count"))
    success_count = finite_number(evidence.get("success_count"))
    success_rate = finite_number(evidence.get("success_rate"))
    values = (
        score,
        score_floor,
        samples_floor,
        sample_count,
        success_count,
        success_rate,
    )
    if any(value is None for value in values):
        return False

    assert score is not None
    assert score_floor is not None
    assert samples_floor is not None
    assert sample_count is not None
    assert success_count is not None
    assert success_rate is not None
    counts_are_integers = all(
        value.is_integer()
        for value in (samples_floor, sample_count, success_count)
    )
    return (
        counts_are_integers
        and samples_floor >= 1
        and sample_count >= samples_floor
        and success_count >= samples_floor
        and success_count <= sample_count
        and score >= score_floor
        and 0.0 <= success_rate <= 1.0
        and success_rate >= 0.8
    )


def trusted_blueprint_summary(blueprint_id: str) -> Optional[Dict[str, Any]]:
    """Resolve an exact Blueprint id and fail closed unless it is trusted."""
    if not blueprint_id:
        return None
    try:
        from flyto_blueprint import get_engine

        for candidate in get_engine().list_blueprints():
            if candidate.get("id") == blueprint_id:
                return candidate if blueprint_is_trusted(candidate) else None
    except Exception:
        return None
    return None


def _match_from_blueprint(msg: str) -> Optional[Dict[str, Any]]:
    """Match only Blueprint experience that has crossed a verification gate."""
    try:
        from flyto_blueprint import get_engine
        engine = get_engine()
        results = engine.search(msg)
        if not results:
            return None

        top = results[0]
        trust_tier = str(top.get("trust_tier", "community"))
        if not blueprint_is_trusted(top):
            return None

        evidence = top.get("evidence_card") or {}

        args_schema = top.get("args", {})
        args = _extract_params_from_message(msg, args_schema)
        missing_required = [
            name for name, meta in args_schema.items()
            if meta.get("required") and name not in args
        ]
        if missing_required:
            return None

        return {
            "intent": "blueprint",
            "blueprint_id": top.get("id", ""),
            "args": args,
            "selection_evidence": {
                "trust_tier": trust_tier,
                "sample_count": int(evidence.get("sample_count", 0) or 0),
                "success_count": int(evidence.get("success_count", 0) or 0),
                "success_rate": float(evidence.get("success_rate", 0.0) or 0.0),
            },
        }
    except Exception:
        return None


def _match_from_registry(msg: str) -> Optional[Dict[str, Any]]:
    """Match user message to a single module using the module registry.

    Searches module names/descriptions for keyword overlap.
    Returns a single-module intent if found.
    """
    try:
        from core.modules.registry import ModuleRegistry
        all_mods = ModuleRegistry.get_all_metadata()
    except ImportError:
        return None

    if not all_mods:
        return None

    # Keyword search — match module_id and description
    q = msg.lower()
    candidates = []
    for mid, meta in all_mods.items():
        desc = (meta.get("description", "") or "").lower()
        mid_lower = mid.lower()
        score = 0
        for word in q.split():
            if len(word) < 2:
                continue
            if word in mid_lower:
                score += 10
            if word in desc:
                score += 5
        if score > 0:
            candidates.append((mid, meta, score))

    if not candidates:
        return None

    # Sort by score, take the top
    candidates.sort(key=lambda x: x[2], reverse=True)
    top_mid, top_meta, top_score = candidates[0]

    # Require a minimum score to avoid false matches
    if top_score < 10:
        return None

    # Get param schema
    try:
        from core.mcp_handler import get_module_info
        info = get_module_info(module_id=top_mid)
        schema = info.get("params_schema", {}) if info else {}
    except Exception:
        schema = {}

    # Try to extract param values from the message using simple heuristics
    params = _extract_params_from_message(msg, schema)

    return {
        "intent": "single_module",
        "module_id": top_mid,
        "params": params,
        "schema": schema,
    }


def _extract_params_from_message(msg: str, schema: dict) -> dict:
    """Extract parameter values from message text based on schema hints.

    Simple heuristic extraction — not LLM, just pattern matching.
    """
    params = {}

    for param_name, param_def in schema.items():
        ptype = param_def.get("type", "")
        desc = (param_def.get("description", "") or "").lower()

        # Explicit ``name=value`` or ``name: value`` wins over heuristics.
        explicit = re.search(
            r"(?:^|\s){}\s*(?:=|:)\s*(\"[^\"]*\"|'[^']*'|[^\s,]+)".format(
                re.escape(param_name),
            ),
            msg,
            flags=re.IGNORECASE,
        )
        if explicit:
            raw = explicit.group(1).strip("\"'")
            if ptype in ("integer", "int"):
                try:
                    params[param_name] = int(raw)
                    continue
                except ValueError:
                    pass
            elif ptype in ("number", "float"):
                try:
                    params[param_name] = float(raw)
                    continue
                except ValueError:
                    pass
            elif ptype in ("boolean", "bool"):
                lowered = raw.lower()
                if lowered in ("true", "yes", "1", "on"):
                    params[param_name] = True
                    continue
                if lowered in ("false", "no", "0", "off"):
                    params[param_name] = False
                    continue
            else:
                params[param_name] = raw
                continue

        # Number extraction (width, height, size, count, etc.)
        if ptype in ("integer", "number", "int", "float"):
            numbers = re.findall(r'\b(\d+(?:\.\d+)?)\b', msg)
            if numbers:
                if "width" in param_name or "寬" in param_name:
                    params[param_name] = int(float(numbers[0]))
                    if len(numbers) > 1:
                        # Look for paired height
                        height_key = param_name.replace("width", "height")
                        if height_key in schema:
                            params[height_key] = int(float(numbers[1]))
                elif "height" in param_name or "高" in param_name:
                    if param_name not in params:
                        params[param_name] = int(float(numbers[-1]))
                elif param_name not in params:
                    params[param_name] = int(float(numbers[0])) if ptype in ("integer", "int") else float(numbers[0])

        # URL extraction
        elif "url" in param_name.lower() or "url" in desc:
            url_match = re.search(r'https?://\S+', msg)
            if url_match:
                params[param_name] = url_match.group(0)

        # File path extraction
        elif "path" in param_name.lower() or "file" in param_name.lower():
            path_match = re.search(r'[/~][\w./\-]+\.\w+', msg)
            if path_match:
                params[param_name] = path_match.group(0)

        # Text/string extraction — use the full message as fallback
        elif ptype == "string" and ("text" in param_name or "content" in param_name or "query" in param_name):
            # Remove common command words, keep the content
            content = re.sub(r'^(?:請|幫我|help me|please)\s*', '', msg, flags=re.IGNORECASE)
            content = re.sub(r'^(?:把|將|convert|resize|download|upload)\s*', '', content, flags=re.IGNORECASE)
            if content and content != msg:
                params[param_name] = content.strip()

    return params


def plan_execution(
    intent: Dict[str, Any],
    has_browser: bool = False,
) -> List[Dict[str, Any]]:
    """Plan a deterministic execution sequence from an intent.

    Returns a list of steps: [{"module": "...", "params": {...}}, ...]
    """
    intent_type = intent.get("intent", "")
    steps = []

    if intent_type == "open_website":
        if not has_browser:
            steps.append({"module": "browser.launch", "params": {"headless": False, "channel": "chrome"}})
        steps.append({"module": "browser.goto", "params": {"url": intent["url"]}})
        steps.append({"module": "browser.snapshot", "params": {}})

    elif intent_type == "open_and_search":
        if not has_browser:
            steps.append({"module": "browser.launch", "params": {"headless": False, "channel": "chrome"}})
        steps.append({"module": "browser.goto", "params": {"url": intent["url"]}})
        steps.append({"module": "browser.snapshot", "params": {}})
        steps.append({
            "module": "browser.type",
            "params": {"text": intent["search_text"], "press_enter": True},
            "needs_selector": True,
            "selector_hint": "search input",
        })
        steps.append({"module": "browser.snapshot", "params": {}})

    elif intent_type == "search_on_website":
        steps.append({"module": "browser.snapshot", "params": {}})
        steps.append({
            "module": "browser.type",
            "params": {"text": intent["search_text"], "press_enter": True},
            "needs_selector": True,
            "selector_hint": "search input",
        })
        # press_enter submits the search — no need for a separate click
        steps.append({"module": "browser.snapshot", "params": {}})

    elif intent_type == "click_element":
        steps.append({"module": "browser.snapshot", "params": {}})
        steps.append({
            "module": "browser.click",
            "params": {},
            "needs_selector": True,
            "selector_hint": intent.get("target", ""),
        })
        steps.append({"module": "browser.snapshot", "params": {}})

    elif intent_type == "single_module":
        steps.append({
            "module": intent["module_id"],
            "params": intent.get("params", {}),
        })

    elif intent_type == "blueprint":
        for bp_step in intent.get("steps", []):
            steps.append({
                "module": bp_step.get("module", bp_step.get("module_id", "")),
                "params": bp_step.get("params", {}),
            })

    return steps


async def execute_plan(
    steps: List[Dict[str, Any]],
    dispatch: Callable,
) -> Tuple[List[Dict[str, Any]], str]:
    """Execute a deterministic plan step by step.

    For steps that need_selector, extracts the selector from the
    previous snapshot result using hint matching.

    Returns (execution_results, summary_text).
    """
    results = []
    last_snapshot = {}  # Full structured snapshot result

    for step_idx, step in enumerate(steps):
        module_id = step["module"]
        params = dict(step.get("params", {}))
        logger.debug("execute_plan step %d: %s", step_idx, module_id)

        # Resolve selector from previous snapshot
        if step.get("needs_selector") and last_snapshot:
            hint = step.get("selector_hint", "")
            element_type = "input" if module_id == "browser.type" else "button" if module_id == "browser.click" else ""
            selector = _find_selector_from_structured(last_snapshot, hint, element_type)
            if selector:
                params["selector"] = selector
            else:
                # Can't find selector — skip this step
                results.append({
                    "module_id": module_id,
                    "ok": False,
                    "error": "Could not find selector for: {}".format(hint),
                })
                continue

        # Execute
        try:
            result = await dispatch("execute_module", {
                "module_id": module_id,
                "params": params,
            })
        except Exception as e:
            logger.error("execute_plan dispatch error step %d (%s): %s", step_idx, module_id, e)
            results.append({"module_id": module_id, "ok": False, "error": str(e)})
            break

        # Determine success: check ok, status, or absence of error
        is_ok = result.get("ok")
        if is_ok is None:
            is_ok = result.get("status") == "success" or (
                "error" not in result and "Error" not in str(result.get("message", ""))
            )

        exec_result = {
            "module_id": module_id,
            "ok": bool(is_ok),
            "error": result.get("error", ""),
        }
        results.append(exec_result)
        logger.debug("step %d %s ok=%s", step_idx, module_id, is_ok)

        # Track snapshot for selector resolution
        if module_id in ("browser.snapshot", "browser.goto") and is_ok:
            last_snapshot = result

        # Stop on failure (except snapshot failures)
        if not is_ok and module_id != "browser.snapshot":
            break

    # Build summary
    ok_count = sum(1 for r in results if r["ok"])
    fail_count = sum(1 for r in results if not r["ok"])
    summary = "Executed {} steps: {} ok, {} failed.".format(
        len(results), ok_count, fail_count,
    )

    return results, summary


def _resolve_url(text: str) -> Optional[str]:
    """Resolve a site name or URL to a full URL. No hardcoded site list."""
    text = text.strip().rstrip("。.，,")

    if text.startswith("http://") or text.startswith("https://"):
        return text

    if "." in text:
        return "https://" + text

    # Treat as site name → add .com (LLM already resolved the name)
    if text and len(text) > 1:
        return "https://www.{}.com".format(text)

    return None


def _find_selector_from_structured(snapshot_result: dict, hint: str, element_type: str = "") -> Optional[str]:
    """Find a CSS selector from structured snapshot data (inputs, buttons, links).

    The snapshot returns structured arrays:
      inputs: [{"selector": "input[name=search_query]", "label": "Search", ...}]
      buttons: [{"selector": "[data-flyto-hint=7]", "text": "Search", ...}]
      links: [{"selector": "a#video-title", "text": "Video Title", ...}]

    This is 100x more reliable than parsing text.
    """
    if not hint or not snapshot_result:
        return None

    hint_lower = hint.lower()

    # Determine which element arrays to search
    search_in = []
    if element_type == "input" or "input" in hint_lower or "search" in hint_lower:
        search_in.append(("inputs", snapshot_result.get("inputs", [])))
    if element_type == "button" or "button" in hint_lower or "search" in hint_lower:
        search_in.append(("buttons", snapshot_result.get("buttons", [])))
    if element_type == "link" or "click" in hint_lower or "play" in hint_lower:
        search_in.append(("links", snapshot_result.get("links", [])))

    # If no specific type, search all
    if not search_in:
        search_in = [
            ("inputs", snapshot_result.get("inputs", [])),
            ("buttons", snapshot_result.get("buttons", [])),
            ("links", snapshot_result.get("links", [])),
        ]

    # Handle ordinal hints (第一個, first, 1 etc.) — pick by index
    ordinal_map = {"第一": 0, "第二": 1, "第三": 2, "第四": 3, "第五": 4,
                   "first": 0, "second": 1, "third": 2, "1": 0, "2": 1, "3": 2}
    for word, idx in ordinal_map.items():
        if word in hint_lower:
            # Pick the Nth visible link/button
            for _, elements in search_in:
                visible = [el for el in elements
                           if el.get("selector") and el.get("rect", {}).get("top", -999) > 0]
                if idx < len(visible):
                    return visible[idx].get("selector")
            break

    best_selector = None
    best_score = 0

    for source_type, elements in search_in:
        for el in elements:
            selector = el.get("selector", "")
            if not selector:
                continue

            # Skip offscreen elements
            rect = el.get("rect", {})
            if rect.get("top", 0) < -500:
                continue

            # Score by matching hint against element attributes
            score = 0
            text = (el.get("text", "") or "").lower()
            label = (el.get("label", "") or "").lower()
            name = (el.get("name", "") or "").lower()
            placeholder = (el.get("placeholder", "") or "").lower()
            el_id = (el.get("id", "") or "").lower()

            searchable = " ".join([text, label, name, placeholder, el_id])

            for word in hint_lower.split():
                if word in searchable:
                    score += 10
                if word in selector.lower():
                    score += 5

            # Bonus for exact matches
            if hint_lower == text or hint_lower == label:
                score += 50
            if hint_lower in text or hint_lower in label:
                score += 20

            # Type-specific bonuses
            if source_type == "inputs" and ("input" in hint_lower or "search" in hint_lower or "type" in hint_lower):
                score += 5
            if source_type == "buttons" and ("button" in hint_lower or "submit" in hint_lower):
                score += 5

            if score > best_score:
                best_score = score
                best_selector = selector

    return best_selector
