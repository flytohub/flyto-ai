"""The verbs that mean "do it", in the languages this product is asked in.

Lifted out of ``planner.py`` because it is the one part of that module that
grows: a verb table over ten languages is an open set, and every gap in it is
a silent failure -- an instruction classified as conversation, answered in
prose, with nothing on screen saying the instruction was not recognised.
"登入kintone" sat in that gap, and the operator had to discover by trial that
"執行 kintone 工作流" worked instead.

Keeping the tables here means adding a verb is a change to a data file rather
than to a safety boundary, and the boundary's own line budget stops paying for
vocabulary it cannot help accumulating.

The classification RULES stay in planner.py. This file only says which words
count.
"""

import re

_EN_ACTION_RE = re.compile(
    r"^\s*(?:(?:please|kindly)\s+|(?:can|could|would)\s+you\s+|"
    r"help\s+me(?:\s+to)?\s+)?"
    r"(?:open|visit|go\s+to|search(?:\s+for)?|run|execute|click|download|"
    r"upload|create|update|delete|remove|fix|repair|push|deploy|send|"
    r"take\s+(?:a\s+)?screenshot|repeat|rewrite|fetch|find|check|write|"
    r"install|commit|rerun|build|apply|read|summari[sz]e|analy[sz]e|"
    r"inspect|list|scrape|extract|save|tell|reuse|convert|import|"
    # Signing in is an action, and its absence here made the most natural way
    # to ask for one -- "log into kintone" -- classify as answer_only: the
    # assistant described the login instead of performing it, and the operator
    # had to rephrase into something like "run the kintone workflow" with
    # nothing explaining why the first wording did nothing.
    r"log\s*in|log\s*into|log\s*on|sign\s*in|sign\s*into|sign\s*on|"
    r"authenticate|connect\s+to|start|launch|trigger|invoke|call|"
    r"restart|stop|cancel|schedule|enable|disable|turn\s+(?:on|off))\b",
    re.IGNORECASE,
)
_CJK_ACTION_RE = re.compile(
    r"^\s*(?:請|请|麻煩|麻烦|幫我|帮我|替我|可以幫我|可以帮我)?\s*"
    r"(?:打開|打开|開啟|开启|前往|搜尋|搜索|查詢|查询|執行|执行|運行|运行|"
    r"點擊|点击|下載|下载|上傳|上传|建立|創建|创建|更新|刪除|删除|修復|修复|"
    r"部署|推送|上去|截圖|截图|重複|重复|重新執行|重新执行|修改|改寫|改写|"
    r"重寫|重写|抓取|尋找|查找|找出|檢查|检查|寫入|写入|安裝|安装|提交|"
    r"讀取|读取|分析|列出|套用|儲存|储存|摘要|截|"
    # Access verbs. "登入kintone" is as plain an instruction as this product
    # receives, and it classified as answer_only because none of these were
    # here -- so the assistant explained the login rather than running it.
    r"登入|登录|登陸|登陆|簽入|签入|連線|连线|連接|连接|"
    r"啟動|启动|開始|开始|觸發|触发|呼叫|调用|呼叫|叫用|"
    r"重啟|重启|停止|取消|排程|排定|啟用|启用|停用|關閉|关闭)"
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


#: Bare agreement. On its own it carries no action -- "yes" is not an
#: instruction -- but immediately after the assistant proposed a specific
#: action it IS one, and classifying it as conversation is what produced the
#: loop the owner hit: the assistant asked whether to run a workflow, "確認"
#: was read as small talk so no tool was exposed, and the only reply it could
#: give was to ask again. Six rounds, no execution.
#:
#: Deliberately narrow. It must not swallow a sentence that agrees with
#: something and then asks for something else -- "好，那先看一下狀態" is a new
#: request, not a confirmation -- so this matches a short utterance that is
#: agreement and nothing more.
_AFFIRMATION_RE = re.compile(
    r"^\s*(?:"
    r"yes|yep|yeah|yup|ok|okay|sure|please|do\s+it|go\s+ahead|confirm(?:ed)?|"
    r"proceed|continue|affirmative|"
    r"是|是的|對|对|好|好的|好啊|可以|沒錯|没错|確認|确认|確定|确定|"
    r"執行|执行|繼續|继续|同意|批准|授權|授权|"
    r"はい|お願いします|"
    r"예|네|"
    r"s[ií]|vale|d'accord|ja|sim"
    r")\s*[.!。！]?\s*$",
    re.IGNORECASE,
)


def is_bare_affirmation(message: str) -> bool:
    """Whether this message is agreement and nothing else."""
    return bool(_AFFIRMATION_RE.match((message or "").strip()))
