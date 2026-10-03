# Copyright 2024 Flyto2
# Licensed under the Apache License, Version 2.0
"""Browser page inspection tool — extracts interactive elements."""
import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)

INSPECT_PAGE_TOOL = {
    "name": "inspect_page",
    "description": (
        "Navigate to a URL and return a compact list "
        "of interactive elements (inputs, buttons, links, selects, textareas) with "
        "their tag, id, class, text, placeholder, aria-label, href, type, and name. "
        "Use this BEFORE generating browser workflow YAML so you can pick correct "
        "selectors from real page structure instead of guessing. Inside a task "
        "that already has a browser, the page opens in a new tab of that browser "
        "(so signed-in pages are visible) and the original tab is restored; "
        "otherwise a headless browser is launched and closed."
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "URL to inspect (must start with http:// or https://)",
            },
            "wait_ms": {
                "type": "number",
                "description": (
                    "Upper bound in ms for dynamic content to settle after page load. "
                    "Inspection starts as soon as the DOM stops changing, so this is "
                    "only reached on pages that keep mutating (default 2000, max 5000)"
                ),
                "default": 2000,
            },
            "browser_channel": {
                "type": "string",
                "enum": ["auto", "chromium", "chrome", "msedge"],
                "description": (
                    "Browser engine selection. 'auto' first tries Core's bundled "
                    "Chromium and then the installed Google Chrome (default auto)."
                ),
                "default": "auto",
            },
        },
        "required": ["url"],
    },
}

_BROWSER_CHANNELS = ("auto", "chromium", "chrome", "msedge")

# Upper bound for the settle, kept from the old fixed wait's cap so a model
# asking for more cannot stall a task.
_MAX_SETTLE_MS = 5000
# How long the DOM must stay unchanged before it counts as rendered. Short
# enough that a static page is inspected almost at once, long enough to span
# the gap between a framework's first paint and its data-driven re-render.
_QUIET_MS = 200

# Waits for the page's own state instead of a clock: a MutationObserver
# resolves once the document has been quiet for QUIET ms. CAP only bounds a
# page that never stops changing (tickers, animations); reaching it is still a
# settled page, so extraction runs and no error is reported.
_SETTLE_JS = """
  const settle = (quietMs, capMs) => new Promise((resolve) => {
    if (capMs <= 0) { resolve({ by: 'skipped', ms: 0 }); return; }
    const started = performance.now();
    let quietTimer = null;
    let capTimer = null;
    let observer = null;
    const done = (by) => {
      if (observer) observer.disconnect();
      clearTimeout(quietTimer);
      clearTimeout(capTimer);
      resolve({ by, ms: Math.round(performance.now() - started) });
    };
    const armQuiet = () => {
      clearTimeout(quietTimer);
      quietTimer = setTimeout(() => done('quiet'), Math.min(quietMs, capMs));
    };
    observer = new MutationObserver(armQuiet);
    observer.observe(document.documentElement || document, {
      childList: true, subtree: true, attributes: true, characterData: true
    });
    armQuiet();
    capTimer = setTimeout(() => done('cap'), capMs);
  });
"""

_INSPECT_JS = """
(async () => {
  const QUIET = __QUIET_MS__;
  const CAP = __CAP_MS__;
""" + _SETTLE_JS + """
  const settled = await settle(QUIET, CAP);
  const MAX = 80;
  const seen = new Set();
  const results = [];
  const els = document.querySelectorAll(
    'input, button, a, select, textarea, [role="button"], [role="link"], ' +
    '[role="tab"], [role="menuitem"], [role="search"], [contenteditable="true"]'
  );
  for (const el of els) {
    if (results.length >= MAX) break;
    const tag = el.tagName.toLowerCase();
    const id = el.id || undefined;
    const cls = el.className
      ? String(el.className).split(/\\s+/).slice(0, 3).join(' ')
      : undefined;
    const name = el.getAttribute('name') || undefined;
    const type = el.getAttribute('type') || undefined;
    const placeholder = el.getAttribute('placeholder') || undefined;
    const ariaLabel = el.getAttribute('aria-label') || undefined;
    const role = el.getAttribute('role') || undefined;
    const href = tag === 'a' ? (el.getAttribute('href') || '').slice(0, 120) : undefined;
    const text = (el.textContent || '').trim().slice(0, 60) || undefined;

    const key = `${tag}|${id || ''}|${name || ''}|${text || ''}`;
    if (seen.has(key)) continue;
    seen.add(key);

    const entry = { tag };
    if (id) entry.id = id;
    if (cls) entry.class = cls;
    if (name) entry.name = name;
    if (type) entry.type = type;
    if (placeholder) entry.placeholder = placeholder;
    if (ariaLabel) entry.ariaLabel = ariaLabel;
    if (role) entry.role = role;
    if (href) entry.href = href;
    if (text && text !== id) entry.text = text;
    results.push(entry);
  }
  return {
    url: location.href,
    title: document.title,
    count: results.length,
    elements: results,
    settle: settled
  };
})()
"""


def _inspect_script(wait_ms: Any) -> str:
    """The inspection script, settling for at most ``wait_ms`` (capped)."""
    try:
        cap = int(wait_ms)
    except (TypeError, ValueError):
        cap = 2000
    cap = max(0, min(cap, _MAX_SETTLE_MS))
    return (
        _INSPECT_JS
        .replace("__QUIET_MS__", str(_QUIET_MS))
        .replace("__CAP_MS__", str(cap))
    )


def _page_data(eval_result: Dict[str, Any]) -> Dict[str, Any]:
    data = eval_result.get("data", {})
    page_data = data.get("result", {}) if isinstance(data, dict) else {}
    return page_data or eval_result.get("result", {})


def _navigated_away(eval_result: Any) -> bool:
    """A redirect during the settle destroys the script's document."""
    if not isinstance(eval_result, dict):
        return False
    error = str(eval_result.get("error", "")).lower()
    return "context was destroyed" in error or "navigat" in error


async def _evaluate_settled(execute: Any, wait_ms: Any, **call: Any) -> Any:
    """Settle and extract; re-run once on the document a redirect landed on."""
    script = _inspect_script(wait_ms)
    result = await execute(module_id="browser.evaluate", params={"script": script}, **call)
    if _navigated_away(result):
        result = await execute(module_id="browser.evaluate", params={"script": script}, **call)
    return result


def _reusable_task_sessions() -> Dict[str, Any] | None:
    """The caller's own browser registry when it holds exactly one browser.

    A Space task or scoped chat runs inside a browser_session_scope; its
    browser carries the operator's signed-in state. The scope is caller
    authority, never a model parameter, and an ambiguous registry (several
    browsers) or a scope already shutting down is not reused.
    """
    from flyto_ai.tools.browser_scope import current_browser_scope

    scope = current_browser_scope()
    if scope is None or scope.closing or scope.closed or len(scope.sessions) != 1:
        return None
    return scope.sessions


async def _inspect_in_task_browser(
    execute: Any,
    sessions: Dict[str, Any],
    url: str,
    wait_ms: Any,
) -> Dict[str, Any] | None:
    """Inspect in a new tab of the task's browser, then restore its tab.

    Returns None when the tab could not be opened, so the caller can fall
    back to a cold launch: nothing in the task browser changed by then.
    """
    from flyto_ai.tools.core_tools import _is_ok

    session_id = next(iter(sessions))
    call = {"context": {"browser_session": session_id}, "browser_sessions": sessions}
    listed = await execute(module_id="browser.tab", params={"action": "list"}, **call)
    if not _is_ok(listed):
        return None
    original_index = listed.get("current_index")
    if not isinstance(original_index, int):
        original_index = (listed.get("data") or {}).get("current_index")
    opened = await execute(module_id="browser.tab", params={"action": "new", "url": url}, **call)
    if not _is_ok(opened):
        if isinstance(opened, dict) and opened.get("error_code") == "SSRF_BLOCKED":
            return {"ok": False, "error": "Failed to navigate: {}".format(opened.get("error", "blocked"))}
        return None
    try:
        eval_result = await _evaluate_settled(execute, wait_ms, **call)
        if not _is_ok(eval_result):
            return {"ok": False, "error": "Failed to inspect: {}".format(
                eval_result.get("error", "unknown")
            )}
        return {"ok": True, "data": _page_data(eval_result), "browser_reused": True}
    finally:
        # Leave the task exactly where it was: close only the tab opened here
        # and put the operator's page back in front.
        try:
            await execute(module_id="browser.tab", params={"action": "close"}, **call)
            if isinstance(original_index, int) and original_index >= 0:
                await execute(
                    module_id="browser.tab",
                    params={"action": "switch", "index": original_index},
                    **call,
                )
        except Exception as exc:
            logger.debug("inspect_page tab restore failed: %s", exc)


async def _launch_browser(
    execute: Any,
    sessions: Dict[str, Any],
    browser_channel: str,
) -> tuple[str | None, str | None]:
    """Launch through Core and return the selected channel plus a bounded error."""
    from flyto_ai.tools.core_tools import _is_ok

    attempts = (
        (("chromium", None), ("chrome", "chrome"))
        if browser_channel == "auto"
        else ((browser_channel, None if browser_channel == "chromium" else browser_channel),)
    )
    errors = []
    for index, (label, channel) in enumerate(attempts):
        params = {"headless": True}
        if channel:
            params["channel"] = channel
        result = await execute(
            module_id="browser.launch",
            params=params,
            browser_sessions=sessions,
        )
        if _is_ok(result):
            return label, None
        detail = result.get("error", "unknown") if isinstance(result, dict) else "unknown"
        errors.append("{}: {}".format(label, detail))
        if index + 1 < len(attempts):
            try:
                await execute(
                    module_id="browser.close",
                    params={},
                    browser_sessions=sessions,
                )
            except Exception as exc:
                logger.debug("inspect_page retry cleanup failed: %s", exc)
            sessions.clear()
    return None, "; ".join(errors)[:1000]


async def inspect_page(
    url: str,
    wait_ms: int = 2000,
    browser_channel: str = "auto",
) -> Dict[str, Any]:
    """Go to URL and extract interactive elements once the DOM has settled.

    Reuses the caller's scoped task browser in a new tab when there is one;
    otherwise launches a headless browser and closes it afterwards.
    """
    from flyto_ai.prompt.policies import is_safe_url
    from flyto_ai.tools.core_tools import _get_mcp_handler, _is_ok

    if not is_safe_url(url):
        return {"ok": False, "error": "URL blocked by SSRF policy: {}".format(url[:100])}
    if browser_channel not in _BROWSER_CHANNELS:
        return {
            "ok": False,
            "error": "browser_channel must be one of: {}".format(", ".join(_BROWSER_CHANNELS)),
        }

    handler = _get_mcp_handler()
    if not handler:
        return {"ok": False, "error": "flyto-core not installed. Run: pip install flyto-core"}
    execute = handler["execute_module"]

    task_sessions = _reusable_task_sessions()
    if task_sessions is not None:
        try:
            reused = await _inspect_in_task_browser(execute, task_sessions, url, wait_ms)
        except Exception as e:
            logger.warning("inspect_page in task browser failed: %s", e)
            reused = None
        if reused is not None:
            return reused

    sessions: Dict[str, Any] = {}

    try:
        selected_channel, launch_error = await _launch_browser(
            execute,
            sessions,
            browser_channel,
        )
        if not selected_channel:
            return {"ok": False, "error": "Failed to launch browser: {}".format(launch_error)}

        goto_result = await execute(
            module_id="browser.goto",
            params={"url": url, "wait_until": "domcontentloaded"},
            browser_sessions=sessions,
        )
        if not _is_ok(goto_result):
            return {"ok": False, "error": "Failed to navigate: {}".format(
                goto_result.get("error", "unknown")
            )}

        # The settle runs inside the extraction script, so the page's own DOM
        # state (not a fixed sleep) decides when the elements are read.
        eval_result = await _evaluate_settled(execute, wait_ms, browser_sessions=sessions)

        if not _is_ok(eval_result):
            return {"ok": False, "error": "Failed to inspect: {}".format(
                eval_result.get("error", "unknown")
            )}

        return {"ok": True, "data": _page_data(eval_result), "browser_channel": selected_channel}

    except Exception as e:
        logger.warning("inspect_page failed: %s", e)
        return {"ok": False, "error": str(e)}

    finally:
        try:
            await execute(
                module_id="browser.close",
                params={},
                browser_sessions=sessions,
            )
        except Exception as e:
            logger.debug("inspect_page browser cleanup failed: %s", e)
        finally:
            sessions.clear()


async def dispatch_inspect_page(name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Dispatch handler for the inspect_page tool."""
    return await inspect_page(
        url=arguments.get("url", ""),
        wait_ms=arguments.get("wait_ms", 2000),
        browser_channel=arguments.get("browser_channel", "auto"),
    )
