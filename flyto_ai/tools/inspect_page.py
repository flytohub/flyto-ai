# Copyright 2024 Flyto2
# Licensed under the Apache License, Version 2.0
"""Browser page inspection tool — extracts interactive elements."""
import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)

INSPECT_PAGE_TOOL = {
    "name": "inspect_page",
    "description": (
        "Launch a headless browser, navigate to a URL, and return a compact list "
        "of interactive elements (inputs, buttons, links, selects, textareas) with "
        "their tag, id, class, text, placeholder, aria-label, href, type, and name. "
        "Use this BEFORE generating browser workflow YAML so you can pick correct "
        "selectors from real page structure instead of guessing."
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
                "description": "Wait time in ms after page load for dynamic content (default 2000)",
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

_INSPECT_JS = """
(() => {
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
    elements: results
  };
})()
"""


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
    """Launch browser, go to URL, extract interactive elements, close browser."""
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

        if wait_ms > 0:
            await execute(
                module_id="browser.wait",
                params={"duration_ms": min(wait_ms, 5000)},
                browser_sessions=sessions,
            )

        eval_result = await execute(
            module_id="browser.evaluate",
            params={"script": _INSPECT_JS},
            browser_sessions=sessions,
        )

        if not _is_ok(eval_result):
            return {"ok": False, "error": "Failed to inspect: {}".format(
                eval_result.get("error", "unknown")
            )}

        page_data = eval_result.get("data", {}).get("result", {})
        if not page_data:
            page_data = eval_result.get("result", {})

        return {"ok": True, "data": page_data, "browser_channel": selected_channel}

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
