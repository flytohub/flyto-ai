# Copyright 2024 Flyto
# Licensed under the Apache License, Version 2.0
"""Resilience layer — snapshot-before-interact guard + selector healing.

Two system-level enforcements:
1. **Snapshot guard**: if browser.type/click/extract is called without a prior
   snapshot, auto-inject one so the LLM sees real selectors.
2. **Selector healing**: if a browser action fails because the selector wasn't
   found, take a snapshot, search for the closest match, and retry.
"""
import json
import logging
import re
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

_FAIL_HINTS = ("not found", "no element", "timeout", "waiting for selector", "failed to find")
_INTERACT_MODULES = frozenset(("browser.click", "browser.type", "browser.extract",
                                "browser.wait", "browser.find", "browser.form", "browser.select"))
_SNAPSHOT_MODULES = frozenset(("browser.snapshot", "browser.extract"))

# Anti-bot detection signals in page content
_ANTIBOT_SIGNALS = (
    "javascript is not available",
    "please enable javascript",
    "enable javascript to continue",
    "browser doesn't support javascript",
    "you need to enable javascript",
    "this site requires javascript",
)


def should_heal(func_name: str, func_args: dict, result: dict) -> bool:
    """Check if a failed browser module call is a candidate for selector healing."""
    if func_name != "execute_module":
        return False
    if func_args.get("module_id", "") not in _INTERACT_MODULES:
        return False
    if result.get("ok", True):
        return False
    error_msg = str(result.get("error", "")).lower()
    return any(hint in error_msg for hint in _FAIL_HINTS)


class SnapshotGuard:
    """Track whether a snapshot has been taken before interaction.

    If browser.type/click is called without a prior snapshot,
    auto-inject one so the result goes back to the LLM as context.
    """

    def __init__(self) -> None:
        self._has_snapshot = False
        self._last_snapshot: Optional[Dict[str, Any]] = None

    def on_tool_call(self, func_name: str, func_args: dict) -> bool:
        """Track snapshot state. Returns True if a snapshot was just recorded."""
        if func_name != "execute_module":
            return False
        module_id = func_args.get("module_id", "")
        if module_id in _SNAPSHOT_MODULES:
            self._has_snapshot = True
            return True
        if module_id in ("browser.goto", "browser.launch"):
            # Navigation resets snapshot state
            self._has_snapshot = False
        return False

    def needs_snapshot(self, func_name: str, func_args: dict) -> bool:
        """Check if a snapshot should be auto-injected before this call."""
        if func_name != "execute_module":
            return False
        module_id = func_args.get("module_id", "")
        if module_id not in _INTERACT_MODULES:
            return False
        return not self._has_snapshot

    def record_snapshot(self, result: Dict[str, Any]) -> None:
        """Store the latest snapshot result for healing reference."""
        self._has_snapshot = True
        self._last_snapshot = result


class AntibotGuard:
    """Detect anti-bot blocking after browser.goto and auto-retry with system Chrome.

    Flow:
    1. browser.goto returns success
    2. Check page content for anti-bot signals ("JavaScript is not available", etc.)
    3. If detected → close browser → re-launch with channel='chrome' → re-navigate
    4. Only downgrades ONCE per session (not infinite loop)
    """

    def __init__(self) -> None:
        self._downgraded = False  # only try once
        self._last_url: Optional[str] = None

    def check_result(self, func_name: str, func_args: dict, result: dict) -> bool:
        """Check if a goto result shows anti-bot blocking. Returns True if blocked."""
        if self._downgraded:
            return False  # already tried system chrome
        if func_name != "execute_module":
            return False
        if func_args.get("module_id") != "browser.goto":
            return False
        if not result.get("ok", False):
            return False

        # Check page text for anti-bot signals
        text = str(result.get("text", "")).lower()
        content = str(result.get("content", "")).lower()
        check_text = text + " " + content

        if any(signal in check_text for signal in _ANTIBOT_SIGNALS):
            self._last_url = func_args.get("params", {}).get("url", "")
            return True

        # Also check: page has almost no interactive elements (JS didn't render)
        links = result.get("links", [])
        if isinstance(links, list) and len(links) <= 1 and len(text) < 200:
            # Very sparse page — might be blocked
            if "enable" in check_text or "javascript" in check_text:
                self._last_url = func_args.get("params", {}).get("url", "")
                return True

        return False

    async def retry_with_system_chrome(self, dispatch: Callable, url: str) -> Optional[Dict[str, Any]]:
        """Close browser, re-launch with system Chrome, re-navigate."""
        self._downgraded = True
        logger.info("Anti-bot detected — retrying with system Chrome (channel='chrome')")

        try:
            # Close current browser
            await dispatch("execute_module", {
                "module_id": "browser.close",
                "params": {},
            })
        except Exception:
            pass

        # Re-launch with system Chrome
        launch_result = await dispatch("execute_module", {
            "module_id": "browser.launch",
            "params": {"headless": False, "channel": "chrome"},
        })
        if not launch_result.get("ok", False):
            logger.warning("System Chrome launch failed: %s", launch_result.get("error"))
            return None

        # Re-navigate
        goto_result = await dispatch("execute_module", {
            "module_id": "browser.goto",
            "params": {"url": url},
        })

        if goto_result.get("ok", False):
            goto_result["_antibot_retry"] = True
            goto_result["message"] = (
                "Anti-bot protection detected. Automatically retried with system Chrome. "
                "Page loaded successfully. Continue with your task."
            )
            logger.info("Anti-bot bypass successful with system Chrome")

        return goto_result


async def try_heal(dispatch: Callable, original_args: dict) -> Optional[Dict[str, Any]]:
    """Attempt to heal a broken selector using snapshot data.

    Strategy cascade:
    1. Take snapshot to see current page state
    2. Extract interactive elements from snapshot (inputs, buttons, links)
    3. Match broken selector intent to actual elements
    4. Retry with the best match
    """
    try:
        params = original_args.get("params", {})
        broken_selector = params.get("selector", "")
        if not broken_selector:
            return None

        module_id = original_args.get("module_id", "")

        # Step 1: Take snapshot
        snap_result = await dispatch("execute_module", {
            "module_id": "browser.snapshot",
            "params": {},
        })
        if not snap_result.get("ok", False):
            return None

        # Step 2: Extract available selectors from snapshot
        snapshot_data = snap_result.get("result", snap_result)
        if isinstance(snapshot_data, str):
            try:
                snapshot_data = json.loads(snapshot_data)
            except (json.JSONDecodeError, TypeError):
                snapshot_data = {}

        # Collect candidate selectors from snapshot hints
        candidates = _extract_candidates(snapshot_data, module_id)

        # Step 3: Find best match for the broken selector
        healed_selector = _find_best_match(broken_selector, candidates)

        # Step 4: Fallback — strip parent specificity
        if not healed_selector and (">" in broken_selector or " " in broken_selector):
            parts = broken_selector.replace(">", " ").split()
            last_part = parts[-1].strip()
            if last_part and last_part != broken_selector:
                healed_selector = last_part

        if not healed_selector:
            return None

        # Step 5: Retry with healed selector
        new_params = dict(params)
        new_params["selector"] = healed_selector
        retry_result = await dispatch("execute_module", {
            "module_id": module_id,
            "params": new_params,
        })
        if retry_result.get("ok", False):
            logger.info("Selector healed: %s -> %s", broken_selector, healed_selector)
            retry_result["_healed_selector"] = {
                "original": broken_selector,
                "healed": healed_selector,
            }
            return retry_result

        return None
    except Exception as e:
        logger.debug("Selector healing failed: %s", e)
        return None


def _extract_candidates(snapshot: Dict[str, Any], module_id: str) -> List[Dict[str, str]]:
    """Extract candidate selectors from snapshot interactive hints."""
    candidates = []

    # Snapshot may have 'inputs', 'buttons', 'links' from _extract_interactive_hints
    for key in ("inputs", "buttons", "links", "selects"):
        items = snapshot.get(key, [])
        if isinstance(items, list):
            for item in items:
                if isinstance(item, dict):
                    sel = item.get("selector", item.get("hint", ""))
                    label = item.get("text", item.get("label", item.get("placeholder", "")))
                    itype = item.get("type", "")
                    if sel:
                        candidates.append({"selector": sel, "label": label, "type": itype})

    # Also try parsing from text content if no structured hints
    text = snapshot.get("text", snapshot.get("content", ""))
    if isinstance(text, str) and not candidates:
        # Extract data-flyto-hint selectors
        for match in re.finditer(r'\[data-flyto-hint="(\d+)"\]', text):
            candidates.append({"selector": '[data-flyto-hint="{}"]'.format(match.group(1)), "label": "", "type": ""})

    return candidates


def _find_best_match(broken_selector: str, candidates: List[Dict[str, str]]) -> Optional[str]:
    """Find the best matching selector from candidates for a broken one.

    Matching heuristics:
    1. Same element type (input, button, a)
    2. Same ID fragment
    3. Same class fragment
    4. Label/text similarity
    """
    if not candidates:
        return None

    broken_lower = broken_selector.lower()

    # Extract hints from broken selector
    element_type = ""
    for tag in ("input", "button", "a", "select", "textarea", "div", "span"):
        if broken_lower.startswith(tag) or ">" + tag in broken_lower or " " + tag in broken_lower:
            element_type = tag
            break

    id_match = re.search(r'#([\w-]+)', broken_selector)
    broken_id = id_match.group(1).lower() if id_match else ""

    class_match = re.search(r'\.([\w-]+)', broken_selector)
    broken_class = class_match.group(1).lower() if class_match else ""

    # Score each candidate
    scored = []
    for cand in candidates:
        sel = cand["selector"]
        label = cand.get("label", "").lower()
        score = 0

        # Element type match
        if element_type and element_type in sel.lower():
            score += 3

        # ID fragment match
        if broken_id and broken_id in sel.lower():
            score += 5

        # Class fragment match
        if broken_class and broken_class in sel.lower():
            score += 2

        # Label contains ID-like text
        if broken_id and broken_id in label:
            score += 3

        # Common keyword matching (login, submit, password, username, email)
        for keyword in ("login", "submit", "password", "username", "email", "account", "signin"):
            if keyword in broken_lower and keyword in (sel.lower() + " " + label):
                score += 4

        if score > 0:
            scored.append((score, sel))

    if not scored:
        return None

    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[0][1]
