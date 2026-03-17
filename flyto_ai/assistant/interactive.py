# Copyright 2024 Flyto
# Licensed under the Apache License, Version 2.0
"""Interactive input detection — extract pending_input from ask_user results."""
import json
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

ASK_USER_MARKER = "__ASK_USER__"


def extract_pending_input(tool_calls: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Scan tool_calls for ASK_USER marker in any tool's result.

    Works for both direct ask_user calls and tools that internally
    trigger ask_user (like navigate_website).
    """
    for tc in tool_calls:
        result = tc.get("result", {})
        if not isinstance(result, dict):
            try:
                result = json.loads(tc.get("result_preview", "{}"))
            except (json.JSONDecodeError, TypeError):
                result = {}

        if result.get(ASK_USER_MARKER):
            return {
                "question": result.get("question", ""),
                "fields": result.get("fields", []),
                "context_key": result.get("context_key", ""),
            }

    return None
