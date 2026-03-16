# Copyright 2024 Flyto
# Licensed under the Apache License, Version 2.0
"""Interactive input detection — extract pending_input from ask_user results."""
import json
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

ASK_USER_MARKER = "__ASK_USER__"


def extract_pending_input(tool_calls: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Scan tool_calls for an ask_user result with the ASK_USER marker.

    Returns the pending_input dict if found, None otherwise.
    """
    for tc in tool_calls:
        if tc.get("function") != "ask_user":
            continue

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
