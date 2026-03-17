# Copyright 2024 Flyto
# Licensed under the Apache License, Version 2.0
"""Output intent tracker — detect 'save to X' and auto-write if LLM forgets.

When the user says "save to /tmp/foo.json", the system:
1. Extracts the target path from the message
2. Tracks whether file.write was called with that path
3. After execution, if file wasn't written, auto-writes the last result

This is fully system-level — no prompt changes needed.
"""
import json
import logging
import re
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# Patterns that indicate the user wants output saved to a file
_SAVE_PATTERNS = [
    r'save\s+(?:it\s+|the\s+|results?\s+)?to\s+(\S+)',
    r'write\s+(?:it\s+|the\s+|results?\s+)?to\s+(\S+)',
    r'output\s+(?:it\s+|the\s+)?to\s+(\S+)',
    r'store\s+(?:it\s+|the\s+)?(?:in|to)\s+(\S+)',
    r'(?:存|寫|輸出|儲存)(?:到|至|入)\s*(\S+)',
]


def extract_output_paths(message: str) -> List[str]:
    """Extract intended output file paths from user message."""
    paths = []
    # Pattern-based extraction
    for pattern in _SAVE_PATTERNS:
        for match in re.finditer(pattern, message, re.IGNORECASE):
            path = match.group(1).strip('.,;:!?"\')]}')
            if '/' in path or '.' in path.split('/')[-1]:
                paths.append(path)
    # Also find any bare file paths in the message (e.g. after "and")
    for match in re.finditer(r'(?:^|\s)(/\S+\.\w{1,5})(?:\s|$|[,.])', message):
        path = match.group(1).strip('.,;:!?')
        if path not in paths:
            paths.append(path)
    return list(dict.fromkeys(paths))


class OutputTracker:
    """Track whether intended output files were actually written."""

    def __init__(self, intended_paths: List[str]) -> None:
        self._intended = set(intended_paths)
        self._written: set = set()

    def on_tool_call(self, func_name: str, func_args: dict, result: dict) -> None:
        """Track file.write calls."""
        if func_name != "execute_module":
            return
        if func_args.get("module_id") != "file.write":
            return
        if not result.get("ok"):
            return
        path = func_args.get("params", {}).get("path", "")
        if path:
            self._written.add(path)

    def get_missing(self) -> List[str]:
        """Return paths that were intended but not written."""
        return [p for p in self._intended if p not in self._written]

    async def auto_write_missing(
        self,
        dispatch: Callable,
        exec_results: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Auto-write missing output files from the last execution result.

        Returns list of auto-write results.
        """
        missing = self.get_missing()
        if not missing:
            return []

        # Find the best content to write from execution results
        content = _extract_best_content(exec_results)
        if not content:
            return []

        results = []
        for path in missing:
            try:
                # Determine format from extension
                if path.endswith('.json'):
                    if isinstance(content, (dict, list)):
                        write_content = json.dumps(content, ensure_ascii=False, indent=2)
                    else:
                        write_content = str(content)
                else:
                    write_content = str(content) if not isinstance(content, str) else content

                r = await dispatch("execute_module", {
                    "module_id": "file.write",
                    "params": {"path": path, "content": write_content},
                })
                if r.get("ok"):
                    logger.info("Auto-saved missing output: %s", path)
                    r["_auto_saved"] = True
                results.append(r)
            except Exception as e:
                logger.debug("Auto-save failed for %s: %s", path, e)

        return results


def _extract_best_content(exec_results: List[Dict[str, Any]]) -> Any:
    """Extract the most meaningful content from execution results."""
    # Walk backwards — last successful result is usually the final output
    for r in reversed(exec_results):
        if not r.get("ok"):
            continue
        preview = r.get("result_preview", "")
        if not preview:
            continue

        try:
            data = json.loads(preview) if isinstance(preview, str) else preview
            if isinstance(data, dict):
                # Skip status-only results
                if set(data.keys()) <= {"status", "message", "ok"}:
                    continue
                # Return the meaningful part
                for key in ("result", "data", "content", "text", "output", "items"):
                    if key in data and data[key]:
                        return data[key]
                # Has other fields beyond status — return whole dict
                meaningful = {k: v for k, v in data.items()
                              if k not in ("status", "ok", "message") and v}
                if meaningful:
                    return meaningful
            elif isinstance(data, (list, str)) and data:
                return data
        except (ValueError, TypeError):
            if isinstance(preview, str) and len(preview) > 5:
                return preview

    return None
