# Copyright 2024 Flyto
# Licensed under the Apache License, Version 2.0
"""YAML recipe verification engine.

Runs flyto-core recipes (browser screenshot, text extraction, etc.) and
optionally compares results against a reference image via multimodal LLM.
"""
import base64
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional

from flyto_ai.agents.models import VerificationResult

logger = logging.getLogger(__name__)

# Lazy-cached workflow engine
_cached_engine_cls: Any = None
_engine_checked = False


def _get_engine_cls() -> Any:
    """Lazily import flyto-core WorkflowEngine."""
    global _cached_engine_cls, _engine_checked
    if _engine_checked:
        return _cached_engine_cls
    _engine_checked = True
    try:
        from core.engine.workflow.engine import WorkflowEngine
        _cached_engine_cls = WorkflowEngine
        logger.debug("flyto-core WorkflowEngine loaded")
    except ImportError:
        _cached_engine_cls = None
        logger.debug("flyto-core not installed — verification engine disabled")
    return _cached_engine_cls


class VerificationEngine:
    """Run a flyto-core recipe and optionally compare screenshots."""

    def __init__(self, timeout: int = 120):
        self._timeout = timeout

    async def verify(
        self,
        recipe: str,
        args: Optional[Dict[str, Any]] = None,
        reference: Optional[str] = None,
    ) -> VerificationResult:
        """Execute a recipe and optionally compare with a reference image.

        Args:
            recipe: Recipe name (e.g. "screenshot") or inline YAML string.
            args: Substitution args for the recipe (e.g. {"url": "..."}).
            reference: Path to reference image for visual comparison.

        Returns:
            VerificationResult with pass/fail and details.
        """
        t0 = time.monotonic()
        args = args or {}

        try:
            result = await self._run_recipe(recipe, args)
        except Exception as e:
            return VerificationResult(
                passed=False,
                recipe_name=recipe,
                duration_ms=int((time.monotonic() - t0) * 1000),
                error="Recipe execution failed: {}".format(e),
            )

        screenshot_path = result.get("screenshot_path")
        extracted_data = result.get("extracted_data")

        # If reference provided, do visual comparison
        comparison = None
        if reference and screenshot_path:
            try:
                comparison = await self._compare_visual(screenshot_path, reference)
            except Exception as e:
                return VerificationResult(
                    passed=False,
                    recipe_name=recipe,
                    screenshot_path=screenshot_path,
                    extracted_data=extracted_data,
                    duration_ms=int((time.monotonic() - t0) * 1000),
                    error="Visual comparison failed: {}".format(e),
                )

        passed = True
        summary = None
        if comparison is not None:
            passed = comparison.get("passed", False)
            summary = comparison.get("summary", "")

        return VerificationResult(
            passed=passed,
            recipe_name=recipe,
            screenshot_path=screenshot_path,
            extracted_data=extracted_data,
            comparison_summary=summary,
            duration_ms=int((time.monotonic() - t0) * 1000),
        )

    async def _run_recipe(self, recipe: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a flyto-core recipe and return its output context."""
        # Try MCP handler first (same pattern as core_tools.py)
        handler = _get_mcp_handler()
        if handler and "run_recipe" in handler:
            result = await handler["run_recipe"](
                recipe_name=recipe,
                args=args,
            )
            if isinstance(result, dict):
                return self._extract_recipe_output(result)

        # Fallback: direct WorkflowEngine
        engine_cls = _get_engine_cls()
        if engine_cls is None:
            raise RuntimeError("flyto-core not installed — cannot run recipes")

        engine = engine_cls()
        result = await engine.run_recipe(recipe, args)
        return self._extract_recipe_output(result)

    def _extract_recipe_output(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        """Pull screenshot_path and extracted_data from recipe result."""
        out: Dict[str, Any] = {}

        # Look for screenshot in step outputs
        steps = raw.get("steps", [])
        if isinstance(steps, list):
            for step in steps:
                data = step.get("data", {}) if isinstance(step, dict) else {}
                if isinstance(data, dict):
                    path = data.get("path") or data.get("screenshot_path")
                    if path and str(path).endswith((".png", ".jpg", ".jpeg", ".webp")):
                        out["screenshot_path"] = str(path)
                    text = data.get("text") or data.get("extracted_text")
                    if text:
                        out.setdefault("extracted_data", {})["text"] = text

        # Direct fields
        if "screenshot_path" not in out:
            p = raw.get("screenshot_path") or raw.get("path")
            if p:
                out["screenshot_path"] = str(p)
        if "extracted_data" not in out:
            ed = raw.get("extracted_data") or raw.get("data")
            if isinstance(ed, dict):
                out["extracted_data"] = ed

        return out

    async def _compare_visual(
        self, actual_path: str, reference_path: str,
    ) -> Dict[str, Any]:
        """Compare two images using a multimodal LLM.

        Returns {"passed": bool, "summary": str, "issues": [...]}.
        """
        actual_b64 = _load_image_b64(actual_path)
        reference_b64 = _load_image_b64(reference_path)

        prompt = (
            "Compare these two UI screenshots. The first is the REFERENCE (expected), "
            "the second is the ACTUAL (current implementation).\n\n"
            "Respond in JSON:\n"
            '{"passed": true/false, "similarity": 0.0-1.0, '
            '"issues": ["issue1", ...], "summary": "brief description"}\n\n'
            "passed=true if the actual matches the reference in layout, text, "
            "and visual appearance (minor color/font differences are OK)."
        )

        try:
            import anthropic
            client = anthropic.AsyncAnthropic()
            resp = await client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=1024,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": reference_b64}},
                        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": actual_b64}},
                        {"type": "text", "text": prompt},
                    ],
                }],
            )
            import json
            text = resp.content[0].text if resp.content else "{}"
            # Extract JSON from possible markdown fence
            if "```" in text:
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            return json.loads(text.strip())
        except ImportError:
            raise RuntimeError("anthropic package required for visual comparison")


def _load_image_b64(path: str) -> str:
    """Read an image file and return base64-encoded content."""
    p = Path(path).expanduser()
    if not p.exists():
        raise FileNotFoundError("Image not found: {}".format(path))
    return base64.b64encode(p.read_bytes()).decode("ascii")


# ── MCP handler cache (same pattern as core_tools.py) ──
_cached_mcp: Any = None
_mcp_checked = False


def _get_mcp_handler() -> Any:
    global _cached_mcp, _mcp_checked
    if _mcp_checked:
        return _cached_mcp
    _mcp_checked = True
    try:
        from core.mcp_handler import run_recipe
        _cached_mcp = {"run_recipe": run_recipe}
    except ImportError:
        _cached_mcp = None
    return _cached_mcp
