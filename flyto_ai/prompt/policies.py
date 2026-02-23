# Copyright 2024 Flyto
# Licensed under the Apache License, Version 2.0
"""Tool/module/URL allowlists and validation — pure logic, no I/O."""
from typing import Any, Dict, Set
from urllib.parse import urlparse

# SSRF protection: base_url domain allowlist
BASE_URL_ALLOWED_DOMAINS = frozenset({
    "api.openai.com",
    "openrouter.ai",
    "api.groq.com",
    "api.together.xyz",
    "api.mistral.ai",
    "api.deepseek.com",
    "api.fireworks.ai",
    "api.perplexity.ai",
    "api.endpoints.anyscale.com",
    "api.cohere.com",
})

# Tool call security
ALLOWED_TOOLS = frozenset({
    "list_modules", "search_modules", "get_module_info", "get_module_examples",
    "execute_module", "validate_params",
    "inspect_page",
    "list_blueprints", "use_blueprint", "save_as_blueprint",
    "report_blueprint_outcome",
})

# Module categories allowed for execute_module
ALLOWED_MODULE_CATEGORIES = frozenset({
    "browser", "element", "stealth",
    "string", "text", "regex",
    "array", "object", "set",
    "data", "convert", "format", "template",
    "math", "stats", "random",
    "logic", "compare", "check", "validate", "verify",
    "flow", "meta", "error",
    "http", "api", "graphql", "webhook",
    "image", "pdf", "archive",
    "encode", "hash", "crypto",
    "file", "path", "env",
    "database", "cache", "queue",
    "datetime", "date", "time",
    "json", "csv", "xml", "yaml",
    "git", "docker", "k8s", "shell", "process",
    "dns", "network", "port", "ssh",
    "email", "notification", "notify",
    "llm", "ai", "agent",
    "output", "sandbox",
    "utility", "training",
})


def validate_base_url(url: str, policies: Dict[str, Any] = None) -> bool:
    """Validate base_url against policy domain allowlist.

    HTTPS required for remote hosts. localhost/127.0.0.1 are always allowed
    (Ollama, vLLM, etc. run locally).
    """
    try:
        parsed = urlparse(url)
        host = (parsed.hostname or "").lower().rstrip(".")
        scheme = (parsed.scheme or "").lower()

        if not host or scheme not in ("http", "https"):
            return False

        # Local development: always allow localhost and loopback
        if host in ("localhost", "127.0.0.1", "0.0.0.0", "::1"):
            return True

        # Remote hosts: HTTPS required
        if scheme != "https":
            return False

        allowed: Set[str] = set(
            policies.get("allowed_domains", []) if policies
            else BASE_URL_ALLOWED_DOMAINS
        )
        if host in allowed:
            return True

        # Azure OpenAI wildcard
        if host.endswith(".openai.azure.com"):
            return True

        return False
    except Exception:
        return False


def is_tool_allowed(name: str, policies: Dict[str, Any] = None) -> bool:
    """Check if a tool name is in the policy allowlist."""
    allowed = set(
        policies.get("allowed_tools", []) if policies
        else ALLOWED_TOOLS
    )
    return name in allowed


def is_module_allowed(module_id: str, policies: Dict[str, Any] = None) -> bool:
    """Check if a module's category is in the policy allowlist."""
    category = module_id.split(".")[0] if "." in module_id else ""
    allowed = set(
        policies.get("allowed_categories", []) if policies
        else ALLOWED_MODULE_CATEGORIES
    )
    return category in allowed


def get_default_policies() -> Dict[str, Any]:
    """Return the default policies dict."""
    return {
        "allowed_domains": sorted(BASE_URL_ALLOWED_DOMAINS),
        "allowed_tools": sorted(ALLOWED_TOOLS),
        "allowed_categories": sorted(ALLOWED_MODULE_CATEGORIES),
    }
