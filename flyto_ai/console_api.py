# Copyright 2024 Flyto
# Licensed under the Apache License, Version 2.0
"""Console API — management endpoints for the flyto-ai dashboard.

Exposes read-only data from existing stores:
- CostTracker / CostController → token usage, spending, budget
- Vault → API key management (masked)
- SQLiteSessionStore → session history
- ChatAuditEntry → execution logs
- Blueprint engine → learned patterns
- ProBridge → license, EMS lessons, module catalog
"""
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Cached references (set by mount_console_api)
_agent = None
_config = None


def set_agent(agent) -> None:
    """Register the running Agent instance for data access."""
    global _agent, _config
    _agent = agent
    _config = agent.config if agent else None


# ── Overview ─────────────────────────────────────────────────

def get_overview() -> Dict[str, Any]:
    """Dashboard overview: session count, cost, license, module count."""
    data = {
        "version": _get_version(),
        "provider": getattr(_config, "provider", "") if _config else "",
        "model": getattr(_config, "resolved_model", "") if _config else "",
        "license_tier": "free",
        "module_count": 0,
        "session_id": getattr(_agent, "session_id", "") if _agent else "",
    }

    # Cost
    if _agent and _agent.cost_tracker:
        data["cost"] = _agent.cost_tracker.summary()
    else:
        data["cost"] = {"session_total_usd": 0, "call_count": 0}

    # Pro cost (multi-resource)
    if _agent and _agent.pro:
        summary = _agent.pro.get_cost_summary()
        if summary:
            data["pro_cost"] = summary
        data["license_tier"] = getattr(_agent.pro, "license_tier", "free")

    # Module count
    try:
        from core.modules.registry import ModuleRegistry
        data["module_count"] = len(ModuleRegistry.get_all_metadata())
    except Exception:
        data["module_count"] = 412

    return data


# ── Cost & Usage ─────────────────────────────────────────────

def get_cost_detail() -> Dict[str, Any]:
    """Detailed cost breakdown."""
    result = {"tracker": None, "controller": None}

    if _agent and _agent.cost_tracker:
        result["tracker"] = _agent.cost_tracker.summary()

    if _agent and _agent.pro:
        result["controller"] = _agent.pro.get_cost_summary()

    return result


# ── API Keys ─────────────────────────────────────────────────

def get_api_keys() -> List[Dict[str, str]]:
    """List configured API keys (masked)."""
    keys = []

    # Environment-based keys
    env_keys = [
        ("OPENAI_API_KEY", "openai"),
        ("ANTHROPIC_API_KEY", "anthropic"),
        ("FLYTO_AI_API_KEY", "flyto-ai"),
    ]
    for env_name, provider in env_keys:
        val = os.environ.get(env_name, "")
        if val:
            keys.append({
                "provider": provider,
                "source": "env",
                "masked": val[:4] + "***" + val[-4:] if len(val) > 8 else "***",
                "env_var": env_name,
            })

    # Vault keys
    if _agent and _agent.vault:
        try:
            for key_name in _agent.vault.list_keys():
                keys.append({
                    "provider": key_name,
                    "source": "vault",
                    "masked": "***" + key_name[-4:] if len(key_name) > 4 else "***",
                })
        except Exception:
            pass

    return keys


def set_api_key(provider: str, api_key: str) -> Dict[str, Any]:
    """Save an API key to the vault."""
    if _agent and _agent.vault:
        try:
            _agent.vault.set(provider, {"api_key": api_key})
            _agent.vault.save()
            return {"ok": True, "provider": provider}
        except Exception as e:
            return {"ok": False, "error": str(e)}
    return {"ok": False, "error": "Vault not available"}


def delete_api_key(provider: str) -> Dict[str, Any]:
    """Remove an API key from the vault."""
    if _agent and _agent.vault:
        try:
            _agent.vault.delete(provider)
            _agent.vault.save()
            return {"ok": True}
        except Exception as e:
            return {"ok": False, "error": str(e)}
    return {"ok": False, "error": "Vault not available"}


# ── Executions ───────────────────────────────────────────────

def get_executions(limit: int = 50) -> List[Dict[str, Any]]:
    """Recent execution history from audit log."""
    from flyto_ai.audit import get_recent_entries
    return get_recent_entries(limit=limit)


# ── Sessions ─────────────────────────────────────────────────

async def get_sessions(limit: int = 20) -> List[Dict[str, Any]]:
    """Recent sessions from memory store."""
    if not _agent or not _agent.memory_store:
        return []
    try:
        sessions = await _agent.memory_store.list_sessions(limit=limit)
        return sessions
    except Exception:
        return []


# ── Blueprints ───────────────────────────────────────────────

def get_blueprints() -> Dict[str, Any]:
    """Blueprint statistics and top patterns."""
    try:
        from flyto_blueprint import get_engine
        engine = get_engine()
        all_bp = engine.list_all()
        return {
            "total": len(all_bp),
            "blueprints": [
                {
                    "id": bp.get("id", ""),
                    "query": bp.get("query", ""),
                    "score": bp.get("score", 0),
                    "use_count": bp.get("use_count", 0),
                    "steps": len(bp.get("steps", [])),
                }
                for bp in sorted(all_bp, key=lambda x: x.get("score", 0), reverse=True)[:20]
            ],
        }
    except Exception:
        return {"total": 0, "blueprints": []}


# ── License ──────────────────────────────────────────────────

def get_license() -> Dict[str, Any]:
    """Current license status."""
    if _agent and _agent.pro:
        return {
            "tier": _agent.pro.license_tier,
            "core_available": _agent.pro.available,
            "premium_available": _agent.pro.premium_available,
        }
    return {"tier": "free", "core_available": False, "premium_available": False}


def activate_license(key: str) -> Dict[str, Any]:
    """Activate a license key."""
    try:
        from pro.license.validator import LicenseValidator
        import asyncio
        loop = asyncio.get_event_loop()
        validator = loop.run_until_complete(LicenseValidator.get_instance())
        result = loop.run_until_complete(validator.activate(key))
        return {"ok": True, "tier": result.tier.value}
    except Exception as e:
        return {"ok": False, "error": str(e)}


# ── EMS ──────────────────────────────────────────────────────

def get_ems_stats() -> Dict[str, Any]:
    """EMS lesson statistics."""
    if not _agent or not _agent.pro:
        return {"available": False}
    ems = _agent.pro.get_ems()
    if ems is None:
        return {"available": False, "reason": "requires pro license"}
    try:
        stats = getattr(ems, "_stats", None)
        return {
            "available": True,
            "errors_recorded": getattr(stats, "errors_recorded", 0) if stats else 0,
            "lessons_learned": len(getattr(ems, "_lessons", [])),
            "fix_attempts": len(getattr(ems, "_fix_attempts", [])),
        }
    except Exception:
        return {"available": True, "errors_recorded": 0}


# ── Modules ──────────────────────────────────────────────────

def get_modules() -> Dict[str, Any]:
    """Module catalog summary."""
    try:
        from core.modules.registry import ModuleRegistry
        metadata = ModuleRegistry.get_all_metadata()
        categories = {}
        for mid, meta in metadata.items():
            cat = mid.split(".")[0] if "." in mid else "other"
            if cat not in categories:
                categories[cat] = {"count": 0, "modules": []}
            categories[cat]["count"] += 1
            categories[cat]["modules"].append(mid)
        return {
            "total": len(metadata),
            "categories": {k: v for k, v in sorted(categories.items())},
        }
    except Exception:
        return {"total": 0, "categories": {}}


# ── Budget ───────────────────────────────────────────────────

def get_budget() -> Dict[str, Any]:
    """Current budget configuration."""
    result = {
        "session_budget_usd": getattr(_config, "session_budget_usd", None) if _config else None,
        "global_budget_usd": getattr(_config, "global_budget_usd", None) if _config else None,
    }
    if _agent and _agent.pro:
        controller = _agent.pro.get_cost_controller()
        if controller:
            result["pro_budget"] = {
                "max_cost_usd": controller.budget.max_cost_usd,
                "max_tokens": controller.budget.max_tokens,
                "max_tool_calls": controller.budget.max_tool_calls,
                "max_llm_calls": controller.budget.max_llm_calls,
                "remaining_cost": controller.remaining_budget,
                "remaining_tokens": controller.remaining_tokens,
            }
    return result


# ── Setup (provider + key configuration) ─────────────────────

def setup_provider(provider: str, api_key: str, base_url: str = "") -> Dict[str, Any]:
    """Configure AI provider. Sets env vars and reinitializes the Agent.

    This is the key function that makes the Setup page work:
    user picks provider + pastes key → this stores it and rebuilds the agent.
    """
    global _agent, _config

    # Map provider to env var
    env_map = {
        "openai": "OPENAI_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "deepseek": "OPENAI_API_KEY",  # DeepSeek uses OpenAI-compatible API
        "ollama": None,
        "custom": "OPENAI_API_KEY",
    }

    env_var = env_map.get(provider)
    if env_var and api_key:
        os.environ[env_var] = api_key

    if provider == "deepseek":
        os.environ["FLYTO_AI_PROVIDER"] = "openai"
        os.environ["FLYTO_AI_MODEL"] = "deepseek-chat"
        os.environ["FLYTO_AI_BASE_URL"] = base_url or "https://api.deepseek.com/v1"
    elif provider == "custom":
        os.environ["FLYTO_AI_PROVIDER"] = "openai"
        if base_url:
            os.environ["FLYTO_AI_BASE_URL"] = base_url
    elif provider == "ollama":
        os.environ["FLYTO_AI_PROVIDER"] = "ollama"
        os.environ["FLYTO_AI_BASE_URL"] = base_url or "http://localhost:11434/v1"
    else:
        os.environ["FLYTO_AI_PROVIDER"] = provider

    # Rebuild agent with new config
    try:
        from flyto_ai import Agent, AgentConfig
        _config = AgentConfig.from_env()
        _agent = Agent(config=_config)
        return {
            "ok": True,
            "provider": _config.provider,
            "model": _config.resolved_model,
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}


def get_setup_status() -> Dict[str, Any]:
    """Check if the agent is properly configured with an API key."""
    if not _agent:
        return {"configured": False, "reason": "no_agent"}
    if not _config:
        return {"configured": False, "reason": "no_config"}
    if not _config.api_key and _config.provider != "ollama":
        return {"configured": False, "reason": "no_api_key"}
    return {
        "configured": True,
        "provider": _config.provider,
        "model": _config.resolved_model,
    }


# ── Chat History (SQLite) ─────────────────────────────────────

async def get_chat_history(session_id: str = "", limit: int = 50) -> Dict[str, Any]:
    """Get chat messages from SQLite. If no session_id, returns current session."""
    if not _agent:
        return {"messages": [], "session_id": ""}

    # Initialize memory if needed
    await _agent._init_memory()

    store = _agent.memory_store
    if not store:
        return {"messages": [], "session_id": ""}

    sid = session_id or _agent.session_id
    try:
        messages = await store.get_messages(sid, limit=limit)
        return {
            "session_id": sid,
            "messages": messages,
            "count": len(messages),
        }
    except Exception as e:
        return {"messages": [], "session_id": sid, "error": str(e)}


async def get_all_sessions() -> List[Dict[str, Any]]:
    """List all chat sessions from SQLite."""
    if not _agent:
        return []

    await _agent._init_memory()
    store = _agent.memory_store
    if not store:
        return []

    try:
        return await store.list_sessions()
    except Exception:
        return []


# ── Budget Management ─────────────────────────────────────────

def set_budget(
    session_budget_usd: Optional[float] = None,
    global_budget_usd: Optional[float] = None,
) -> Dict[str, Any]:
    """Set budget limits. Applies immediately to the running agent."""
    global _config
    if not _agent or not _config:
        return {"ok": False, "error": "No agent running"}

    if session_budget_usd is not None:
        _config.session_budget_usd = session_budget_usd
        if _agent.cost_tracker:
            _agent.cost_tracker.session_budget_usd = session_budget_usd

    if global_budget_usd is not None:
        _config.global_budget_usd = global_budget_usd
        if _agent.cost_tracker:
            _agent.cost_tracker.global_budget_usd = global_budget_usd

    return {
        "ok": True,
        "session_budget_usd": _config.session_budget_usd,
        "global_budget_usd": _config.global_budget_usd,
    }


def reset_budget() -> Dict[str, Any]:
    """Reset session cost counters (keep budget limits)."""
    if _agent and _agent.cost_tracker:
        _agent.cost_tracker.reset_session()
        return {"ok": True, "message": "Session counters reset"}
    return {"ok": False, "error": "No cost tracker"}


# ── Third-Party Channels ─────────────────────────────────────

def get_channels() -> Dict[str, Any]:
    """Get status of third-party chat integrations."""
    channels = {}

    # Telegram
    tg_token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    channels["telegram"] = {
        "configured": bool(tg_token),
        "token_set": bool(tg_token),
        "env_var": "TELEGRAM_BOT_TOKEN",
    }

    # Slack
    slack_token = os.environ.get("SLACK_BOT_TOKEN", "")
    channels["slack"] = {
        "configured": bool(slack_token),
        "token_set": bool(slack_token),
        "env_var": "SLACK_BOT_TOKEN",
    }

    # Discord
    discord_token = os.environ.get("DISCORD_BOT_TOKEN", "")
    channels["discord"] = {
        "configured": bool(discord_token),
        "token_set": bool(discord_token),
        "env_var": "DISCORD_BOT_TOKEN",
    }

    # Webhook
    webhook_url = os.environ.get("FLYTO_WEBHOOK_URL", "")
    channels["webhook"] = {
        "configured": bool(webhook_url),
        "url": webhook_url[:50] + "..." if len(webhook_url) > 50 else webhook_url,
        "env_var": "FLYTO_WEBHOOK_URL",
    }

    return channels


def set_channel(channel: str, token: str) -> Dict[str, Any]:
    """Configure a third-party channel token."""
    env_map = {
        "telegram": "TELEGRAM_BOT_TOKEN",
        "slack": "SLACK_BOT_TOKEN",
        "discord": "DISCORD_BOT_TOKEN",
        "webhook": "FLYTO_WEBHOOK_URL",
    }
    env_var = env_map.get(channel)
    if not env_var:
        return {"ok": False, "error": "Unknown channel: {}".format(channel)}

    os.environ[env_var] = token
    return {"ok": True, "channel": channel, "configured": True}


# ── Helpers ──────────────────────────────────────────────────

def _get_version() -> str:
    try:
        from flyto_ai import __version__
        return __version__
    except Exception:
        return "unknown"
