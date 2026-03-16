# Copyright 2024 Flyto
# Licensed under the Apache License, Version 2.0
"""ask_user tool — pause agent execution to request user input.

Features:
- Vault auto-fill: if credentials exist, skip asking
- Multi-account: if multiple accounts saved, show selection
- Third-party login: detected options injected as select field
- Preference auto-fill: non-sensitive fields pre-filled
"""
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

ASK_USER_MARKER = "__ASK_USER__"

TOOL_DEF = {
    "name": "ask_user",
    "description": (
        "Pause execution and ask the user for input. Use when you need: "
        "login credentials, a choice between options, confirmation, or any "
        "information you cannot determine on your own. "
        "The execution will pause until the user responds."
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "The question to ask the user (in their language)",
            },
            "fields": {
                "type": "array",
                "description": "Input fields to collect from the user",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string", "description": "Field identifier"},
                        "type": {"type": "string", "enum": ["text", "password", "select", "confirm", "number", "date"]},
                        "label": {"type": "string", "description": "Display label"},
                        "options": {"type": "array", "items": {"type": "string"}, "description": "Options for select type"},
                        "default": {"description": "Default value"},
                        "required": {"type": "boolean", "default": True},
                    },
                    "required": ["id", "type", "label"],
                },
            },
            "context_key": {
                "type": "string",
                "description": (
                    "Key for credential vault lookup (e.g. 'x_login', 'github'). "
                    "If credentials exist in vault, they are auto-filled. "
                    "Use domain-based keys like 'x_login' for site-specific credentials."
                ),
            },
            "login_options": {
                "type": "array",
                "description": (
                    "Third-party login options detected on the page "
                    "(e.g. ['Sign in with Google', 'Sign in with Apple']). "
                    "These are presented as alternatives to username/password login."
                ),
                "items": {"type": "string"},
            },
        },
        "required": ["question", "fields"],
    },
}


def _check_vault(context_key: str) -> Optional[Dict[str, Any]]:
    """Check if credentials exist in vault for the given context_key."""
    try:
        from flyto_ai.vault import Vault
        vault = Vault()
        vault.load()
        stored = vault.get(context_key)
        if stored and isinstance(stored, dict):
            logger.info("Vault hit for context_key=%s", context_key)
            return stored
    except Exception as e:
        logger.debug("Vault lookup failed: %s", e)
    return None


def _find_saved_accounts(domain: str) -> List[Dict[str, Any]]:
    """Find all saved accounts for a domain prefix in vault.

    e.g. domain='x' matches 'x_login', 'x_login_work', 'x_personal'
    """
    try:
        from flyto_ai.vault import Vault
        vault = Vault()
        vault.load()
        accounts = []
        for key in vault.list_keys():
            if key.startswith(domain):
                data = vault.get(key)
                if isinstance(data, dict):
                    # Show username/email but never password
                    display = data.get("username", data.get("email", data.get("id", key)))
                    accounts.append({
                        "context_key": key,
                        "display": display,
                        "label": key.replace("_", " ").replace(domain, "").strip() or "default",
                    })
        return accounts
    except Exception:
        return []


def save_to_vault(context_key: str, data: Any) -> bool:
    """Save user-provided data to encrypted vault."""
    try:
        from flyto_ai.vault import Vault
        vault = Vault()
        vault.load()
        vault.set(context_key, data)
        vault.save()
        logger.info("Saved to vault: context_key=%s", context_key)
        return True
    except Exception as e:
        logger.warning("Vault save failed: %s", e)
        return False


def learn_preferences(context_key: str, fields: List[Dict], values: Dict[str, Any]) -> None:
    """Learn user preferences from an ask_user response."""
    if not context_key:
        return
    try:
        from flyto_ai.preferences import PreferenceStore
        prefs = PreferenceStore()
        prefs.learn_from_response(context_key, fields, values)
    except Exception as e:
        logger.debug("Preference learning failed: %s", e)


async def dispatch_ask_user(name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """Dispatch ask_user tool call.

    Priority:
    1. Vault exact match → auto-fill, continue execution
    2. Vault prefix match → inject saved_accounts for user selection
    3. Preferences → prefill non-sensitive fields
    4. No match → return ASK_USER_MARKER, pause execution
    """
    question = arguments.get("question", "")
    fields = arguments.get("fields", [])
    context_key = arguments.get("context_key", "")
    login_options = arguments.get("login_options", [])

    # 1. Vault exact match → auto-fill
    if context_key:
        vault_data = _check_vault(context_key)
        if vault_data:
            filled = {}
            for field in fields:
                fid = field.get("id", "")
                if fid in vault_data:
                    filled[fid] = vault_data[fid]
            if filled:
                return {
                    "ok": True,
                    "auto_filled": True,
                    "source": "vault",
                    "context_key": context_key,
                    "data": filled,
                    "message": "Credentials auto-filled from secure vault.",
                }

    # 2. Multi-account: find all saved accounts for this domain
    saved_accounts = []
    if context_key:
        domain = context_key.split("_")[0]  # 'x_login' → 'x'
        saved_accounts = _find_saved_accounts(domain)

    if saved_accounts:
        # Inject account selection as a field
        account_options = ["{} ({})".format(a["display"], a["label"]) for a in saved_accounts]
        account_options.append("+ Add new account")
        fields = [
            {
                "id": "_saved_account",
                "type": "select",
                "label": "Saved accounts",
                "options": account_options,
                "required": True,
            }
        ] + [f for f in fields]  # keep original fields for "Add new"

    # 3. Third-party login options → inject as select field
    if login_options:
        login_options_with_manual = login_options + ["Username & Password"]
        fields = [
            {
                "id": "_login_method",
                "type": "select",
                "label": "Login method",
                "options": login_options_with_manual,
                "required": True,
            }
        ] + [f for f in fields]

    # 4. Preference auto-fill for non-sensitive fields
    if context_key:
        try:
            from flyto_ai.preferences import PreferenceStore
            prefs = PreferenceStore()
            pref_data = prefs.get_all(context_key)
            if pref_data:
                for field in fields:
                    fid = field.get("id", "")
                    ftype = field.get("type", "")
                    if ftype != "password" and fid in pref_data:
                        field["prefill"] = pref_data[fid]
        except Exception:
            pass

    # Return marker for tool loop to break
    return {
        "ok": True,
        ASK_USER_MARKER: True,
        "question": question,
        "fields": fields,
        "context_key": context_key,
        "saved_accounts": [{"key": a["context_key"], "display": a["display"]} for a in saved_accounts],
    }
