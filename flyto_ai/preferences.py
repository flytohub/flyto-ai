# Copyright 2024 Flyto
# Licensed under the Apache License, Version 2.0
"""User preference store — learn from ask_user responses.

Stores non-sensitive preferences (seat choice, time preference, etc.)
separately from credentials (which go to Vault).
"""
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_DEFAULT_PATH = "~/.flyto/preferences.json"


class PreferenceStore:
    """Simple JSON-based preference store for non-sensitive user choices."""

    def __init__(self, path: Optional[str] = None) -> None:
        self._path = Path(os.path.expanduser(path or _DEFAULT_PATH))
        self._data: Dict[str, Dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        if self._path.exists():
            try:
                with open(self._path, "r", encoding="utf-8") as f:
                    self._data = json.load(f)
            except Exception as e:
                logger.debug("Preferences load failed: %s", e)
                self._data = {}

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, ensure_ascii=False, indent=2)

    def get(self, context_key: str, field_id: str) -> Optional[Any]:
        """Get a stored preference value."""
        return self._data.get(context_key, {}).get(field_id)

    def get_all(self, context_key: str) -> Dict[str, Any]:
        """Get all preferences for a context key."""
        return dict(self._data.get(context_key, {}))

    def set(self, context_key: str, field_id: str, value: Any) -> None:
        """Store a preference value."""
        if context_key not in self._data:
            self._data[context_key] = {}
        self._data[context_key][field_id] = value
        self._save()

    def set_many(self, context_key: str, values: Dict[str, Any]) -> None:
        """Store multiple preference values at once."""
        if context_key not in self._data:
            self._data[context_key] = {}
        self._data[context_key].update(values)
        self._save()

    def learn_from_response(self, context_key: str, fields: List[Dict], values: Dict[str, Any]) -> None:
        """Learn preferences from an ask_user response.

        Only stores non-sensitive fields (not password type).
        Sensitive fields should go to Vault instead.
        """
        for field in fields:
            fid = field.get("id", "")
            ftype = field.get("type", "")
            if ftype == "password":
                continue  # passwords go to vault, not preferences
            if fid in values and values[fid] is not None:
                self.set(context_key, fid, values[fid])
