# Copyright 2024 Flyto
# Licensed under the Apache License, Version 2.0
"""Session store with TTL, per-user limits, and locks."""
import asyncio
import time
from typing import Dict, List, Optional

from flyto_ai.models import ChatMessage

MAX_SESSIONS = 100
MAX_SESSIONS_PER_USER = 5
SESSION_TTL_SECONDS = 3600  # 1 hour


class SessionStore:
    """In-memory session store with TTL and per-user limits."""

    def __init__(
        self,
        max_sessions: int = MAX_SESSIONS,
        max_per_user: int = MAX_SESSIONS_PER_USER,
        ttl_seconds: int = SESSION_TTL_SECONDS,
    ) -> None:
        self._sessions: Dict[str, List[ChatMessage]] = {}
        self._timestamps: Dict[str, float] = {}
        self._owners: Dict[str, str] = {}
        self._locks: Dict[str, asyncio.Lock] = {}
        self._max_sessions = max_sessions
        self._max_per_user = max_per_user
        self._ttl = ttl_seconds

    def cleanup(self) -> None:
        """Remove expired sessions and enforce max limit."""
        now = time.time()
        expired = [
            sid for sid, ts in self._timestamps.items()
            if now - ts > self._ttl
        ]
        for sid in expired:
            self._remove(sid)

        if len(self._sessions) > self._max_sessions:
            sorted_sids = sorted(self._timestamps, key=self._timestamps.get)
            for sid in sorted_sids[:len(self._sessions) - self._max_sessions]:
                self._remove(sid)

    def get_lock(self, session_id: str) -> asyncio.Lock:
        """Get or create a per-session async lock."""
        if session_id not in self._locks:
            self._locks[session_id] = asyncio.Lock()
        return self._locks[session_id]

    def get_owner(self, session_id: str) -> Optional[str]:
        """Return the user ID that owns a session."""
        return self._owners.get(session_id)

    def exists(self, session_id: str) -> bool:
        return session_id in self._sessions

    def create(self, session_id: str, user_id: str) -> None:
        """Create a new session, evicting oldest if user is at limit."""
        if self._count_user_sessions(user_id) >= self._max_per_user:
            self._evict_oldest(user_id)
        self._sessions[session_id] = []
        self._owners[session_id] = user_id
        self._timestamps[session_id] = time.time()

    def get_messages(self, session_id: str) -> List[ChatMessage]:
        return self._sessions.get(session_id, [])

    def append(self, session_id: str, message: ChatMessage) -> None:
        if session_id in self._sessions:
            self._sessions[session_id].append(message)
            self._timestamps[session_id] = time.time()
            # Keep last 20 messages
            if len(self._sessions[session_id]) > 20:
                self._sessions[session_id] = self._sessions[session_id][-20:]

    def delete(self, session_id: str, user_id: str) -> bool:
        """Delete a session. Returns True if deleted."""
        if session_id not in self._sessions:
            return False
        owner = self._owners.get(session_id)
        if owner is not None and owner != user_id:
            return False
        self._remove(session_id)
        return True

    def _remove(self, session_id: str) -> None:
        self._sessions.pop(session_id, None)
        self._timestamps.pop(session_id, None)
        self._owners.pop(session_id, None)
        self._locks.pop(session_id, None)

    def _count_user_sessions(self, user_id: str) -> int:
        return sum(1 for uid in self._owners.values() if uid == user_id)

    def _evict_oldest(self, user_id: str) -> None:
        user_sids = [
            sid for sid, uid in self._owners.items() if uid == user_id
        ]
        if not user_sids:
            return
        oldest = min(user_sids, key=lambda s: self._timestamps.get(s, 0))
        self._remove(oldest)
