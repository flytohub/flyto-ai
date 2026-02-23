# Copyright 2024 Flyto
# Licensed under the Apache License, Version 2.0
"""SQLite-backed session store — persistent across restarts."""
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional

import aiosqlite

logger = logging.getLogger(__name__)

_SCHEMA = """\
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    created_at REAL,
    updated_at REAL
);

CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT REFERENCES sessions(session_id),
    role TEXT,
    content TEXT,
    timestamp REAL,
    metadata TEXT
);

CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, timestamp);

CREATE TABLE IF NOT EXISTS summaries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT REFERENCES sessions(session_id),
    content TEXT,
    source_count INTEGER,
    created_at REAL
);

CREATE INDEX IF NOT EXISTS idx_summaries_session ON summaries(session_id);
"""


class SQLiteSessionStore:
    """Async SQLite session store — drop-in replacement for in-memory SessionStore."""

    def __init__(self, db_path: str = "~/.flyto/memory.db") -> None:
        self._db_path = os.path.expanduser(db_path)
        self._db: Optional[aiosqlite.Connection] = None

    async def init(self) -> None:
        """Open database and create tables."""
        os.makedirs(os.path.dirname(self._db_path), exist_ok=True)
        self._db = await aiosqlite.connect(self._db_path)
        await self._db.executescript(_SCHEMA)
        await self._db.commit()

    async def close(self) -> None:
        """Close the database connection."""
        if self._db:
            await self._db.close()
            self._db = None

    async def _ensure_db(self) -> aiosqlite.Connection:
        if self._db is None:
            await self.init()
        return self._db

    # ── Session CRUD ──────────────────────────────────────────

    async def create_session(self, session_id: str) -> None:
        """Create a new session."""
        db = await self._ensure_db()
        now = time.time()
        await db.execute(
            "INSERT OR REPLACE INTO sessions (session_id, created_at, updated_at) VALUES (?, ?, ?)",
            (session_id, now, now),
        )
        await db.commit()

    async def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add a message to a session. Auto-creates session if needed."""
        db = await self._ensure_db()
        now = time.time()
        # Ensure session exists
        await db.execute(
            "INSERT OR IGNORE INTO sessions (session_id, created_at, updated_at) VALUES (?, ?, ?)",
            (session_id, now, now),
        )
        await db.execute(
            "UPDATE sessions SET updated_at = ? WHERE session_id = ?",
            (now, session_id),
        )
        meta_json = json.dumps(metadata) if metadata else None
        await db.execute(
            "INSERT INTO messages (session_id, role, content, timestamp, metadata) VALUES (?, ?, ?, ?, ?)",
            (session_id, role, content, now, meta_json),
        )
        await db.commit()

    async def get_messages(
        self, session_id: str, limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Get messages for a session, ordered by timestamp."""
        db = await self._ensure_db()
        if limit:
            cursor = await db.execute(
                "SELECT role, content, timestamp, metadata FROM messages "
                "WHERE session_id = ? ORDER BY timestamp ASC LIMIT ?",
                (session_id, limit),
            )
        else:
            cursor = await db.execute(
                "SELECT role, content, timestamp, metadata FROM messages "
                "WHERE session_id = ? ORDER BY timestamp ASC",
                (session_id,),
            )
        rows = await cursor.fetchall()
        return [
            {
                "role": r[0],
                "content": r[1],
                "timestamp": r[2],
                "metadata": json.loads(r[3]) if r[3] else None,
            }
            for r in rows
        ]

    async def list_sessions(self) -> List[Dict[str, Any]]:
        """List all sessions with metadata."""
        db = await self._ensure_db()
        cursor = await db.execute(
            "SELECT s.session_id, s.created_at, s.updated_at, COUNT(m.id) as msg_count "
            "FROM sessions s LEFT JOIN messages m ON s.session_id = m.session_id "
            "GROUP BY s.session_id ORDER BY s.updated_at DESC",
        )
        rows = await cursor.fetchall()
        return [
            {
                "session_id": r[0],
                "created_at": r[1],
                "updated_at": r[2],
                "message_count": r[3],
            }
            for r in rows
        ]

    async def delete_session(self, session_id: str) -> bool:
        """Delete a session and all its messages."""
        db = await self._ensure_db()
        cursor = await db.execute(
            "SELECT 1 FROM sessions WHERE session_id = ?", (session_id,)
        )
        if not await cursor.fetchone():
            return False
        await db.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
        await db.execute("DELETE FROM summaries WHERE session_id = ?", (session_id,))
        await db.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
        await db.commit()
        return True

    async def session_exists(self, session_id: str) -> bool:
        """Check if a session exists."""
        db = await self._ensure_db()
        cursor = await db.execute(
            "SELECT 1 FROM sessions WHERE session_id = ?", (session_id,)
        )
        return await cursor.fetchone() is not None

    # ── Summary support (Sprint B) ────────────────────────────

    async def replace_old_with_summary(
        self, session_id: str, summary: str, keep_recent: int = 10
    ) -> None:
        """Replace old messages with a summary, keeping the most recent ones."""
        db = await self._ensure_db()
        now = time.time()

        # Get all message IDs ordered by timestamp
        cursor = await db.execute(
            "SELECT id FROM messages WHERE session_id = ? ORDER BY timestamp ASC",
            (session_id,),
        )
        all_ids = [r[0] for r in await cursor.fetchall()]

        if len(all_ids) <= keep_recent:
            return

        # IDs to delete (old messages)
        old_ids = all_ids[:-keep_recent]
        source_count = len(old_ids)

        # Delete old messages
        placeholders = ",".join("?" * len(old_ids))
        await db.execute(
            "DELETE FROM messages WHERE id IN ({})".format(placeholders),
            old_ids,
        )

        # Insert summary
        await db.execute(
            "INSERT INTO summaries (session_id, content, source_count, created_at) VALUES (?, ?, ?, ?)",
            (session_id, summary, source_count, now),
        )
        await db.commit()

    async def get_summaries(self, session_id: str) -> List[Dict[str, Any]]:
        """Get all summaries for a session."""
        db = await self._ensure_db()
        cursor = await db.execute(
            "SELECT content, source_count, created_at FROM summaries "
            "WHERE session_id = ? ORDER BY created_at ASC",
            (session_id,),
        )
        rows = await cursor.fetchall()
        return [
            {"content": r[0], "source_count": r[1], "created_at": r[2]}
            for r in rows
        ]

    async def get_message_count(self, session_id: str) -> int:
        """Get the number of messages in a session."""
        db = await self._ensure_db()
        cursor = await db.execute(
            "SELECT COUNT(*) FROM messages WHERE session_id = ?", (session_id,)
        )
        row = await cursor.fetchone()
        return row[0] if row else 0

    async def get_all_contents(self) -> List[Dict[str, Any]]:
        """Get all message contents + summaries for indexing."""
        db = await self._ensure_db()
        results = []

        # Messages
        cursor = await db.execute(
            "SELECT session_id, content, timestamp FROM messages ORDER BY timestamp ASC"
        )
        for r in await cursor.fetchall():
            results.append({
                "session_id": r[0],
                "content": r[1],
                "timestamp": r[2],
                "type": "message",
            })

        # Summaries
        cursor = await db.execute(
            "SELECT session_id, content, created_at FROM summaries ORDER BY created_at ASC"
        )
        for r in await cursor.fetchall():
            results.append({
                "session_id": r[0],
                "content": r[1],
                "timestamp": r[2],
                "type": "summary",
            })

        return results
