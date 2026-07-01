# Copyright 2024 Flyto
# Licensed under the Apache License, Version 2.0
"""SQLite-backed job queue for Telegram tasks."""
import logging
import os
import time
import uuid
from typing import Any, Dict, List, Optional

import aiosqlite

logger = logging.getLogger(__name__)

_SCHEMA = """\
CREATE TABLE IF NOT EXISTS jobs (
    job_id TEXT PRIMARY KEY,
    chat_id INTEGER NOT NULL,
    text TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    result TEXT,
    error TEXT,
    created_at REAL NOT NULL,
    started_at REAL,
    completed_at REAL
);

CREATE INDEX IF NOT EXISTS idx_jobs_chat_status ON jobs(chat_id, status);
"""


class JobQueue:
    """Persistent job queue backed by SQLite.

    Lifecycle: enqueue → start → complete/fail/cancel.
    On server restart, ``resume_incomplete()`` marks running jobs as failed.
    """

    def __init__(self, db_path: str = "~/.flyto/tg_jobs.db") -> None:
        self._db_path = os.path.expanduser(db_path)
        self._db: Optional[aiosqlite.Connection] = None

    async def init(self) -> None:
        """Open database and create tables."""
        os.makedirs(os.path.dirname(self._db_path), exist_ok=True)
        self._db = await aiosqlite.connect(self._db_path)
        await self._db.executescript(_SCHEMA)
        await self._db.commit()

    async def close(self) -> None:
        if self._db:
            await self._db.close()
            self._db = None

    async def _ensure_db(self) -> aiosqlite.Connection:
        if self._db is None:
            await self.init()
        return self._db  # type: ignore[return-value]

    # ── Lifecycle ──────────────────────────────────────────────

    async def enqueue(self, chat_id: int, text: str) -> str:
        """Create a new pending job. Returns job_id."""
        db = await self._ensure_db()
        job_id = uuid.uuid4().hex[:12]
        now = time.time()
        await db.execute(
            "INSERT INTO jobs (job_id, chat_id, text, status, created_at) VALUES (?, ?, ?, 'pending', ?)",
            (job_id, chat_id, text, now),
        )
        await db.commit()
        logger.info("Job %s enqueued for chat %s", job_id, chat_id)
        return job_id

    async def start(self, job_id: str) -> None:
        """Mark a job as running."""
        db = await self._ensure_db()
        await db.execute(
            "UPDATE jobs SET status = 'running', started_at = ? WHERE job_id = ?",
            (time.time(), job_id),
        )
        await db.commit()

    async def complete(self, job_id: str, result: str = "") -> None:
        """Mark a job as completed."""
        db = await self._ensure_db()
        await db.execute(
            "UPDATE jobs SET status = 'completed', result = ?, completed_at = ? WHERE job_id = ?",
            (result, time.time(), job_id),
        )
        await db.commit()

    async def fail(self, job_id: str, error: str = "") -> None:
        """Mark a job as failed."""
        db = await self._ensure_db()
        await db.execute(
            "UPDATE jobs SET status = 'failed', error = ?, completed_at = ? WHERE job_id = ?",
            (error, time.time(), job_id),
        )
        await db.commit()

    async def cancel(self, job_id: str) -> bool:
        """Cancel a pending/running job. Returns True if cancelled."""
        db = await self._ensure_db()
        cursor = await db.execute(
            "UPDATE jobs SET status = 'cancelled', completed_at = ? "
            "WHERE job_id = ? AND status IN ('pending', 'running')",
            (time.time(), job_id),
        )
        await db.commit()
        return cursor.rowcount > 0

    # ── Queries ────────────────────────────────────────────────

    async def get(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get a single job by ID."""
        db = await self._ensure_db()
        cursor = await db.execute(
            "SELECT job_id, chat_id, text, status, result, error, "
            "created_at, started_at, completed_at FROM jobs WHERE job_id = ?",
            (job_id,),
        )
        row = await cursor.fetchone()
        if not row:
            return None
        return _row_to_dict(row)

    async def get_active(self, chat_id: int) -> Optional[Dict[str, Any]]:
        """Get the active (pending/running) job for a chat, if any."""
        db = await self._ensure_db()
        cursor = await db.execute(
            "SELECT job_id, chat_id, text, status, result, error, "
            "created_at, started_at, completed_at FROM jobs "
            "WHERE chat_id = ? AND status IN ('pending', 'running') "
            "ORDER BY created_at DESC LIMIT 1",
            (chat_id,),
        )
        row = await cursor.fetchone()
        if not row:
            return None
        return _row_to_dict(row)

    async def get_recent(self, chat_id: int, limit: int = 5) -> List[Dict[str, Any]]:
        """Get recent jobs for a chat."""
        db = await self._ensure_db()
        cursor = await db.execute(
            "SELECT job_id, chat_id, text, status, result, error, "
            "created_at, started_at, completed_at FROM jobs "
            "WHERE chat_id = ? ORDER BY created_at DESC LIMIT ?",
            (chat_id, limit),
        )
        rows = await cursor.fetchall()
        return [_row_to_dict(r) for r in rows]

    async def resume_incomplete(self) -> int:
        """On restart: mark all running jobs as failed. Returns count."""
        db = await self._ensure_db()
        cursor = await db.execute(
            "UPDATE jobs SET status = 'failed', error = 'Server restarted', completed_at = ? "
            "WHERE status = 'running'",
            (time.time(),),
        )
        await db.commit()
        count = cursor.rowcount
        if count:
            logger.warning("Marked %d incomplete jobs as failed after restart", count)
        return count


def _row_to_dict(row) -> Dict[str, Any]:
    return {
        "job_id": row[0],
        "chat_id": row[1],
        "text": row[2],
        "status": row[3],
        "result": row[4],
        "error": row[5],
        "created_at": row[6],
        "started_at": row[7],
        "completed_at": row[8],
    }
