"""Small durable definition/claim catalog; execution authority lives in MissionStore."""
from __future__ import annotations

import fcntl
import json
import os
import sqlite3
import stat
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Optional

from flyto_ai.scheduler.tasks import ScheduledTask

SCHEMA = 1
MAX_CATALOG_BYTES = 16 * 1024 * 1024
MAX_TASKS = 1000
MAX_OCCURRENCES = 5000
HISTORY_LIMIT = 50
_TASK_COLUMNS = ("task_id", "definition", "enabled", "cursor", "generation", "mission_id", "root_id", "item_count")
_OCC_COLUMNS = ("task_id", "slot", "state", "mission_id", "work_item_id", "fence", "result")
_DIR_FLAGS = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
_FILE_FLAGS = os.O_RDONLY | os.O_NOFOLLOW
_TABLE_SQL = {
    "tasks": "CREATE TABLE tasks(task_id TEXT PRIMARY KEY, definition TEXT NOT NULL, "
             "enabled INTEGER NOT NULL CHECK(enabled IN (0,1)), cursor REAL NOT NULL, "
             "generation INTEGER NOT NULL, mission_id TEXT, root_id TEXT, item_count INTEGER NOT NULL)",
    "occurrences": "CREATE TABLE occurrences(task_id TEXT NOT NULL, slot INTEGER NOT NULL, "
                   "state TEXT NOT NULL CHECK(state IN ('claimed','materialized','closed')), "
                   "mission_id TEXT, work_item_id TEXT, fence INTEGER, result TEXT, "
                   "PRIMARY KEY(task_id,slot), FOREIGN KEY(task_id) REFERENCES tasks(task_id))",
}


class CatalogError(RuntimeError):
    pass


class ScheduleCatalog:
    def __init__(self, state_root: os.PathLike[str] | str) -> None:
        self.state_root = Path(os.path.abspath(os.fspath(state_root)))
        self.root = self.state_root / "scheduler-catalog"
        self.db = self.root / "catalog.sqlite3"
        self._state_identity: Optional[tuple[int, int]] = None
        self._root_identity: Optional[tuple[int, int]] = None

    @staticmethod
    def _private(info: os.stat_result, *, directory: bool) -> None:
        kind = stat.S_ISDIR(info.st_mode) if directory else stat.S_ISREG(info.st_mode)
        if not kind or info.st_uid != os.getuid() or info.st_mode & 0o077:
            noun = "directories" if directory else "file"
            raise CatalogError(f"scheduler catalog {noun} must be owner-only")

    def _open_state_root(self) -> int:
        """Walk without following links, creating missing components privately."""
        current = os.open(os.sep, _DIR_FLAGS)
        try:
            for component in self.state_root.parts[1:]:
                try:
                    child = os.open(component, _DIR_FLAGS, dir_fd=current)
                except FileNotFoundError:
                    try:
                        os.mkdir(component, 0o700, dir_fd=current)
                    except FileExistsError:
                        pass
                    child = os.open(component, _DIR_FLAGS, dir_fd=current)
                except OSError as exc:
                    raise CatalogError("scheduler catalog path must not contain a symlink") from exc
                os.close(current)
                current = child
            info = os.fstat(current)
            self._private(info, directory=True)
            identity = (info.st_dev, info.st_ino)
            if self._state_identity is not None and identity != self._state_identity:
                raise CatalogError("scheduler catalog configured root was displaced")
            self._state_identity = identity
            return current
        except BaseException:
            os.close(current)
            raise

    @staticmethod
    def _open_private_file(parent_fd: int, name: str) -> int:
        try:
            fd = os.open(name, _FILE_FLAGS, dir_fd=parent_fd)
        except OSError as exc:
            raise CatalogError("scheduler catalog file is unsafe") from exc
        try:
            ScheduleCatalog._private(os.fstat(fd), directory=False)
            return fd
        except BaseException:
            os.close(fd)
            raise

    def _open_root(self, state_fd: int) -> int:
        try:
            os.mkdir("scheduler-catalog", 0o700, dir_fd=state_fd)
        except FileExistsError:
            pass
        try:
            root_fd = os.open("scheduler-catalog", _DIR_FLAGS, dir_fd=state_fd)
        except OSError as exc:
            raise CatalogError("scheduler catalog path must not contain a symlink") from exc
        try:
            info = os.fstat(root_fd)
            self._private(info, directory=True)
            identity = (info.st_dev, info.st_ino)
            if self._root_identity is not None and identity != self._root_identity:
                raise CatalogError("scheduler catalog path was displaced")
            self._root_identity = identity
            return root_fd
        except BaseException:
            os.close(root_fd)
            raise

    def _rewalk_bound_paths(self, state_fd: int, root_fd: int) -> tuple[int, int]:
        """Reopen and hold both configured directories after identity checks."""
        walked_state_fd = os.open(os.sep, _DIR_FLAGS)
        walked_root_fd = -1
        try:
            for component in self.state_root.parts[1:]:
                child = os.open(component, _DIR_FLAGS, dir_fd=walked_state_fd)
                os.close(walked_state_fd)
                walked_state_fd = child
            walked_root_fd = os.open("scheduler-catalog", _DIR_FLAGS, dir_fd=walked_state_fd)
        except OSError as exc:
            os.close(walked_state_fd)
            raise CatalogError("scheduler catalog path was displaced") from exc
        try:
            state_identities = {
                (os.fstat(fd).st_dev, os.fstat(fd).st_ino)
                for fd in (state_fd, walked_state_fd)
            }
            root_identities = {
                (os.fstat(fd).st_dev, os.fstat(fd).st_ino)
                for fd in (root_fd, walked_root_fd)
            }
            if state_identities != {self._state_identity}:
                raise CatalogError("scheduler catalog configured root was displaced")
            if root_identities != {self._root_identity}:
                raise CatalogError("scheduler catalog path was displaced")
            return walked_state_fd, walked_root_fd
        except BaseException:
            os.close(walked_root_fd)
            os.close(walked_state_fd)
            raise

    @staticmethod
    def _read_all(fd: int) -> bytes:
        info = os.fstat(fd)
        if info.st_size > MAX_CATALOG_BYTES:
            raise CatalogError("scheduler catalog exceeds its bound")
        chunks = []
        remaining = MAX_CATALOG_BYTES + 1
        while remaining:
            chunk = os.read(fd, min(1024 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        data = b"".join(chunks)
        if len(data) > MAX_CATALOG_BYTES:
            raise CatalogError("scheduler catalog exceeds its bound")
        return data

    @classmethod
    def _verify_final_file(
        cls,
        root_fd: int,
        *,
        expected_identity: Optional[tuple[int, int]],
        expected_bytes: bytes,
    ) -> tuple[int, int]:
        if expected_identity is None:
            try:
                os.stat("catalog.sqlite3", dir_fd=root_fd, follow_symlinks=False)
            except FileNotFoundError:
                return (-1, -1)
            raise CatalogError("scheduler catalog file was displaced")
        fd = cls._open_private_file(root_fd, "catalog.sqlite3")
        try:
            info = os.fstat(fd)
            identity = (info.st_dev, info.st_ino)
            if identity != expected_identity or cls._read_all(fd) != expected_bytes:
                raise CatalogError("scheduler catalog file was displaced")
            return identity
        finally:
            os.close(fd)

    def _publish(self, root_fd: int, data: bytes) -> tuple[int, int]:
        if len(data) > MAX_CATALOG_BYTES:
            raise CatalogError("scheduler catalog exceeds its bound")
        name = f".catalog.{os.getpid()}.{id(data)}.tmp"
        fd = os.open(name, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=root_fd)
        try:
            view = memoryview(data)
            while view:
                written = os.write(fd, view)
                view = view[written:]
            os.fsync(fd)
            os.rename(name, "catalog.sqlite3", src_dir_fd=root_fd, dst_dir_fd=root_fd)
            os.fsync(root_fd)
            published_fd = self._open_private_file(root_fd, "catalog.sqlite3")
            try:
                if self._read_all(published_fd) != data:
                    raise CatalogError("published scheduler catalog bytes differ")
                info = os.fstat(published_fd)
                return (info.st_dev, info.st_ino)
            finally:
                os.close(published_fd)
        finally:
            os.close(fd)
            try:
                os.unlink(name, dir_fd=root_fd)
            except FileNotFoundError:
                pass

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        state_fd = self._open_state_root()
        root_fd = -1
        db_fd = -1
        walked_state_fd = -1
        walked_root_fd = -1
        conn: Optional[sqlite3.Connection] = None
        try:
            fcntl.flock(state_fd, fcntl.LOCK_EX)
            root_fd = self._open_root(state_fd)
            exists = True
            try:
                db_fd = self._open_private_file(root_fd, "catalog.sqlite3")
                original = self._read_all(db_fd)
            except CatalogError as exc:
                try:
                    os.stat("catalog.sqlite3", dir_fd=root_fd, follow_symlinks=False)
                except FileNotFoundError:
                    exists = False
                    original = b""
                else:
                    raise exc
            conn = sqlite3.connect(":memory:", isolation_level=None)
            if exists:
                if not original:
                    raise CatalogError("invalid scheduler catalog")
                conn.deserialize(original)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA foreign_keys=ON")
            if exists:
                self._check_integrity(conn)
            self._ensure_schema(conn)
            conn.execute("BEGIN IMMEDIATE")
            yield conn
            self._ensure_schema(conn)
            conn.execute("COMMIT")
            updated = conn.serialize()
            if len(updated) > MAX_CATALOG_BYTES:
                raise CatalogError("scheduler catalog exceeds its bound")
            walked_state_fd, walked_root_fd = self._rewalk_bound_paths(state_fd, root_fd)
            original_identity = None
            if db_fd >= 0:
                opened = os.fstat(db_fd)
                original_identity = (opened.st_dev, opened.st_ino)
            self._verify_final_file(
                walked_root_fd,
                expected_identity=original_identity,
                expected_bytes=original,
            )
            if updated != original:
                self._check_serialized(updated)
                published_identity = self._publish(walked_root_fd, updated)
                os.close(walked_root_fd)
                walked_root_fd = -1
                os.close(walked_state_fd)
                walked_state_fd = -1
                walked_state_fd, walked_root_fd = self._rewalk_bound_paths(state_fd, root_fd)
                self._verify_final_file(
                    walked_root_fd,
                    expected_identity=published_identity,
                    expected_bytes=updated,
                )
        except sqlite3.Error as exc:
            if conn is not None:
                try:
                    conn.execute("ROLLBACK")
                except sqlite3.Error:
                    pass
            raise CatalogError("invalid scheduler catalog") from exc
        except BaseException:
            if conn is not None:
                try:
                    conn.execute("ROLLBACK")
                except sqlite3.Error:
                    pass
            raise
        finally:
            if conn is not None:
                conn.close()
            if db_fd >= 0:
                os.close(db_fd)
            if walked_root_fd >= 0:
                os.close(walked_root_fd)
            if walked_state_fd >= 0:
                os.close(walked_state_fd)
            if root_fd >= 0:
                os.close(root_fd)
            fcntl.flock(state_fd, fcntl.LOCK_UN)
            os.close(state_fd)

    @staticmethod
    def _check_integrity(conn: sqlite3.Connection) -> None:
        integrity = conn.execute("PRAGMA integrity_check(1)").fetchone()
        if integrity is None or integrity[0] != "ok":
            raise CatalogError("scheduler catalog integrity check failed")
        if conn.execute("PRAGMA foreign_key_check").fetchone() is not None:
            raise CatalogError("scheduler catalog foreign key check failed")

    @classmethod
    def _check_serialized(cls, data: bytes) -> None:
        check = sqlite3.connect(":memory:", isolation_level=None)
        try:
            check.deserialize(data)
            check.execute("PRAGMA foreign_keys=ON")
            cls._check_integrity(check)
        except sqlite3.Error as exc:
            raise CatalogError("invalid scheduler catalog") from exc
        finally:
            check.close()

    @staticmethod
    def _ensure_schema(conn: sqlite3.Connection) -> None:
        version = conn.execute("PRAGMA user_version").fetchone()[0]
        objects = {(row[0], row[1]) for row in conn.execute(
            "SELECT type,name FROM sqlite_schema WHERE name NOT LIKE 'sqlite_%'"
        )}
        names = {name for kind, name in objects if kind == "table"}
        if version == 0 and not names:
            if objects:
                raise CatalogError("unknown scheduler catalog schema objects")
            conn.executescript("""
                CREATE TABLE tasks(task_id TEXT PRIMARY KEY, definition TEXT NOT NULL,
                    enabled INTEGER NOT NULL CHECK(enabled IN (0,1)), cursor REAL NOT NULL,
                    generation INTEGER NOT NULL, mission_id TEXT, root_id TEXT,
                    item_count INTEGER NOT NULL);
                CREATE TABLE occurrences(task_id TEXT NOT NULL, slot INTEGER NOT NULL,
                    state TEXT NOT NULL CHECK(state IN ('claimed','materialized','closed')),
                    mission_id TEXT, work_item_id TEXT, fence INTEGER, result TEXT,
                    PRIMARY KEY(task_id,slot), FOREIGN KEY(task_id) REFERENCES tasks(task_id));
                PRAGMA user_version=1;
            """)
            return
        if version != SCHEMA or objects != {("table", "tasks"), ("table", "occurrences")}:
            raise CatalogError("unknown scheduler catalog schema")
        for table, expected in (("tasks", _TASK_COLUMNS), ("occurrences", _OCC_COLUMNS)):
            actual = tuple(row[1] for row in conn.execute(f"PRAGMA table_info({table})"))
            if actual != expected:
                raise CatalogError("unknown scheduler catalog fields")
            sql = conn.execute("SELECT sql FROM sqlite_schema WHERE type='table' AND name=?", (table,)).fetchone()[0]
            if " ".join(sql.split()) != _TABLE_SQL[table]:
                raise CatalogError("unknown scheduler catalog schema")
        if conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] > MAX_TASKS:
            raise CatalogError("scheduler task bound exceeded")
        if conn.execute("SELECT COUNT(*) FROM occurrences").fetchone()[0] > MAX_OCCURRENCES:
            raise CatalogError("scheduler occurrence bound exceeded")

    @staticmethod
    def decode_task(row: sqlite3.Row) -> ScheduledTask:
        try:
            raw = json.loads(row["definition"])
            if not isinstance(raw, dict):
                raise ValueError
            task = ScheduledTask.from_dict(raw)
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            raise CatalogError("corrupt scheduler task definition") from exc
        task.enabled = bool(row["enabled"])
        return task

    def put(self, task: ScheduledTask, *, now: Optional[float] = None) -> None:
        if task.schedule.type.value == "one_shot" and task.schedule.run_at == 0:
            resolved = time.time() if now is None else now
            definition_data = task.to_definition()
            definition_data["schedule"] = {"type": "one_shot", "run_at": resolved}
            # Re-parse before persistence so injected and runtime timestamps share
            # the public numeric bounds and the stored definition is self-validating.
            persisted_task = ScheduledTask.from_dict(definition_data)
            if persisted_task.schedule.run_at == 0:
                raise ValueError("resolved one-shot run_at must be positive")
        else:
            persisted_task = task
        definition = json.dumps(persisted_task.to_definition(), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        with self.transaction() as conn:
            existing = conn.execute("SELECT 1 FROM tasks WHERE task_id=?", (task.task_id,)).fetchone()
            if existing:
                raise ValueError("task_id already exists")
            if conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0] >= MAX_TASKS:
                raise CatalogError("scheduler task bound exceeded")
            conn.execute("INSERT INTO tasks VALUES(?,?,?,?,0,NULL,NULL,0)",
                         (task.task_id, definition, int(task.enabled), 0.0))

    def rows(self) -> list[sqlite3.Row]:
        with self.transaction() as conn:
            return list(conn.execute("SELECT * FROM tasks ORDER BY task_id"))

    def row(self, task_id: str) -> Optional[sqlite3.Row]:
        with self.transaction() as conn:
            return conn.execute("SELECT * FROM tasks WHERE task_id=?", (task_id,)).fetchone()

    def enabled(self, task_id: str, value: bool) -> bool:
        with self.transaction() as conn:
            changed = conn.execute("UPDATE tasks SET enabled=? WHERE task_id=?", (int(value), task_id)).rowcount
            return bool(changed)

    def remove(self, task_id: str) -> bool:
        with self.transaction() as conn:
            open_row = conn.execute("SELECT 1 FROM occurrences WHERE task_id=? AND state!='closed'", (task_id,)).fetchone()
            if open_row:
                raise CatalogError("task has an unreconciled occurrence")
            conn.execute("DELETE FROM occurrences WHERE task_id=?", (task_id,))
            return bool(conn.execute("DELETE FROM tasks WHERE task_id=?", (task_id,)).rowcount)

    def claim_due(self, now: float) -> list[tuple[ScheduledTask, int]]:
        claimed: list[tuple[ScheduledTask, int]] = []
        with self.transaction() as conn:
            rows = list(conn.execute("SELECT * FROM tasks WHERE enabled=1 ORDER BY task_id"))
            for row in rows:
                task = self.decode_task(row)
                slot = task.schedule.next_slot(float(row["cursor"]), now=now)
                if slot > now:
                    continue
                # At most one missed occurrence per pass; advance to now to prevent storms.
                if task.schedule.type.value == "interval" and now - slot > task.schedule.interval_seconds:
                    steps = int((now - slot) // task.schedule.interval_seconds)
                    slot += steps * task.schedule.interval_seconds
                cursor = slot
                if task.schedule.type.value == "cron" and now - slot > 60:
                    # Admit one deterministic overdue slot and skip the remaining
                    # historical window; the next pass starts strictly after now.
                    cursor = now
                key = int(slot)
                if conn.execute("SELECT COUNT(*) FROM occurrences").fetchone()[0] >= MAX_OCCURRENCES:
                    conn.execute(
                        "DELETE FROM occurrences WHERE rowid IN (SELECT rowid FROM occurrences "
                        "WHERE state='closed' ORDER BY slot LIMIT 1000)"
                    )
                if conn.execute("SELECT COUNT(*) FROM occurrences").fetchone()[0] >= MAX_OCCURRENCES:
                    raise CatalogError("scheduler occurrence bound exceeded")
                inserted = conn.execute("INSERT OR IGNORE INTO occurrences VALUES(?,?,'claimed',NULL,NULL,NULL,NULL)",
                                        (task.task_id, key)).rowcount
                conn.execute("UPDATE tasks SET cursor=? WHERE task_id=?", (cursor, task.task_id))
                if inserted:
                    claimed.append((task, key))
            # Reconciliation includes claims from a process that crashed before materialization.
            for row in conn.execute("SELECT o.*,t.definition,t.enabled,t.cursor,t.generation,t.root_id,t.item_count FROM occurrences o JOIN tasks t USING(task_id) WHERE o.state!='closed' ORDER BY o.task_id,o.slot"):
                pair = (self.decode_task(row), int(row["slot"]))
                if all((item.task_id, slot) != (pair[0].task_id, pair[1]) for item, slot in claimed):
                    claimed.append(pair)
        return claimed

    def public_results(self, task_id: str) -> list[dict[str, Any]]:
        with self.transaction() as conn:
            rows = conn.execute("SELECT result FROM occurrences WHERE task_id=? AND state='closed' ORDER BY slot DESC LIMIT ?",
                                (task_id, HISTORY_LIMIT)).fetchall()
        result = []
        for row in rows:
            try:
                value = json.loads(row[0])
            except (TypeError, json.JSONDecodeError) as exc:
                raise CatalogError("corrupt scheduler result") from exc
            if not isinstance(value, dict):
                raise CatalogError("corrupt scheduler result")
            result.append(value)
        return result
