"""Session store — SQLite-backed persistence for chat sessions."""

from __future__ import annotations

import asyncio
import json
import logging
import sqlite3
import uuid
from datetime import UTC, datetime
from pathlib import Path

import aiosqlite

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    model TEXT NOT NULL,
    skill TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    metadata TEXT DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    model TEXT,
    token_count INTEGER DEFAULT 0,
    created_at TEXT NOT NULL,
    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS context_files (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    file_path TEXT NOT NULL,
    file_type TEXT NOT NULL,
    size_bytes INTEGER DEFAULT 0,
    added_at TEXT NOT NULL,
    FOREIGN KEY (session_id) REFERENCES sessions(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id);
CREATE INDEX IF NOT EXISTS idx_context_session ON context_files(session_id);
"""


class SessionStore:
    """SQLite-backed session persistence."""

    def __init__(self, db_path: str | Path) -> None:
        self._db_path = Path(db_path).expanduser()
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection | None = None
        self._aconn: aiosqlite.Connection | None = None
        self._async_lock = asyncio.Lock()

    def _get_conn(self) -> sqlite3.Connection:
        """Get or create the database connection."""
        if self._conn is None:
            self._conn = sqlite3.connect(str(self._db_path))
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
            self._conn.executescript(_SCHEMA)
            logger.debug("Database opened: %s", self._db_path)
        return self._conn

    def create_session(
        self,
        name: str,
        model: str,
        skill: str | None = None,
        metadata: dict | None = None,
    ) -> str:
        """Create a new session. Returns the session ID."""
        conn = self._get_conn()
        session_id = str(uuid.uuid4())
        now = datetime.now(UTC).isoformat()

        conn.execute(
            "INSERT INTO sessions (id, name, model, skill, created_at, updated_at, metadata) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (session_id, name, model, skill, now, now, json.dumps(metadata or {})),
        )
        conn.commit()
        logger.info("Created session: %s (%s)", name, session_id[:8])
        return session_id

    def add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        model: str | None = None,
        token_count: int = 0,
    ) -> None:
        """Add a message to a session."""
        conn = self._get_conn()
        now = datetime.now(UTC).isoformat()

        conn.execute(
            "INSERT INTO messages (session_id, role, content, model, token_count, created_at) VALUES (?, ?, ?, ?, ?, ?)",
            (session_id, role, content, model, token_count, now),
        )
        conn.execute(
            "UPDATE sessions SET updated_at = ? WHERE id = ?",
            (now, session_id),
        )
        conn.commit()

    def get_messages(self, session_id: str, *, limit: int | None = None) -> list[dict]:
        """Get all messages for a session, ordered by creation time."""
        conn = self._get_conn()
        query = "SELECT role, content, model, token_count, created_at FROM messages WHERE session_id = ? ORDER BY id ASC"
        params: list = [session_id]

        if limit:
            query += " LIMIT ?"
            params.append(limit)

        rows = conn.execute(query, params).fetchall()
        return [dict(row) for row in rows]

    def list_sessions(self, *, limit: int = 20) -> list[dict]:
        """List recent sessions."""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT id, name, model, skill, created_at, updated_at, "
            "(SELECT COUNT(*) FROM messages WHERE session_id = sessions.id) as message_count "
            "FROM sessions ORDER BY updated_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(row) for row in rows]

    def get_session(self, session_id: str) -> dict | None:
        """Get session details by ID."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM sessions WHERE id = ?",
            (session_id,),
        ).fetchone()
        return dict(row) if row else None

    def delete_session(self, session_id: str) -> bool:
        """Delete a session and all its messages.

        Returns:
            True if the session existed and was deleted, False otherwise.
        """
        conn = self._get_conn()
        cursor = conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        conn.commit()
        return cursor.rowcount > 0

    def add_context_file(
        self,
        session_id: str,
        file_path: str,
        file_type: str,
        size_bytes: int = 0,
    ) -> None:
        """Record a context file added to a session."""
        conn = self._get_conn()
        now = datetime.now(UTC).isoformat()
        conn.execute(
            "INSERT INTO context_files (session_id, file_path, file_type, size_bytes, added_at) VALUES (?, ?, ?, ?, ?)",
            (session_id, file_path, file_type, size_bytes, now),
        )
        conn.commit()

    def get_context_files(self, session_id: str) -> list[dict]:
        """Get all context files for a session."""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT file_path, file_type, size_bytes, added_at FROM context_files WHERE session_id = ? ORDER BY added_at",
            (session_id,),
        ).fetchall()
        return [dict(row) for row in rows]

    def update_metadata(self, session_id: str, metadata: dict) -> bool:
        """Update the metadata JSON for a session.

        Performs a **merge** — existing keys are preserved unless overwritten
        by the new *metadata* dict.

        Returns:
            True if the session existed and was updated, False otherwise.
        """
        conn = self._get_conn()

        row = conn.execute(
            "SELECT metadata FROM sessions WHERE id = ?",
            (session_id,),
        ).fetchone()
        if row is None:
            return False

        existing: dict = json.loads(row["metadata"] or "{}")
        existing.update(metadata)

        now = datetime.now(UTC).isoformat()
        conn.execute(
            "UPDATE sessions SET metadata = ?, updated_at = ? WHERE id = ?",
            (json.dumps(existing), now, session_id),
        )
        conn.commit()
        return True

    def get_metadata(self, session_id: str) -> dict | None:
        """Get parsed metadata dict for a session.

        Returns:
            The metadata dict, or None if the session does not exist.
        """
        conn = self._get_conn()
        row = conn.execute(
            "SELECT metadata FROM sessions WHERE id = ?",
            (session_id,),
        ).fetchone()
        if row is None:
            return None
        return json.loads(row["metadata"] or "{}")

    def rename_session(self, session_id: str, new_name: str) -> bool:
        """Rename a session.

        Returns:
            True if the session existed and was renamed, False otherwise.
        """
        conn = self._get_conn()
        now = datetime.now(UTC).isoformat()
        cursor = conn.execute(
            "UPDATE sessions SET name = ?, updated_at = ? WHERE id = ?",
            (new_name, now, session_id),
        )
        conn.commit()
        return cursor.rowcount > 0

    def search_sessions(self, query: str) -> list[dict]:
        """Search sessions by name or message content."""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT DISTINCT s.id, s.name, s.model, s.skill, s.created_at, s.updated_at, "
            "(SELECT COUNT(*) FROM messages WHERE session_id = s.id) as message_count "
            "FROM sessions s "
            "LEFT JOIN messages m ON m.session_id = s.id "
            "WHERE s.name LIKE ? OR m.content LIKE ? "
            "ORDER BY s.updated_at DESC",
            (f"%{query}%", f"%{query}%"),
        ).fetchall()
        return [dict(row) for row in rows]

    def get_last_session(self) -> dict | None:
        """Get the most recently updated session."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT id, name, model, skill, created_at, updated_at, "
            "(SELECT COUNT(*) FROM messages WHERE session_id = sessions.id) as message_count "
            "FROM sessions ORDER BY updated_at DESC LIMIT 1",
        ).fetchone()
        return dict(row) if row else None

    def close(self) -> None:
        """Close the database connection.

        Also cleans up any orphaned aiosqlite connection to prevent
        worker-thread leaks (e.g. when sync close is called after async usage).
        """
        if self._conn:
            self._conn.close()
            self._conn = None
        if self._aconn:
            # Can't await in sync code, so use aiosqlite's stop() which
            # puts a sentinel on the worker thread's queue, causing it to
            # close the connection and exit cleanly.
            try:
                self._aconn.stop()
            except Exception:  # noqa: BLE001
                pass
            self._aconn = None

    # ── Async connection management ──────────────────────────

    async def _get_aconn(self) -> aiosqlite.Connection:
        """Get or create the async database connection (lazy, WAL mode)."""
        if self._aconn is None:
            self._aconn = await aiosqlite.connect(str(self._db_path))
            self._aconn.row_factory = aiosqlite.Row
            await self._aconn.execute("PRAGMA journal_mode=WAL")
            await self._aconn.execute("PRAGMA foreign_keys=ON")
            await self._aconn.executescript(_SCHEMA)
            logger.debug("Async database opened: %s", self._db_path)
        return self._aconn

    # ── Async public methods ─────────────────────────────────

    async def async_create_session(
        self,
        name: str,
        model: str,
        skill: str | None = None,
        metadata: dict | None = None,
    ) -> str:
        """Async version of :meth:`create_session`."""
        async with self._async_lock:
            conn = await self._get_aconn()
            session_id = str(uuid.uuid4())
            now = datetime.now(UTC).isoformat()

            await conn.execute(
                "INSERT INTO sessions (id, name, model, skill, created_at, updated_at, metadata) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (session_id, name, model, skill, now, now, json.dumps(metadata or {})),
            )
            await conn.commit()
            logger.info("Created session (async): %s (%s)", name, session_id[:8])
            return session_id

    async def async_add_message(
        self,
        session_id: str,
        role: str,
        content: str,
        model: str | None = None,
        token_count: int = 0,
    ) -> None:
        """Async version of :meth:`add_message`."""
        async with self._async_lock:
            conn = await self._get_aconn()
            now = datetime.now(UTC).isoformat()

            await conn.execute(
                "INSERT INTO messages (session_id, role, content, model, token_count, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (session_id, role, content, model, token_count, now),
            )
            await conn.execute(
                "UPDATE sessions SET updated_at = ? WHERE id = ?",
                (now, session_id),
            )
            await conn.commit()

    async def async_get_messages(self, session_id: str, *, limit: int | None = None) -> list[dict]:
        """Async version of :meth:`get_messages`."""
        async with self._async_lock:
            conn = await self._get_aconn()
            query = "SELECT role, content, model, token_count, created_at FROM messages WHERE session_id = ? ORDER BY id ASC"
            params: list = [session_id]

            if limit:
                query += " LIMIT ?"
                params.append(limit)

            cursor = await conn.execute(query, params)
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]

    async def async_list_sessions(self, *, limit: int = 20) -> list[dict]:
        """Async version of :meth:`list_sessions`."""
        async with self._async_lock:
            conn = await self._get_aconn()
            cursor = await conn.execute(
                "SELECT id, name, model, skill, created_at, updated_at, "
                "(SELECT COUNT(*) FROM messages WHERE session_id = sessions.id) as message_count "
                "FROM sessions ORDER BY updated_at DESC LIMIT ?",
                (limit,),
            )
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]

    async def async_get_session(self, session_id: str) -> dict | None:
        """Async version of :meth:`get_session`."""
        async with self._async_lock:
            conn = await self._get_aconn()
            cursor = await conn.execute(
                "SELECT * FROM sessions WHERE id = ?",
                (session_id,),
            )
            row = await cursor.fetchone()
            return dict(row) if row else None

    async def async_delete_session(self, session_id: str) -> bool:
        """Async version of :meth:`delete_session`."""
        async with self._async_lock:
            conn = await self._get_aconn()
            cursor = await conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
            await conn.commit()
            return cursor.rowcount > 0

    async def async_add_context_file(
        self,
        session_id: str,
        file_path: str,
        file_type: str,
        size_bytes: int = 0,
    ) -> None:
        """Async version of :meth:`add_context_file`."""
        async with self._async_lock:
            conn = await self._get_aconn()
            now = datetime.now(UTC).isoformat()
            await conn.execute(
                "INSERT INTO context_files (session_id, file_path, file_type, size_bytes, added_at) VALUES (?, ?, ?, ?, ?)",
                (session_id, file_path, file_type, size_bytes, now),
            )
            await conn.commit()

    async def async_get_context_files(self, session_id: str) -> list[dict]:
        """Async version of :meth:`get_context_files`."""
        async with self._async_lock:
            conn = await self._get_aconn()
            cursor = await conn.execute(
                "SELECT file_path, file_type, size_bytes, added_at FROM context_files WHERE session_id = ? ORDER BY added_at",
                (session_id,),
            )
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]

    async def async_update_metadata(self, session_id: str, metadata: dict) -> bool:
        """Async version of :meth:`update_metadata`.

        Performs a **merge** — existing keys are preserved unless overwritten.
        """
        async with self._async_lock:
            conn = await self._get_aconn()

            cursor = await conn.execute(
                "SELECT metadata FROM sessions WHERE id = ?",
                (session_id,),
            )
            row = await cursor.fetchone()
            if row is None:
                return False

            existing: dict = json.loads(row["metadata"] or "{}")
            existing.update(metadata)

            now = datetime.now(UTC).isoformat()
            await conn.execute(
                "UPDATE sessions SET metadata = ?, updated_at = ? WHERE id = ?",
                (json.dumps(existing), now, session_id),
            )
            await conn.commit()
            return True

    async def async_get_metadata(self, session_id: str) -> dict | None:
        """Async version of :meth:`get_metadata`."""
        async with self._async_lock:
            conn = await self._get_aconn()
            cursor = await conn.execute(
                "SELECT metadata FROM sessions WHERE id = ?",
                (session_id,),
            )
            row = await cursor.fetchone()
            if row is None:
                return None
            return json.loads(row["metadata"] or "{}")

    async def async_rename_session(self, session_id: str, new_name: str) -> bool:
        """Async version of :meth:`rename_session`."""
        async with self._async_lock:
            conn = await self._get_aconn()
            now = datetime.now(UTC).isoformat()
            cursor = await conn.execute(
                "UPDATE sessions SET name = ?, updated_at = ? WHERE id = ?",
                (new_name, now, session_id),
            )
            await conn.commit()
            return cursor.rowcount > 0

    async def async_search_sessions(self, query: str) -> list[dict]:
        """Async version of :meth:`search_sessions`."""
        async with self._async_lock:
            conn = await self._get_aconn()
            cursor = await conn.execute(
                "SELECT DISTINCT s.id, s.name, s.model, s.skill, s.created_at, s.updated_at, "
                "(SELECT COUNT(*) FROM messages WHERE session_id = s.id) as message_count "
                "FROM sessions s "
                "LEFT JOIN messages m ON m.session_id = s.id "
                "WHERE s.name LIKE ? OR m.content LIKE ? "
                "ORDER BY s.updated_at DESC",
                (f"%{query}%", f"%{query}%"),
            )
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]

    async def async_get_last_session(self) -> dict | None:
        """Async version of :meth:`get_last_session`."""
        async with self._async_lock:
            conn = await self._get_aconn()
            cursor = await conn.execute(
                "SELECT id, name, model, skill, created_at, updated_at, "
                "(SELECT COUNT(*) FROM messages WHERE session_id = sessions.id) as message_count "
                "FROM sessions ORDER BY updated_at DESC LIMIT 1",
            )
            row = await cursor.fetchone()
            return dict(row) if row else None

    async def async_close(self) -> None:
        """Close the async database connection."""
        if self._aconn:
            await self._aconn.close()
            self._aconn = None
