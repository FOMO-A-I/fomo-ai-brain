"""Durable user-scoped memories stored in SQLite."""

from __future__ import annotations

import json
import math
import os
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


class LongTermMemory:
    """Store explicit memories with SQLite parameterization and user isolation."""

    def __init__(self, path: str | Path) -> None:
        path = Path(path).expanduser()
        if str(path) == ":memory:":
            raise ValueError("use a file path; each operation opens its own connection")
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
        except FileExistsError:
            os.chmod(path, 0o600)
        else:
            os.close(descriptor)
        self.path = str(path)
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    content TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS memories_user_created_idx "
                "ON memories(user_id, created_at DESC)"
            )

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    @staticmethod
    def _validate_user(user_id: str) -> None:
        if not isinstance(user_id, str) or not user_id.strip() or len(user_id) > 200:
            raise ValueError("user_id must contain 1–200 characters")

    def add(
        self, user_id: str, content: str, metadata: dict[str, Any] | None = None
    ) -> str:
        self._validate_user(user_id)
        if not isinstance(content, str) or not content.strip() or len(content) > 20_000:
            raise ValueError("memory content must contain 1–20,000 characters")
        if metadata is not None and not isinstance(metadata, dict):
            raise ValueError("metadata must be an object")
        metadata_json = json.dumps(metadata or {}, ensure_ascii=False, allow_nan=False)
        identifier = str(uuid.uuid4())
        created_at = datetime.now(timezone.utc).isoformat()
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO memories (id,user_id,content,metadata_json,created_at) "
                "VALUES (?,?,?,?,?)",
                (identifier, user_id, content.strip(), metadata_json, created_at),
            )
        return identifier

    def list(self, user_id: str, limit: int = 50) -> list[dict[str, Any]]:
        self._validate_user(user_id)
        if type(limit) is not int or not 1 <= limit <= 500:
            raise ValueError("limit must be an integer between 1 and 500")
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT id,user_id,content,metadata_json,created_at FROM memories "
                "WHERE user_id=? ORDER BY created_at DESC,id DESC LIMIT ?",
                (user_id, limit),
            ).fetchall()
        return [
            {
                "id": row["id"],
                "user_id": row["user_id"],
                "content": row["content"],
                "metadata": json.loads(row["metadata_json"]),
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    def delete(self, user_id: str, memory_id: str) -> bool:
        self._validate_user(user_id)
        if not isinstance(memory_id, str) or not memory_id:
            raise ValueError("memory_id must be non-empty text")
        with self._connect() as connection:
            cursor = connection.execute(
                "DELETE FROM memories WHERE user_id=? AND id=?",
                (user_id, memory_id),
            )
            return cursor.rowcount == 1

    def delete_user(self, user_id: str) -> int:
        """Delete all durable memories for a user (for privacy/retention workflows)."""
        self._validate_user(user_id)
        with self._connect() as connection:
            cursor = connection.execute("DELETE FROM memories WHERE user_id=?", (user_id,))
            return cursor.rowcount