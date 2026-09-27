"""SQLite persistence and cosine-similarity search over real embeddings."""

from __future__ import annotations

import json
import math
import os
import sqlite3
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Sequence


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if not left or len(left) != len(right):
        raise ValueError("vectors must be non-empty and have the same dimension")
    a = [float(value) for value in left]
    b = [float(value) for value in right]
    if not all(math.isfinite(value) for value in (*a, *b)):
        raise ValueError("vectors must contain only finite numbers")
    scale_a = max(abs(value) for value in a)
    scale_b = max(abs(value) for value in b)
    if scale_a == 0 or scale_b == 0:
        raise ValueError("zero-length vectors cannot be compared")
    scaled_a = [value / scale_a for value in a]
    scaled_b = [value / scale_b for value in b]
    norm_a = math.sqrt(sum(value * value for value in scaled_a))
    norm_b = math.sqrt(sum(value * value for value in scaled_b))
    normalized_a = [value / norm_a for value in scaled_a]
    normalized_b = [value / norm_b for value in scaled_b]
    return sum(x * y for x, y in zip(normalized_a, normalized_b))


class SQLiteVectorStore:
    """Persist vectors and metadata, then rank candidates using cosine similarity."""

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
                CREATE TABLE IF NOT EXISTS vectors (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    content TEXT NOT NULL,
                    embedding_json TEXT NOT NULL,
                    metadata_json TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS vectors_user_idx ON vectors(user_id)"
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

    def add(
        self,
        user_id: str,
        content: str,
        embedding: Sequence[float],
        metadata: dict | None = None,
        *,
        item_id: str | None = None,
    ) -> str:
        if not isinstance(user_id, str) or not user_id.strip():
            raise ValueError("user_id must be non-empty text")
        if not isinstance(content, str) or not content.strip() or len(content) > 20_000:
            raise ValueError("content must contain 1–20,000 characters")
        vector = [float(value) for value in embedding]
        if not vector or not all(math.isfinite(value) for value in vector):
            raise ValueError("embedding must be a non-empty finite numeric vector")
        if metadata is not None and not isinstance(metadata, dict):
            raise ValueError("metadata must be an object")
        identifier = item_id or str(uuid.uuid4())
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO vectors (id,user_id,content,embedding_json,metadata_json) "
                "VALUES (?,?,?,?,?)",
                (
                    identifier,
                    user_id,
                    content.strip(),
                    json.dumps(vector, allow_nan=False),
                    json.dumps(metadata or {}, ensure_ascii=False, allow_nan=False),
                ),
            )
        return identifier

    def search(
        self,
        user_id: str,
        query_embedding: Sequence[float],
        *,
        limit: int = 5,
        min_score: float = -1.0,
    ) -> list[dict]:
        if not isinstance(user_id, str) or not user_id.strip():
            raise ValueError("user_id must be non-empty text")
        if type(limit) is not int or not 1 <= limit <= 100:
            raise ValueError("limit must be an integer between 1 and 100")
        if not math.isfinite(min_score) or not -1 <= min_score <= 1:
            raise ValueError("min_score must be between -1 and 1")
        query = [float(value) for value in query_embedding]
        if not query or not all(math.isfinite(value) for value in query):
            raise ValueError("query embedding must be a non-empty finite vector")
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT id,content,embedding_json,metadata_json,created_at "
                "FROM vectors WHERE user_id=?",
                (user_id,),
            ).fetchall()
        matches = []
        for row in rows:
            embedding = json.loads(row["embedding_json"])
            score = cosine_similarity(query, embedding)
            if score >= min_score:
                matches.append(
                    {
                        "id": row["id"],
                        "content": row["content"],
                        "metadata": json.loads(row["metadata_json"]),
                        "created_at": row["created_at"],
                        "score": score,
                    }
                )
        matches.sort(key=lambda item: (-item["score"], item["id"]))
        return matches[:limit]

    def list(self, user_id: str, limit: int = 50) -> list[dict]:
        if not isinstance(user_id, str) or not user_id.strip():
            raise ValueError("user_id must be non-empty text")
        if type(limit) is not int or not 1 <= limit <= 100:
            raise ValueError("limit must be an integer between 1 and 100")
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT id,content,metadata_json,created_at FROM vectors "
                "WHERE user_id=? ORDER BY created_at DESC,id DESC LIMIT ?",
                (user_id, limit),
            ).fetchall()
        return [
            {
                "id": row["id"],
                "content": row["content"],
                "metadata": json.loads(row["metadata_json"]),
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    def delete(self, user_id: str, item_id: str) -> bool:
        with self._connect() as connection:
            cursor = connection.execute(
                "DELETE FROM vectors WHERE user_id=? AND id=?", (user_id, item_id)
            )
            return cursor.rowcount == 1