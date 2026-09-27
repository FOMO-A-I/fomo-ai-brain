"""Embedding-backed retrieval; embedding failures are never replaced by keyword guesses."""

from __future__ import annotations

from .embeddings import Embedder
from .vector_store import SQLiteVectorStore


class MemoryRetriever:
    def __init__(self, embedder: Embedder, store: SQLiteVectorStore) -> None:
        self.embedder = embedder
        self.store = store

    def remember(
        self, user_id: str, content: str, metadata: dict | None = None
    ) -> str:
        vector = self.embedder.embed(content)
        return self.store.add(user_id, content, vector, metadata)

    def retrieve(
        self,
        user_id: str,
        query: str,
        *,
        limit: int = 5,
        min_score: float = -1.0,
    ) -> list[dict]:
        vector = self.embedder.embed(query)
        return self.store.search(
            user_id, vector, limit=limit, min_score=min_score
        )