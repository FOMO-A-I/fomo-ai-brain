"""Strict local embedding interface; never invent vectors or silently download."""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Protocol, Sequence


class EmbeddingUnavailable(RuntimeError):
    """Raised when a configured embedding model cannot be loaded locally."""


class Embedder(Protocol):
    def embed(self, text: str) -> list[float]: ...


class LocalSentenceTransformerEmbedder:
    """Load a pre-downloaded sentence-transformer directory with networking off."""

    def __init__(self, model_path: str | Path, *, device: str | None = None) -> None:
        path = Path(model_path).expanduser()
        if not path.is_dir():
            raise EmbeddingUnavailable(
                f"Local embedding model directory does not exist: {path}"
            )
        # The model must already be present. These settings prevent Hub fallback.
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise EmbeddingUnavailable(
                "Install the optional sentence-transformers package to use embeddings"
            ) from exc
        try:
            self._model = SentenceTransformer(str(path), device=device)
            self.dimension = int(self._model.get_sentence_embedding_dimension())
        except Exception as exc:
            raise EmbeddingUnavailable(
                f"Could not load local embedding model at {path}: {exc}"
            ) from exc
        if self.dimension < 1:
            raise EmbeddingUnavailable("embedding model reported an invalid dimension")

    def embed(self, text: str) -> list[float]:
        if not isinstance(text, str) or not text.strip():
            raise ValueError("embedding input must be non-empty text")
        try:
            raw: Sequence[float] = self._model.encode(
                text, normalize_embeddings=False, convert_to_numpy=False
            )
            vector = [float(value) for value in raw]
        except Exception as exc:
            raise EmbeddingUnavailable(f"Local embedding inference failed: {exc}") from exc
        if len(vector) != self.dimension or not all(math.isfinite(v) for v in vector):
            raise EmbeddingUnavailable("embedding model returned an invalid vector")
        return vector