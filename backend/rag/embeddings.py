"""Sentence-transformer embedding infrastructure."""

from __future__ import annotations

from typing import Protocol, Sequence

import numpy as np
from numpy.typing import NDArray

from backend.config import settings


class Embedder(Protocol):
    """Interface used by the vector store for production and test embedders."""

    def encode(self, texts: Sequence[str]) -> NDArray[np.float32]:
        """Return one normalized embedding per input text."""


class SentenceTransformerEmbedder:
    """Lazy sentence-transformer model wrapper."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self._model: object | None = None

    def encode(self, texts: Sequence[str]) -> NDArray[np.float32]:
        """Generate L2-normalized float32 sentence embeddings."""
        if not texts:
            return np.empty((0, 0), dtype=np.float32)

        model = self._get_model()
        vectors = model.encode(  # type: ignore[attr-defined]
            list(texts),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.ascontiguousarray(vectors, dtype=np.float32)

    def _get_model(self) -> object:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model


embedding_service = SentenceTransformerEmbedder(settings.embedding_model_name)

