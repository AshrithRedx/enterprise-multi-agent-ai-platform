"""Persistent FAISS vector store backed by database chunk identifiers."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Sequence

import faiss
import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.config import settings
from backend.models.document import DocumentChunk
from backend.rag.embeddings import Embedder, embedding_service


class FaissVectorStore:
    """Store and retrieve normalized chunk embeddings using cosine similarity."""

    def __init__(self, index_path: Path, embedder: Embedder) -> None:
        self.index_path = index_path
        self.embedder = embedder
        self._index: faiss.IndexIDMap2 | None = None
        self._lock = threading.RLock()
        self.load()

    @property
    def count(self) -> int:
        """Return the number of indexed vectors."""
        with self._lock:
            return 0 if self._index is None else self._index.ntotal

    def load(self) -> None:
        """Load a previously persisted FAISS index when present."""
        with self._lock:
            if not self.index_path.exists():
                self._index = None
                return

            try:
                loaded = faiss.read_index(str(self.index_path))
            except RuntimeError:
                # Startup synchronization will rebuild a damaged index from SQL.
                self._index = None
                return
            if not isinstance(loaded, faiss.IndexIDMap2):
                self._index = None
                return
            self._index = loaded

    def add(self, chunk_ids: Sequence[int], texts: Sequence[str]) -> None:
        """Embed and persist chunks using their database IDs as FAISS IDs."""
        if len(chunk_ids) != len(texts):
            raise ValueError("chunk_ids and texts must contain the same number of items")
        if not chunk_ids:
            return

        vectors = self._validate_vectors(self.embedder.encode(texts), len(texts))
        ids = np.ascontiguousarray(chunk_ids, dtype=np.int64)

        with self._lock:
            index = self._ensure_index(vectors.shape[1])
            existing_ids = self._index_ids(index)
            duplicates = existing_ids.intersection(int(value) for value in ids)
            if duplicates:
                index.remove_ids(
                    np.ascontiguousarray(list(duplicates), dtype=np.int64)
                )
            index.add_with_ids(vectors, ids)
            self._persist()

    def remove(self, chunk_ids: Sequence[int]) -> None:
        """Remove chunk vectors and persist the updated index."""
        if not chunk_ids:
            return

        with self._lock:
            if self._index is None:
                return
            ids = np.ascontiguousarray(chunk_ids, dtype=np.int64)
            self._index.remove_ids(ids)
            self._persist()

    def search(self, query: str, limit: int) -> list[tuple[int, float]]:
        """Return matching chunk IDs and cosine-similarity scores."""
        if not query.strip() or limit <= 0:
            return []

        query_vector = self._validate_vectors(self.embedder.encode([query]), 1)
        with self._lock:
            if self._index is None or self._index.ntotal == 0:
                return []
            if query_vector.shape[1] != self._index.d:
                raise ValueError(
                    "Embedding dimension does not match the persisted FAISS index"
                )
            scores, ids = self._index.search(
                query_vector,
                min(limit, self._index.ntotal),
            )

        return [
            (int(chunk_id), float(score))
            for chunk_id, score in zip(ids[0], scores[0], strict=True)
            if chunk_id >= 0
        ]

    def synchronize(self, session: Session) -> None:
        """Rebuild the index when it differs from database chunk metadata."""
        chunks = session.scalars(
            select(DocumentChunk).order_by(DocumentChunk.id)
        ).all()
        database_ids = {chunk.id for chunk in chunks}

        with self._lock:
            indexed_ids = (
                set() if self._index is None else self._index_ids(self._index)
            )
            if database_ids == indexed_ids:
                return

        self.rebuild(chunks)

    def rebuild(self, chunks: Sequence[DocumentChunk]) -> None:
        """Replace the index from authoritative database chunk records."""
        if not chunks:
            with self._lock:
                self._index = None
                self._delete_persisted_index()
            return

        vectors = self._validate_vectors(
            self.embedder.encode([chunk.text for chunk in chunks]),
            len(chunks),
        )
        ids = np.ascontiguousarray([chunk.id for chunk in chunks], dtype=np.int64)
        replacement = faiss.IndexIDMap2(faiss.IndexFlatIP(vectors.shape[1]))
        replacement.add_with_ids(vectors, ids)

        with self._lock:
            self._index = replacement
            self._persist()

    def _ensure_index(self, dimension: int) -> faiss.IndexIDMap2:
        if self._index is None:
            self._index = faiss.IndexIDMap2(faiss.IndexFlatIP(dimension))
        elif self._index.d != dimension:
            raise ValueError(
                "Embedding dimension does not match the persisted FAISS index"
            )
        return self._index

    def _persist(self) -> None:
        if self._index is None:
            return

        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = self.index_path.with_suffix(
            f"{self.index_path.suffix}.tmp"
        )
        faiss.write_index(self._index, str(temporary_path))
        temporary_path.replace(self.index_path)

    def _delete_persisted_index(self) -> None:
        self.index_path.unlink(missing_ok=True)

    @staticmethod
    def _index_ids(index: faiss.IndexIDMap2) -> set[int]:
        return {
            int(value)
            for value in faiss.vector_to_array(index.id_map).tolist()
        }

    @staticmethod
    def _validate_vectors(
        vectors: np.ndarray,
        expected_count: int,
    ) -> np.ndarray:
        normalized = np.ascontiguousarray(vectors, dtype=np.float32)
        if normalized.ndim != 2 or normalized.shape[0] != expected_count:
            raise ValueError("Embedder returned an invalid vector array")
        if normalized.shape[1] == 0:
            raise ValueError("Embeddings must have at least one dimension")
        return normalized


vector_store = FaissVectorStore(
    index_path=settings.vector_index_path,
    embedder=embedding_service,
)
