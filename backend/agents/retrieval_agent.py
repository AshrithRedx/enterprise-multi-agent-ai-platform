"""Retrieval agent for resolving FAISS results to database chunks."""

from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.models.document import DocumentChunk
from backend.rag.vector_store import FaissVectorStore


@dataclass(frozen=True)
class RetrievedChunk:
    """A ranked document chunk returned by semantic retrieval."""

    text: str
    score: float
    document_id: str


class RetrievalAgent:
    """Search semantic vectors and hydrate results from SQLAlchemy models."""

    def __init__(self, vector_store: FaissVectorStore) -> None:
        self.vector_store = vector_store

    def search(
        self,
        *,
        query: str,
        session: Session,
        limit: int = 5,
    ) -> list[RetrievedChunk]:
        """Return ranked chunks for a user query."""
        matches = self.vector_store.search(query, limit)
        if not matches:
            return []

        chunk_ids = [chunk_id for chunk_id, _ in matches]
        chunks = session.scalars(
            select(DocumentChunk).where(DocumentChunk.id.in_(chunk_ids))
        ).all()
        chunks_by_id = {chunk.id: chunk for chunk in chunks}

        return [
            RetrievedChunk(
                text=chunks_by_id[chunk_id].text,
                score=score,
                document_id=chunks_by_id[chunk_id].document_id,
            )
            for chunk_id, score in matches
            if chunk_id in chunks_by_id
        ]

