"""FAISS persistence and synchronization tests."""

from pathlib import Path
from typing import Sequence

import numpy as np
import pytest
from numpy.typing import NDArray
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from backend.agents.retrieval_agent import RetrievalAgent
from backend.database import Base
from backend.models.document import Document, DocumentChunk
from backend.rag.vector_store import FaissVectorStore


class KeywordEmbedder:
    """Small deterministic embedder used without external model downloads."""

    vocabulary = ("database", "python", "finance")

    def encode(self, texts: Sequence[str]) -> NDArray[np.float32]:
        vectors: list[list[float]] = []
        for text in texts:
            lowered = text.lower()
            vector = [float(lowered.count(word)) for word in self.vocabulary]
            if not any(vector):
                vector = [1.0, 1.0, 1.0]
            values = np.asarray(vector, dtype=np.float32)
            values /= np.linalg.norm(values)
            vectors.append(values.tolist())
        return np.asarray(vectors, dtype=np.float32)


def create_test_engine():
    """Create an isolated in-memory SQLAlchemy engine."""
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    return engine


def test_persists_and_loads_faiss_index(tmp_path: Path) -> None:
    index_path = tmp_path / "chunks.faiss"
    store = FaissVectorStore(index_path, KeywordEmbedder())
    store.add(
        chunk_ids=[10, 20],
        texts=["Python application code", "Database indexing guide"],
    )

    reloaded = FaissVectorStore(index_path, KeywordEmbedder())
    matches = reloaded.search("database", limit=2)

    assert reloaded.count == 2
    assert matches[0][0] == 20
    assert matches[0][1] == pytest.approx(1.0)


def test_retrieval_agent_resolves_ranked_database_chunks(tmp_path: Path) -> None:
    engine = create_test_engine()
    store = FaissVectorStore(tmp_path / "chunks.faiss", KeywordEmbedder())

    with Session(engine) as session:
        document = Document(
            id="document-1",
            original_filename="knowledge.txt",
            stored_filename="document-1.txt",
            file_format="txt",
            content_type="text/plain",
            size_bytes=100,
            file_path="uploads/document-1.txt",
            chunk_count=2,
            chunks=[
                DocumentChunk(
                    chunk_index=0,
                    text="Python service architecture",
                    character_count=27,
                ),
                DocumentChunk(
                    chunk_index=1,
                    text="Database indexing and SQL",
                    character_count=25,
                ),
            ],
        )
        session.add(document)
        session.commit()
        store.synchronize(session)

        results = RetrievalAgent(store).search(
            query="database",
            session=session,
            limit=2,
        )

    assert results[0].text == "Database indexing and SQL"
    assert results[0].document_id == "document-1"
    assert results[0].score == pytest.approx(1.0)


def test_synchronize_rebuilds_an_outdated_index(tmp_path: Path) -> None:
    engine = create_test_engine()
    store = FaissVectorStore(tmp_path / "chunks.faiss", KeywordEmbedder())
    store.add([999], ["finance"])

    with Session(engine) as session:
        document = Document(
            id="document-2",
            original_filename="database.txt",
            stored_filename="document-2.txt",
            file_format="txt",
            content_type="text/plain",
            size_bytes=20,
            file_path="uploads/document-2.txt",
            chunk_count=1,
            chunks=[
                DocumentChunk(
                    chunk_index=0,
                    text="database",
                    character_count=8,
                )
            ],
        )
        session.add(document)
        session.commit()
        expected_chunk_id = document.chunks[0].id

        store.synchronize(session)
        matches = store.search("database", limit=5)

    assert store.count == 1
    assert matches[0][0] == expected_chunk_id
