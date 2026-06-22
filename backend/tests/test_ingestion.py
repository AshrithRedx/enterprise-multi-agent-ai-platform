"""Document ingestion service tests."""

from pathlib import Path
from typing import Sequence

import numpy as np
from numpy.typing import NDArray
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from backend.database import Base
from backend.models.document import Document, DocumentChunk
from backend.rag.ingestion import DocumentIngestionService
from backend.rag.vector_store import FaissVectorStore


class TestEmbedder:
    """Deterministic embedder for ingestion tests."""

    __test__ = False

    def encode(self, texts: Sequence[str]) -> NDArray[np.float32]:
        return np.asarray(
            [
                [float("enterprise" in text.lower()), 1.0]
                for text in texts
            ],
            dtype=np.float32,
        )


def test_ingests_txt_document(tmp_path: Path) -> None:
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    service = DocumentIngestionService(
        upload_directory=tmp_path,
        chunk_size=30,
        chunk_overlap=5,
    )

    with Session(engine) as session:
        result = service.ingest(
            filename="example.txt",
            content=b"Enterprise AI document ingestion stores useful text safely.",
            content_type="text/plain",
            session=session,
        )

        document = session.get(Document, result.document_id)
        chunks = session.scalars(
            select(DocumentChunk)
            .where(DocumentChunk.document_id == result.document_id)
            .order_by(DocumentChunk.chunk_index)
        ).all()

    assert document is not None
    assert document.original_filename == "example.txt"
    assert document.chunk_count == result.chunk_count
    assert len(chunks) == result.chunk_count
    assert (tmp_path / document.stored_filename).exists()


def test_extracts_csv_as_tabular_text(tmp_path: Path) -> None:
    service = DocumentIngestionService(upload_directory=tmp_path)

    text = service.extract_text(b"name,role\nAda,Engineer", "csv")

    assert text == "name\trole\nAda\tEngineer"


def test_ingestion_automatically_indexes_new_chunks(tmp_path: Path) -> None:
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    store = FaissVectorStore(tmp_path / "chunks.faiss", TestEmbedder())
    service = DocumentIngestionService(
        upload_directory=tmp_path / "uploads",
        vector_store=store,
    )

    with Session(engine) as session:
        result = service.ingest(
            filename="enterprise.txt",
            content=b"Enterprise retrieval infrastructure",
            content_type="text/plain",
            session=session,
        )
        chunk_id = session.scalar(
            select(DocumentChunk.id).where(
                DocumentChunk.document_id == result.document_id
            )
        )

    matches = store.search("enterprise", limit=1)

    assert store.count == result.chunk_count
    assert matches[0][0] == chunk_id

