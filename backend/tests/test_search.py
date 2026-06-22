"""Semantic search endpoint tests."""

from fastapi.testclient import TestClient

from backend.agents.retrieval_agent import RetrievedChunk
from backend.api.routes import search
from backend.app import app


def test_search_endpoint_returns_retrieved_chunks(monkeypatch) -> None:
    monkeypatch.setattr(
        search.retrieval_agent,
        "search",
        lambda **_: [
            RetrievedChunk(
                text="Relevant enterprise knowledge",
                score=0.91,
                document_id="document-1",
            )
        ],
    )

    with TestClient(app) as client:
        response = client.post("/search", json={"query": "enterprise AI"})

    assert response.status_code == 200
    assert response.json() == {
        "chunks": [
            {
                "text": "Relevant enterprise knowledge",
                "score": 0.91,
                "document_id": "document-1",
            }
        ]
    }

