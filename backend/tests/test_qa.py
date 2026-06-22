"""Question-answering agent and endpoint tests."""

from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from backend.agents.qa_agent import (
    INSUFFICIENT_CONTEXT_ANSWER,
    QAAgent,
)
from backend.agents.retrieval_agent import RetrievedChunk
from backend.api.routes import qa
from backend.models.qa import AskRequest


class FakeRetrievalAgent:
    """Return predefined chunks without FAISS or database access."""

    def __init__(self, chunks):
        self.chunks = chunks

    def search(self, **_):
        return self.chunks


class FakeGeminiClient:
    """Return a deterministic generated response."""

    def __init__(self, **_):
        self.models = self
        self.calls = []

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(text="A grounded answer.")


def test_qa_agent_retrieves_context_and_calls_gemini(monkeypatch) -> None:
    client = FakeGeminiClient()
    monkeypatch.setattr(
        "backend.agents.qa_agent.genai.Client",
        lambda **_: client,
    )
    agent = QAAgent(
        retrieval_agent=FakeRetrievalAgent(
            [
                RetrievedChunk("Relevant text", 0.9, "document-1"),
                RetrievedChunk("More text", 0.8, "document-1"),
            ]
        ),
        api_key="test-key",
        model="gemini-2.5-flash",
    )

    result = agent.answer(question="What is relevant?", session=None, limit=5)

    assert result.answer == "A grounded answer."
    assert result.document_ids == ["document-1"]
    assert "Relevant text" in client.calls[0]["contents"]


def test_qa_agent_returns_fallback_without_calling_gemini(monkeypatch) -> None:
    client = FakeGeminiClient()
    monkeypatch.setattr(
        "backend.agents.qa_agent.genai.Client",
        lambda **_: client,
    )
    agent = QAAgent(
        retrieval_agent=FakeRetrievalAgent([]),
        api_key="test-key",
        model="gemini-2.5-flash",
    )

    result = agent.answer(question="Unknown?", session=None, limit=5)

    assert result.answer == INSUFFICIENT_CONTEXT_ANSWER
    assert result.document_ids == []
    assert client.calls == []


def test_ask_returns_503_without_api_key(monkeypatch) -> None:
    monkeypatch.setattr(qa.settings, "gemini_api_key", None)

    with pytest.raises(HTTPException) as error:
        qa.ask_question(AskRequest(question="Question"), session=None)

    assert error.value.status_code == 503

