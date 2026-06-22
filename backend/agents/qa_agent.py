"""Grounded question answering with Gemini."""

from dataclasses import dataclass

from google import genai
from sqlalchemy.orm import Session

from backend.agents.retrieval_agent import RetrievalAgent

INSUFFICIENT_CONTEXT_ANSWER = (
    "I don't have enough information in the uploaded documents to answer that."
)


@dataclass(frozen=True)
class QAResult:
    """Generated answer and the source documents used for context."""

    answer: str
    document_ids: list[str]


class QAAgent:
    """Retrieve document context and generate a grounded Gemini answer."""

    def __init__(
        self,
        retrieval_agent: RetrievalAgent,
        api_key: str,
        model: str,
    ) -> None:
        self.retrieval_agent = retrieval_agent
        self.model = model
        self.client = genai.Client(api_key=api_key)

    def answer(
        self,
        *,
        question: str,
        session: Session,
        limit: int,
    ) -> QAResult:
        """Answer a question using only retrieved document chunks."""
        chunks = self.retrieval_agent.search(
            query=question,
            session=session,
            limit=limit,
        )
        if not chunks:
            return QAResult(
                answer=INSUFFICIENT_CONTEXT_ANSWER,
                document_ids=[],
            )

        context = "\n\n".join(
            f"[Source {index}; document_id={chunk.document_id}]\n{chunk.text}"
            for index, chunk in enumerate(chunks, start=1)
        )
        prompt = (
            "Answer the question using only the supplied context. "
            "If the context is insufficient, say that you do not have enough "
            "information in the uploaded documents. Do not use outside knowledge.\n\n"
            f"Context:\n{context}\n\nQuestion:\n{question}"
        )
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
        )
        answer = (response.text or "").strip() or INSUFFICIENT_CONTEXT_ANSWER
        document_ids = list(dict.fromkeys(chunk.document_id for chunk in chunks))
        return QAResult(answer=answer, document_ids=document_ids)

