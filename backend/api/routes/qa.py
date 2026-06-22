"""Grounded question-answering endpoint."""

from fastapi import APIRouter, Depends, HTTPException, status
from google.genai.errors import APIError
from sqlalchemy.orm import Session

from backend.agents.qa_agent import QAAgent
from backend.agents.retrieval_agent import RetrievalAgent
from backend.config import settings
from backend.database import get_db
from backend.models.qa import AskRequest, AskResponse
from backend.rag.vector_store import vector_store

router = APIRouter()
retrieval_agent = RetrievalAgent(vector_store)


@router.post(
    "/ask",
    response_model=AskResponse,
    summary="Answer a question using uploaded documents",
)
def ask_question(
    request: AskRequest,
    session: Session = Depends(get_db),
) -> AskResponse:
    """Retrieve relevant chunks and generate a grounded Gemini answer."""
    if not settings.gemini_api_key:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Gemini API key is not configured",
        )

    try:
        agent = QAAgent(
            retrieval_agent=retrieval_agent,
            api_key=settings.gemini_api_key,
            model=settings.gemini_model,
        )
        result = agent.answer(
            question=request.question,
            session=session,
            limit=settings.search_result_limit,
        )
    except APIError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Gemini could not generate an answer",
        ) from exc

    return AskResponse(
        answer=result.answer,
        document_ids=result.document_ids,
    )
