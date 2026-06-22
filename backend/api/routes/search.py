"""Semantic document search endpoint."""

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from backend.agents.retrieval_agent import RetrievalAgent
from backend.config import settings
from backend.database import get_db
from backend.models.search import SearchChunk, SearchRequest, SearchResponse
from backend.rag.vector_store import vector_store

router = APIRouter()
retrieval_agent = RetrievalAgent(vector_store)


@router.post(
    "/search",
    response_model=SearchResponse,
    summary="Search ingested document chunks",
)
def search_documents(
    request: SearchRequest,
    session: Session = Depends(get_db),
) -> SearchResponse:
    """Return the most semantically relevant stored chunks."""
    results = retrieval_agent.search(
        query=request.query,
        session=session,
        limit=settings.search_result_limit,
    )
    return SearchResponse(
        chunks=[
            SearchChunk(
                text=result.text,
                score=result.score,
                document_id=result.document_id,
            )
            for result in results
        ]
    )

