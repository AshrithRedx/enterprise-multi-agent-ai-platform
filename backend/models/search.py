"""Semantic search API schemas."""

from pydantic import BaseModel, Field, field_validator


class SearchRequest(BaseModel):
    """Semantic search request."""

    query: str = Field(min_length=1, max_length=2_000)

    @field_validator("query")
    @classmethod
    def validate_query(cls, value: str) -> str:
        """Reject whitespace-only searches and normalize surrounding space."""
        normalized = value.strip()
        if not normalized:
            raise ValueError("query must not be empty")
        return normalized


class SearchChunk(BaseModel):
    """A retrieved source chunk."""

    text: str
    score: float
    document_id: str


class SearchResponse(BaseModel):
    """Ranked semantic search results."""

    chunks: list[SearchChunk]
