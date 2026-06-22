"""Question-answering API schemas."""

from pydantic import BaseModel, Field, field_validator


class AskRequest(BaseModel):
    """A grounded question-answering request."""

    question: str = Field(min_length=1, max_length=2_000)

    @field_validator("question")
    @classmethod
    def validate_question(cls, value: str) -> str:
        """Reject whitespace-only questions."""
        normalized = value.strip()
        if not normalized:
            raise ValueError("question must not be empty")
        return normalized


class AskResponse(BaseModel):
    """A grounded answer and its source document identifiers."""

    answer: str
    document_ids: list[str]

