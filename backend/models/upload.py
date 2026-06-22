"""Document upload API schemas."""

from pydantic import BaseModel


class UploadResponse(BaseModel):
    """Result returned after successful document ingestion."""

    document_id: str
    chunk_count: int

