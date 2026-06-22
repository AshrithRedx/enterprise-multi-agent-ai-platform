"""Document upload and ingestion endpoint."""

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from backend.config import settings
from backend.database import get_db
from backend.models.upload import UploadResponse
from backend.rag.ingestion import DocumentIngestionService, IngestionError

router = APIRouter()

ingestion_service = DocumentIngestionService(
    upload_directory=settings.upload_directory,
    chunk_size=settings.chunk_size,
    chunk_overlap=settings.chunk_overlap,
)


@router.post(
    "/upload",
    response_model=UploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload and ingest a document",
)
async def upload_document(
    file: UploadFile = File(...),
    session: Session = Depends(get_db),
) -> UploadResponse:
    """Upload a supported document and store its extracted text chunks."""
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="A filename is required",
        )

    content = await file.read(settings.max_upload_size_bytes + 1)
    await file.close()

    if not content:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The uploaded file is empty",
        )
    if len(content) > settings.max_upload_size_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="The uploaded file exceeds the configured size limit",
        )

    try:
        result = ingestion_service.ingest(
            filename=file.filename,
            content=content,
            content_type=file.content_type,
            session=session,
        )
    except IngestionError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc

    return UploadResponse(
        document_id=result.document_id,
        chunk_count=result.chunk_count,
    )

