"""Service health endpoints."""

from fastapi import APIRouter, status

from backend.models.health import HealthResponse

router = APIRouter()


@router.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    summary="Check service health",
)
def health_check() -> HealthResponse:
    """Return the current API health state."""
    return HealthResponse(status="healthy")

