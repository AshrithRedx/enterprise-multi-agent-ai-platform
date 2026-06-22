"""Top-level API router."""

from fastapi import APIRouter

from backend.api.routes.health import router as health_router
from backend.api.routes.upload import router as upload_router

root_router = APIRouter()
root_router.include_router(upload_router, tags=["documents"])

api_router = APIRouter()
api_router.include_router(health_router, tags=["health"])
