"""FastAPI application entry point."""

from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

from fastapi import FastAPI

from backend.api.router import api_router, root_router
from backend.config import settings
from backend.database import SessionLocal, init_database
from backend.rag.vector_store import vector_store


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Initialize and release application resources."""
    init_database()
    vector_store.load()
    with SessionLocal() as session:
        vector_store.synchronize(session)
    yield


def create_app() -> FastAPI:
    """Application factory used by production servers and tests."""
    application = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        debug=settings.debug,
        lifespan=lifespan,
    )
    application.include_router(root_router)
    application.include_router(api_router, prefix=settings.api_prefix)
    return application


app = create_app()
