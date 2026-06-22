"""SQLAlchemy database configuration and session management."""

from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from backend.config import settings


class Base(DeclarativeBase):
    """Base class for SQLAlchemy ORM models."""


engine_options: dict[str, object] = {"pool_pre_ping": True}
if settings.database_url.startswith("sqlite"):
    engine_options["connect_args"] = {"check_same_thread": False}

engine = create_engine(settings.database_url, **engine_options)
SessionLocal = sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def get_db() -> Generator[Session, None, None]:
    """Provide a transactional database session to request handlers."""
    with SessionLocal() as session:
        try:
            yield session
        except Exception:
            session.rollback()
            raise


def init_database() -> None:
    """Create tables registered on the shared metadata."""
    # Import model modules before table creation so their metadata is registered.
    from backend.models import document  # noqa: F401

    Base.metadata.create_all(bind=engine)
