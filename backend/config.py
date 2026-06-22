"""Application configuration."""

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Environment-backed application settings."""

    app_name: str = "Enterprise Multi-Agent AI Platform"
    app_version: str = "0.1.0"
    api_prefix: str = "/api/v1"
    debug: bool = False
    database_url: str = Field(
        default="sqlite:///./enterprise_ai.db",
        description="SQLAlchemy-compatible database connection URL.",
    )
    upload_directory: Path = Path("backend/uploads")
    max_upload_size_bytes: int = 25 * 1024 * 1024
    chunk_size: int = 1_000
    chunk_overlap: int = 150
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    vector_index_path: Path = Path("backend/rag/data/chunks.faiss")
    search_result_limit: int = 5
    gemini_api_key: str | None = None


    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="APP_",
        case_sensitive=False,
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    """Return a cached settings instance."""
    return Settings()


settings = get_settings()
