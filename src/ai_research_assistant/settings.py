from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class DatabaseSettings(BaseSettings):
    database_url: str

    database_pool_size: int = Field(default=5, ge=1)
    database_max_overflow: int = Field(default=5, ge=0)
    database_pool_timeout: int = Field(default=30, ge=1)
    database_pool_recycle: int = Field(default=1800, ge=0)

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
    )


class Settings(BaseSettings):
    gemini_api_key: str
    gemini_model: str
    gemini_embedding_model: str

    database_url: str
    database_test_url: str

    redis_url: str
    redis_embedding_ttl: int = Field(default=86400, ge=1)

    retrieval_similarity_threshold: float = Field(
        default=0.70,
        ge=0.0,
        le=1.0,
    )

    gemini_timeout_seconds: float = Field(
        default=30.0,
        gt=0.0,
    )

    gemini_max_retries: int = Field(
        default=2,
        ge=0,
        le=5,
    )

    gemini_retry_base_delay_seconds: float = Field(
        default=0.5,
        gt=0.0,
    )

    redis_connect_timeout_seconds: float = Field(
        default=2.0,
        gt=0.0,
    )

    redis_socket_timeout_seconds: float = Field(
        default=2.0,
        gt=0.0,
    )

    model_config = SettingsConfigDict(
        env_file=".env",
    )