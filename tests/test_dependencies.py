import os
from unittest.mock import patch

from ai_research_assistant.dependencies import (
    get_embedding_cache,
    get_llm_provider,
    get_settings,
)


MOCK_ENV = {
    "GEMINI_API_KEY": "test-key",
    "GEMINI_MODEL": "gemini-pro",
    "GEMINI_EMBEDDING_MODEL": "embedding-001",
    "DATABASE_URL": "postgresql://user:pass@localhost/db",
    "DATABASE_TEST_URL": "postgresql://user:pass@localhost/test_db",
    "REDIS_URL": "redis://localhost:6379",
}


@patch.dict(os.environ, MOCK_ENV)
def test_settings_is_cached():
    get_settings.cache_clear()

    first = get_settings()
    second = get_settings()

    assert first is second

    get_settings.cache_clear()


@patch.dict(os.environ, MOCK_ENV)
def test_embedding_cache_is_cached():
    get_settings.cache_clear()
    get_embedding_cache.cache_clear()

    first = get_embedding_cache()
    second = get_embedding_cache()

    assert first is second

    get_embedding_cache.cache_clear()
    get_settings.cache_clear()


@patch.dict(os.environ, MOCK_ENV)
def test_llm_provider_is_cached():
    get_settings.cache_clear()
    get_llm_provider.cache_clear()

    first = get_llm_provider()
    second = get_llm_provider()

    assert first is second

    get_llm_provider.cache_clear()
    get_settings.cache_clear()