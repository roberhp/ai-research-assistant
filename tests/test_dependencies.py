from ai_research_assistant.dependencies import (
    get_embedding_cache,
    get_llm_provider,
    get_settings,
)


def test_settings_is_cached():
    get_settings.cache_clear()

    first = get_settings()
    second = get_settings()

    assert first is second


def test_embedding_cache_is_cached():
    get_embedding_cache.cache_clear()

    first = get_embedding_cache()
    second = get_embedding_cache()

    assert first is second


def test_llm_provider_is_cached():
    get_llm_provider.cache_clear()

    first = get_llm_provider()
    second = get_llm_provider()

    assert first is second