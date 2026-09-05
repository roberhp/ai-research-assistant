from types import SimpleNamespace

from ai_research_assistant.rag.embeddings.embedding_service import (
    EmbeddingService,
)

from unittest.mock import Mock
import pytest
from ai_research_assistant.exceptions import LLMProviderError
from ai_research_assistant.llm.gemini_provider import GeminiProvider
from ai_research_assistant.settings import Settings


class FakeEmbeddingCache:
    def __init__(self):
        self.values = {}
        self.get_calls = 0
        self.set_calls = 0
        self.last_key = None
        self.last_embedding = None
        self.last_ttl = None

    def get(self, key: str):
        self.get_calls += 1
        self.last_key = key

        return self.values.get(key)

    def set(
        self,
        key: str,
        embedding: list[float],
        ttl: int,
    ):
        self.set_calls += 1
        self.last_key = key
        self.last_embedding = embedding
        self.last_ttl = ttl

        self.values[key] = embedding


class FakeGeminiClient:
    def __init__(self):
        self.calls = 0

    def embed_content(
        self,
        model: str,
        contents: str,
    ):
        self.calls += 1

        return SimpleNamespace(
            embeddings=[
                SimpleNamespace(
                    values=[1.0, 2.0, 3.0]
                )
            ]
        )


class FakeModels:
    def __init__(self, client):
        self.client = client

    def embed_content(
        self,
        model: str,
        contents: str,
    ):
        return self.client.embed_content(
            model=model,
            contents=contents,
        )


class FakeClient:
    def __init__(self):
        self.embedding_client = FakeGeminiClient()
        self.models = FakeModels(
            self.embedding_client
        )


class FakeSettings:
    gemini_api_key = "test-key"
    gemini_embedding_model = "test-model"
    redis_embedding_ttl = 3600


def create_service(cache=None):
    service = EmbeddingService(
        settings=FakeSettings(),
        cache=cache,
    )

    service.client = FakeClient()

    return service


def test_embedding_service_generates_embedding():
    service = create_service()

    result = service.generate(
        "What is RAG?"
    )

    assert result == [1.0, 2.0, 3.0]


def test_embedding_service_returns_cached_embedding():
    cache = FakeEmbeddingCache()

    cache.values["cached-key"] = [
        10.0,
        20.0,
        30.0,
    ]

    service = create_service(
        cache=cache
    )

    key = service._build_cache_key(
        "What is RAG?"
    )

    cache.values[key] = [
        10.0,
        20.0,
        30.0,
    ]

    fake_client = service.client

    result = service.generate(
        "What is RAG?"
    )

    assert result == [
        10.0,
        20.0,
        30.0,
    ]

    assert cache.get_calls == 1
    assert cache.set_calls == 0

    assert fake_client.embedding_client.calls == 0


def test_embedding_service_generates_and_caches_on_cache_miss():
    cache = FakeEmbeddingCache()

    service = create_service(
        cache=cache
    )

    result = service.generate(
        "What is RAG?"
    )

    assert result == [
        1.0,
        2.0,
        3.0,
    ]

    assert cache.get_calls == 1
    assert cache.set_calls == 1

    assert cache.last_embedding == [
        1.0,
        2.0,
        3.0,
    ]

    assert cache.last_ttl == 3600

    assert service.client.embedding_client.calls == 1


def test_embedding_service_generates_without_cache():
    service = create_service()

    result = service.generate(
        "What is RAG?"
    )

    assert result == [
        1.0,
        2.0,
        3.0,
    ]

    assert service.client.embedding_client.calls == 1

def create_settings():
    return Settings(
        gemini_api_key="test-key",
        gemini_model="test-model",
        gemini_embedding_model="test-embedding-model",
        database_url="postgresql://test",
        database_test_url="postgresql://test",
        redis_url="redis://localhost:6379",
    )


def test_generate_returns_response():
    provider = GeminiProvider(create_settings())

    response = Mock()
    response.text = "Test response"

    provider.client.models.generate_content = Mock(
        return_value=response
    )

    result = provider.generate("Test prompt")

    assert result == "Test response"

    provider.client.models.generate_content.assert_called_once_with(
        model="test-model",
        contents="Test prompt",
    )


def test_generate_retries_after_failure():
    settings = create_settings()

    provider = GeminiProvider(settings)
    provider.max_retries = 2
    provider.retry_base_delay = 0

    response = Mock()
    response.text = "Success"

    provider.client.models.generate_content = Mock(
        side_effect=[
            RuntimeError("temporary failure"),
            RuntimeError("temporary failure"),
            response,
        ]
    )

    result = provider.generate("Test prompt")

    assert result == "Success"

    assert provider.client.models.generate_content.call_count == 3


def test_generate_raises_provider_error_after_retries():
    settings = create_settings()

    provider = GeminiProvider(settings)
    provider.max_retries = 2
    provider.retry_base_delay = 0

    provider.client.models.generate_content = Mock(
        side_effect=RuntimeError("provider unavailable")
    )

    with pytest.raises(LLMProviderError):
        provider.generate("Test prompt")

    assert provider.client.models.generate_content.call_count == 3