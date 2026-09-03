from types import SimpleNamespace

from ai_research_assistant.rag.embeddings.embedding_service import (
    EmbeddingService,
)


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