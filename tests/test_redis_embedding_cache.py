import os

import pytest
import redis

from ai_research_assistant.rag.embeddings.redis_embedding_cache import (
    RedisEmbeddingCache,
)


@pytest.fixture
def redis_cache():
    redis_url = os.environ["REDIS_URL"]

    client = redis.Redis.from_url(
        redis_url,
    )

    try:
        client.ping()
    except redis.ConnectionError:
        pytest.skip("Redis is not available.")

    cache = RedisEmbeddingCache(
        redis_url=redis_url,
    )

    yield cache

    client.flushdb()
    client.close()


def test_redis_embedding_cache_returns_none_for_missing_key(
    redis_cache,
):
    result = redis_cache.get("missing-key")

    assert result is None


def test_redis_embedding_cache_stores_and_retrieves_embedding(
    redis_cache,
):
    embedding = [
        0.1,
        0.2,
        0.3,
    ]

    redis_cache.set(
        key="embedding-key",
        embedding=embedding,
        ttl=60,
    )

    result = redis_cache.get("embedding-key")

    assert result == embedding


def test_redis_embedding_cache_sets_ttl(
    redis_cache,
):
    redis_cache.set(
        key="ttl-key",
        embedding=[
            0.1,
            0.2,
            0.3,
        ],
        ttl=60,
    )

    ttl = redis_cache.client.ttl("ttl-key")

    assert 0 < ttl <= 60


def test_redis_embedding_cache_uses_configured_timeouts(
    redis_cache,
):
    connection_kwargs = (
        redis_cache.client.connection_pool.connection_kwargs
    )

    assert connection_kwargs["socket_connect_timeout"] == 2.0
    assert connection_kwargs["socket_timeout"] == 2.0


def test_redis_embedding_cache_get_returns_none_when_redis_fails(
    redis_cache,
    monkeypatch,
):
    def failing_get(key):
        raise redis.ConnectionError("Redis unavailable")

    monkeypatch.setattr(
        redis_cache.client,
        "get",
        failing_get,
    )

    result = redis_cache.get("test-key")

    assert result is None


def test_redis_embedding_cache_set_does_not_raise_when_redis_fails(
    redis_cache,
    monkeypatch,
):
    def failing_set(key, value, ex):
        raise redis.ConnectionError("Redis unavailable")

    monkeypatch.setattr(
        redis_cache.client,
        "set",
        failing_set,
    )

    redis_cache.set(
        key="test-key",
        embedding=[0.1, 0.2, 0.3],
        ttl=60,
    )