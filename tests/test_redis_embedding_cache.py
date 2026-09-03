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
        redis_url
    )

    try:
        client.ping()
    except redis.ConnectionError:
        pytest.skip(
            "Redis is not available."
        )

    cache = RedisEmbeddingCache(
        redis_url=redis_url
    )

    yield cache

    client.flushdb()
    client.close()


def test_redis_embedding_cache_returns_none_for_missing_key(
    redis_cache,
):
    result = redis_cache.get(
        "missing-key"
    )

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

    result = redis_cache.get(
        "embedding-key"
    )

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

    client = redis.Redis.from_url(
        os.environ["REDIS_URL"]
    )

    ttl = client.ttl(
        "ttl-key"
    )

    client.close()

    assert 0 < ttl <= 60