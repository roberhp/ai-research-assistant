import json

import redis

from ai_research_assistant.rag.embeddings.embedding_cache import (
    EmbeddingCache,
)


class RedisEmbeddingCache(EmbeddingCache):
    def __init__(self, redis_url: str):
        self.client = redis.Redis.from_url(
            redis_url,
            decode_responses=True,
        )

    def get(self, key: str) -> list[float] | None:
        value = self.client.get(key)

        if value is None:
            return None

        return json.loads(value)

    def set(
        self,
        key: str,
        embedding: list[float],
        ttl: int,
    ) -> None:
        self.client.set(
            key,
            json.dumps(embedding),
            ex=ttl,
        )