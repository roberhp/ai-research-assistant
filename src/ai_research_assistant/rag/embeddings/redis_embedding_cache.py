import json
import logging

import redis

from ai_research_assistant.rag.embeddings.embedding_cache import EmbeddingCache

logger = logging.getLogger(__name__)


class RedisEmbeddingCache(EmbeddingCache):
    def __init__(
        self,
        redis_url: str,
        connect_timeout: float = 2.0,
        socket_timeout: float = 2.0,
    ):
        self.client = redis.Redis.from_url(
            redis_url,
            decode_responses=True,
            socket_connect_timeout=connect_timeout,
            socket_timeout=socket_timeout,
        )

    def get(self, key: str) -> list[float] | None:
        try:
            value = self.client.get(key)

            if value is None:
                return None

            return json.loads(value)

        except Exception:
            logger.exception(
                "embedding_cache_get_failed"
            )
            return None

    def set(
        self,
        key: str,
        embedding: list[float],
        ttl: int,
    ) -> None:
        try:
            self.client.set(
                key,
                json.dumps(embedding),
                ex=ttl,
            )
        except Exception:
            logger.exception(
                "embedding_cache_set_failed"
            )