import hashlib
import logging
import time

from google import genai

from ai_research_assistant.rag.embeddings.embedding_cache import (
    EmbeddingCache,
)
from ai_research_assistant.settings import Settings


logger = logging.getLogger(
    "ai_research_assistant.embedding"
)


class EmbeddingService:
    def __init__(
        self,
        settings: Settings,
        cache: EmbeddingCache | None = None,
    ):
        self.client = genai.Client(
            api_key=settings.gemini_api_key
        )
        self.model = settings.gemini_embedding_model
        self.cache = cache
        self.cache_ttl = settings.redis_embedding_ttl

    def generate(self, text: str) -> list[float]:
        cache_key = self._build_cache_key(text)

        if self.cache:
            cached_embedding = self.cache.get(
                cache_key
            )

            if cached_embedding is not None:
                logger.info(
                    "embedding_cache_hit key=%s",
                    cache_key,
                )

                return cached_embedding

        logger.info(
            "embedding_cache_miss key=%s",
            cache_key,
        )

        start_time = time.perf_counter()

        response = self.client.models.embed_content(
            model=self.model,
            contents=text,
        )

        embedding = response.embeddings[0].values

        elapsed_ms = (
            time.perf_counter() - start_time
        ) * 1000

        logger.info(
            "embedding_generated "
            "model=%s latency_ms=%.2f",
            self.model,
            elapsed_ms,
        )

        if self.cache:
            self.cache.set(
                cache_key,
                embedding,
                self.cache_ttl,
            )

            logger.info(
                "embedding_cache_set key=%s ttl=%s",
                cache_key,
                self.cache_ttl,
            )

        return embedding

    def _build_cache_key(
        self,
        text: str,
    ) -> str:
        return hashlib.sha256(
            text.encode("utf-8")
        ).hexdigest()