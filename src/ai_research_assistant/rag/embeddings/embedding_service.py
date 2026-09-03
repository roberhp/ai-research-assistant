import hashlib

from google import genai

from ai_research_assistant.rag.embeddings.embedding_cache import (
    EmbeddingCache,
)
from ai_research_assistant.settings import Settings


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
            cached_embedding = self.cache.get(cache_key)

            if cached_embedding is not None:
                return cached_embedding

        response = self.client.models.embed_content(
            model=self.model,
            contents=text,
        )

        embedding = response.embeddings[0].values

        if self.cache:
            self.cache.set(
                cache_key,
                embedding,
                self.cache_ttl,
            )

        return embedding

    def _build_cache_key(self, text: str) -> str:
        return hashlib.sha256(
            text.encode("utf-8")
        ).hexdigest()