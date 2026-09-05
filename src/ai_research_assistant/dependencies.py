from collections.abc import Generator
from functools import lru_cache

from fastapi import Depends
from sqlalchemy.orm import Session

from ai_research_assistant.database import SessionLocal
from ai_research_assistant.llm.gemini_provider import GeminiProvider
from ai_research_assistant.rag.chunking.text_chunker import TextChunker
from ai_research_assistant.rag.embeddings.embedding_cache import EmbeddingCache
from ai_research_assistant.rag.embeddings.embedding_service import EmbeddingService
from ai_research_assistant.rag.embeddings.redis_embedding_cache import (
    RedisEmbeddingCache,
)
from ai_research_assistant.rag.generation.citation_validator import CitationValidator
from ai_research_assistant.rag.generation.context_builder import ContextBuilder
from ai_research_assistant.rag.generation.rag_service import RagService
from ai_research_assistant.rag.ingestion.ingestion_service import IngestionService
from ai_research_assistant.rag.retrieval.retrieval_service import RetrievalService
from ai_research_assistant.repositories.chunk_repository import ChunkRepository
from ai_research_assistant.repositories.document_repository import DocumentRepository
from ai_research_assistant.services.chat_service import ChatService
from ai_research_assistant.settings import Settings


@lru_cache
def get_settings() -> Settings:
    return Settings()


def get_db() -> Generator:
    db = SessionLocal()

    try:
        yield db
    finally:
        db.close()


@lru_cache
def get_embedding_cache() -> EmbeddingCache:
    settings = get_settings()

    return RedisEmbeddingCache(
        redis_url=settings.redis_url,
        connect_timeout=settings.redis_connect_timeout_seconds,
        socket_timeout=settings.redis_socket_timeout_seconds,
    )


def get_embedding_service(
    cache: EmbeddingCache = Depends(get_embedding_cache),
) -> EmbeddingService:
    settings = get_settings()

    return EmbeddingService(
        settings=settings,
        cache=cache,
    )


def get_retrieval_service(
    db: Session = Depends(get_db),
    embedding_service: EmbeddingService = Depends(get_embedding_service),
) -> RetrievalService:
    settings = get_settings()
    return RetrievalService(
        embedding_service=embedding_service,
        chunk_repository=ChunkRepository(db),
        similarity_threshold=settings.retrieval_similarity_threshold,
    )


def get_ingestion_service(
    db: Session = Depends(get_db),
    embedding_service: EmbeddingService = Depends(get_embedding_service),
) -> IngestionService:
    return IngestionService(
        session=db,
        chunker=TextChunker(),
        embedding_service=embedding_service,
        document_repository=DocumentRepository(db),
        chunk_repository=ChunkRepository(db),
    )


@lru_cache
def get_llm_provider() -> GeminiProvider:
    settings = get_settings()
    return GeminiProvider(settings)


def get_chat_service(
    llm_provider: GeminiProvider = Depends(get_llm_provider),
) -> ChatService:
    return ChatService(llm_provider)


def get_rag_service(
    retrieval_service: RetrievalService = Depends(get_retrieval_service),
    llm_provider: GeminiProvider = Depends(get_llm_provider),
) -> RagService:
    return RagService(
        retrieval_service=retrieval_service,
        llm_provider=llm_provider,
        context_builder=ContextBuilder(),
        citation_validator=CitationValidator(),
    )