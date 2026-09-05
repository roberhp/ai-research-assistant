from ai_research_assistant.database import SessionLocal
from ai_research_assistant.rag.embeddings.embedding_service import EmbeddingService
from ai_research_assistant.rag.retrieval.retrieval_service import RetrievalService
from ai_research_assistant.repositories.chunk_repository import ChunkRepository
from ai_research_assistant.settings import Settings


def create_evaluation_retrieval_service(
    similarity_threshold: float = 0.70,
) -> RetrievalService:
    settings = Settings()

    db = SessionLocal()

    embedding_service = EmbeddingService(
        settings=settings,
    )

    chunk_repository = ChunkRepository(db)

    return RetrievalService(
        embedding_service=embedding_service,
        chunk_repository=chunk_repository,
        similarity_threshold=similarity_threshold,
    )