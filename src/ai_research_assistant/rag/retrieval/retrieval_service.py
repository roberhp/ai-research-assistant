from ai_research_assistant.rag.embeddings.embedding_service import EmbeddingService
from ai_research_assistant.rag.retrieval.retrieval_result import RetrievalResult
from ai_research_assistant.repositories.chunk_repository import ChunkRepository


class RetrievalService:
    def __init__(
        self,
        embedding_service: EmbeddingService,
        chunk_repository: ChunkRepository,
    ):
        self.embedding_service = embedding_service
        self.chunk_repository = chunk_repository

    def retrieve(
        self,
        query: str,
        limit: int = 5,
    ) -> list[RetrievalResult]:
        query_embedding = self.embedding_service.generate(query)

        results = self.chunk_repository.similarity_search(
            query_embedding=query_embedding,
            limit=limit,
        )

        return [
            RetrievalResult(
                content=chunk.content,
                source=chunk.document.source,
                chunk_index=chunk.chunk_index,
                score=1 - distance,
            )
            for chunk, distance in results
        ]