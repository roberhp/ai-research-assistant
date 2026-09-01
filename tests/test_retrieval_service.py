from ai_research_assistant.rag.ingestion.embedded_chunk import EmbeddedChunk
from ai_research_assistant.rag.retrieval.retrieval_service import RetrievalService
from ai_research_assistant.repositories.chunk_repository import ChunkRepository
from ai_research_assistant.repositories.document_repository import DocumentRepository


class FakeEmbeddingService:
    def generate(self, text: str) -> list[float]:
        return [1.0] + [0.0] * 3071


def test_retrieval_returns_relevant_chunks(db_session):
    document_repository = DocumentRepository(db_session)
    chunk_repository = ChunkRepository(db_session)

    document = document_repository.create(
        "retrieval-test.txt"
    )

    chunk_repository.create_many(
        document_id=document.id,
        chunks=[
            EmbeddedChunk(
                content="RAG combines retrieval and generation.",
                chunk_index=0,
                embedding=[1.0] + [0.0] * 3071,
            ),
            EmbeddedChunk(
                content="Java is an object oriented programming language.",
                chunk_index=1,
                embedding=[-1.0] + [0.0] * 3071,
            ),
        ],
    )

    db_session.commit()

    service = RetrievalService(
        embedding_service=FakeEmbeddingService(),
        chunk_repository=chunk_repository,
    )

    results = service.retrieve(
        query="What is RAG?",
        limit=1,
    )

    assert len(results) == 1

    result = results[0]

    assert result.content == (
        "RAG combines retrieval and generation."
    )
    assert result.source == "retrieval-test.txt"
    assert result.chunk_index == 0
    assert result.score > 0.99


def test_retrieval_respects_limit(db_session):
    document_repository = DocumentRepository(db_session)
    chunk_repository = ChunkRepository(db_session)

    document = document_repository.create(
        "limit-test.txt"
    )

    chunk_repository.create_many(
        document_id=document.id,
        chunks=[
            EmbeddedChunk(
                content=f"Chunk {i}",
                chunk_index=i,
                embedding=[1.0] * 3072,
            )
            for i in range(5)
        ],
    )

    db_session.commit()

    service = RetrievalService(
        embedding_service=FakeEmbeddingService(),
        chunk_repository=chunk_repository,
    )

    results = service.retrieve(
        query="test",
        limit=3,
    )

    assert len(results) == 3