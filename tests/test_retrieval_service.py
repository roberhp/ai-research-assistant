from types import SimpleNamespace

from ai_research_assistant.rag.embeddings.embedding_service import EmbeddingService
from ai_research_assistant.rag.ingestion.embedded_chunk import EmbeddedChunk
from ai_research_assistant.rag.retrieval.retrieval_service import RetrievalService
from ai_research_assistant.repositories.chunk_repository import ChunkRepository
from ai_research_assistant.repositories.document_repository import DocumentRepository


class FakeEmbeddingService:
    def generate(self, text):
        return [1.0] + [0.0] * 3071


class FakeChunkRepository:
    def similarity_search(self, query_embedding, limit):
        return [
            (
                SimpleNamespace(
                    content="Relevant content",
                    chunk_index=0,
                    document=SimpleNamespace(
                        source="doc1.txt",
                    ),
                ),
                0.10,
            ),
            (
                SimpleNamespace(
                    content="Not relevant enough",
                    chunk_index=1,
                    document=SimpleNamespace(
                        source="doc2.txt",
                    ),
                ),
                0.40,
            ),
        ][:limit]


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
    assert results[0].content == "RAG combines retrieval and generation."
    assert results[0].source == "retrieval-test.txt"
    assert results[0].chunk_index == 0
    assert results[0].score == 1.0


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
                embedding=[1.0] + [0.0] * 3071,
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


def test_retrieval_service_filters_results_below_threshold():
    service = RetrievalService(
        embedding_service=FakeEmbeddingService(),
        chunk_repository=FakeChunkRepository(),
        similarity_threshold=0.80,
    )

    results = service.retrieve(
        query="What is RAG?",
        limit=5,
    )

    assert len(results) == 1
    assert results[0].source == "doc1.txt"
    assert results[0].score == 0.90


def test_retrieval_service_accepts_custom_similarity_threshold():
    class CustomFakeChunkRepository:
        def similarity_search(self, query_embedding, limit):
            return [
                (
                    SimpleNamespace(
                        content="Relevant content",
                        chunk_index=0,
                        document=SimpleNamespace(
                            source="doc1.txt",
                        ),
                    ),
                    0.40,
                ),
            ]

    service = RetrievalService(
        embedding_service=FakeEmbeddingService(),
        chunk_repository=CustomFakeChunkRepository(),
        similarity_threshold=0.50,
    )

    results = service.retrieve(
        query="What is RAG?",
        limit=5,
    )

    assert len(results) == 1
    assert results[0].source == "doc1.txt"
    assert results[0].score == 0.60