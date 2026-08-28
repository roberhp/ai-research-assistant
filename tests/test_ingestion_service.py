from sqlalchemy import select

from ai_research_assistant.models.chunk import ChunkModel
from ai_research_assistant.rag.document import Document
from ai_research_assistant.rag.ingestion.ingestion_service import IngestionService


class FakeEmbeddingService:
    def generate(self, text: str) -> list[float]:
        return [0.1] * 3072


def test_ingestion_persists_document_and_chunks(db_session):
    from ai_research_assistant.rag.chunking.text_chunker import TextChunker
    from ai_research_assistant.repositories.chunk_repository import ChunkRepository
    from ai_research_assistant.repositories.document_repository import DocumentRepository

    ingestion_service = IngestionService(
        session=db_session,
        chunker=TextChunker(chunk_size=20, overlap=5),
        embedding_service=FakeEmbeddingService(),
        document_repository=DocumentRepository(db_session),
        chunk_repository=ChunkRepository(db_session),
    )

    document = Document(
        content="This is a document that should be split into multiple chunks.",
        source="test.txt",
    )

    result = ingestion_service.ingest(document)

    assert result.id is not None
    assert result.source == "test.txt"

    chunks = db_session.scalars(
        select(ChunkModel).where(
            ChunkModel.document_id == result.id
        )
    ).all()

    assert len(chunks) > 1
    assert all(len(chunk.embedding) == 3072 for chunk in chunks)