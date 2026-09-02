from types import SimpleNamespace

from ai_research_assistant.rag.document import Document
from ai_research_assistant.rag.ingestion.embedded_chunk import EmbeddedChunk
from ai_research_assistant.rag.ingestion.ingestion_service import IngestionService


class FakeChunker:
    def chunk(self, document: Document):
        return [
            SimpleNamespace(
                content=document.content,
                source=document.source,
                chunk_index=0,
            )
        ]


class FakeEmbeddingService:
    def __init__(self):
        self.calls = 0

    def generate(self, text: str) -> list[float]:
        self.calls += 1
        return [1.0] + [0.0] * 3071


class FakeDocumentRepository:
    def __init__(self):
        self.documents = {}

    def find_by_source(self, source: str):
        return self.documents.get(source)

    def create(self, source: str):
        document = SimpleNamespace(
            id="document-id",
            source=source,
        )

        self.documents[source] = document

        return document


class FakeChunkRepository:
    def __init__(self):
        self.created_chunks = []

    def create_many(
        self,
        document_id,
        chunks: list[EmbeddedChunk],
    ):
        self.created_chunks.extend(chunks)


class FakeSession:
    def __init__(self):
        self.commit_calls = 0
        self.rollback_calls = 0

    def commit(self):
        self.commit_calls += 1

    def rollback(self):
        self.rollback_calls += 1


def create_service(
    session,
    chunker,
    embedding_service,
    document_repository,
    chunk_repository,
):
    return IngestionService(
        session=session,
        chunker=chunker,
        embedding_service=embedding_service,
        document_repository=document_repository,
        chunk_repository=chunk_repository,
    )


def test_ingestion_creates_document_and_chunks():
    session = FakeSession()
    chunker = FakeChunker()
    embedding_service = FakeEmbeddingService()
    document_repository = FakeDocumentRepository()
    chunk_repository = FakeChunkRepository()

    service = create_service(
        session=session,
        chunker=chunker,
        embedding_service=embedding_service,
        document_repository=document_repository,
        chunk_repository=chunk_repository,
    )

    document = Document(
        source="test.txt",
        content="Test document",
    )

    document_model, chunk_count = service.ingest(document)

    assert document_model.source == "test.txt"
    assert chunk_count == 1

    assert len(chunk_repository.created_chunks) == 1
    assert embedding_service.calls == 1
    assert session.commit_calls == 1


def test_ingestion_skips_existing_document():
    session = FakeSession()
    chunker = FakeChunker()
    embedding_service = FakeEmbeddingService()
    document_repository = FakeDocumentRepository()
    chunk_repository = FakeChunkRepository()

    existing_document = document_repository.create(
        "existing.txt"
    )

    service = create_service(
        session=session,
        chunker=chunker,
        embedding_service=embedding_service,
        document_repository=document_repository,
        chunk_repository=chunk_repository,
    )

    document = Document(
        source="existing.txt",
        content="This content should not be ingested.",
    )

    document_model, chunk_count = service.ingest(document)

    assert document_model is existing_document
    assert chunk_count == 0

    assert len(chunk_repository.created_chunks) == 0
    assert embedding_service.calls == 0
    assert session.commit_calls == 0