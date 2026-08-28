import uuid

import pytest
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from ai_research_assistant.models.chunk import ChunkModel
from ai_research_assistant.repositories.chunk_repository import ChunkRepository
from ai_research_assistant.repositories.document_repository import DocumentRepository
from ai_research_assistant.rag.ingestion.embedded_chunk import EmbeddedChunk


def test_create_document(db_session):
    repository = DocumentRepository(db_session)

    document = repository.create("rag-test.txt")

    assert document.id is not None
    assert isinstance(document.id, uuid.UUID)
    assert document.source == "rag-test.txt"


def test_create_chunks(db_session):
    document_repository = DocumentRepository(db_session)
    chunk_repository = ChunkRepository(db_session)

    document = document_repository.create("rag-test.txt")

    chunks = chunk_repository.create_many(
        document_id=document.id,
        chunks=[
            EmbeddedChunk(
                content="RAG combines retrieval and generation.",
                chunk_index=0,
                embedding=[0.1] * 3072,
            ),
            EmbeddedChunk(
                content="Embeddings represent text as vectors.",
                chunk_index=1,
                embedding=[0.2] * 3072,
            ),
        ],
    )

    assert len(chunks) == 2
    assert chunks[0].chunk_index == 0
    assert chunks[1].chunk_index == 1
    assert len(chunks[0].embedding) == 3072


def test_create_chunk_requires_existing_document(db_session):
    repository = ChunkRepository(db_session)

    with pytest.raises(IntegrityError):
        repository.create_many(
            document_id=uuid.uuid4(),
            chunks=[
                EmbeddedChunk(
                    content="This document does not exist.",
                    chunk_index=0,
                    embedding=[0.1] * 3072,
                ),
            ],
        )


def test_embedding_is_persisted_as_vector(db_session):
    document_repository = DocumentRepository(db_session)
    chunk_repository = ChunkRepository(db_session)

    document = document_repository.create("embedding-test.txt")

    chunk_repository.create_many(
        document_id=document.id,
        chunks=[
            EmbeddedChunk(
                content ="Vector test",
                chunk_index=0,
                embedding=[0.5] * 3072,
            ),
        ],
    )

    chunk = db_session.scalar(
        select(ChunkModel).where(
            ChunkModel.document_id == document.id
        )
    )

    assert chunk is not None
    assert len(chunk.embedding) == 3072