from sqlalchemy.orm import Session

from ai_research_assistant.rag.document import Document
from ai_research_assistant.rag.chunking.text_chunker import TextChunker
from ai_research_assistant.rag.embeddings.embedding_service import EmbeddingService
from ai_research_assistant.rag.ingestion.embedded_chunk import EmbeddedChunk
from ai_research_assistant.repositories.chunk_repository import ChunkRepository
from ai_research_assistant.repositories.document_repository import DocumentRepository


class IngestionService:
    def __init__(
        self,
        session: Session,
        chunker: TextChunker,
        embedding_service: EmbeddingService,
        document_repository: DocumentRepository,
        chunk_repository: ChunkRepository,
    ):
        self.session = session
        self.chunker = chunker
        self.embedding_service = embedding_service
        self.document_repository = document_repository
        self.chunk_repository = chunk_repository

    def ingest(self, document: Document):
        try:
            existing_document = self.document_repository.find_by_source(
                document.source
            )

            if existing_document:
                return existing_document, 0

            document_model = self.document_repository.create(
                source=document.source,
            )

            chunks = self.chunker.chunk(document)

            embedded_chunks = [
                EmbeddedChunk(
                    content=chunk.content,
                    chunk_index=chunk.chunk_index,
                    embedding=self.embedding_service.generate(
                        chunk.content
                    ),
                )
                for chunk in chunks
            ]

            self.chunk_repository.create_many(
                document_id=document_model.id,
                chunks=embedded_chunks,
            )

            self.session.commit()

            return document_model, len(embedded_chunks)

        except Exception:
            self.session.rollback()
            raise