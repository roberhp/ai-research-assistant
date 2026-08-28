from uuid import UUID

from sqlalchemy.orm import Session

from ai_research_assistant.models.chunk import ChunkModel
from ai_research_assistant.rag.ingestion.embedded_chunk import EmbeddedChunk

class ChunkRepository:
    def __init__(self, session: Session):
        self.session = session

    def create_many(
        self,
        document_id: UUID,
        chunks: list[EmbeddedChunk],
    ) -> list[ChunkModel]:
        chunk_models = [
            ChunkModel(
                document_id=document_id,
                content=chunk.content,
                chunk_index=chunk.chunk_index,
                embedding=chunk.embedding,
            )
            for chunk in chunks
        ]

        self.session.add_all(chunk_models)
        self.session.flush()

        return chunk_models