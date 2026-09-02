from sqlalchemy.orm import Session

from ai_research_assistant.evaluation.evaluation_documents import (
    EVALUATION_DOCUMENTS,
)
from ai_research_assistant.rag.embeddings.embedding_service import (
    EmbeddingService,
)
from ai_research_assistant.rag.ingestion.ingestion_service import (
    IngestionService,
)
from ai_research_assistant.rag.chunking.text_chunker import TextChunker
from ai_research_assistant.repositories.chunk_repository import ChunkRepository
from ai_research_assistant.repositories.document_repository import (
    DocumentRepository,
)


class EvaluationSetup:
    def __init__(
        self,
        session: Session,
        embedding_service: EmbeddingService,
    ):
        self.session = session
        self.ingestion_service = IngestionService(
            session=session,
            chunker=TextChunker(),
            embedding_service=embedding_service,
            document_repository=DocumentRepository(session),
            chunk_repository=ChunkRepository(session),
        )

    def prepare(self) -> None:
        for document in EVALUATION_DOCUMENTS:
            self.ingestion_service.ingest(document)