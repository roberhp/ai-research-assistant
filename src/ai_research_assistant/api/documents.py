from fastapi import APIRouter, Depends, status

from ai_research_assistant.dependencies import get_ingestion_service
from ai_research_assistant.rag.document import Document
from ai_research_assistant.rag.ingestion.ingestion_service import IngestionService
from ai_research_assistant.schemas.document import (
    DocumentCreateRequest,
    DocumentCreateResponse,
)


router = APIRouter(
    prefix="/api/v1",
    tags=["documents"],
)


@router.post(
    "/documents",
    response_model=DocumentCreateResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_document(
    request: DocumentCreateRequest,
    ingestion_service: IngestionService = Depends(get_ingestion_service),
):
    document = Document(
        source=request.source,
        content=request.content,
    )

    document_model, chunk_count = ingestion_service.ingest(document)

    return DocumentCreateResponse(
        id=str(document_model.id),
        source=document_model.source,
        chunk_count=chunk_count,
    )