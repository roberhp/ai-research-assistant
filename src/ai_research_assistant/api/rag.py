from fastapi import APIRouter, Depends

from ai_research_assistant.dependencies import get_rag_service
from ai_research_assistant.rag.generation.rag_service import RagService
from ai_research_assistant.schemas.rag import (
    RagRequest,
    RagResponse,
    RagSource,
)


router = APIRouter(
    prefix="/api/v1",
    tags=["rag"],
)


@router.post(
    "/chat",
    response_model=RagResponse,
)
def chat(
    request: RagRequest,
    rag_service: RagService = Depends(get_rag_service),
):
    result = rag_service.answer(
        query=request.query,
        limit=request.limit,
    )

    sources = [
        RagSource(
            source=source.source,
            chunk_index=source.chunk_index,
            score=source.score,
        )
        for source in result.sources
    ]

    return RagResponse(
        answer=result.answer,
        sources=sources,
    )