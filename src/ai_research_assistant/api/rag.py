from fastapi import APIRouter, Depends

from ai_research_assistant.dependencies import get_rag_service
from ai_research_assistant.rag.generation.rag_service import RagService
from ai_research_assistant.schemas.rag import RagRequest, RagResponse


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
    answer = rag_service.answer(
        query=request.query,
        limit=request.limit,
    )

    return RagResponse(
        answer=answer,
    )