from fastapi import APIRouter, Depends

from ai_research_assistant.dependencies import get_retrieval_service
from ai_research_assistant.rag.retrieval.retrieval_service import RetrievalService
from ai_research_assistant.schemas.search import (
    SearchRequest,
    SearchResponse,
    SearchResultResponse,
)


router = APIRouter(
    prefix="/api/v1",
    tags=["search"],
)


@router.post(
    "/search",
    response_model=SearchResponse,
)
def search(
    request: SearchRequest,
    retrieval_service: RetrievalService = Depends(get_retrieval_service),
):
    results = retrieval_service.retrieve(
        query=request.query,
        limit=request.limit,
    )

    return SearchResponse(
        results=[
            SearchResultResponse(
                content=result.content,
                source=result.source,
                chunk_index=result.chunk_index,
                score=result.score,
            )
            for result in results
        ]
    )