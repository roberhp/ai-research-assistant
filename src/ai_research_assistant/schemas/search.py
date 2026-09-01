from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    query: str = Field(
        min_length=1,
        description="Question or text used to search the knowledge base.",
    )

    limit: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum number of results to return.",
    )


class SearchResultResponse(BaseModel):
    content: str
    source: str
    chunk_index: int
    score: float


class SearchResponse(BaseModel):
    results: list[SearchResultResponse]