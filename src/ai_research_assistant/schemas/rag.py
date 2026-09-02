from pydantic import BaseModel, Field


class RagRequest(BaseModel):
    query: str = Field(
        min_length=1,
        description="Question to answer using the knowledge base.",
    )
    limit: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum number of chunks used as context.",
    )


class RagSource(BaseModel):
    source: str
    chunk_index: int
    score: float


class RagResponse(BaseModel):
    answer: str
    sources: list[RagSource]