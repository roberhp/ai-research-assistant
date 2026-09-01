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


class RagResponse(BaseModel):
    answer: str