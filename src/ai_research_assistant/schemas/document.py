from pydantic import BaseModel, Field


class DocumentCreateRequest(BaseModel):
    source: str = Field(
        min_length=1,
        max_length=500,
        description="Document source or filename.",
    )

    content: str = Field(
        min_length=1,
        description="Text content of the document.",
    )


class DocumentCreateResponse(BaseModel):
    id: str
    source: str
    chunk_count: int