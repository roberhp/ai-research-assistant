from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(
        ...,
        description="Message to send to the AI assistant.",
        examples=["What is retrieval-augmented generation? Explain it in one line"],
    )


class ChatResponse(BaseModel):
    response: str