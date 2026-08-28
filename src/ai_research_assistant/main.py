from fastapi import Depends, FastAPI

from ai_research_assistant.dependencies import get_chat_service
from ai_research_assistant.schemas.chat import ChatRequest, ChatResponse
from ai_research_assistant.services.chat_service import ChatService

app = FastAPI()

@app.get("/")
def root():
    return {"message": "AI Research Assistant"}

@app.post("/chat", response_model=ChatResponse)
def chat(
    request: ChatRequest,
    service: ChatService = Depends(get_chat_service),
):
    response = service.chat(request.message)

    return ChatResponse(response=response)