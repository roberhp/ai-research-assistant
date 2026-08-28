from ai_research_assistant.llm.gemini_provider import GeminiProvider
from ai_research_assistant.services.chat_service import ChatService
from ai_research_assistant.settings import Settings

from collections.abc import Generator
from ai_research_assistant.database import SessionLocal


def get_settings() -> Settings:
    return Settings()


def get_chat_service() -> ChatService:
    settings = get_settings()
    provider = GeminiProvider(settings)

    return ChatService(provider)

def get_db() -> Generator:
    db = SessionLocal()

    try:
        yield db
    finally:
        db.close()