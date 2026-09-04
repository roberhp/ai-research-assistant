from ai_research_assistant.services.chat_service import ChatService
from tests.fakes.fake_llm_provider import FakeLLMProvider


def test_chat_returns_llm_response():
    provider = FakeLLMProvider()
    service = ChatService(provider)

    response = service.chat("Hello")

    assert response == "Fake LLM response"


def test_chat_sends_message_to_llm_provider():
    provider = FakeLLMProvider()
    service = ChatService(provider)

    service.chat("What is RAG?")

    assert provider.prompts == ["What is RAG?"]