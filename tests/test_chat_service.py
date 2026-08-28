from ai_research_assistant.services.chat_service import ChatService


class FakeLLMProvider:
    def __init__(self):
        self.last_prompt = None

    def generate(self, prompt: str) -> str:
        self.last_prompt = prompt
        return "Fake response"

def test_chat_returns_llm_response():
    provider = FakeLLMProvider()
    service = ChatService(provider)

    response = service.chat("Hello")

    assert response == "Fake response"

def test_chat_sends_message_to_llm_provider():
    provider = FakeLLMProvider()
    service = ChatService(provider)

    service.chat("What is RAG?")

    assert provider.last_prompt == "What is RAG?"