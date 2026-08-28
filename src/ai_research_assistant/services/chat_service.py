from ai_research_assistant.llm.provider import LLMProvider


class ChatService:
    def __init__(self, provider: LLMProvider):
        self.provider = provider

    def chat(self, message: str) -> str:
        return self.provider.generate(message)