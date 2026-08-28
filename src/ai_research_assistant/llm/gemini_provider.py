from google import genai

from ai_research_assistant.llm.provider import LLMProvider
from ai_research_assistant.settings import Settings


class GeminiProvider:
    def __init__(self, settings: Settings):
        self.client = genai.Client(api_key=settings.gemini_api_key)
        self.model = settings.gemini_model

    def generate(self, prompt: str) -> str:
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
        )

        return response.text