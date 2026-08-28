from ai_research_assistant.llm.gemini_provider import GeminiProvider
from ai_research_assistant.settings import Settings


settings = Settings()
provider = GeminiProvider(settings)

response = provider.generate("Say hello in one sentence.")

print(response)