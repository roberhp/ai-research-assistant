import logging
import time

from google import genai
from google.genai import types

from ai_research_assistant.exceptions import LLMProviderError
from ai_research_assistant.llm.provider import LLMProvider
from ai_research_assistant.settings import Settings

logger = logging.getLogger(__name__)


class GeminiProvider(LLMProvider):
    def __init__(self, settings: Settings):
        self.client = genai.Client(
            api_key=settings.gemini_api_key,
            http_options=types.HttpOptions(
                timeout=int(settings.gemini_timeout_seconds * 1000),
            ),
        )

        self.model = settings.gemini_model
        self.max_retries = settings.gemini_max_retries
        self.retry_base_delay = settings.gemini_retry_base_delay_seconds

    def generate(self, prompt: str) -> str:
        for attempt in range(self.max_retries + 1):
            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                )

                if not response.text:
                    raise LLMProviderError(
                        "LLM provider returned an empty response."
                    )

                return response.text

            except LLMProviderError:
                raise

            except Exception:
                if attempt >= self.max_retries:
                    logger.exception(
                        "llm_provider_failed "
                        "model=%s "
                        "attempt=%s",
                        self.model,
                        attempt + 1,
                    )

                    raise LLMProviderError(
                        "LLM provider request failed."
                    )

                delay = self.retry_base_delay * (2**attempt)

                logger.warning(
                    "llm_provider_retry "
                    "model=%s "
                    "attempt=%s "
                    "max_retries=%s "
                    "delay_seconds=%.2f",
                    self.model,
                    attempt + 1,
                    self.max_retries,
                    delay,
                )

                time.sleep(delay)