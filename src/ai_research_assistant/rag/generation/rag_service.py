import logging
import time

from ai_research_assistant.llm.provider import LLMProvider
from ai_research_assistant.rag.generation.citation_validator import CitationValidator
from ai_research_assistant.rag.generation.context_builder import ContextBuilder
from ai_research_assistant.rag.generation.rag_answer import RagAnswer
from ai_research_assistant.rag.retrieval.retrieval_service import RetrievalService


logger = logging.getLogger(__name__)


class RagService:
    def __init__(
        self,
        retrieval_service: RetrievalService,
        llm_provider: LLMProvider,
        context_builder: ContextBuilder,
        citation_validator: CitationValidator,
    ):
        self.retrieval_service = retrieval_service
        self.llm_provider = llm_provider
        self.context_builder = context_builder
        self.citation_validator = citation_validator

    def answer(
        self,
        query: str,
        limit: int = 5,
    ) -> RagAnswer:
        start_time = time.perf_counter()

        results = self.retrieval_service.retrieve(
            query=query,
            limit=limit,
        )

        if not results:
            elapsed_ms = (time.perf_counter() - start_time) * 1000

            logger.info(
                "rag_no_results latency_ms=%.2f",
                elapsed_ms,
            )

            return RagAnswer(
                answer="I don't have enough information to answer this question.",
                sources=[],
                citations=[],
            )

        context = self.context_builder.build(results)

        prompt = self._build_prompt(
            query=query,
            context=context,
        )

        llm_start_time = time.perf_counter()

        answer = self.llm_provider.generate(prompt)

        llm_elapsed_ms = (
            time.perf_counter() - llm_start_time
        ) * 1000

        citations = self.citation_validator.validate(
            answer=answer,
            sources=results,
        )

        total_elapsed_ms = (
            time.perf_counter() - start_time
        ) * 1000

        logger.info(
            "rag_completed "
            "sources=%s "
            "citations=%s "
            "llm_latency_ms=%.2f "
            "total_latency_ms=%.2f",
            len(results),
            len(citations),
            llm_elapsed_ms,
            total_elapsed_ms,
        )

        return RagAnswer(
            answer=answer,
            sources=results,
            citations=citations,
        )

    def _build_prompt(
        self,
        query: str,
        context: str,
    ) -> str:
        return f"""
You are a research assistant.

Answer the user's question using only the provided context.

Rules:
- Do not use information that is not present in the context.
- If the context does not contain enough information to answer the question, say that you don't have enough information.
- When making a factual claim supported by the context, cite the corresponding source using [Source N].
- Only use source identifiers that exist in the provided context.
- Do not invent source identifiers.
- Do not cite a source that does not support the claim.

Context:
{context}

Question:
{query}
""".strip()