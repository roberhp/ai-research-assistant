import logging
import time

from ai_research_assistant.llm.provider import LLMProvider
from ai_research_assistant.rag.generation.context_builder import (
    ContextBuilder,
)
from ai_research_assistant.rag.generation.rag_answer import (
    RagAnswer,
)
from ai_research_assistant.rag.retrieval.retrieval_service import (
    RetrievalService,
)


logger = logging.getLogger(
    "ai_research_assistant.rag"
)


class RagService:
    def __init__(
        self,
        retrieval_service: RetrievalService,
        llm_provider: LLMProvider,
        context_builder: ContextBuilder,
    ):
        self.retrieval_service = retrieval_service
        self.llm_provider = llm_provider
        self.context_builder = context_builder

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
            elapsed_ms = (
                time.perf_counter() - start_time
            ) * 1000

            logger.info(
                "rag_no_results latency_ms=%.2f",
                elapsed_ms,
            )

            return RagAnswer(
                answer=(
                    "I don't have enough information "
                    "to answer this question."
                ),
                sources=[],
            )

        context = self.context_builder.build(
            results
        )

        prompt = self._build_prompt(
            query=query,
            context=context,
        )

        llm_start_time = time.perf_counter()

        answer = self.llm_provider.generate(
            prompt
        )

        llm_elapsed_ms = (
            time.perf_counter()
            - llm_start_time
        ) * 1000

        total_elapsed_ms = (
            time.perf_counter()
            - start_time
        ) * 1000

        logger.info(
            "rag_completed "
            "sources=%s "
            "llm_latency_ms=%.2f "
            "total_latency_ms=%.2f",
            len(results),
            llm_elapsed_ms,
            total_elapsed_ms,
        )

        return RagAnswer(
            answer=answer,
            sources=results,
        )

    def _build_prompt(
        self,
        query: str,
        context: str,
    ) -> str:
        return f"""
You are a research assistant.

Answer the user's question using only the provided context.

When an answer is supported by information from the context,
cite the corresponding source using its identifier, such as [Source 1]
or [Source 2].

Use only source identifiers that exist in the provided context.
Do not invent source identifiers.

If the context does not contain enough information to answer the question,
say that you don't have enough information.

Context:
{context}

Question:
{query}
""".strip()