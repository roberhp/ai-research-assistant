from ai_research_assistant.llm.provider import LLMProvider
from ai_research_assistant.rag.generation.context_builder import ContextBuilder
from ai_research_assistant.rag.retrieval.retrieval_service import RetrievalService


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

    def answer(self, query: str, limit: int = 5) -> str:
        results = self.retrieval_service.retrieve(
            query=query,
            limit=limit,
        )

        context = self.context_builder.build(results)

        prompt = self._build_prompt(
            query=query,
            context=context,
        )

        return self.llm_provider.generate(prompt)

    def _build_prompt(self, query: str, context: str) -> str:
        return f"""
            You are a research assistant.

            Answer the user's question using only the provided context.

            If the context does not contain enough information to answer the question,
            say that you don't have enough information.

            Context:
            {context}

            Question:
            {query}
            """.strip()