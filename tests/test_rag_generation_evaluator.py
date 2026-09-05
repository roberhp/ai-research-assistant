from ai_research_assistant.evaluation.llm_generation_evaluator import (
    LLMGenerationEvaluator,
)
from ai_research_assistant.evaluation.rag_generation_evaluator import (
    RagGenerationEvaluator,
)
from ai_research_assistant.rag.generation.context_builder import ContextBuilder
from ai_research_assistant.rag.generation.rag_answer import (
    RagAnswer,
    RagCitation,
)
from ai_research_assistant.rag.retrieval.retrieval_result import (
    RetrievalResult,
)


class FakeRagService:
    def answer(self, query: str, limit: int = 5):
        return RagAnswer(
            answer="RAG combines retrieval and generation [Source 1].",
            sources=[
                RetrievalResult(
                    content="RAG combines retrieval and generation.",
                    source="rag-guide.txt",
                    chunk_index=0,
                    score=0.95,
                )
            ],
            citations=[
                RagCitation(
                    source="rag-guide.txt",
                    chunk_index=0,
                    score=0.95,
                )
            ],
        )


class FakeLLMProvider:
    def generate(self, prompt: str) -> str:
        return """
        {
            "relevance": 1.0,
            "grounding": 1.0,
            "citation_correctness": 1.0
        }
        """


def test_rag_generation_evaluator_evaluates_rag_response():
    rag_service = FakeRagService()

    llm_evaluator = LLMGenerationEvaluator(
        FakeLLMProvider()
    )

    evaluator = RagGenerationEvaluator(
        rag_service=rag_service,
        llm_evaluator=llm_evaluator,
        context_builder=ContextBuilder(),
    )

    result = evaluator.evaluate(
        query="What is RAG?"
    )

    assert result.relevance == 1.0
    assert result.grounding == 1.0
    assert result.citation_correctness == 1.0