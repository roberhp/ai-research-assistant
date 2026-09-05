from ai_research_assistant.evaluation.llm_generation_evaluator import (
    LLMGenerationEvaluator,
)


class FakeLLMProvider:
    def __init__(self, response: str):
        self.response = response

    def generate(self, prompt: str) -> str:
        return self.response


def test_llm_generation_evaluator_parses_evaluation():
    provider = FakeLLMProvider(
        """
        {
            "relevance": 0.9,
            "grounding": 0.8,
            "citation_correctness": 1.0
        }
        """
    )

    evaluator = LLMGenerationEvaluator(provider)

    result = evaluator.evaluate(
        query="What is RAG?",
        context="RAG combines retrieval and generation.",
        answer="RAG combines retrieval and generation [Source 1].",
    )

    assert result.relevance == 0.9
    assert result.grounding == 0.8
    assert result.citation_correctness == 1.0

def test_llm_generation_evaluator_handles_zero_scores():
    provider = FakeLLMProvider(
        """
        {
            "relevance": 0.0,
            "grounding": 0.0,
            "citation_correctness": 0.0
        }
        """
    )

    evaluator = LLMGenerationEvaluator(provider)

    result = evaluator.evaluate(
        query="What is Kubernetes?",
        context="RAG combines retrieval and generation.",
        answer="Kubernetes is a container orchestration platform.",
    )

    assert result.relevance == 0.0
    assert result.grounding == 0.0
    assert result.citation_correctness == 0.0