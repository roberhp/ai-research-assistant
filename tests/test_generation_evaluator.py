from ai_research_assistant.evaluation.generation_case import (
    GenerationEvaluationCase,
)
from ai_research_assistant.evaluation.generation_evaluator import (
    GenerationEvaluator,
)
from ai_research_assistant.rag.generation.rag_answer import (
    RagAnswer,
    RagCitation,
)
from ai_research_assistant.rag.retrieval.retrieval_result import (
    RetrievalResult,
)


class FakeRagService:
    def __init__(self, results_by_query):
        self.results_by_query = results_by_query

    def answer(self, query: str, limit: int = 5):
        return self.results_by_query[query]


def test_generation_evaluator_calculates_metrics():
    rag_service = FakeRagService(
        results_by_query={
            "What is RAG?": RagAnswer(
                answer=(
                    "RAG combines retrieval and generation "
                    "[Source 1]."
                ),
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
        }
    )

    evaluator = GenerationEvaluator(rag_service)

    case = GenerationEvaluationCase(
        query="What is RAG?",
        expected_answer=(
            "RAG combines retrieval and generation."
        ),
        expected_sources={"rag-guide.txt"},
    )

    report = evaluator.evaluate([case])

    assert report["answer_relevance"] > 0.0
    assert report["citation_precision"] == 1.0
    assert report["citation_recall"] == 1.0


def test_generation_evaluator_handles_missing_citations():
    rag_service = FakeRagService(
        results_by_query={
            "What is RAG?": RagAnswer(
                answer="I don't have enough information.",
                sources=[],
                citations=[],
            )
        }
    )

    evaluator = GenerationEvaluator(rag_service)

    case = GenerationEvaluationCase(
        query="What is RAG?",
        expected_answer=(
            "RAG combines retrieval and generation."
        ),
        expected_sources={"rag-guide.txt"},
    )

    report = evaluator.evaluate([case])

    assert report["citation_precision"] == 0.0
    assert report["citation_recall"] == 0.0


def test_generation_evaluator_returns_zero_for_empty_dataset():
    rag_service = FakeRagService(
        results_by_query={}
    )

    evaluator = GenerationEvaluator(rag_service)

    report = evaluator.evaluate([])

    assert report["answer_relevance"] == 0.0
    assert report["citation_precision"] == 0.0
    assert report["citation_recall"] == 0.0