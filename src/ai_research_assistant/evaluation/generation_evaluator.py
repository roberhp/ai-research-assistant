from ai_research_assistant.evaluation.generation_case import (
    GenerationEvaluationCase,
)
from ai_research_assistant.rag.generation.rag_service import RagService


class GenerationEvaluator:
    def __init__(self, rag_service: RagService):
        self.rag_service = rag_service

    def evaluate(
        self,
        cases: list[GenerationEvaluationCase],
    ) -> dict[str, float]:
        if not cases:
            return {
                "answer_relevance": 0.0,
                "citation_precision": 0.0,
                "citation_recall": 0.0,
            }

        relevance_scores = []
        citation_precisions = []
        citation_recalls = []

        for case in cases:
            result = self.rag_service.answer(
                query=case.query,
                limit=5,
            )

            answer_relevance = self._calculate_answer_relevance(
                result.answer,
                case.expected_answer,
            )

            actual_sources = {
                citation.source
                for citation in result.citations
            }

            expected_sources = case.expected_sources

            citation_precision = (
                len(actual_sources & expected_sources)
                / len(actual_sources)
                if actual_sources
                else 0.0
            )

            citation_recall = (
                len(actual_sources & expected_sources)
                / len(expected_sources)
                if expected_sources
                else 0.0
            )

            relevance_scores.append(answer_relevance)
            citation_precisions.append(citation_precision)
            citation_recalls.append(citation_recall)

        return {
            "answer_relevance": sum(relevance_scores)
            / len(relevance_scores),
            "citation_precision": sum(citation_precisions)
            / len(citation_precisions),
            "citation_recall": sum(citation_recalls)
            / len(citation_recalls),
        }

    def _calculate_answer_relevance(
        self,
        answer: str,
        expected_answer: str,
    ) -> float:
        expected_words = set(
            expected_answer.lower().split()
        )

        answer_words = set(
            answer.lower().split()
        )

        if not expected_words:
            return 0.0

        overlap = expected_words & answer_words

        return len(overlap) / len(expected_words)