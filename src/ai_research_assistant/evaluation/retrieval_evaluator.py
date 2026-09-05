from ai_research_assistant.evaluation.evaluation_case import EvaluationCase
from ai_research_assistant.rag.retrieval.retrieval_service import RetrievalService


class RetrievalEvaluator:
    def __init__(self, retrieval_service: RetrievalService):
        self.retrieval_service = retrieval_service

    def evaluate(
        self,
        cases: list[EvaluationCase],
        k: int = 5,
    ) -> dict[str, float]:
        if not cases:
            return {
                "hit_at_k": 0.0,
                "mrr_at_k": 0.0,
                "precision_at_k": 0.0,
                "recall_at_k": 0.0,
            }

        hits = []
        reciprocal_ranks = []
        precisions = []
        recalls = []

        for case in cases:
            results = self.retrieval_service.retrieve(
                query=case.query,
                limit=k,
            )

            retrieved_sources = [
                result.source
                for result in results
            ]

            expected_sources = case.expected_sources

            unique_retrieved_sources = list(
                dict.fromkeys(retrieved_sources)
            )

            relevant_sources = [
                source
                for source in unique_retrieved_sources
                if source in expected_sources
            ]

            hit = 1.0 if relevant_sources else 0.0

            reciprocal_rank = 0.0

            for index, source in enumerate(
                retrieved_sources,
                start=1,
            ):
                if source in expected_sources:
                    reciprocal_rank = 1.0 / index
                    break

            precision = (
                len(relevant_sources) / len(unique_retrieved_sources)
                if unique_retrieved_sources
                else 0.0
            )

            recall = (
                len(relevant_sources) / len(expected_sources)
                if expected_sources
                else 0.0
            )

            hits.append(hit)
            reciprocal_ranks.append(reciprocal_rank)
            precisions.append(precision)
            recalls.append(recall)

        return {
            "hit_at_k": sum(hits) / len(hits),
            "mrr_at_k": sum(reciprocal_ranks) / len(reciprocal_ranks),
            "precision_at_k": sum(precisions) / len(precisions),
            "recall_at_k": sum(recalls) / len(recalls),
        }