from ai_research_assistant.evaluation.evaluation_case import EvaluationCase
from ai_research_assistant.evaluation.evaluation_report import (
    EvaluationCaseReport,
    EvaluationReport,
)
from ai_research_assistant.rag.retrieval.retrieval_result import RetrievalResult


class RetrievalEvaluator:
    def hit_at_k(
        self,
        case: EvaluationCase,
        results: list[RetrievalResult],
        k: int,
    ) -> float:
        retrieved_sources = {
            result.source
            for result in results[:k]
        }

        return float(
            bool(retrieved_sources & case.expected_sources)
        )

    def mrr_at_k(
        self,
        case: EvaluationCase,
        results: list[RetrievalResult],
        k: int,
    ) -> float:
        for rank, result in enumerate(results[:k], start=1):
            if result.source in case.expected_sources:
                return 1.0 / rank

        return 0.0

    def evaluate(
        self,
        cases: list[EvaluationCase],
        retrieval_results: list[list[RetrievalResult]],
        k: int,
    ) -> EvaluationReport:
        if len(cases) != len(retrieval_results):
            raise ValueError(
                "Cases and retrieval results must have the same length."
            )

        if not cases:
            return EvaluationReport(
                hit_at_k=0.0,
                mrr_at_k=0.0,
                total_cases=0,
                cases=[],
            )

        case_reports = [
            EvaluationCaseReport(
                query=case.query,
                hit_at_k=self.hit_at_k(case, results, k),
                reciprocal_rank=self.mrr_at_k(case, results, k),
            )
            for case, results in zip(
                cases,
                retrieval_results,
            )
        ]

        hit_at_k = sum(
            case_report.hit_at_k
            for case_report in case_reports
        ) / len(case_reports)

        mrr_at_k = sum(
            case_report.reciprocal_rank
            for case_report in case_reports
        ) / len(case_reports)

        return EvaluationReport(
            hit_at_k=hit_at_k,
            mrr_at_k=mrr_at_k,
            total_cases=len(case_reports),
            cases=case_reports,
        )