from ai_research_assistant.evaluation.evaluation_case import EvaluationCase
from ai_research_assistant.evaluation.retrieval_evaluator import RetrievalEvaluator
from ai_research_assistant.rag.retrieval.retrieval_result import RetrievalResult
from tests.fakes.fake_retrieval_service import FakeRetrievalService


def test_evaluator_calculates_hit_mrr_precision_and_recall():
    retrieval_service = FakeRetrievalService(
        results_by_query={
            "question": [
                RetrievalResult(
                    content="Relevant",
                    source="doc1.txt",
                    chunk_index=0,
                    score=0.95,
                ),
                RetrievalResult(
                    content="Irrelevant",
                    source="doc2.txt",
                    chunk_index=0,
                    score=0.80,
                ),
                RetrievalResult(
                    content="Relevant",
                    source="doc3.txt",
                    chunk_index=0,
                    score=0.75,
                ),
            ]
        }
    )

    evaluator = RetrievalEvaluator(retrieval_service)

    case = EvaluationCase(
        query="question",
        expected_sources={"doc1.txt", "doc3.txt"},
    )

    report = evaluator.evaluate([case], k=3)

    assert report["hit_at_k"] == 1.0
    assert report["mrr_at_k"] == 1.0
    assert report["precision_at_k"] == 2 / 3
    assert report["recall_at_k"] == 1.0


def test_evaluator_returns_zero_metrics_without_results():
    retrieval_service = FakeRetrievalService(
        results_by_query={"question": []}
    )

    evaluator = RetrievalEvaluator(retrieval_service)

    case = EvaluationCase(
        query="question",
        expected_sources={"doc1.txt"},
    )

    report = evaluator.evaluate([case], k=5)

    assert report["hit_at_k"] == 0.0
    assert report["mrr_at_k"] == 0.0
    assert report["precision_at_k"] == 0.0
    assert report["recall_at_k"] == 0.0


def test_evaluator_only_considers_top_k_results():
    retrieval_service = FakeRetrievalService(
        results_by_query={
            "question": [
                RetrievalResult(
                    content="Irrelevant",
                    source="doc1.txt",
                    chunk_index=0,
                    score=0.95,
                ),
                RetrievalResult(
                    content="Irrelevant",
                    source="doc2.txt",
                    chunk_index=0,
                    score=0.90,
                ),
                RetrievalResult(
                    content="Relevant",
                    source="doc3.txt",
                    chunk_index=0,
                    score=0.85,
                ),
            ]
        }
    )

    evaluator = RetrievalEvaluator(retrieval_service)

    case = EvaluationCase(
        query="question",
        expected_sources={"doc3.txt"},
    )

    report = evaluator.evaluate([case], k=2)

    assert report["hit_at_k"] == 0.0
    assert report["mrr_at_k"] == 0.0
    assert report["precision_at_k"] == 0.0
    assert report["recall_at_k"] == 0.0


def test_evaluator_calculates_reciprocal_rank_when_relevant_source_is_second():
    retrieval_service = FakeRetrievalService(
        results_by_query={
            "question": [
                RetrievalResult(
                    content="Irrelevant",
                    source="doc1.txt",
                    chunk_index=0,
                    score=0.95,
                ),
                RetrievalResult(
                    content="Relevant",
                    source="doc2.txt",
                    chunk_index=0,
                    score=0.90,
                ),
            ]
        }
    )

    evaluator = RetrievalEvaluator(retrieval_service)

    case = EvaluationCase(
        query="question",
        expected_sources={"doc2.txt"},
    )

    report = evaluator.evaluate([case], k=2)

    assert report["hit_at_k"] == 1.0
    assert report["mrr_at_k"] == 0.5


def test_evaluator_averages_metrics_across_cases():
    retrieval_service = FakeRetrievalService(
        results_by_query={
            "question 1": [
                RetrievalResult(
                    content="Relevant",
                    source="doc1.txt",
                    chunk_index=0,
                    score=0.95,
                )
            ],
            "question 2": [
                RetrievalResult(
                    content="Irrelevant",
                    source="doc3.txt",
                    chunk_index=0,
                    score=0.90,
                ),
                RetrievalResult(
                    content="Relevant",
                    source="doc2.txt",
                    chunk_index=0,
                    score=0.85,
                ),
            ],
        }
    )

    evaluator = RetrievalEvaluator(retrieval_service)

    cases = [
        EvaluationCase(
            query="question 1",
            expected_sources={"doc1.txt"},
        ),
        EvaluationCase(
            query="question 2",
            expected_sources={"doc2.txt"},
        ),
    ]

    report = evaluator.evaluate(cases, k=2)

    assert report["hit_at_k"] == 1.0
    assert report["mrr_at_k"] == (1.0 + 0.5) / 2
    assert report["precision_at_k"] == (1.0 + 0.5) / 2
    assert report["recall_at_k"] == 1.0


def test_evaluator_returns_zero_metrics_for_empty_dataset():
    retrieval_service = FakeRetrievalService(results_by_query={})
    evaluator = RetrievalEvaluator(retrieval_service)

    report = evaluator.evaluate(cases=[], k=3)

    assert report["hit_at_k"] == 0.0
    assert report["mrr_at_k"] == 0.0
    assert report["precision_at_k"] == 0.0
    assert report["recall_at_k"] == 0.0


def test_evaluator_deduplicates_sources_for_precision_and_recall():
    retrieval_service = FakeRetrievalService(
        results_by_query={
            "question": [
                RetrievalResult(
                    content="Relevant chunk 1",
                    source="doc1.txt",
                    chunk_index=0,
                    score=0.95,
                ),
                RetrievalResult(
                    content="Relevant chunk 2",
                    source="doc1.txt",
                    chunk_index=1,
                    score=0.90,
                ),
                RetrievalResult(
                    content="Irrelevant",
                    source="doc2.txt",
                    chunk_index=0,
                    score=0.80,
                ),
            ]
        }
    )

    evaluator = RetrievalEvaluator(retrieval_service)

    case = EvaluationCase(
        query="question",
        expected_sources={"doc1.txt"},
    )

    report = evaluator.evaluate([case], k=3)

    assert report["hit_at_k"] == 1.0
    assert report["mrr_at_k"] == 1.0
    assert report["precision_at_k"] == 0.5
    assert report["recall_at_k"] == 1.0