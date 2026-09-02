from ai_research_assistant.evaluation.evaluation_case import EvaluationCase
from ai_research_assistant.evaluation.retrieval_evaluator import RetrievalEvaluator
from ai_research_assistant.rag.retrieval.retrieval_result import RetrievalResult


def test_hit_at_k_returns_one_when_expected_source_is_retrieved():
    evaluator = RetrievalEvaluator()

    case = EvaluationCase(
        query="What is RAG?",
        expected_sources={"rag.txt"},
    )

    results = [
        RetrievalResult(
            content="RAG content",
            source="rag.txt",
            chunk_index=0,
            score=0.95,
        ),
        RetrievalResult(
            content="Embeddings content",
            source="embeddings.txt",
            chunk_index=0,
            score=0.80,
        ),
    ]

    assert evaluator.hit_at_k(case, results, k=2) == 1.0


def test_hit_at_k_returns_zero_when_expected_source_is_not_retrieved():
    evaluator = RetrievalEvaluator()

    case = EvaluationCase(
        query="What is RAG?",
        expected_sources={"rag.txt"},
    )

    results = [
        RetrievalResult(
            content="Java content",
            source="java.txt",
            chunk_index=0,
            score=0.95,
        ),
        RetrievalResult(
            content="Python content",
            source="python.txt",
            chunk_index=0,
            score=0.80,
        ),
    ]

    assert evaluator.hit_at_k(case, results, k=2) == 0.0


def test_hit_at_k_only_considers_top_k_results():
    evaluator = RetrievalEvaluator()

    case = EvaluationCase(
        query="What is RAG?",
        expected_sources={"rag.txt"},
    )

    results = [
        RetrievalResult(
            content="Java content",
            source="java.txt",
            chunk_index=0,
            score=0.95,
        ),
        RetrievalResult(
            content="Python content",
            source="python.txt",
            chunk_index=0,
            score=0.80,
        ),
        RetrievalResult(
            content="RAG content",
            source="rag.txt",
            chunk_index=0,
            score=0.70,
        ),
    ]

    assert evaluator.hit_at_k(case, results, k=2) == 0.0
    assert evaluator.hit_at_k(case, results, k=3) == 1.0


def test_mrr_at_k_returns_one_when_relevant_source_is_first():
    evaluator = RetrievalEvaluator()

    case = EvaluationCase(
        query="What is RAG?",
        expected_sources={"rag.txt"},
    )

    results = [
        RetrievalResult(
            content="RAG content",
            source="rag.txt",
            chunk_index=0,
            score=0.95,
        ),
        RetrievalResult(
            content="Java content",
            source="java.txt",
            chunk_index=0,
            score=0.80,
        ),
    ]

    assert evaluator.mrr_at_k(case, results, k=2) == 1.0


def test_mrr_at_k_returns_reciprocal_rank():
    evaluator = RetrievalEvaluator()

    case = EvaluationCase(
        query="What is RAG?",
        expected_sources={"rag.txt"},
    )

    results = [
        RetrievalResult(
            content="Java content",
            source="java.txt",
            chunk_index=0,
            score=0.95,
        ),
        RetrievalResult(
            content="RAG content",
            source="rag.txt",
            chunk_index=0,
            score=0.80,
        ),
    ]

    assert evaluator.mrr_at_k(case, results, k=2) == 0.5


def test_mrr_at_k_returns_zero_when_relevant_source_is_not_found():
    evaluator = RetrievalEvaluator()

    case = EvaluationCase(
        query="What is RAG?",
        expected_sources={"rag.txt"},
    )

    results = [
        RetrievalResult(
            content="Java content",
            source="java.txt",
            chunk_index=0,
            score=0.95,
        ),
    ]

    assert evaluator.mrr_at_k(case, results, k=1) == 0.0


def test_evaluate_returns_average_metrics():
    evaluator = RetrievalEvaluator()

    cases = [
        EvaluationCase(
            query="What is RAG?",
            expected_sources={"rag.txt"},
        ),
        EvaluationCase(
            query="What are embeddings?",
            expected_sources={"embeddings.txt"},
        ),
        EvaluationCase(
            query="What is Java?",
            expected_sources={"java.txt"},
        ),
    ]

    retrieval_results = [
        [
            RetrievalResult(
                content="RAG content",
                source="rag.txt",
                chunk_index=0,
                score=0.95,
            )
        ],
        [
            RetrievalResult(
                content="Wrong content",
                source="python.txt",
                chunk_index=0,
                score=0.90,
            ),
            RetrievalResult(
                content="Embeddings content",
                source="embeddings.txt",
                chunk_index=0,
                score=0.80,
            ),
        ],
        [
            RetrievalResult(
                content="Java content",
                source="java.txt",
                chunk_index=0,
                score=0.70,
            )
        ],
    ]

    report = evaluator.evaluate(
        cases=cases,
        retrieval_results=retrieval_results,
        k=2,
    )

    assert report.total_cases == 3
    assert report.hit_at_k == 1.0
    assert report.mrr_at_k == (1.0 + 0.5 + 1.0) / 3


def test_evaluate_rejects_different_number_of_cases_and_results():
    evaluator = RetrievalEvaluator()

    cases = [
        EvaluationCase(
            query="What is RAG?",
            expected_sources={"rag.txt"},
        )
    ]

    retrieval_results = []

    try:
        evaluator.evaluate(
            cases=cases,
            retrieval_results=retrieval_results,
            k=3,
        )
        assert False
    except ValueError as error:
        assert str(error) == (
            "Cases and retrieval results must have the same length."
        )


def test_evaluate_returns_zero_metrics_for_empty_dataset():
    evaluator = RetrievalEvaluator()

    report = evaluator.evaluate(
        cases=[],
        retrieval_results=[],
        k=3,
    )

    assert report.total_cases == 0
    assert report.hit_at_k == 0.0
    assert report.mrr_at_k == 0.0