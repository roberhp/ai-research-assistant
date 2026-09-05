from ai_research_assistant.evaluation.dataset import EVALUATION_CASES
from ai_research_assistant.evaluation.evaluation_setup import (
    create_evaluation_retrieval_service,
)
from ai_research_assistant.evaluation.retrieval_evaluator import RetrievalEvaluator


K_VALUES = [1, 3, 5, 10]

THRESHOLD_VALUES = [
    0.50,
    0.60,
    0.70,
    0.80,
    0.85,
]


def print_report(
    k: int,
    report: dict[str, float],
) -> None:
    print(f"\nK={k}")
    print(f"  Hit@K:        {report['hit_at_k']:.3f}")
    print(f"  MRR@K:        {report['mrr_at_k']:.3f}")
    print(f"  Precision@K:  {report['precision_at_k']:.3f}")
    print(f"  Recall@K:     {report['recall_at_k']:.3f}")


def evaluate_k_values() -> None:
    print("=" * 60)
    print("RETRIEVAL EVALUATION - K VALUES")
    print("=" * 60)

    retrieval_service = create_evaluation_retrieval_service()
    evaluator = RetrievalEvaluator(retrieval_service)

    for k in K_VALUES:
        report = evaluator.evaluate(
            cases=EVALUATION_CASES,
            k=k,
        )

        print_report(k, report)


def evaluate_threshold_values() -> None:
    print("\n" + "=" * 60)
    print("RETRIEVAL EVALUATION - THRESHOLD VALUES")
    print("=" * 60)

    for threshold in THRESHOLD_VALUES:
        retrieval_service = create_evaluation_retrieval_service(
            similarity_threshold=threshold,
        )

        evaluator = RetrievalEvaluator(retrieval_service)

        report = evaluator.evaluate(
            cases=EVALUATION_CASES,
            k=5,
        )

        print(f"\nThreshold={threshold:.2f}")
        print(f"  Hit@5:        {report['hit_at_k']:.3f}")
        print(f"  MRR@5:        {report['mrr_at_k']:.3f}")
        print(f"  Precision@5:  {report['precision_at_k']:.3f}")
        print(f"  Recall@5:     {report['recall_at_k']:.3f}")


def main() -> None:
    evaluate_k_values()
    evaluate_threshold_values()


if __name__ == "__main__":
    main()