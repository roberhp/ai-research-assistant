from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from ai_research_assistant.evaluation.dataset import EVALUATION_DATASET
from ai_research_assistant.evaluation.evaluation_setup import EvaluationSetup
from ai_research_assistant.evaluation.retrieval_evaluator import (
    RetrievalEvaluator,
)
from ai_research_assistant.rag.embeddings.embedding_service import (
    EmbeddingService,
)
from ai_research_assistant.rag.retrieval.retrieval_service import (
    RetrievalService,
)
from ai_research_assistant.repositories.chunk_repository import ChunkRepository
from ai_research_assistant.settings import Settings


def run(k: int = 3):
    settings = Settings()

    engine = create_engine(settings.database_url)

    embedding_service = EmbeddingService(settings)

    with Session(engine) as session:
        setup = EvaluationSetup(
            session=session,
            embedding_service=embedding_service,
        )

        setup.prepare()

        retrieval_service = RetrievalService(
            embedding_service=embedding_service,
            chunk_repository=ChunkRepository(session),
        )

        evaluator = RetrievalEvaluator()

        retrieval_results = [
            retrieval_service.retrieve(
                query=case.query,
                limit=k,
            )
            for case in EVALUATION_DATASET
        ]

        return evaluator.evaluate(
            cases=EVALUATION_DATASET,
            retrieval_results=retrieval_results,
            k=k,
        )


if __name__ == "__main__":
    k = 3

    report = run(k=k)

    print()
    print("RAG Retrieval Evaluation")
    print("========================")
    print(f"Total cases: {report.total_cases}")
    print(f"Hit@{k}:      {report.hit_at_k:.3f}")
    print(f"MRR@{k}:      {report.mrr_at_k:.3f}")

    print()
    print("Cases")
    print("-----")

    for case in report.cases:
        print(f"Query: {case.query}")
        print(f"Hit@{k}: {case.hit_at_k:.3f}")
        print(f"RR:     {case.reciprocal_rank:.3f}")
        print()