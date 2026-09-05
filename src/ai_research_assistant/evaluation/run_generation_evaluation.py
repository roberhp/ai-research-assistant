from ai_research_assistant.database import SessionLocal
from ai_research_assistant.evaluation.generation_dataset import (
    GENERATION_EVALUATION_CASES,
)
from ai_research_assistant.evaluation.llm_generation_evaluator import (
    LLMGenerationEvaluator,
)
from ai_research_assistant.evaluation.rag_generation_evaluator import (
    RagGenerationEvaluator,
)
from ai_research_assistant.llm.gemini_provider import GeminiProvider
from ai_research_assistant.rag.embeddings.embedding_service import EmbeddingService
from ai_research_assistant.rag.generation.citation_validator import (
    CitationValidator,
)
from ai_research_assistant.rag.generation.context_builder import ContextBuilder
from ai_research_assistant.rag.generation.rag_service import RagService
from ai_research_assistant.rag.retrieval.retrieval_service import RetrievalService
from ai_research_assistant.repositories.chunk_repository import ChunkRepository
from ai_research_assistant.settings import Settings


def create_evaluator() -> RagGenerationEvaluator:
    settings = Settings()

    db = SessionLocal()

    embedding_service = EmbeddingService(
        settings=settings,
    )

    retrieval_service = RetrievalService(
        embedding_service=embedding_service,
        chunk_repository=ChunkRepository(db),
        similarity_threshold=0.70,
    )

    rag_service = RagService(
        retrieval_service=retrieval_service,
        llm_provider=GeminiProvider(settings),
        context_builder=ContextBuilder(),
        citation_validator=CitationValidator(),
    )

    llm_evaluator = LLMGenerationEvaluator(
        GeminiProvider(settings)
    )

    return RagGenerationEvaluator(
        rag_service=rag_service,
        llm_evaluator=llm_evaluator,
        context_builder=ContextBuilder(),
    )


def main() -> None:
    evaluator = create_evaluator()

    relevance_scores = []
    grounding_scores = []
    citation_scores = []

    for case in GENERATION_EVALUATION_CASES:
        result = evaluator.evaluate(
            query=case.query,
        )

        relevance_scores.append(result.relevance)
        grounding_scores.append(result.grounding)
        citation_scores.append(result.citation_correctness)

        print(f"\nQuery: {case.query}")
        print(f"  Relevance:            {result.relevance:.3f}")
        print(f"  Grounding:            {result.grounding:.3f}")
        print(
            f"  Citation correctness: "
            f"{result.citation_correctness:.3f}"
        )

    print("\n" + "=" * 60)
    print("GENERATION EVALUATION")
    print("=" * 60)

    print(
        f"Average relevance: "
        f"{sum(relevance_scores) / len(relevance_scores):.3f}"
    )

    print(
        f"Average grounding: "
        f"{sum(grounding_scores) / len(grounding_scores):.3f}"
    )

    print(
        f"Average citation correctness: "
        f"{sum(citation_scores) / len(citation_scores):.3f}"
    )


if __name__ == "__main__":
    main()