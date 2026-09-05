from ai_research_assistant.evaluation.generation_evaluation_result import (
    GenerationEvaluationResult,
)
from ai_research_assistant.evaluation.llm_generation_evaluator import (
    LLMGenerationEvaluator,
)
from ai_research_assistant.rag.generation.context_builder import ContextBuilder
from ai_research_assistant.rag.generation.rag_service import RagService


class RagGenerationEvaluator:
    def __init__(
        self,
        rag_service: RagService,
        llm_evaluator: LLMGenerationEvaluator,
        context_builder: ContextBuilder,
    ):
        self.rag_service = rag_service
        self.llm_evaluator = llm_evaluator
        self.context_builder = context_builder

    def evaluate(
        self,
        query: str,
        limit: int = 5,
    ) -> GenerationEvaluationResult:
        result = self.rag_service.answer(
            query=query,
            limit=limit,
        )

        context = self.context_builder.build(
            result.sources
        )

        return self.llm_evaluator.evaluate(
            query=query,
            context=context,
            answer=result.answer,
        )