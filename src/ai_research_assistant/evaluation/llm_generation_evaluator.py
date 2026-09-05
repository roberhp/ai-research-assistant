import json

from ai_research_assistant.evaluation.generation_evaluation_result import (
    GenerationEvaluationResult,
)
from ai_research_assistant.llm.provider import LLMProvider


class LLMGenerationEvaluator:
    def __init__(self, llm_provider: LLMProvider):
        self.llm_provider = llm_provider

    def evaluate(
        self,
        query: str,
        context: str,
        answer: str,
    ) -> GenerationEvaluationResult:
        prompt = self._build_prompt(
            query=query,
            context=context,
            answer=answer,
        )

        response = self.llm_provider.generate(prompt)

        evaluation = json.loads(response)

        return GenerationEvaluationResult(
            relevance=float(evaluation["relevance"]),
            grounding=float(evaluation["grounding"]),
            citation_correctness=float(
                evaluation["citation_correctness"]
            ),
        )

    def _build_prompt(
        self,
        query: str,
        context: str,
        answer: str,
    ) -> str:
        return f"""
You are evaluating the quality of a RAG system response.

Evaluate the response using only the provided context.

Question:
{query}

Context:
{context}

Answer:
{answer}

Evaluate three dimensions using a score between 0.0 and 1.0.

1. relevance:
How well does the answer address the user's question?

2. grounding:
How well is the answer supported by the provided context?
A response containing claims that are not supported by the context
should receive a lower score.

3. citation_correctness:
Are the citations in the answer supported by the provided context?
If the answer contains no citations, return 0.0.

Return ONLY valid JSON using exactly this structure:

{{
  "relevance": 0.0,
  "grounding": 0.0,
  "citation_correctness": 0.0
}}
""".strip()