from dataclasses import dataclass


@dataclass(frozen=True)
class GenerationEvaluationResult:
    relevance: float
    grounding: float
    citation_correctness: float