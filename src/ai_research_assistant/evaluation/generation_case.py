from dataclasses import dataclass


@dataclass(frozen=True)
class GenerationEvaluationCase:
    query: str
    expected_answer: str
    expected_sources: set[str]