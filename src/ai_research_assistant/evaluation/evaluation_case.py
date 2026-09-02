from dataclasses import dataclass


@dataclass(frozen=True)
class EvaluationCase:
    query: str
    expected_sources: set[str]