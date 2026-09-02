from dataclasses import dataclass


@dataclass(frozen=True)
class EvaluationCaseReport:
    query: str
    hit_at_k: float
    reciprocal_rank: float


@dataclass(frozen=True)
class EvaluationReport:
    hit_at_k: float
    mrr_at_k: float
    total_cases: int
    cases: list[EvaluationCaseReport]