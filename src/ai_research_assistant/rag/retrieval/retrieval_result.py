from dataclasses import dataclass


@dataclass(frozen=True)
class RetrievalResult:
    content: str
    source: str
    chunk_index: int
    score: float