from dataclasses import dataclass

from ai_research_assistant.rag.retrieval.retrieval_result import RetrievalResult


@dataclass(frozen=True)
class RagCitation:
    source: str
    chunk_index: int
    score: float


@dataclass(frozen=True)
class RagAnswer:
    answer: str
    sources: list[RetrievalResult]
    citations: list[RagCitation]