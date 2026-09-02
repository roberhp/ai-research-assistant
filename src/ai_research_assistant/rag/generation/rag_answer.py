from dataclasses import dataclass

from ai_research_assistant.rag.retrieval.retrieval_result import RetrievalResult


@dataclass(frozen=True)
class RagAnswer:
    answer: str
    sources: list[RetrievalResult]