import re

from ai_research_assistant.rag.generation.rag_answer import RagCitation
from ai_research_assistant.rag.retrieval.retrieval_result import RetrievalResult


class CitationValidator:
    CITATION_PATTERN = re.compile(r"\[Source (\d+)\]")

    def validate(
        self,
        answer: str,
        sources: list[RetrievalResult],
    ) -> list[RagCitation]:
        citations = []

        for match in self.CITATION_PATTERN.finditer(answer):
            source_number = int(match.group(1))

            if source_number < 1 or source_number > len(sources):
                continue

            source = sources[source_number - 1]

            citation = RagCitation(
                source=source.source,
                chunk_index=source.chunk_index,
                score=source.score,
            )

            if citation not in citations:
                citations.append(citation)

        return citations