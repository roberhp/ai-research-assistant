from ai_research_assistant.rag.retrieval.retrieval_result import RetrievalResult


class ContextBuilder:
    def build(self, results: list[RetrievalResult]) -> str:
        return "\n\n".join(
            (
                f"[Source: {result.source}, "
                f"Chunk: {result.chunk_index}]\n"
                f"{result.content}"
            )
            for result in results
        )