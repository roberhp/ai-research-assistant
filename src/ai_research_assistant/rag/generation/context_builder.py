from ai_research_assistant.rag.retrieval.retrieval_result import RetrievalResult


class ContextBuilder:
    def build(self, results: list[RetrievalResult]) -> str:
        return "\n\n".join(
            (
                f"[Source {index}]\n"
                f"Document: {result.source}\n"
                f"Chunk: {result.chunk_index}\n\n"
                f"{result.content}"
            )
            for index, result in enumerate(results, start=1)
        )