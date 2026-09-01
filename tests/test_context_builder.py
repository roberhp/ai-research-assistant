from ai_research_assistant.rag.generation.context_builder import ContextBuilder
from ai_research_assistant.rag.retrieval.retrieval_result import RetrievalResult


def test_context_builder_formats_results():
    builder = ContextBuilder()

    results = [
        RetrievalResult(
            content="RAG combines retrieval and generation.",
            source="rag.txt",
            chunk_index=0,
            score=0.95,
        ),
        RetrievalResult(
            content="Embeddings represent text as vectors.",
            source="embeddings.txt",
            chunk_index=2,
            score=0.90,
        ),
    ]

    context = builder.build(results)

    assert "[Source: rag.txt, Chunk: 0]" in context
    assert "RAG combines retrieval and generation." in context
    assert "[Source: embeddings.txt, Chunk: 2]" in context
    assert "Embeddings represent text as vectors." in context