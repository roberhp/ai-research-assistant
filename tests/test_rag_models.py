from ai_research_assistant.rag.chunk import Chunk
from ai_research_assistant.rag.document import Document


def test_document_creation():
    document = Document(
        content="RAG combines retrieval with generation.",
        source="rag.txt",
    )

    assert document.content == "RAG combines retrieval with generation."
    assert document.source == "rag.txt"


def test_chunk_creation():
    chunk = Chunk(
        content="RAG combines retrieval with generation.",
        source="rag.txt",
        chunk_index=0,
    )

    assert chunk.chunk_index == 0
    assert chunk.source == "rag.txt"