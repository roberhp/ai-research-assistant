import pytest

from ai_research_assistant.rag.chunk import Chunk
from ai_research_assistant.rag.chunking.text_chunker import TextChunker
from ai_research_assistant.rag.document import Document


def test_document_is_split_into_chunks():
    document = Document(
        content="ABCDEFGHIJKLMNOPQRSTUVWXYZ",
        source="test.txt",
    )

    chunker = TextChunker(chunk_size=10, overlap=2)

    chunks = chunker.chunk(document)

    assert len(chunks) == 4
    assert chunks[0].content == "ABCDEFGHIJ"
    assert chunks[1].content == "IJKLMNOPQR"
    assert chunks[2].content == "QRSTUVWXYZ"

def test_chunks_have_overlap():
    document = Document(
        content="ABCDEFGHIJKLMNOPQRSTUVWXYZ",
        source="test.txt",
    )

    chunker = TextChunker(chunk_size=10, overlap=2)

    chunks = chunker.chunk(document)

    assert chunks[0].content[-2:] == chunks[1].content[:2]
    assert chunks[1].content[-2:] == chunks[2].content[:2]


def test_overlap_must_be_smaller_than_chunk_size():
    with pytest.raises(ValueError):
        TextChunker(chunk_size=10, overlap=10)


def test_overlap_cannot_be_larger_than_chunk_size():
    with pytest.raises(ValueError):
        TextChunker(chunk_size=10, overlap=15)