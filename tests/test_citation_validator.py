from ai_research_assistant.rag.generation.citation_validator import CitationValidator
from ai_research_assistant.rag.retrieval.retrieval_result import RetrievalResult


def test_validator_extracts_valid_citations():
    validator = CitationValidator()

    sources = [
        RetrievalResult(
            content="RAG combines retrieval and generation.",
            source="rag-guide.txt",
            chunk_index=0,
            score=0.95,
        ),
        RetrievalResult(
            content="Embeddings represent semantic information.",
            source="embeddings-guide.txt",
            chunk_index=1,
            score=0.90,
        ),
    ]

    citations = validator.validate(
        answer="RAG combines retrieval and generation [Source 1].",
        sources=sources,
    )

    assert len(citations) == 1
    assert citations[0].source == "rag-guide.txt"
    assert citations[0].chunk_index == 0


def test_validator_extracts_multiple_citations():
    validator = CitationValidator()

    sources = [
        RetrievalResult(
            content="RAG information.",
            source="rag-guide.txt",
            chunk_index=0,
            score=0.95,
        ),
        RetrievalResult(
            content="Embedding information.",
            source="embeddings-guide.txt",
            chunk_index=1,
            score=0.90,
        ),
    ]

    citations = validator.validate(
        answer="RAG uses embeddings [Source 1] [Source 2].",
        sources=sources,
    )

    assert len(citations) == 2
    assert citations[0].source == "rag-guide.txt"
    assert citations[1].source == "embeddings-guide.txt"


def test_validator_ignores_invalid_source_numbers():
    validator = CitationValidator()

    sources = [
        RetrievalResult(
            content="RAG information.",
            source="rag-guide.txt",
            chunk_index=0,
            score=0.95,
        ),
    ]

    citations = validator.validate(
        answer="This information is from [Source 1] and [Source 99].",
        sources=sources,
    )

    assert len(citations) == 1
    assert citations[0].source == "rag-guide.txt"


def test_validator_deduplicates_citations():
    validator = CitationValidator()

    sources = [
        RetrievalResult(
            content="RAG information.",
            source="rag-guide.txt",
            chunk_index=0,
            score=0.95,
        ),
    ]

    citations = validator.validate(
        answer="RAG is useful [Source 1]. It retrieves context [Source 1].",
        sources=sources,
    )

    assert len(citations) == 1