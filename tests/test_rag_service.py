from unittest.mock import Mock

from ai_research_assistant.rag.generation.context_builder import ContextBuilder
from ai_research_assistant.rag.generation.rag_service import RagService
from ai_research_assistant.rag.retrieval.retrieval_result import RetrievalResult
from tests.fakes.fake_llm_provider import FakeLLMProvider


def test_rag_service_generates_answer():
    retrieval_service = Mock()

    retrieval_service.retrieve.return_value = [
        RetrievalResult(
            content="RAG retrieves relevant context before generating an answer.",
            source="rag-guide.txt",
            chunk_index=0,
            score=0.91,
        )
    ]

    llm_provider = FakeLLMProvider(
        response="RAG retrieves relevant context before generating an answer."
    )

    service = RagService(
        retrieval_service=retrieval_service,
        llm_provider=llm_provider,
        context_builder=ContextBuilder(),
    )

    result = service.answer(
        query="What does RAG do before generating an answer?"
    )

    assert result.answer == (
        "RAG retrieves relevant context before generating an answer."
    )

    assert len(result.sources) == 1
    assert result.sources[0].source == "rag-guide.txt"

    assert len(llm_provider.prompts) == 1
    assert "What does RAG do before generating an answer?" in (
        llm_provider.prompts[0]
    )


def test_rag_service_returns_fallback_when_no_results():
    retrieval_service = Mock()
    retrieval_service.retrieve.return_value = []

    llm_provider = FakeLLMProvider()

    service = RagService(
        retrieval_service=retrieval_service,
        llm_provider=llm_provider,
        context_builder=ContextBuilder(),
    )

    result = service.answer(
        query="Question without matching context"
    )

    assert result.answer == (
        "I don't have enough information to answer this question."
    )

    assert result.sources == []
    assert llm_provider.prompts == []