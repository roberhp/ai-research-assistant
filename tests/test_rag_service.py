from ai_research_assistant.rag.generation.context_builder import ContextBuilder
from ai_research_assistant.rag.generation.rag_service import RagService
from ai_research_assistant.rag.retrieval.retrieval_result import RetrievalResult


class FakeRetrievalService:
    def retrieve(self, query: str, limit: int):
        return [
            RetrievalResult(
                content="RAG combines retrieval and generation.",
                source="rag.txt",
                chunk_index=0,
                score=0.95,
            )
        ]


class EmptyRetrievalService:
    def retrieve(self, query: str, limit: int):
        return []


class FakeLLMProvider:
    def __init__(self):
        self.last_prompt = None

    def generate(self, prompt: str) -> str:
        self.last_prompt = prompt

        return "RAG retrieves relevant information before generating an answer."


def test_rag_service_uses_retrieved_context():
    retrieval_service = FakeRetrievalService()
    llm_provider = FakeLLMProvider()

    service = RagService(
        retrieval_service=retrieval_service,
        llm_provider=llm_provider,
        context_builder=ContextBuilder(),
    )

    result = service.answer(
        query="What is RAG?",
        limit=5,
    )

    assert result.answer == (
        "RAG retrieves relevant information before generating an answer."
    )

    assert len(result.sources) == 1

    assert result.sources[0].source == "rag.txt"
    assert result.sources[0].chunk_index == 0
    assert result.sources[0].score == 0.95

    assert "RAG combines retrieval and generation." in (
        llm_provider.last_prompt
    )

    assert "What is RAG?" in llm_provider.last_prompt

    assert "[Source 1]" in llm_provider.last_prompt
    assert "Do not invent source identifiers." in llm_provider.last_prompt


def test_rag_service_does_not_call_llm_without_relevant_results():
    retrieval_service = EmptyRetrievalService()
    llm_provider = FakeLLMProvider()

    service = RagService(
        retrieval_service=retrieval_service,
        llm_provider=llm_provider,
        context_builder=ContextBuilder(),
    )

    result = service.answer(
        query="What is quantum computing?",
        limit=5,
    )

    assert result.answer == (
        "I don't have enough information to answer this question."
    )

    assert result.sources == []

    assert llm_provider.last_prompt is None