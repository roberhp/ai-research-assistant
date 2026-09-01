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

    answer = service.answer(
        query="What is RAG?",
        limit=5,
    )

    assert answer == (
        "RAG retrieves relevant information before generating an answer."
    )

    assert "RAG combines retrieval and generation." in (
        llm_provider.last_prompt
    )

    assert "What is RAG?" in llm_provider.last_prompt