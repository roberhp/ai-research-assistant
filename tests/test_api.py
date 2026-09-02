from fastapi.testclient import TestClient

from ai_research_assistant.main import app
from ai_research_assistant.dependencies import get_retrieval_service
from ai_research_assistant.rag.retrieval.retrieval_result import RetrievalResult
from ai_research_assistant.dependencies import get_ingestion_service
from ai_research_assistant.dependencies import get_rag_service
from ai_research_assistant.rag.generation.rag_answer import RagAnswer
from ai_research_assistant.rag.retrieval.retrieval_result import RetrievalResult




client = TestClient(app)


class FakeChatService:
    def chat(self, message: str) -> str:
        return f"Fake response to: {message}"


class FakeRetrievalService:
    def retrieve(self, query: str, limit: int):
        return [
            RetrievalResult(
                content="RAG combines retrieval and generation.",
                source="test.pdf",
                chunk_index=3,
                score=0.95,
            )
        ]
class FakeIngestionService:
    def ingest(self, document):
        class FakeDocument:
            id = "12345678-1234-1234-1234-123456789012"
            source = document.source

        return FakeDocument(), 3

class FakeRagService:
    def answer(self, query: str, limit: int):
        return RagAnswer(
            answer="RAG retrieves relevant information before generating an answer.",
            sources=[
                RetrievalResult(
                    content="RAG combines retrieval and generation.",
                    source="rag.txt",
                    chunk_index=0,
                    score=0.95,
                )
            ],
        )

def test_root():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {"message": "AI Research Assistant"}


def test_chat():
    from ai_research_assistant.dependencies import get_chat_service

    app.dependency_overrides[get_chat_service] = lambda: FakeChatService()

    response = client.post(
        "/chat",
        json={"message": "What is RAG?"}
    )

    app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json() == {"response": "Fake response to: What is RAG?"}

def test_search_endpoint_returns_results():
    app.dependency_overrides[get_retrieval_service] = (
        lambda: FakeRetrievalService()
    )

    try:
        response = client.post(
            "/api/v1/search",
            json={
                "query": "What is RAG?",
                "limit": 3,
            },
        )

        assert response.status_code == 200

        data = response.json()

        assert len(data["results"]) == 1
        assert data["results"][0]["content"] == (
            "RAG combines retrieval and generation."
        )
        assert data["results"][0]["source"] == "test.pdf"
        assert data["results"][0]["chunk_index"] == 3
        assert data["results"][0]["score"] == 0.95

    finally:
        app.dependency_overrides.clear()

def test_search_endpoint_validates_limit():
    app.dependency_overrides[get_retrieval_service] = (
        lambda: FakeRetrievalService()
    )

    try:
        response = client.post(
            "/api/v1/search",
            json={
                "query": "What is RAG?",
                "limit": 100,
            },
        )

        assert response.status_code == 422

    finally:
        app.dependency_overrides.clear()

def test_create_document_endpoint():
    app.dependency_overrides[get_ingestion_service] = (
        lambda: FakeIngestionService()
    )

    try:
        response = client.post(
            "/api/v1/documents",
            json={
                "source": "test.txt",
                "content": "This is a test document.",
            },
        )

        assert response.status_code == 201

        data = response.json()

        assert data["id"] == "12345678-1234-1234-1234-123456789012"
        assert data["source"] == "test.txt"
        assert data["chunk_count"] == 3

    finally:
        app.dependency_overrides.clear()

def test_create_document_validates_empty_content():
    response = client.post(
        "/api/v1/documents",
        json={
            "source": "test.txt",
            "content": "",
        },
    )

    assert response.status_code == 422

def test_chat_endpoint_uses_rag_service():
    app.dependency_overrides[get_rag_service] = (
        lambda: FakeRagService()
    )

    try:
        response = client.post(
            "/api/v1/chat",
            json={
                "query": "What is RAG?",
                "limit": 5,
            },
        )

        assert response.status_code == 200

        data = response.json()

        assert response.json()["sources"] == [
            {
                "source": "rag.txt",
                "chunk_index": 0,
                "score": 0.95,
            }
        ]

    finally:
        app.dependency_overrides.clear()