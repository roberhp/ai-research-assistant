from fastapi.testclient import TestClient

from ai_research_assistant.main import app


client = TestClient(app)


def test_root():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {
        "message": "AI Research Assistant"
    }

class FakeChatService:
    def chat(self, message: str) -> str:
        return f"Fake response to: {message}"
    
def test_chat():
    from ai_research_assistant.dependencies import get_chat_service

    app.dependency_overrides[get_chat_service] = lambda: FakeChatService()

    response = client.post(
        "/chat",
        json={"message": "What is RAG?"}
    )

    app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json() == {
        "response": "Fake response to: What is RAG?"
    }


