from ai_research_assistant.rag.embeddings.embedding_service import EmbeddingService
from ai_research_assistant.settings import Settings


def test_generate_embedding():
    settings = Settings()
    service = EmbeddingService(settings)

    embedding = service.generate(
        "PostgreSQL is a relational database."
    )

    assert isinstance(embedding, list)
    assert len(embedding) > 0
    assert all(isinstance(value, float) for value in embedding)