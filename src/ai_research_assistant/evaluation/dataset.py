from ai_research_assistant.evaluation.evaluation_case import EvaluationCase


EVALUATION_DATASET = [
    EvaluationCase(
        query="What does a RAG system do before generating an answer?",
        expected_sources={"rag-guide.txt"},
    ),
    EvaluationCase(
        query="What are embeddings used for?",
        expected_sources={"embeddings-guide.txt"},
    ),
    EvaluationCase(
        query="What is vector similarity search?",
        expected_sources={"vector-search-guide.txt"},
    ),
]