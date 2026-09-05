from ai_research_assistant.evaluation.evaluation_case import EvaluationCase


EVALUATION_CASES = [
    EvaluationCase(
        query="What is retrieval augmented generation?",
        expected_sources={"rag-guide.txt"},
    ),
    EvaluationCase(
        query="How does RAG combine retrieval and generation?",
        expected_sources={"rag-guide.txt"},
    ),
    EvaluationCase(
        query="What are the main steps in a RAG pipeline?",
        expected_sources={"rag-guide.txt"},
    ),
    EvaluationCase(
        query="Why is grounding important in RAG systems?",
        expected_sources={"rag-guide.txt"},
    ),
    EvaluationCase(
        query="What is chunking in a RAG system?",
        expected_sources={"rag-guide.txt"},
    ),
    EvaluationCase(
        query="What is an embedding?",
        expected_sources={"embeddings-guide.txt"},
    ),
    EvaluationCase(
        query="Why are embeddings useful for semantic search?",
        expected_sources={"embeddings-guide.txt"},
    ),
    EvaluationCase(
        query="How are text embeddings generated?",
        expected_sources={"embeddings-guide.txt"},
    ),
    EvaluationCase(
        query="What is the purpose of an embedding model?",
        expected_sources={"embeddings-guide.txt"},
    ),
    EvaluationCase(
        query="Why can embeddings represent semantic similarity?",
        expected_sources={"embeddings-guide.txt"},
    ),
    EvaluationCase(
        query="What is vector search?",
        expected_sources={"vector-search-guide.txt"},
    ),
    EvaluationCase(
        query="How does vector similarity search work?",
        expected_sources={"vector-search-guide.txt"},
    ),
    EvaluationCase(
        query="Why is cosine similarity useful for vector search?",
        expected_sources={"vector-search-guide.txt"},
    ),
    EvaluationCase(
        query="What is a vector database?",
        expected_sources={"vector-search-guide.txt"},
    ),
    EvaluationCase(
        query="How are vectors compared during similarity search?",
        expected_sources={"vector-search-guide.txt"},
    ),
    EvaluationCase(
        query="What is Kubernetes?",
        expected_sources=set(),
    ),
    EvaluationCase(
        query="How does Kafka work?",
        expected_sources=set(),
    ),
    EvaluationCase(
        query="What is a relational database?",
        expected_sources=set(),
    ),
    EvaluationCase(
        query="How does OAuth authentication work?",
        expected_sources=set(),
    ),
    EvaluationCase(
        query="What is a Docker container?",
        expected_sources=set(),
    ),
]