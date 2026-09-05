from ai_research_assistant.evaluation.generation_case import (
    GenerationEvaluationCase,
)


GENERATION_EVALUATION_CASES = [
    GenerationEvaluationCase(
        query="What is retrieval augmented generation?",
        expected_answer=(
            "Retrieval augmented generation combines information retrieval "
            "with language generation by providing retrieved context to a language model."
        ),
        expected_sources={"rag-guide.txt"},
    ),
    GenerationEvaluationCase(
        query="What is an embedding?",
        expected_answer=(
            "An embedding is a numerical representation of data that captures "
            "semantic information."
        ),
        expected_sources={"embeddings-guide.txt"},
    ),
    GenerationEvaluationCase(
        query="What is vector search?",
        expected_answer=(
            "Vector search finds similar items by comparing their vector representations."
        ),
        expected_sources={"vector-search-guide.txt"},
    ),
]