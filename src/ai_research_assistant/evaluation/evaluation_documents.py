from ai_research_assistant.rag.document import Document


EVALUATION_DOCUMENTS = [
    Document(
        source="rag-guide.txt",
        content="""
        Retrieval-Augmented Generation, or RAG, combines information retrieval
        with language model generation. Before generating an answer, a RAG
        system retrieves relevant information from a knowledge base.
        The retrieved information is then provided to the language model
        as context.
        """.strip(),
    ),
    Document(
        source="embeddings-guide.txt",
        content="""
        Embeddings represent text as numerical vectors. They allow systems
        to compare semantic similarity between pieces of text. Texts with
        similar meanings tend to have vectors that are closer together
        in the embedding space.
        """.strip(),
    ),
    Document(
        source="vector-search-guide.txt",
        content="""
        Vector similarity search finds the vectors that are most similar
        to a query vector. In a RAG system, vector search can be used to
        retrieve relevant document chunks from a vector database.
        """.strip(),
    ),
]