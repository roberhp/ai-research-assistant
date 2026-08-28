from ai_research_assistant.rag.chunk import Chunk
from ai_research_assistant.rag.document import Document


class TextChunker:
    def __init__(self, chunk_size: int, overlap: int):
        if overlap >= chunk_size:
            raise ValueError("Overlap must be smaller than chunk size.")

        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, document: Document) -> list[Chunk]:
        chunks = []
        start = 0
        chunk_index = 0

        step = self.chunk_size - self.overlap

        while start < len(document.content):
            end = start + self.chunk_size

            chunks.append(
                Chunk(
                    content=document.content[start:end],
                    source=document.source,
                    chunk_index=chunk_index,
                )
            )

            start += step
            chunk_index += 1

        return chunks