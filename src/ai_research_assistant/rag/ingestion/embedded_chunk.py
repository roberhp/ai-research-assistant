from dataclasses import dataclass


@dataclass(frozen=True)
class EmbeddedChunk:
    content: str
    chunk_index: int
    embedding: list[float]