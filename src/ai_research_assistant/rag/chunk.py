from dataclasses import dataclass


@dataclass
class Chunk:
    content: str
    source: str
    chunk_index: int