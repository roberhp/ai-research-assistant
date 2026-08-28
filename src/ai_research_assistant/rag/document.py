from dataclasses import dataclass


@dataclass
class Document:
    content: str
    source: str