from typing import Protocol


class EmbeddingCache(Protocol):
    def get(self, key: str) -> list[float] | None:
        ...

    def set(
        self,
        key: str,
        embedding: list[float],
        ttl: int,
    ) -> None:
        ...