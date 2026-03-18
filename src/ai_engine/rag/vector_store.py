"""Vector store interface for the RAG pipeline."""

from __future__ import annotations

from abc import ABC, abstractmethod

from ai_engine.rag.document import Document


class VectorStore(ABC):
    """Abstract base class for vector databases.

    Subclasses must implement :meth:`add`, :meth:`search`, and :meth:`clear`.
    """

    @abstractmethod
    def add(self, documents: list[Document], embeddings: list[list[float]]) -> None:
        """Store documents together with their embeddings.

        Args:
            documents: The documents to store.
            embeddings: Pre-computed embedding vectors (one per document).
        """

    @abstractmethod
    def search(
        self, query_embedding: list[float], top_k: int = 5
    ) -> list[tuple[Document, float]]:
        """Retrieve the *top_k* most similar documents.

        Args:
            query_embedding: The embedding of the query.
            top_k: Number of results to return.

        Returns:
            A list of ``(document, score)`` tuples ordered by descending score.
        """

    @abstractmethod
    def clear(self) -> None:
        """Remove all stored documents and embeddings."""


class InMemoryVectorStore(VectorStore):
    """Simple in-memory vector store using cosine similarity.

    Suitable for development and testing without external dependencies.
    """

    def __init__(self) -> None:
        self._documents: list[Document] = []
        self._embeddings: list[list[float]] = []

    def add(self, documents: list[Document], embeddings: list[list[float]]) -> None:
        if len(documents) != len(embeddings):
            raise ValueError("documents and embeddings must have the same length")
        self._documents.extend(documents)
        self._embeddings.extend(embeddings)

    def search(
        self, query_embedding: list[float], top_k: int = 5
    ) -> list[tuple[Document, float]]:
        if not self._documents:
            return []

        scores = [
            self._cosine_similarity(query_embedding, emb) for emb in self._embeddings
        ]
        ranked = sorted(zip(self._documents, scores), key=lambda x: x[1], reverse=True)
        return list(ranked[:top_k])

    def clear(self) -> None:
        self._documents.clear()
        self._embeddings.clear()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x**2 for x in a) ** 0.5
        norm_b = sum(x**2 for x in b) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)
