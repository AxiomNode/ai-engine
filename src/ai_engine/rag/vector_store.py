"""Vector store interface for the RAG pipeline."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

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
    Uses numpy for vectorized similarity computation.
    """

    def __init__(self) -> None:
        self._documents: list[Document] = []
        self._embeddings_list: list[list[float]] = []
        self._embeddings_matrix: np.ndarray | None = None
        self._norms: np.ndarray | None = None
        self._dirty: bool = False

    def add(self, documents: list[Document], embeddings: list[list[float]]) -> None:
        if len(documents) != len(embeddings):
            raise ValueError("documents and embeddings must have the same length")
        self._documents.extend(documents)
        self._embeddings_list.extend(embeddings)
        self._dirty = True

    def _rebuild_matrix(self) -> None:
        """Rebuild the numpy matrix and norms cache from the embeddings list."""
        if not self._embeddings_list:
            self._embeddings_matrix = None
            self._norms = None
        else:
            self._embeddings_matrix = np.array(self._embeddings_list, dtype=np.float32)
            self._norms = np.linalg.norm(self._embeddings_matrix, axis=1)
            # Avoid division by zero
            self._norms[self._norms == 0] = 1.0
        self._dirty = False

    def search(
        self, query_embedding: list[float], top_k: int = 5
    ) -> list[tuple[Document, float]]:
        if not self._documents:
            return []

        if self._dirty or self._embeddings_matrix is None:
            self._rebuild_matrix()

        assert self._embeddings_matrix is not None
        assert self._norms is not None

        query_vec = np.array(query_embedding, dtype=np.float32)
        query_norm = np.linalg.norm(query_vec)
        if query_norm == 0:
            return []

        # Vectorized cosine similarity: (M @ q) / (||M|| * ||q||)
        scores = self._embeddings_matrix @ query_vec / (self._norms * query_norm)

        # Use heapq.nlargest instead of full sort for O(n log k)
        if top_k >= len(self._documents):
            top_indices = np.argsort(scores)[::-1]
        else:
            top_indices = np.argpartition(scores, -top_k)[-top_k:]
            top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

        return [(self._documents[i], float(scores[i])) for i in top_indices]

    def clear(self) -> None:
        self._documents.clear()
        self._embeddings_list.clear()
        self._embeddings_matrix = None
        self._norms = None
        self._dirty = False
