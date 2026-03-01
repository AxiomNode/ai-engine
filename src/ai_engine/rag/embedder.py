"""Embedder interface for the RAG pipeline."""

from __future__ import annotations

from abc import ABC, abstractmethod

from ai_engine.rag.document import Document


class Embedder(ABC):
    """Abstract base class for text embedding models.

    Subclasses must implement :meth:`embed_text` and :meth:`embed_documents`.
    """

    @abstractmethod
    def embed_text(self, text: str) -> list[float]:
        """Return a vector embedding for a single text string.

        Args:
            text: The text to embed.

        Returns:
            A list of floats representing the embedding vector.
        """

    def embed_documents(self, documents: list[Document]) -> list[list[float]]:
        """Return embeddings for a list of documents.

        The default implementation calls :meth:`embed_text` for each document.
        Subclasses may override this for batched efficiency.

        Args:
            documents: The documents to embed.

        Returns:
            A list of embedding vectors, one per document.
        """
        return [self.embed_text(doc.content) for doc in documents]
