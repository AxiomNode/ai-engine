"""Retriever for the RAG pipeline."""

from __future__ import annotations

from ai_engine.rag.document import Document
from ai_engine.rag.embedder import Embedder
from ai_engine.rag.vector_store import VectorStore


class Retriever:
    """Retrieves the most relevant documents for a given query.

    Args:
        embedder: The embedder used to encode the query.
        vector_store: The vector store to search.
        top_k: Default number of documents to retrieve.
    """

    def __init__(
        self,
        embedder: Embedder,
        vector_store: VectorStore,
        top_k: int = 5,
    ) -> None:
        self.embedder = embedder
        self.vector_store = vector_store
        self.top_k = top_k

    def retrieve(self, query: str, top_k: int | None = None) -> list[Document]:
        """Return the top documents relevant to *query*.

        Args:
            query: The user question or search phrase.
            top_k: Override the instance-level ``top_k`` for this call.

        Returns:
            A list of :class:`Document` instances ordered by relevance.
        """
        k = top_k if top_k is not None else self.top_k
        query_embedding = self.embedder.embed_text(query)
        results = self.vector_store.search(query_embedding, top_k=k)
        return [doc for doc, _score in results]
