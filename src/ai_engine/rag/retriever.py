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
        min_score: Minimum cosine similarity score to include a document.
            Documents below this threshold are discarded even if fewer
            than *top_k* results remain.  Set to ``0.0`` to disable.
    """

    def __init__(
        self,
        embedder: Embedder,
        vector_store: VectorStore,
        top_k: int = 5,
        min_score: float = 0.3,
    ) -> None:
        self.embedder = embedder
        self.vector_store = vector_store
        self.top_k = top_k
        self.min_score = min_score
        self.last_scores: list[float] = []

    def retrieve(self, query: str, top_k: int | None = None) -> list[Document]:
        """Return the top documents relevant to *query*.

        Args:
            query: The user question or search phrase.
            top_k: Override the instance-level ``top_k`` for this call.

        Returns:
            A list of :class:`Document` instances ordered by relevance.
            Documents with similarity below *min_score* are excluded.
        """
        k = top_k if top_k is not None else self.top_k
        query_embedding = self.embedder.embed_text(query)
        results = self.vector_store.search(query_embedding, top_k=k)
        filtered = [
            (doc, score) for doc, score in results if score >= self.min_score
        ]
        self.last_scores = [score for _, score in filtered]
        return [doc for doc, _score in filtered]
