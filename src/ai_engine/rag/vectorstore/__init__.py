"""Persistence-backed vector store implementations.

Available stores:

- :class:`~ai_engine.rag.vectorstore.chroma.ChromaVectorStore` – ChromaDB-backed
  persistent store (requires the ``rag`` extra: ``pip install ai-engine[rag]``).

For a dependency-free in-memory store suitable for development and
testing, use :class:`ai_engine.rag.vector_store.InMemoryVectorStore`.
"""

try:
    from ai_engine.rag.vectorstore.chroma import ChromaVectorStore
except ImportError:
    ChromaVectorStore = None  # type: ignore[assignment,misc]

__all__ = [
    "ChromaVectorStore",
]
