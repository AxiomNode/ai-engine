"""Persistent vector store implementations for RAG.

Preferred imports:

    from ai_engine.rag.vectorstores import ChromaVectorStore

Backward-compatible imports from `ai_engine.rag.vectorstore` remain available.
"""

try:
    from ai_engine.rag.vectorstores.chroma import ChromaVectorStore
except ImportError:
    ChromaVectorStore = None  # type: ignore[assignment,misc]

__all__ = [
    "ChromaVectorStore",
]
