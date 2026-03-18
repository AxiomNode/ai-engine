"""Backward-compatible shim for ChromaVectorStore import path.

Prefer:
    from ai_engine.rag.vectorstores.chroma import ChromaVectorStore
"""

from ai_engine.rag.vectorstores.chroma import ChromaVectorStore

__all__ = ["ChromaVectorStore"]
