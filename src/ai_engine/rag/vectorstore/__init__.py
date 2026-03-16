"""Backward-compatible import path for persistent vector stores.

Prefer importing from `ai_engine.rag.vectorstores`.
"""

try:
    from ai_engine.rag.vectorstores.chroma import ChromaVectorStore
except ImportError:
    ChromaVectorStore = None  # type: ignore[assignment,misc]

__all__ = [
    "ChromaVectorStore",
]
