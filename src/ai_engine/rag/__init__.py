"""RAG (Retrieval-Augmented Generation) module."""

from ai_engine.rag.chunker import Chunker
from ai_engine.rag.document import Document
from ai_engine.rag.embedder import Embedder
from ai_engine.rag.pipeline import RAGPipeline
from ai_engine.rag.retriever import Retriever
from ai_engine.rag.vector_store import VectorStore

try:
    from ai_engine.rag.vectorstores import ChromaVectorStore
except ImportError:
    ChromaVectorStore = None  # type: ignore[assignment,misc]

__all__ = [
    "Document",
    "Chunker",
    "Embedder",
    "VectorStore",
    "Retriever",
    "RAGPipeline",
    "ChromaVectorStore",
]
