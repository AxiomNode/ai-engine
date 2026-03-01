"""Text chunking utilities for the RAG pipeline."""

from __future__ import annotations

from ai_engine.rag.document import Document


class Chunker:
    """Splits a :class:`Document` into smaller, overlapping chunks.

    Args:
        chunk_size: Maximum number of characters per chunk.
        chunk_overlap: Number of characters shared between consecutive chunks.
    """

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap must be non-negative")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, document: Document) -> list[Document]:
        """Split *document* into a list of chunk documents.

        Each chunk inherits the parent document's metadata and adds
        ``chunk_index`` and ``chunk_total`` keys.

        Args:
            document: The document to split.

        Returns:
            A list of :class:`Document` instances representing the chunks.
        """
        text = document.content
        if not text:
            return []

        step = self.chunk_size - self.chunk_overlap
        starts = range(0, len(text), step)
        raw_chunks = [text[s : s + self.chunk_size] for s in starts]
        raw_chunks = [c for c in raw_chunks if c.strip()]

        chunks: list[Document] = []
        for i, chunk_text in enumerate(raw_chunks):
            meta = {**document.metadata, "chunk_index": i, "chunk_total": len(raw_chunks)}
            chunk_id = f"{document.doc_id}#{i}" if document.doc_id else None
            chunks.append(Document(content=chunk_text, metadata=meta, doc_id=chunk_id))

        return chunks
