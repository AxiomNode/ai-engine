"""Chroma vector store wrapper implementing :class:`VectorStore`.

This file provides a thin adapter around `chromadb` to satisfy the project's
`VectorStore` interface.
"""

from __future__ import annotations

from ai_engine.rag.document import Document
from ai_engine.rag.vector_store import VectorStore

try:
    import chromadb
except Exception:  # pragma: no cover - optional dependency
    chromadb = None


class ChromaVectorStore(VectorStore):
    def __init__(self, persist_dir: str | None = None, collection_name: str = "ai_engine") -> None:
        if chromadb is None:
            raise RuntimeError(
                "chromadb is not installed. Install extras 'rag' or 'pip install chromadb'."
            )
        self.client = chromadb.Client()
        # create or get collection
        try:
            self.collection = self.client.get_collection(collection_name)
        except Exception:
            self.collection = self.client.create_collection(name=collection_name)

    def add(self, documents: list[Document], embeddings: list[list[float]]) -> None:
        ids = [d.doc_id or str(i) for i, d in enumerate(documents)]
        metadatas = [d.metadata for d in documents]
        docs = [d.content for d in documents]
        self.collection.add(ids=ids, embeddings=embeddings, metadatas=metadatas, documents=docs)

    def search(self, query_embedding: list[float], top_k: int = 5) -> list[tuple[Document, float]]:
        res = self.collection.query(query_embeddings=[query_embedding], n_results=top_k)
        results = []
        # chromadb returns lists for 'documents','metadatas','ids','distances'
        docs = res.get("documents", [[]])[0]
        metadatas = res.get("metadatas", [[]])[0]
        ids = res.get("ids", [[]])[0]
        distances = res.get("distances", [[]])[0]
        for doc_text, meta, _id, dist in zip(docs, metadatas, ids, distances):
            # convert distance to a simple score (higher better)
            try:
                score = 1.0 / (1.0 + float(dist))
            except Exception:
                score = 0.0
            document = Document(content=doc_text, metadata=meta or {}, doc_id=_id)
            results.append((document, score))
        return results

    def clear(self) -> None:
        # chroma does not expose a simple clear; recreate collection
        name = self.collection.name
        self.client.delete_collection(name)
        self.collection = self.client.create_collection(name=name)
