"""ChromaDB-backed vector store for the RAG pipeline.

Uses `chromadb <https://docs.trychroma.com/>`_ as the persistence layer.

Usage::

    # Preferred import path
    from ai_engine.rag.vectorstores.chroma import ChromaVectorStore
    store = ChromaVectorStore(collection_name="lectures", path="./chroma_db")

    # Ephemeral in-memory store (useful for testing)
    store = ChromaVectorStore(collection_name="test", path=None)
"""

from __future__ import annotations

import uuid
from typing import Mapping, cast

try:
    import chromadb
except ImportError as _err:  # pragma: no cover
    raise ImportError(
        "chromadb is required for ChromaVectorStore.  "
        "Install it with:  pip install ai-engine[rag]"
    ) from _err

from ai_engine.rag.document import Document
from ai_engine.rag.vector_store import VectorStore


class ChromaVectorStore(VectorStore):
    """Persistent vector store backed by ChromaDB.

    Documents are stored in a named ChromaDB collection.  When *path* is
    ``None`` an ephemeral in-memory client is used (ideal for tests); when
    *path* is a directory the data is persisted there.

    Args:
        collection_name: Name of the ChromaDB collection to use.
            Defaults to ``"ai_engine_default"``.
        path: Filesystem path for a persistent Chroma database, or ``None``
            for an ephemeral in-memory client.
    """

    DEFAULT_COLLECTION = "ai_engine_default"

    def __init__(
        self,
        collection_name: str = DEFAULT_COLLECTION,
        path: str | None = None,
    ) -> None:
        self.collection_name = collection_name
        self._path = path

        if path is None:
            self._client: chromadb.ClientAPI = chromadb.EphemeralClient()
        else:
            self._client = chromadb.PersistentClient(path=path)

        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    _MetadataValue = str | int | float | bool | list[str | int | float | bool] | None

    def add(self, documents: list[Document], embeddings: list[list[float]]) -> None:
        """Store documents together with their embeddings."""
        if len(documents) != len(embeddings):
            raise ValueError(
                f"documents and embeddings must have the same length, "
                f"got {len(documents)} documents and {len(embeddings)} embeddings"
            )

        if not documents:
            return

        ids = [doc.doc_id or str(uuid.uuid4()) for doc in documents]
        metadatas: list[Mapping[str, ChromaVectorStore._MetadataValue]] = []
        for doc, doc_id in zip(documents, ids):
            meta: dict[str, ChromaVectorStore._MetadataValue] = {
                "_content": doc.content,
                "_doc_id": doc_id,
            }
            for k, v in (doc.metadata or {}).items():
                if isinstance(v, (str, int, float, bool)):
                    meta[k] = v
                else:
                    meta[k] = str(v)
            metadatas.append(meta)

        self._collection.add(
            ids=ids,
            embeddings=embeddings,  # type: ignore[arg-type]
            metadatas=cast(list[dict[str, object]], metadatas),  # type: ignore[arg-type]
            documents=[doc.content for doc in documents],
        )

    def search(
        self, query_embedding: list[float], top_k: int = 5
    ) -> list[tuple[Document, float]]:
        """Retrieve the *top_k* most similar documents."""
        count = self._collection.count()
        if count == 0:
            return []

        effective_k = min(top_k, count)
        result = self._collection.query(
            query_embeddings=[query_embedding],  # type: ignore[arg-type]
            n_results=effective_k,
            include=["documents", "metadatas", "distances"],
        )

        docs_out: list[tuple[Document, float]] = []
        for content, meta, distance in zip(
            result["documents"][0],  # type: ignore[index]
            result["metadatas"][0],  # type: ignore[index]
            result["distances"][0],  # type: ignore[index]
        ):
            score = 1.0 - distance / 2.0
            safe_meta = meta if isinstance(meta, dict) else {}
            doc_content = str(safe_meta.get("_content", content))
            doc_id_value = safe_meta.get("_doc_id")
            recovered_doc_id = str(doc_id_value) if doc_id_value is not None else None
            doc_metadata = {
                k: v for k, v in safe_meta.items() if k not in ("_content", "_doc_id")
            }
            docs_out.append(
                (
                    Document(
                        content=doc_content,
                        metadata=doc_metadata,
                        doc_id=recovered_doc_id,
                    ),
                    score,
                )
            )

        return docs_out

    def clear(self) -> None:
        """Remove all documents from the collection."""
        all_ids = self._collection.get(include=[])["ids"]
        if all_ids:
            self._collection.delete(ids=all_ids)
