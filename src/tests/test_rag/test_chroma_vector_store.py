"""Tests for ChromaVectorStore import paths and behavior.

ChromaDB is run entirely in-memory (no external server) by passing
``path=None`` (ephemeral client).  All tests are therefore fast and
deterministic; no cleanup of on-disk files is required.
"""

from __future__ import annotations

import uuid

import pytest

try:
    import chromadb  # noqa: F401

    _HAS_CHROMA = True
except ImportError:
    _HAS_CHROMA = False

pytestmark = pytest.mark.skipif(not _HAS_CHROMA, reason="chromadb not installed")

if _HAS_CHROMA:
    from ai_engine.rag.document import Document
    from ai_engine.rag.vectorstore.chroma import ChromaVectorStore as LegacyChroma
    from ai_engine.rag.vectorstores.chroma import ChromaVectorStore


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _store() -> ChromaVectorStore:
    """Return a fresh ephemeral ChromaVectorStore with a unique collection name."""
    return ChromaVectorStore(collection_name=f"test-{uuid.uuid4().hex}", path=None)


def _docs_and_embeddings() -> tuple[list[Document], list[list[float]]]:
    docs = [
        Document(content="Python programming language", doc_id="d1"),
        Document(content="Machine learning basics", doc_id="d2"),
        Document(content="Data structures and algorithms", doc_id="d3"),
    ]
    embeddings: list[list[float]] = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
    return docs, embeddings


# ------------------------------------------------------------------
# Initialisation
# ------------------------------------------------------------------


class TestChromaVectorStoreInit:

    def test_legacy_import_path_remains_compatible(self) -> None:
        """Legacy vectorstore import path should resolve to the same class."""
        assert LegacyChroma is ChromaVectorStore

    def test_default_collection_name(self) -> None:
        """Default collection name is used when none is supplied."""
        store = ChromaVectorStore(path=None)
        assert store.collection_name == "ai_engine_default"

    def test_custom_collection_name(self) -> None:
        """Custom collection name is preserved."""
        store = ChromaVectorStore(collection_name="my-collection", path=None)
        assert store.collection_name == "my-collection"


# ------------------------------------------------------------------
# add / search
# ------------------------------------------------------------------


class TestChromaVectorStoreAddSearch:

    def test_search_after_add_returns_results(self) -> None:
        """Documents added to the store are retrievable via search."""
        store = _store()
        docs, embeddings = _docs_and_embeddings()
        store.add(docs, embeddings)
        results = store.search([1.0, 0.0, 0.0], top_k=1)
        assert len(results) == 1
        assert results[0][0].doc_id == "d1"

    def test_search_returns_top_k(self) -> None:
        """search respects the top_k limit."""
        store = _store()
        docs, embeddings = _docs_and_embeddings()
        store.add(docs, embeddings)
        results = store.search([1.0, 0.0, 0.0], top_k=2)
        assert len(results) == 2

    def test_search_scores_are_descending(self) -> None:
        """Results are ordered from most to least similar."""
        store = _store()
        docs, embeddings = _docs_and_embeddings()
        store.add(docs, embeddings)
        results = store.search([0.0, 1.0, 0.0], top_k=3)
        scores = [score for _, score in results]
        assert scores == sorted(scores, reverse=True)

    def test_search_empty_store_returns_empty_list(self) -> None:
        """Searching an empty store returns an empty list."""
        store = _store()
        assert store.search([1.0, 0.0, 0.0], top_k=3) == []

    def test_add_length_mismatch_raises(self) -> None:
        """add raises ValueError when documents and embeddings lengths differ."""
        store = _store()
        docs = [Document(content="a", doc_id="x")]
        with pytest.raises(ValueError, match="same length"):
            store.add(docs, [[1.0], [2.0]])

    def test_document_content_preserved(self) -> None:
        """Original document content is recovered from search results."""
        store = _store()
        docs, embeddings = _docs_and_embeddings()
        store.add(docs, embeddings)
        results = store.search([1.0, 0.0, 0.0], top_k=1)
        assert results[0][0].content == "Python programming language"

    def test_document_metadata_preserved(self) -> None:
        """Metadata attached to documents survives a round-trip through Chroma."""
        store = _store()
        doc = Document(
            content="Test doc", doc_id="m1", metadata={"source": "book.pdf", "page": 3}
        )
        store.add([doc], [[0.5, 0.5, 0.0]])
        results = store.search([0.5, 0.5, 0.0], top_k=1)
        assert results[0][0].metadata.get("source") == "book.pdf"

    def test_add_document_without_doc_id_uses_generated_id(self) -> None:
        """Documents without a doc_id are assigned a generated identifier."""
        store = _store()
        doc = Document(content="No ID doc")
        store.add([doc], [[1.0, 0.0, 0.0]])
        results = store.search([1.0, 0.0, 0.0], top_k=1)
        assert len(results) == 1
        assert results[0][0].content == "No ID doc"

    def test_add_is_cumulative(self) -> None:
        """Multiple calls to add accumulate documents."""
        store = _store()
        store.add([Document(content="first", doc_id="a")], [[1.0, 0.0]])
        store.add([Document(content="second", doc_id="b")], [[0.0, 1.0]])
        results = store.search([1.0, 0.0], top_k=2)
        assert len(results) == 2

    def test_add_upserts_existing_doc_id(self) -> None:
        """Repeated corpus priming should update existing Chroma ids safely."""
        store = _store()
        store.add([Document(content="old content", doc_id="same")], [[1.0, 0.0]])
        store.add([Document(content="new content", doc_id="same")], [[0.0, 1.0]])

        results = store.search([0.0, 1.0], top_k=3)

        assert len(results) == 1
        assert results[0][0].doc_id == "same"
        assert results[0][0].content == "new content"


# ------------------------------------------------------------------
# clear
# ------------------------------------------------------------------


class TestChromaVectorStoreClear:

    def test_clear_removes_all_documents(self) -> None:
        """After clear, search returns an empty list."""
        store = _store()
        docs, embeddings = _docs_and_embeddings()
        store.add(docs, embeddings)
        store.clear()
        assert store.search([1.0, 0.0, 0.0], top_k=5) == []

    def test_clear_then_add_works(self) -> None:
        """The store can be reused after clearing."""
        store = _store()
        docs, embeddings = _docs_and_embeddings()
        store.add(docs, embeddings)
        store.clear()

        new_doc = Document(content="Fresh start", doc_id="fresh")
        store.add([new_doc], [[1.0, 0.0, 0.0]])
        results = store.search([1.0, 0.0, 0.0], top_k=1)
        assert results[0][0].doc_id == "fresh"
