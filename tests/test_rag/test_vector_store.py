"""Tests for ai_engine.rag.vector_store (InMemoryVectorStore)."""

import pytest

from ai_engine.rag.document import Document
from ai_engine.rag.vector_store import InMemoryVectorStore


def _store_with_docs() -> tuple[InMemoryVectorStore, list[Document]]:
    store = InMemoryVectorStore()
    docs = [
        Document(content="Python programming", doc_id="1"),
        Document(content="Machine learning basics", doc_id="2"),
        Document(content="Data structures and algorithms", doc_id="3"),
    ]
    embeddings = [
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
    store.add(docs, embeddings)
    return store, docs


def test_search_returns_top_k():
    store, docs = _store_with_docs()
    results = store.search([1.0, 0.0, 0.0], top_k=1)
    assert len(results) == 1
    assert results[0][0].doc_id == "1"


def test_search_ordering():
    store, _ = _store_with_docs()
    results = store.search([0.0, 1.0, 0.0], top_k=3)
    scores = [score for _, score in results]
    assert scores == sorted(scores, reverse=True)


def test_search_empty_store():
    store = InMemoryVectorStore()
    assert store.search([1.0, 0.0], top_k=3) == []


def test_clear():
    store, _ = _store_with_docs()
    store.clear()
    assert store.search([1.0, 0.0, 0.0], top_k=1) == []


def test_add_length_mismatch_raises():
    store = InMemoryVectorStore()
    docs = [Document(content="a")]
    with pytest.raises(ValueError):
        store.add(docs, [[1.0], [2.0]])
