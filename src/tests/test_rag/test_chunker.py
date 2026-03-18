"""Tests for ai_engine.rag.chunker."""

import pytest

from ai_engine.rag.chunker import Chunker
from ai_engine.rag.document import Document


def _make_doc(text: str, doc_id: str = "d1") -> Document:
    return Document(content=text, doc_id=doc_id)


def test_chunker_splits_long_text():
    chunker = Chunker(chunk_size=10, chunk_overlap=0)
    doc = _make_doc("0123456789abcdefghij")  # 20 chars → 2 chunks of 10
    chunks = chunker.split(doc)
    assert len(chunks) == 2
    assert chunks[0].content == "0123456789"
    assert chunks[1].content == "abcdefghij"


def test_chunker_overlap():
    chunker = Chunker(chunk_size=10, chunk_overlap=5)
    doc = _make_doc("0123456789abcde")  # step = 5
    chunks = chunker.split(doc)
    # starts: 0, 5, 10 → chunks "0123456789", "56789abcde", "abcde"
    assert chunks[0].content[:5] == "01234"
    assert chunks[1].content[:5] == "56789"


def test_chunker_empty_document():
    chunker = Chunker()
    chunks = chunker.split(Document(content=""))
    assert chunks == []


def test_chunker_chunk_ids():
    chunker = Chunker(chunk_size=5, chunk_overlap=0)
    doc = _make_doc("0123456789", doc_id="parent")
    chunks = chunker.split(doc)
    assert chunks[0].doc_id == "parent#0"
    assert chunks[1].doc_id == "parent#1"


def test_chunker_metadata_propagation():
    chunker = Chunker(chunk_size=5, chunk_overlap=0)
    doc = Document(content="0123456789", metadata={"source": "test"}, doc_id="d")
    chunks = chunker.split(doc)
    for chunk in chunks:
        assert chunk.metadata["source"] == "test"
        assert "chunk_index" in chunk.metadata
        assert "chunk_total" in chunk.metadata


def test_chunker_invalid_params():
    with pytest.raises(ValueError):
        Chunker(chunk_size=0)
    with pytest.raises(ValueError):
        Chunker(chunk_overlap=-1)
    with pytest.raises(ValueError):
        Chunker(chunk_size=5, chunk_overlap=5)
