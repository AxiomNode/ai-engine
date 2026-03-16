"""Tests for ai_engine.rag.document."""

import pytest

from ai_engine.rag.document import Document


def test_document_basic():
    doc = Document(content="Hello world")
    assert doc.content == "Hello world"
    assert doc.metadata == {}
    assert doc.doc_id is None


def test_document_with_metadata():
    doc = Document(content="text", metadata={"source": "wiki"}, doc_id="doc-1")
    assert doc.metadata["source"] == "wiki"
    assert doc.doc_id == "doc-1"


def test_document_len():
    doc = Document(content="abcde")
    assert len(doc) == 5


def test_document_requires_string_content():
    with pytest.raises(TypeError):
        Document(content=123)  # type: ignore[arg-type]
