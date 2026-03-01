"""Tests for ai_engine.kbd.entry."""

import pytest
from ai_engine.kbd.entry import KnowledgeEntry


def _make_entry(**kwargs) -> KnowledgeEntry:
    defaults = {
        "entry_id": "e1",
        "title": "Sample entry",
        "content": "Some content",
        "tags": ["tag1"],
        "metadata": {"author": "test"},
    }
    defaults.update(kwargs)
    return KnowledgeEntry(**defaults)


def test_entry_creation():
    entry = _make_entry()
    assert entry.entry_id == "e1"
    assert entry.title == "Sample entry"
    assert entry.tags == ["tag1"]


def test_entry_empty_id_raises():
    with pytest.raises(ValueError):
        _make_entry(entry_id="")


def test_entry_empty_title_raises():
    with pytest.raises(ValueError):
        _make_entry(title="")


def test_entry_invalid_content_raises():
    with pytest.raises(TypeError):
        _make_entry(content=123)  # type: ignore[arg-type]


def test_to_dict_round_trip():
    entry = _make_entry()
    data = entry.to_dict()
    restored = KnowledgeEntry.from_dict(data)
    assert restored.entry_id == entry.entry_id
    assert restored.title == entry.title
    assert restored.content == entry.content
    assert restored.tags == entry.tags
    assert restored.metadata == entry.metadata


def test_from_dict_missing_optional_fields():
    data = {"entry_id": "x", "title": "T", "content": "C"}
    entry = KnowledgeEntry.from_dict(data)
    assert entry.tags == []
    assert entry.metadata == {}
