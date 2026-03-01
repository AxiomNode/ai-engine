"""Tests for ai_engine.kbd.knowledge_base."""

import pytest
from ai_engine.kbd.entry import KnowledgeEntry
from ai_engine.kbd.knowledge_base import KnowledgeBase


def _entry(entry_id: str, title: str = "T", content: str = "C", tags: list | None = None):
    return KnowledgeEntry(entry_id=entry_id, title=title, content=content, tags=tags or [])


def test_add_and_get():
    kb = KnowledgeBase()
    e = _entry("1", title="Python Intro", content="Python is a language")
    kb.add(e)
    assert kb.get("1") is e


def test_get_missing_returns_none():
    kb = KnowledgeBase()
    assert kb.get("missing") is None


def test_update():
    kb = KnowledgeBase()
    kb.add(_entry("1", title="Old title"))
    updated = _entry("1", title="New title")
    kb.update(updated)
    assert kb.get("1").title == "New title"


def test_update_missing_raises():
    kb = KnowledgeBase()
    with pytest.raises(KeyError):
        kb.update(_entry("nonexistent"))


def test_delete():
    kb = KnowledgeBase()
    kb.add(_entry("1"))
    kb.delete("1")
    assert kb.get("1") is None
    assert len(kb) == 0


def test_delete_missing_raises():
    kb = KnowledgeBase()
    with pytest.raises(KeyError):
        kb.delete("nonexistent")


def test_list_all():
    kb = KnowledgeBase()
    kb.add(_entry("1"))
    kb.add(_entry("2"))
    assert len(kb.list_all()) == 2


def test_search_by_tag():
    kb = KnowledgeBase()
    kb.add(_entry("1", tags=["Python", "AI"]))
    kb.add(_entry("2", tags=["java"]))
    results = kb.search_by_tag("python")  # case-insensitive
    assert len(results) == 1
    assert results[0].entry_id == "1"


def test_search_by_keyword():
    kb = KnowledgeBase()
    kb.add(_entry("1", title="Python basics", content="Python is great"))
    kb.add(_entry("2", title="Java overview", content="Java is compiled"))
    results = kb.search_by_keyword("python")
    assert len(results) == 1


def test_contains_operator():
    kb = KnowledgeBase()
    kb.add(_entry("1"))
    assert "1" in kb
    assert "99" not in kb


def test_len():
    kb = KnowledgeBase()
    assert len(kb) == 0
    kb.add(_entry("1"))
    assert len(kb) == 1
