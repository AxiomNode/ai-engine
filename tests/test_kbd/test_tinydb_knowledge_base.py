"""Tests for ai_engine.kbd.tinydb_knowledge_base – TinyDBKnowledgeBase.

Uses a TinyDB database backed by an in-memory storage so tests run
without touching the filesystem.
"""

from __future__ import annotations

import pytest

try:
    import tinydb  # noqa: F401

    _HAS_TINYDB = True
except ImportError:
    _HAS_TINYDB = False

pytestmark = pytest.mark.skipif(not _HAS_TINYDB, reason="tinydb not installed")

if _HAS_TINYDB:
    from ai_engine.kbd.entry import KnowledgeEntry
    from ai_engine.kbd.tinydb_knowledge_base import TinyDBKnowledgeBase


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _kb() -> TinyDBKnowledgeBase:
    """Return a fresh in-memory TinyDBKnowledgeBase."""
    return TinyDBKnowledgeBase(path=None)


def _entry(
    entry_id: str,
    title: str = "Title",
    content: str = "Content",
    tags: list[str] | None = None,
) -> KnowledgeEntry:
    return KnowledgeEntry(
        entry_id=entry_id,
        title=title,
        content=content,
        tags=tags or [],
    )


# ------------------------------------------------------------------
# Initialisation
# ------------------------------------------------------------------


class TestTinyDBKnowledgeBaseInit:

    def test_empty_on_creation(self) -> None:
        """A fresh knowledge base contains no entries."""
        kb = _kb()
        assert len(kb) == 0

    def test_persistent_path_accepted(self, tmp_path) -> None:
        """A filesystem path is accepted and creates a DB file."""
        db_path = str(tmp_path / "kb.json")
        kb = TinyDBKnowledgeBase(path=db_path)
        kb.add(_entry("1"))
        assert len(kb) == 1


# ------------------------------------------------------------------
# CRUD
# ------------------------------------------------------------------


class TestTinyDBKnowledgeBaseCRUD:

    def test_add_and_get(self) -> None:
        """An added entry is retrievable by its ID."""
        kb = _kb()
        e = _entry("1", title="Python Intro", content="Python is great")
        kb.add(e)
        result = kb.get("1")
        assert result is not None
        assert result.entry_id == "1"
        assert result.title == "Python Intro"

    def test_get_missing_returns_none(self) -> None:
        """Getting a non-existent entry returns None."""
        kb = _kb()
        assert kb.get("missing") is None

    def test_add_replaces_existing(self) -> None:
        """Adding an entry with an existing ID overwrites it."""
        kb = _kb()
        kb.add(_entry("1", title="Old"))
        kb.add(_entry("1", title="New"))
        assert kb.get("1").title == "New"  # type: ignore[union-attr]
        assert len(kb) == 1

    def test_update(self) -> None:
        """update replaces the entry content."""
        kb = _kb()
        kb.add(_entry("1", title="Original"))
        kb.update(_entry("1", title="Updated"))
        assert kb.get("1").title == "Updated"  # type: ignore[union-attr]

    def test_update_missing_raises(self) -> None:
        """Updating a non-existent entry raises KeyError."""
        kb = _kb()
        with pytest.raises(KeyError):
            kb.update(_entry("nonexistent"))

    def test_delete(self) -> None:
        """A deleted entry is no longer retrievable."""
        kb = _kb()
        kb.add(_entry("1"))
        kb.delete("1")
        assert kb.get("1") is None
        assert len(kb) == 0

    def test_delete_missing_raises(self) -> None:
        """Deleting a non-existent entry raises KeyError."""
        kb = _kb()
        with pytest.raises(KeyError):
            kb.delete("nonexistent")

    def test_list_all(self) -> None:
        """list_all returns every stored entry."""
        kb = _kb()
        kb.add(_entry("1"))
        kb.add(_entry("2"))
        kb.add(_entry("3"))
        assert len(kb.list_all()) == 3

    def test_contains_operator(self) -> None:
        """The ``in`` operator checks membership by entry_id."""
        kb = _kb()
        kb.add(_entry("42"))
        assert "42" in kb
        assert "99" not in kb

    def test_len(self) -> None:
        """len() reflects the number of stored entries."""
        kb = _kb()
        assert len(kb) == 0
        kb.add(_entry("a"))
        kb.add(_entry("b"))
        assert len(kb) == 2


# ------------------------------------------------------------------
# Querying
# ------------------------------------------------------------------


class TestTinyDBKnowledgeBaseSearch:

    def test_search_by_tag_case_insensitive(self) -> None:
        """search_by_tag is case-insensitive."""
        kb = _kb()
        kb.add(_entry("1", tags=["Python", "AI"]))
        kb.add(_entry("2", tags=["java"]))
        results = kb.search_by_tag("python")
        assert len(results) == 1
        assert results[0].entry_id == "1"

    def test_search_by_tag_no_match(self) -> None:
        """search_by_tag returns empty list when no entry matches."""
        kb = _kb()
        kb.add(_entry("1", tags=["Python"]))
        assert kb.search_by_tag("ruby") == []

    def test_search_by_keyword_in_title(self) -> None:
        """search_by_keyword matches the entry title."""
        kb = _kb()
        kb.add(_entry("1", title="Python basics", content="Learn Python"))
        kb.add(_entry("2", title="Java intro", content="Learn Java"))
        results = kb.search_by_keyword("python")
        assert len(results) == 1
        assert results[0].entry_id == "1"

    def test_search_by_keyword_in_content(self) -> None:
        """search_by_keyword also matches the entry content."""
        kb = _kb()
        kb.add(_entry("1", title="Intro", content="Python is versatile"))
        kb.add(_entry("2", title="Intro", content="Java is verbose"))
        results = kb.search_by_keyword("versatile")
        assert len(results) == 1

    def test_search_by_keyword_case_insensitive(self) -> None:
        """search_by_keyword is case-insensitive."""
        kb = _kb()
        kb.add(_entry("1", title="PYTHON TUTORIAL", content="stuff"))
        results = kb.search_by_keyword("python")
        assert len(results) == 1


# ------------------------------------------------------------------
# Persistence
# ------------------------------------------------------------------


class TestTinyDBKnowledgeBasePersistence:

    def test_data_persists_across_instances(self, tmp_path) -> None:
        """Entries written to a path-backed store survive re-opening."""
        db_path = str(tmp_path / "kb.json")

        kb1 = TinyDBKnowledgeBase(path=db_path)
        kb1.add(_entry("p1", title="Persisted Entry"))

        kb2 = TinyDBKnowledgeBase(path=db_path)
        result = kb2.get("p1")
        assert result is not None
        assert result.title == "Persisted Entry"

    def test_metadata_round_trip(self) -> None:
        """Entry metadata is preserved through add/get."""
        kb = _kb()
        e = KnowledgeEntry(
            entry_id="meta1",
            title="Meta test",
            content="content",
            metadata={"source": "book.pdf", "page": 5},
        )
        kb.add(e)
        result = kb.get("meta1")
        assert result is not None
        assert result.metadata["source"] == "book.pdf"
        assert result.metadata["page"] == 5
