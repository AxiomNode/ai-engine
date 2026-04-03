"""TinyDB-backed persistent knowledge base.

Uses `TinyDB <https://tinydb.readthedocs.io/>`_ as the storage engine.

Usage::

    # Persistent on-disk knowledge base
    from ai_engine.kbd.tinydb_knowledge_base import TinyDBKnowledgeBase
    kb = TinyDBKnowledgeBase(path="./knowledge_base.json")

    # Ephemeral in-memory knowledge base (useful for testing)
    kb = TinyDBKnowledgeBase(path=None)
"""

from __future__ import annotations

from typing import Any, cast

try:
    from tinydb import TinyDB, where
    from tinydb.storages import MemoryStorage
except ImportError as _err:  # pragma: no cover
    raise ImportError(
        "tinydb is required for TinyDBKnowledgeBase.  "
        "Install it with:  pip install ai-engine[kbd]"
    ) from _err

from ai_engine.kbd.entry import KnowledgeEntry


def _as_entry_dict(row: object) -> dict[str, Any] | None:
    """Return a TinyDB row as a plain dict when it has the expected shape."""
    if not isinstance(row, dict):
        return None
    data = cast(dict[str, Any], row)
    required = ("entry_id", "title", "content")
    if not all(key in data for key in required):
        return None
    return data


class TinyDBKnowledgeBase:
    """Persistent knowledge base backed by TinyDB.

    Mirrors the API of the in-memory :class:`~ai_engine.kbd.knowledge_base.KnowledgeBase`
    so the two are interchangeable.  When *path* is ``None`` an ephemeral
    in-memory database is used; when *path* is a file path the data is
    persisted to that JSON file.

    Args:
        path: Path to the TinyDB JSON file, or ``None`` for in-memory storage.

    Example::

        kb = TinyDBKnowledgeBase(path="./kb.json")
        kb.add(KnowledgeEntry("1", "Python Intro", "Python is a language"))
        entry = kb.get("1")
    """

    _TABLE = "entries"

    def __init__(self, path: str | None = None) -> None:
        if path is None:
            self._db = TinyDB(storage=MemoryStorage)
        else:
            self._db = TinyDB(path)
        self._table = self._db.table(self._TABLE)

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def add(self, entry: KnowledgeEntry) -> None:
        """Add or replace an entry in the knowledge base.

        If an entry with the same ``entry_id`` already exists it is
        overwritten (upsert semantics).

        Args:
            entry: The :class:`~ai_engine.kbd.entry.KnowledgeEntry` to store.
        """
        self._table.upsert(entry.to_dict(), where("entry_id") == entry.entry_id)

    def get(self, entry_id: str) -> KnowledgeEntry | None:
        """Return the entry with *entry_id*, or ``None`` if not found.

        Args:
            entry_id: The identifier to look up.
        """
        row = self._table.get(where("entry_id") == entry_id)
        data = _as_entry_dict(row)
        if data is None:
            return None
        return KnowledgeEntry.from_dict(data)

    def update(self, entry: KnowledgeEntry) -> None:
        """Update an existing entry.

        Args:
            entry: The updated :class:`~ai_engine.kbd.entry.KnowledgeEntry`.

        Raises:
            KeyError: If no entry with the given ``entry_id`` exists.
        """
        existing = self._table.get(where("entry_id") == entry.entry_id)
        if existing is None:
            raise KeyError(f"Entry '{entry.entry_id}' not found")
        self._table.update(entry.to_dict(), where("entry_id") == entry.entry_id)

    def delete(self, entry_id: str) -> None:
        """Remove an entry from the knowledge base.

        Args:
            entry_id: The identifier of the entry to remove.

        Raises:
            KeyError: If no entry with the given ``entry_id`` exists.
        """
        if not self._table.contains(where("entry_id") == entry_id):
            raise KeyError(f"Entry '{entry_id}' not found")
        self._table.remove(where("entry_id") == entry_id)

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def list_all(self) -> list[KnowledgeEntry]:
        """Return all entries in the knowledge base."""
        entries: list[KnowledgeEntry] = []
        for row in self._table.all():
            data = _as_entry_dict(row)
            if data is not None:
                entries.append(KnowledgeEntry.from_dict(data))
        return entries

    def search_by_tag(self, tag: str) -> list[KnowledgeEntry]:
        """Return all entries that include *tag* in their tags list.

        The comparison is case-insensitive.

        Args:
            tag: The tag to filter by.
        """
        tag_lower = tag.lower()
        rows = self._table.search(
            where("tags").test(lambda tags: tag_lower in (t.lower() for t in tags))
        )
        entries: list[KnowledgeEntry] = []
        for row in rows:
            data = _as_entry_dict(row)
            if data is not None:
                entries.append(KnowledgeEntry.from_dict(data))
        return entries

    def search_by_keyword(self, keyword: str) -> list[KnowledgeEntry]:
        """Return entries whose title or content contains *keyword*.

        The search is case-insensitive.

        Args:
            keyword: The keyword to search for.
        """
        kw = keyword.lower()
        rows = self._table.search(
            where("title").test(lambda t: isinstance(t, str) and kw in t.lower())
            | where("content").test(lambda c: isinstance(c, str) and kw in c.lower())
        )
        entries: list[KnowledgeEntry] = []
        for row in rows:
            data = _as_entry_dict(row)
            if data is not None:
                entries.append(KnowledgeEntry.from_dict(data))
        return entries

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._table)

    def __contains__(self, entry_id: str) -> bool:
        return self._table.contains(where("entry_id") == entry_id)
