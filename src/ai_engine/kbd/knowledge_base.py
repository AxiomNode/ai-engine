"""Knowledge base manager."""

from __future__ import annotations

from ai_engine.kbd.entry import KnowledgeEntry


class KnowledgeBase:
    """In-memory knowledge base for storing and querying :class:`KnowledgeEntry` objects.

    Entries are indexed by their ``entry_id`` for O(1) lookup.

    Example::

        kb = KnowledgeBase()
        kb.add(KnowledgeEntry("1", "Python basics", "Python is a language..."))
        entry = kb.get("1")
        results = kb.search_by_tag("python")
    """

    def __init__(self) -> None:
        self._entries: dict[str, KnowledgeEntry] = {}

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def add(self, entry: KnowledgeEntry) -> None:
        """Add or replace an entry in the knowledge base.

        Args:
            entry: The :class:`KnowledgeEntry` to store.
        """
        self._entries[entry.entry_id] = entry

    def get(self, entry_id: str) -> KnowledgeEntry | None:
        """Return the entry with *entry_id*, or ``None`` if not found.

        Args:
            entry_id: The identifier to look up.
        """
        return self._entries.get(entry_id)

    def update(self, entry: KnowledgeEntry) -> None:
        """Update an existing entry.

        Args:
            entry: The updated :class:`KnowledgeEntry`.

        Raises:
            KeyError: If no entry with the given ``entry_id`` exists.
        """
        if entry.entry_id not in self._entries:
            raise KeyError(f"Entry '{entry.entry_id}' not found")
        self._entries[entry.entry_id] = entry

    def delete(self, entry_id: str) -> None:
        """Remove an entry from the knowledge base.

        Args:
            entry_id: The identifier of the entry to remove.

        Raises:
            KeyError: If no entry with the given ``entry_id`` exists.
        """
        if entry_id not in self._entries:
            raise KeyError(f"Entry '{entry_id}' not found")
        del self._entries[entry_id]

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def list_all(self) -> list[KnowledgeEntry]:
        """Return all entries in the knowledge base."""
        return list(self._entries.values())

    def search_by_tag(self, tag: str) -> list[KnowledgeEntry]:
        """Return all entries that include *tag* in their tags list.

        The comparison is case-insensitive.

        Args:
            tag: The tag to filter by.
        """
        tag_lower = tag.lower()
        return [
            e
            for e in self._entries.values()
            if tag_lower in (t.lower() for t in e.tags)
        ]

    def search_by_keyword(self, keyword: str) -> list[KnowledgeEntry]:
        """Return entries whose title or content contains *keyword*.

        The search is case-insensitive.

        Args:
            keyword: The keyword to search for.
        """
        kw = keyword.lower()
        return [
            e
            for e in self._entries.values()
            if kw in e.title.lower() or kw in e.content.lower()
        ]

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, entry_id: str) -> bool:
        return entry_id in self._entries
