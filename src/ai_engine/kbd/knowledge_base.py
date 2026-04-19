"""Knowledge base manager."""

from __future__ import annotations

import re

from ai_engine.kbd.entry import KnowledgeEntry

_TOKEN_RE = re.compile(r"[a-z0-9]{3,}")


def _normalize_tokens(text: str) -> set[str]:
    return set(_TOKEN_RE.findall(str(text or "").lower()))


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
        self._tag_index: dict[str, set[str]] = {}
        self._keyword_index: dict[str, set[str]] = {}

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def add(self, entry: KnowledgeEntry) -> None:
        """Add or replace an entry in the knowledge base.

        Args:
            entry: The :class:`KnowledgeEntry` to store.
        """
        existing = self._entries.get(entry.entry_id)
        if existing is not None:
            self._deindex_entry(existing)
        self._entries[entry.entry_id] = entry
        self._index_entry(entry)

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
        self._deindex_entry(self._entries[entry.entry_id])
        self._entries[entry.entry_id] = entry
        self._index_entry(entry)

    def delete(self, entry_id: str) -> None:
        """Remove an entry from the knowledge base.

        Args:
            entry_id: The identifier of the entry to remove.

        Raises:
            KeyError: If no entry with the given ``entry_id`` exists.
        """
        if entry_id not in self._entries:
            raise KeyError(f"Entry '{entry_id}' not found")
        entry = self._entries.pop(entry_id)
        self._deindex_entry(entry)

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
        entry_ids = self._tag_index.get(tag_lower, set())
        return [self._entries[entry_id] for entry_id in entry_ids if entry_id in self._entries]

    def search_by_keyword(self, keyword: str) -> list[KnowledgeEntry]:
        """Return entries whose title or content contains *keyword*.

        The search is case-insensitive.

        Args:
            keyword: The keyword to search for.
        """
        tokens = _normalize_tokens(keyword)
        if not tokens:
            return []

        candidate_ids: set[str] = set()
        for token in tokens:
            candidate_ids.update(self._keyword_index.get(token, set()))

        ranked: list[tuple[int, KnowledgeEntry]] = []
        for entry_id in candidate_ids:
            entry = self._entries.get(entry_id)
            if entry is None:
                continue
            title_tokens = _normalize_tokens(entry.title)
            content_tokens = _normalize_tokens(entry.content)
            score = (2 * len(tokens & title_tokens)) + len(tokens & content_tokens)
            if score > 0:
                ranked.append((score, entry))

        ranked.sort(key=lambda item: (-item[0], item[1].title.lower(), item[1].entry_id))
        return [entry for _, entry in ranked]

    def _index_entry(self, entry: KnowledgeEntry) -> None:
        for tag in entry.tags:
            normalized_tag = tag.lower().strip()
            if normalized_tag:
                self._tag_index.setdefault(normalized_tag, set()).add(entry.entry_id)

        for token in _normalize_tokens(f"{entry.title} {entry.content}"):
            self._keyword_index.setdefault(token, set()).add(entry.entry_id)

    def _deindex_entry(self, entry: KnowledgeEntry) -> None:
        for tag in entry.tags:
            normalized_tag = tag.lower().strip()
            if normalized_tag in self._tag_index:
                self._tag_index[normalized_tag].discard(entry.entry_id)
                if not self._tag_index[normalized_tag]:
                    del self._tag_index[normalized_tag]

        for token in _normalize_tokens(f"{entry.title} {entry.content}"):
            if token in self._keyword_index:
                self._keyword_index[token].discard(entry.entry_id)
                if not self._keyword_index[token]:
                    del self._keyword_index[token]

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, entry_id: str) -> bool:
        return entry_id in self._entries
