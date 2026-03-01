"""Persistent KnowledgeBase backed by TinyDB.

This adapter provides the same public API as :class:`KnowledgeBase` but
persists entries to disk. It can optionally attach to a RAG pipeline so that
new or updated entries are indexed automatically.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

from tinydb import TinyDB, Query

from ai_engine.kbd.entry import KnowledgeEntry
from ai_engine.kbd.knowledge_base import KnowledgeBase
from ai_engine.rag.document import Document


class PersistentKnowledgeBase(KnowledgeBase):
    """Persistent knowledge base using TinyDB.

    Args:
        db_path: Path to the TinyDB JSON file.
        table_name: TinyDB table name to use.
        rag_pipeline: Optional RAGPipeline instance to auto-ingest entries.
    """

    def __init__(self, db_path: str | Path = "data/kb.json", table_name: str = "entries", rag_pipeline=None) -> None:
        super().__init__()
        self.db_path = Path(db_path)
        self.table_name = table_name
        self._db = TinyDB(str(self.db_path))
        self._table = self._db.table(self.table_name)
        self._rag_pipeline = rag_pipeline

        # Load existing entries into memory index
        for rec in self._table.all():
            entry = KnowledgeEntry.from_dict(rec)
            self._entries[entry.entry_id] = entry

    # ------------------------------------------------------------------
    # CRUD overrides to persist changes
    # ------------------------------------------------------------------

    def add(self, entry: KnowledgeEntry) -> None:
        super().add(entry)
        self._table.upsert(entry.to_dict(), Query().entry_id == entry.entry_id)
        # optionally ingest to RAG
        if self._rag_pipeline is not None:
            try:
                doc = Document(content=entry.content, metadata={"entry_id": entry.entry_id, "title": entry.title}, doc_id=entry.entry_id)
                self._rag_pipeline.ingest([doc])
            except Exception:
                # do not fail persistence if ingestion fails; log could be added later
                pass

    def update(self, entry: KnowledgeEntry) -> None:
        super().update(entry)
        self._table.update(entry.to_dict(), Query().entry_id == entry.entry_id)
        if self._rag_pipeline is not None:
            try:
                doc = Document(content=entry.content, metadata={"entry_id": entry.entry_id, "title": entry.title}, doc_id=entry.entry_id)
                # re-indexing by ingesting again
                self._rag_pipeline.ingest([doc])
            except Exception:
                pass

    def delete(self, entry_id: str) -> None:
        super().delete(entry_id)
        self._table.remove(Query().entry_id == entry_id)

    # ------------------------------------------------------------------
    # Additional helpers
    # ------------------------------------------------------------------

    def attach_rag_pipeline(self, rag_pipeline) -> None:
        """Attach a RAG pipeline instance to auto-ingest entries.

        When attached, subsequent `add`/`update` operations will attempt to
        ingest the entry content into the pipeline's vector store.
        """
        self._rag_pipeline = rag_pipeline

    def close(self) -> None:
        """Close the underlying TinyDB file handle."""
        try:
            self._db.close()
        except Exception:
            pass

    def bulk_import(self, entries: Iterable[KnowledgeEntry]) -> None:
        """Insert multiple entries efficiently."""
        docs = [e.to_dict() for e in entries]
        self._table.insert_multiple(docs)
        for e in entries:
            self._entries[e.entry_id] = e
