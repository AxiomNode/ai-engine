import os
from pathlib import Path

import pytest

from ai_engine.kbd.entry import KnowledgeEntry


def test_persistent_kb_roundtrip(tmp_path):
    pytest.importorskip("tinydb")
    from ai_engine.kbd.persistent_kb import PersistentKnowledgeBase

    db_file = tmp_path / "kb.json"
    kb = PersistentKnowledgeBase(db_path=db_file)

    entry = KnowledgeEntry(entry_id="e1", title="T", content="Content about X", tags=["x"])
    kb.add(entry)
    assert "e1" in kb

    kb2 = PersistentKnowledgeBase(db_path=db_file)
    assert kb2.get("e1") is not None
    assert kb2.get("e1").title == "T"

    kb.close()
    kb2.close()


def test_attach_rag_pipeline_calls_ingest(tmp_path):
    pytest.importorskip("tinydb")
    from ai_engine.kbd.persistent_kb import PersistentKnowledgeBase

    class DummyRag:
        def __init__(self):
            self.calls = 0

        def ingest(self, docs):
            self.calls += len(docs)

    db_file = tmp_path / "kb2.json"
    rag = DummyRag()
    kb = PersistentKnowledgeBase(db_path=db_file, rag_pipeline=rag)

    entry = KnowledgeEntry(entry_id="e2", title="Title2", content="Some content")
    kb.add(entry)
    assert rag.calls == 1
    kb.close()
