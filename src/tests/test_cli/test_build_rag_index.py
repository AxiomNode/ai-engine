from __future__ import annotations

import json

from ai_engine.cli import build_rag_index
from ai_engine.rag.document import Document
from ai_engine.rag.embedder import Embedder


class FakeEmbedder(Embedder):
    def embed_text(self, text: str) -> list[float]:
        lowered = text.lower()
        return [
            1.0 if "database" in lowered or "vector" in lowered else 0.0,
            1.0 if "history" in lowered or "war" in lowered else 0.0,
            1.0 if "science" in lowered or "photosynthesis" in lowered else 0.0,
        ]


class RecordingVectorStore:
    def __init__(self) -> None:
        self.cleared = False
        self.documents: list[Document] = []
        self.embeddings: list[list[float]] = []

    def add(self, documents: list[Document], embeddings: list[list[float]]) -> None:
        self.documents.extend(documents)
        self.embeddings.extend(embeddings)

    def search(self, query_embedding: list[float], top_k: int = 5):
        return [(document, 1.0) for document in self.documents[:top_k]]

    def clear(self) -> None:
        self.cleared = True


def test_build_rag_index_uses_curated_corpus(monkeypatch, capsys) -> None:
    store = RecordingVectorStore()
    monkeypatch.setattr(
        build_rag_index, "_build_embedder", lambda *args, **kwargs: FakeEmbedder()
    )
    monkeypatch.setattr(
        build_rag_index, "_build_vector_store", lambda *args, **kwargs: store
    )

    exit_code = build_rag_index.main(
        ["--backend", "memory", "--clear", "--query", "vector database"]
    )

    assert exit_code == 0
    assert store.cleared is True
    assert len(store.documents) >= 10
    assert any(
        document.doc_id == "seed-computers-databases-en#0"
        for document in store.documents
    )
    output = capsys.readouterr().out
    summary = json.loads(output.split("\n[1]", 1)[0])
    assert summary["backend"] == "memory"
    assert summary["seed_corpus"]["documents"] >= 10
