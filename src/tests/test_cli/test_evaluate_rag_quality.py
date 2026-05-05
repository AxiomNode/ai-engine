from __future__ import annotations

import json

from ai_engine.cli import evaluate_rag_quality
from ai_engine.rag.document import Document
from ai_engine.rag.embedder import Embedder


class FakeEmbedder(Embedder):
    def embed_text(self, text: str) -> list[float]:
        lowered = text.lower()
        tokens = [
            ("photosynthesis", "chloroplast"),
            ("database", "vector"),
            ("literary", "fiction"),
            ("gameplay", "mechanics"),
            ("industrial", "factory"),
            ("vertebrate", "backbone"),
            ("tempo", "music"),
            ("democracy", "election"),
            ("median", "statistics"),
        ]
        return [
            1.0 if any(token in lowered for token in group) else 0.0 for group in tokens
        ]


class FakeVectorStore:
    def __init__(self) -> None:
        self.documents: list[Document] = []
        self.embeddings: list[list[float]] = []

    def add(self, documents: list[Document], embeddings: list[list[float]]) -> None:
        self.documents.extend(documents)
        self.embeddings.extend(embeddings)

    def search(self, query_embedding: list[float], top_k: int = 5):
        scored = [
            (
                document,
                sum(left * right for left, right in zip(query_embedding, embedding)),
            )
            for document, embedding in zip(self.documents, self.embeddings)
        ]
        scored.sort(key=lambda item: item[1], reverse=True)
        return [(document, score) for document, score in scored[:top_k] if score > 0]

    def clear(self) -> None:
        self.documents.clear()
        self.embeddings.clear()


def test_evaluate_rag_quality_cli_outputs_json(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        evaluate_rag_quality, "_build_embedder", lambda *args, **kwargs: FakeEmbedder()
    )
    monkeypatch.setattr(
        evaluate_rag_quality,
        "_build_vector_store",
        lambda *args, **kwargs: FakeVectorStore(),
    )

    exit_code = evaluate_rag_quality.main(["--format", "json"])

    assert exit_code == 0
    output = json.loads(capsys.readouterr().out)
    assert output["failed"] == 0
    assert output["hit_rate"] == 1.0


def test_evaluate_rag_quality_cli_outputs_text(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        evaluate_rag_quality, "_build_embedder", lambda *args, **kwargs: FakeEmbedder()
    )
    monkeypatch.setattr(
        evaluate_rag_quality,
        "_build_vector_store",
        lambda *args, **kwargs: FakeVectorStore(),
    )

    exit_code = evaluate_rag_quality.main(["--format", "text"])

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "RAG retrieval quality:" in output
    assert "Hit rate:" in output
    assert "MRR:" in output
