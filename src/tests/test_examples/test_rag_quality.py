from __future__ import annotations

from ai_engine.examples.rag_quality import (
    RetrievalQualityCase,
    evaluate_retrieval_quality,
)
from ai_engine.rag.document import Document


class FakePipeline:
    def __init__(self, documents: list[Document]) -> None:
        self._documents = documents

    def retrieve(self, query: str, top_k: int = 5) -> list[Document]:
        return self._documents[:top_k]


def test_evaluate_retrieval_quality_reports_hit_rate_and_mrr() -> None:
    pipeline = FakePipeline(
        [
            Document(
                content="Vector databases store embeddings for semantic retrieval.",
                doc_id="seed-computers-databases-en#0",
                metadata={
                    "category": "Science: Computers",
                    "topic": "databases and vector search",
                    "language": "en",
                },
            )
        ]
    )

    report = evaluate_retrieval_quality(
        pipeline,
        [
            RetrievalQualityCase(
                query="vector database retrieval",
                expected_category="Science: Computers",
                expected_topic="databases and vector search",
                expected_doc_id="seed-computers-databases-en#0",
            )
        ],
    )

    assert report["passed"] == 1
    assert report["failed"] == 0
    assert report["hit_rate"] == 1.0
    assert report["mrr"] == 1.0
    assert report["cases"][0]["retrieved"][0]["language"] == "en"


def test_evaluate_retrieval_quality_reports_failures() -> None:
    pipeline = FakePipeline(
        [
            Document(
                content="A history document",
                doc_id="history#0",
                metadata={"category": "History", "topic": "Cold War", "language": "en"},
            )
        ]
    )

    report = evaluate_retrieval_quality(
        pipeline,
        [
            RetrievalQualityCase(
                query="photosynthesis",
                expected_category="Science & Nature",
                expected_topic="photosynthesis",
            )
        ],
    )

    assert report["passed"] == 0
    assert report["failed"] == 1
    assert report["mrr"] == 0.0
