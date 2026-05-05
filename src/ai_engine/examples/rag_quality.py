"""Retrieval-quality benchmarks for the curated RAG corpus."""

from __future__ import annotations

import time
from dataclasses import dataclass
from statistics import mean
from typing import Any, Iterable

from ai_engine.rag.document import Document


@dataclass(frozen=True)
class RetrievalQualityCase:
    """One expected retrieval behavior for the curated corpus."""

    query: str
    expected_category: str
    expected_topic: str | None = None
    expected_doc_id: str | None = None
    top_k: int = 5


DEFAULT_RETRIEVAL_QUALITY_CASES = [
    RetrievalQualityCase(
        query="how plants use chloroplasts to make glucose",
        expected_category="Science & Nature",
        expected_topic="photosynthesis",
        expected_doc_id="seed-science-photosynthesis-en#0",
    ),
    RetrievalQualityCase(
        query="vector databases retrieve similar embedded documents for RAG",
        expected_category="Science: Computers",
        expected_topic="databases and vector search",
        expected_doc_id="seed-computers-databases-en#0",
    ),
    RetrievalQualityCase(
        query="literary genres fiction poetry drama nonfiction",
        expected_category="Entertainment: Books",
        expected_topic="literary genres",
        expected_doc_id="seed-books-literary-genres-en#0",
    ),
    RetrievalQualityCase(
        query="game mechanics levels core gameplay loop",
        expected_category="Entertainment: Video Games",
        expected_topic="gameplay systems",
        expected_doc_id="seed-video-games-gameplay-systems-en#0",
    ),
    RetrievalQualityCase(
        query="industrial revolution factories steam power social consequences",
        expected_category="History",
        expected_topic="Industrial Revolution",
        expected_doc_id="seed-history-industrial-revolution-en#0",
    ),
    RetrievalQualityCase(
        query="vertebrates invertebrates animal backbone classification",
        expected_category="Animals",
        expected_topic="animal classification",
        expected_doc_id="seed-animals-classification-en#0",
    ),
    RetrievalQualityCase(
        query="tempo pitch rhythm music notation staff",
        expected_category="Entertainment: Music",
        expected_topic="music notation",
        expected_doc_id="seed-music-notation-en#0",
    ),
    RetrievalQualityCase(
        query="democracy rule of law free elections civil liberties",
        expected_category="Politics",
        expected_topic="democracy",
        expected_doc_id="seed-politics-democracy-en#0",
    ),
    RetrievalQualityCase(
        query="median mean mode biased sample statistics",
        expected_category="Science: Mathematics",
        expected_topic="basic statistics",
        expected_doc_id="seed-math-statistics-en#0",
    ),
]


def evaluate_retrieval_quality(
    pipeline: Any,
    cases: Iterable[RetrievalQualityCase] | None = None,
) -> dict[str, Any]:
    """Run retrieval quality cases against a RAG pipeline."""
    selected_cases = list(cases or DEFAULT_RETRIEVAL_QUALITY_CASES)
    results: list[dict[str, Any]] = []
    latencies_ms: list[float] = []

    for case in selected_cases:
        started = time.perf_counter()
        documents = pipeline.retrieve(case.query, top_k=case.top_k)
        latency_ms = (time.perf_counter() - started) * 1000.0
        latencies_ms.append(latency_ms)
        results.append(_score_case(case, documents, latency_ms=latency_ms))

    passed = sum(1 for result in results if result["passed"])
    total = len(results)
    reciprocal_ranks = [result["reciprocal_rank"] for result in results]
    return {
        "suite": "RAG Retrieval Quality",
        "total": total,
        "passed": passed,
        "failed": total - passed,
        "hit_rate": round(passed / total, 4) if total else 0.0,
        "mrr": round(mean(reciprocal_ranks), 4) if reciprocal_ranks else 0.0,
        "latency_ms": {
            "avg": round(mean(latencies_ms), 2) if latencies_ms else 0.0,
            "p95": round(_percentile(latencies_ms, 95), 2) if latencies_ms else 0.0,
            "max": round(max(latencies_ms), 2) if latencies_ms else 0.0,
        },
        "cases": results,
    }


def _score_case(
    case: RetrievalQualityCase,
    documents: list[Document],
    *,
    latency_ms: float,
) -> dict[str, Any]:
    ranks = []
    for index, document in enumerate(documents, start=1):
        if _document_matches(case, document):
            ranks.append(index)

    rank = min(ranks) if ranks else None
    return {
        "query": case.query,
        "expected_category": case.expected_category,
        "expected_topic": case.expected_topic,
        "expected_doc_id": case.expected_doc_id,
        "passed": rank is not None,
        "rank": rank,
        "reciprocal_rank": round(1.0 / rank, 4) if rank else 0.0,
        "latency_ms": round(latency_ms, 2),
        "retrieved": [
            {
                "doc_id": document.doc_id,
                "category": document.metadata.get("category"),
                "topic": document.metadata.get("topic")
                or document.metadata.get("sub_topic"),
                "language": document.metadata.get("language"),
            }
            for document in documents
        ],
    }


def _document_matches(case: RetrievalQualityCase, document: Document) -> bool:
    metadata = document.metadata or {}
    if case.expected_doc_id and document.doc_id == case.expected_doc_id:
        return True
    if metadata.get("category") != case.expected_category:
        return False
    if case.expected_topic and metadata.get("topic") != case.expected_topic:
        return False
    return True


def _percentile(values: list[float], percentile: int) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(
        len(ordered) - 1, max(0, round((percentile / 100) * (len(ordered) - 1)))
    )
    return ordered[index]
