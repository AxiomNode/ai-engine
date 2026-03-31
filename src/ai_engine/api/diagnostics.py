"""Diagnostics endpoints for the ai-engine API.

Provides:
- ``GET  /diagnostics/rag/stats``    – RAG vector store health & coverage metrics.
- ``POST /diagnostics/tests/run``    – Execute hallucination & quality test suites.
- ``GET  /diagnostics/tests/status`` – Poll current test execution status/results.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import threading
import time
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# RAG statistics helper
# ---------------------------------------------------------------------------


def compute_rag_stats(rag_pipeline: Any) -> dict[str, Any]:
    """Compute RAG vector store health and coverage metrics.

    Returns a dictionary with document count, chunk count, embedding
    dimensions, per-source breakdown, and a coverage assessment.
    """
    store = rag_pipeline.vector_store
    retriever = rag_pipeline.retriever
    embedder = rag_pipeline.embedder

    documents = getattr(store, "_documents", [])
    embeddings_list = getattr(store, "_embeddings_list", [])

    total_chunks = len(documents)
    embedding_dim = len(embeddings_list[0]) if embeddings_list else 0

    # Per-source breakdown
    source_stats: dict[str, dict[str, Any]] = {}
    total_chars = 0
    for doc in documents:
        source = (doc.metadata or {}).get("source", "unknown")
        if source not in source_stats:
            source_stats[source] = {"chunks": 0, "total_chars": 0, "doc_ids": set()}
        source_stats[source]["chunks"] += 1
        chars = len(doc.content)
        source_stats[source]["total_chars"] += chars
        total_chars += chars
        if doc.doc_id:
            source_stats[source]["doc_ids"].add(doc.doc_id)

    # Serialize sets to counts
    sources_breakdown = []
    unique_doc_ids: set[str] = set()
    for source, stats in source_stats.items():
        unique_doc_ids.update(stats["doc_ids"])
        sources_breakdown.append({
            "source": source,
            "chunks": stats["chunks"],
            "total_chars": stats["total_chars"],
            "unique_documents": len(stats["doc_ids"]),
            "avg_chunk_chars": round(stats["total_chars"] / stats["chunks"], 1) if stats["chunks"] else 0,
        })

    sources_breakdown.sort(key=lambda x: x["chunks"], reverse=True)

    avg_chunk_chars = round(total_chars / total_chunks, 1) if total_chunks else 0

    # Coverage assessment
    if total_chunks == 0:
        coverage_level = "empty"
        coverage_message = "No documents found. Ingest documentation to activate RAG."
    elif total_chunks < 10:
        coverage_level = "critical"
        coverage_message = "Very few chunks. RAG will not be able to generate quality content."
    elif total_chunks < 50:
        coverage_level = "low"
        coverage_message = "Low coverage. Consider ingesting more documentation."
    elif total_chunks < 200:
        coverage_level = "moderate"
        coverage_message = "Moderate coverage. Functional for most queries."
    elif total_chunks < 500:
        coverage_level = "good"
        coverage_message = "Good document coverage."
    else:
        coverage_level = "excellent"
        coverage_message = "Excellent coverage."

    # Retriever config
    retriever_config = {
        "top_k": getattr(retriever, "top_k", None),
        "min_score": getattr(retriever, "min_score", None),
    }

    return {
        "total_chunks": total_chunks,
        "total_chars": total_chars,
        "unique_documents": len(unique_doc_ids),
        "embedding_dimensions": embedding_dim,
        "avg_chunk_chars": avg_chunk_chars,
        "coverage_level": coverage_level,
        "coverage_message": coverage_message,
        "retriever_config": retriever_config,
        "sources": sources_breakdown,
    }


# ---------------------------------------------------------------------------
# Test runner — runs tests in a background thread with streaming results
# ---------------------------------------------------------------------------

# A single global lock protects the active test run.
_run_lock = threading.Lock()
_current_run: dict[str, Any] | None = None


def _reset_current_run() -> dict[str, Any]:
    """Initialise a fresh test run state dict."""
    return {
        "status": "running",
        "started_at": time.time(),
        "finished_at": None,
        "suites": {},
        "summary": {
            "total": 0,
            "passed": 0,
            "failed": 0,
            "skipped": 0,
            "errors": 0,
        },
    }


# ---- individual test suites ------------------------------------------------

def _run_rag_retrieval_suite(rag_pipeline: Any) -> dict[str, Any]:
    """Test RAG retrieval quality using known queries."""

    from ai_engine.rag.document import Document

    results: list[dict[str, Any]] = []

    # Ingest test corpus into a temporary pipeline copy
    from ai_engine.rag.chunker import Chunker
    from ai_engine.rag.retriever import Retriever
    from ai_engine.rag.vector_store import InMemoryVectorStore

    embedder = rag_pipeline.embedder
    test_store = InMemoryVectorStore()
    test_retriever = Retriever(embedder, test_store, top_k=5)
    chunker = Chunker(chunk_size=500, chunk_overlap=50)

    test_docs = [
        Document(
            content=(
                "La fotosíntesis es el proceso mediante el cual las plantas convierten "
                "la luz solar, el agua y el dióxido de carbono en glucosa y oxígeno. "
                "Ocurre en los cloroplastos, en la clorofila. Fases: luminosa (tilacoides) "
                "y ciclo de Calvin (estroma)."
            ),
            doc_id="test-fotosintesis",
            metadata={"source": "test", "subject": "biología"},
        ),
        Document(
            content=(
                "La Revolución Francesa comenzó en 1789 con la Toma de la Bastilla. "
                "Causas: crisis económica, desigualdad, ideas de Voltaire, Rousseau y "
                "Montesquieu. Lema: Libertad, Igualdad, Fraternidad."
            ),
            doc_id="test-rev-francesa",
            metadata={"source": "test", "subject": "historia"},
        ),
        Document(
            content=(
                "El teorema de Pitágoras: en un triángulo rectángulo, a² + b² = c². "
                "Ejemplo: catetos 3 y 4, hipotenusa 5. Ternas: (3,4,5), (5,12,13)."
            ),
            doc_id="test-pitagoras",
            metadata={"source": "test", "subject": "matemáticas"},
        ),
    ]

    chunks: list[Document] = []
    for doc in test_docs:
        chunks.extend(chunker.split(doc))
    embeddings = embedder.embed_documents(chunks)
    test_store.add(chunks, embeddings)

    # Test 1: On-topic retrieval
    def _test_on_topic() -> dict[str, Any]:
        docs = test_retriever.retrieve("fotosíntesis en plantas cloroplastos")
        scores = test_retriever.last_scores
        passed = len(docs) > 0 and all(s >= 0.3 for s in scores)
        return {
            "name": "On-topic retrieval returns relevant results",
            "passed": passed,
            "details": {
                "docs_returned": len(docs),
                "scores": [round(s, 4) for s in scores],
                "threshold": 0.3,
            },
        }

    # Test 2: Similarity score quality
    def _test_similarity_quality() -> dict[str, Any]:
        test_retriever.retrieve("fases de la fotosíntesis luminosa Calvin")
        scores = test_retriever.last_scores
        mean_score = sum(scores) / len(scores) if scores else 0
        passed = mean_score > 0.4
        return {
            "name": "Mean similarity > 0.4 for on-topic queries",
            "passed": passed,
            "details": {"mean_similarity": round(mean_score, 4), "threshold": 0.4},
        }

    # Test 3: Off-topic filtering
    def _test_off_topic_filter() -> dict[str, Any]:
        off_topic_queries = [
            "recetas de cocina italiana carbonara",
            "programación en rust async",
        ]
        max_docs = 0
        for q in off_topic_queries:
            docs = test_retriever.retrieve(q)
            max_docs = max(max_docs, len(docs))
        passed = max_docs <= 2
        return {
            "name": "Off-topic queries return ≤2 docs (min_score filter)",
            "passed": passed,
            "details": {"max_docs_returned": max_docs, "expected_max": 2},
        }

    # Test 4: Cross-topic isolation
    def _test_cross_topic() -> dict[str, Any]:
        docs = test_retriever.retrieve("teorema de Pitágoras hipotenusa")
        context = "\n".join(d.content for d in docs).lower()
        math_hits = sum(1 for w in ["pitágoras", "hipotenusa", "catetos", "triángulo"] if w in context)
        bio_hits = sum(1 for w in ["fotosíntesis", "cloroplastos", "clorofila"] if w in context)
        passed = math_hits >= bio_hits
        return {
            "name": "Cross-topic isolation (math vs bio)",
            "passed": passed,
            "details": {"math_hits": math_hits, "bio_hits": bio_hits},
        }

    for test_fn in [_test_on_topic, _test_similarity_quality, _test_off_topic_filter, _test_cross_topic]:
        try:
            results.append(test_fn())
        except Exception as exc:
            results.append({
                "name": test_fn.__name__,
                "passed": False,
                "error": str(exc),
            })

    passed = sum(1 for r in results if r.get("passed"))
    return {
        "suite": "RAG Retrieval Quality",
        "total": len(results),
        "passed": passed,
        "failed": len(results) - passed,
        "tests": results,
    }


def _run_prompt_grounding_suite() -> dict[str, Any]:
    """Test that prompt templates include anti-hallucination instructions."""
    from ai_engine.games.prompts import _SYSTEM, get_prompt

    results: list[dict[str, Any]] = []

    # Test 1: System prompt contains grounding instruction
    has_exclusive = "exclusively" in _SYSTEM.lower()
    has_fabricate = "fabricat" in _SYSTEM.lower() or "invent" in _SYSTEM.lower()
    results.append({
        "name": "System prompt contains grounding instruction",
        "passed": has_exclusive and has_fabricate,
        "details": {
            "has_exclusively": has_exclusive,
            "has_anti_fabrication": has_fabricate,
        },
    })

    # Test 2: Context injection in all game types
    for game_type in ["quiz", "word-pass", "true_false"]:
        prompt = get_prompt(game_type, context="__CONTEXT_MARKER__", language="es")
        passed = "__CONTEXT_MARKER__" in prompt
        results.append({
            "name": f"Prompt '{game_type}' injects context",
            "passed": passed,
        })

    passed = sum(1 for r in results if r.get("passed"))
    return {
        "suite": "Prompt Grounding",
        "total": len(results),
        "passed": passed,
        "failed": len(results) - passed,
        "tests": results,
    }


def _run_generation_params_suite() -> dict[str, Any]:
    """Test LLM generation parameters for factual content."""
    from ai_engine.llm.llama_client import LlamaClient, JSON_GRAMMAR

    results: list[dict[str, Any]] = []

    client = LlamaClient(api_url="http://test:8080")

    results.append({
        "name": "Temperature ≤ 0.4 (hallucination reduction)",
        "passed": client.temperature <= 0.4,
        "details": {"temperature": client.temperature, "max_recommended": 0.4},
    })

    results.append({
        "name": "top_p ≤ 0.95 (controlled generation)",
        "passed": client.top_p <= 0.95,
        "details": {"top_p": client.top_p, "max_recommended": 0.95},
    })

    results.append({
        "name": "JSON Grammar available for structured output",
        "passed": "object" in JSON_GRAMMAR and "string" in JSON_GRAMMAR,
    })

    passed = sum(1 for r in results if r.get("passed"))
    return {
        "suite": "Generation Parameters",
        "total": len(results),
        "passed": passed,
        "failed": len(results) - passed,
        "tests": results,
    }


def _run_cache_integrity_suite() -> dict[str, Any]:
    """Test cache key differentiation."""
    from ai_engine.api.optimization import GenerationOptimizationService
    from ai_engine.api.schemas import GenerateRequest

    results: list[dict[str, Any]] = []

    svc = GenerationOptimizationService.__new__(GenerationOptimizationService)
    svc._cache_namespace = "v1"

    req_easy = GenerateRequest(query="fotosíntesis", game_type="quiz", difficulty_percentage=20)
    req_hard = GenerateRequest(query="fotosíntesis", game_type="quiz", difficulty_percentage=80)
    req_same = GenerateRequest(query="fotosíntesis", game_type="quiz", difficulty_percentage=20)

    key_easy = svc._cache_key(req_easy)
    key_hard = svc._cache_key(req_hard)
    key_same = svc._cache_key(req_same)

    results.append({
        "name": "Different difficulty → different cache key",
        "passed": key_easy != key_hard,
    })

    results.append({
        "name": "Same params → same cache key",
        "passed": key_easy == key_same,
    })

    passed = sum(1 for r in results if r.get("passed"))
    return {
        "suite": "Cache Integrity",
        "total": len(results),
        "passed": passed,
        "failed": len(results) - passed,
        "tests": results,
    }


def _run_metrics_suite() -> dict[str, Any]:
    """Test observability metrics completeness."""
    from ai_engine.observability.collector import StatsCollector

    results: list[dict[str, Any]] = []
    collector = StatsCollector()
    summary = collector.summary()

    required_keys = ["retry_rate", "retry_used_count", "avg_rag_similarity", "avg_rag_context_length_chars"]
    for key in required_keys:
        results.append({
            "name": f"Metric '{key}' present in summary",
            "passed": key in summary,
        })

    passed = sum(1 for r in results if r.get("passed"))
    return {
        "suite": "Metrics Completeness",
        "total": len(results),
        "passed": passed,
        "failed": len(results) - passed,
        "tests": results,
    }


# ---- runner orchestrator ---------------------------------------------------

_SUITE_REGISTRY: list[tuple[str, Any]] = [
    ("rag_retrieval", None),       # requires rag_pipeline
    ("prompt_grounding", None),
    ("generation_params", None),
    ("cache_integrity", None),
    ("metrics", None),
]


def _run_all_suites(rag_pipeline: Any) -> None:
    """Execute all test suites, updating _current_run in place."""
    global _current_run
    if _current_run is None:
        return

    suite_fns = [
        ("rag_retrieval", lambda: _run_rag_retrieval_suite(rag_pipeline)),
        ("prompt_grounding", _run_prompt_grounding_suite),
        ("generation_params", _run_generation_params_suite),
        ("cache_integrity", _run_cache_integrity_suite),
        ("metrics", _run_metrics_suite),
    ]

    for suite_key, fn in suite_fns:
        try:
            result = fn()
        except Exception as exc:
            logger.exception("Suite %s failed with error", suite_key)
            result = {
                "suite": suite_key,
                "total": 1,
                "passed": 0,
                "failed": 1,
                "tests": [{"name": suite_key, "passed": False, "error": str(exc)}],
            }

        _current_run["suites"][suite_key] = result
        _current_run["summary"]["total"] += result["total"]
        _current_run["summary"]["passed"] += result["passed"]
        _current_run["summary"]["failed"] += result["failed"]

    _current_run["status"] = "completed"
    _current_run["finished_at"] = time.time()


def start_test_run(rag_pipeline: Any) -> dict[str, Any]:
    """Start a test run in a background thread.

    Returns immediately with run status. Poll ``get_test_status()`` for results.
    """
    global _current_run

    if not _run_lock.acquire(blocking=False):
        return {
            "status": "already_running",
            "message": "A test run is already in progress. Check /diagnostics/tests/status.",
        }

    try:
        _current_run = _reset_current_run()

        def _worker() -> None:
            try:
                _run_all_suites(rag_pipeline)
            except Exception:
                logger.exception("Test run failed")
                if _current_run is not None:
                    _current_run["status"] = "error"
                    _current_run["finished_at"] = time.time()
            finally:
                _run_lock.release()

        thread = threading.Thread(target=_worker, daemon=True)
        thread.start()

        return {"status": "started", "message": "Test run started."}
    except Exception:
        _run_lock.release()
        raise


def get_test_status() -> dict[str, Any]:
    """Return current test run status and results collected so far."""
    if _current_run is None:
        return {
            "status": "idle",
            "message": "No active test run. Execute POST /diagnostics/tests/run.",
            "suites": {},
            "summary": {"total": 0, "passed": 0, "failed": 0, "skipped": 0, "errors": 0},
        }
    return dict(_current_run)
