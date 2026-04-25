"""Diagnostics endpoints for the ai-engine API.

Provides:
- ``GET  /diagnostics/rag/stats``    – RAG vector store health & coverage metrics.
- ``POST /diagnostics/tests/run``    – Execute hallucination & quality test suites.
- ``GET  /diagnostics/tests/status`` – Poll current test execution status/results.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import threading
import time
from typing import Any, Callable

from ai_engine.config import get_settings

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
        sources_breakdown.append(
            {
                "source": source,
                "chunks": stats["chunks"],
                "total_chars": stats["total_chars"],
                "unique_documents": len(stats["doc_ids"]),
                "avg_chunk_chars": (
                    round(stats["total_chars"] / stats["chunks"], 1)
                    if stats["chunks"]
                    else 0
                ),
            }
        )

    sources_breakdown.sort(key=lambda x: x["chunks"], reverse=True)

    avg_chunk_chars = round(total_chars / total_chunks, 1) if total_chunks else 0

    # Coverage assessment
    if total_chunks == 0:
        coverage_level = "empty"
        coverage_message = "No documents found. Ingest documentation to activate RAG."
    elif total_chunks < 10:
        coverage_level = "critical"
        coverage_message = (
            "Very few chunks. RAG will not be able to generate quality content."
        )
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

_RETRIEVAL_CASE_TARGET_MS = 600.0
_RETRIEVAL_P95_TARGET_MS = 800.0
_GENERATION_CASE_TARGET_MS = 7000.0
_GENERATION_P95_TARGET_MS = 9000.0
_GENERATION_SUCCESS_RATE_TARGET = 0.67
_GENERATION_TIMEOUT_SECONDS = 18.0
_NON_PROD_DISTRIBUTIONS = {"dev", "stg", "stage", "staging", "sandbox", "local"}


def _generation_performance_targets() -> dict[str, float]:
    """Return generation performance thresholds adapted to the active distribution."""
    distribution = (get_settings().distribution or "").strip().lower()

    if distribution in _NON_PROD_DISTRIBUTIONS:
        # Staging/dev clusters can run on constrained CPU nodes.
        return {
            "case_target_ms": 35000.0,
            "p95_target_ms": 40000.0,
            "success_rate_target": _GENERATION_SUCCESS_RATE_TARGET,
            "timeout_seconds": 40.0,
        }

    return {
        "case_target_ms": _GENERATION_CASE_TARGET_MS,
        "p95_target_ms": _GENERATION_P95_TARGET_MS,
        "success_rate_target": _GENERATION_SUCCESS_RATE_TARGET,
        "timeout_seconds": _GENERATION_TIMEOUT_SECONDS,
    }


def _reset_current_run() -> dict[str, Any]:
    """Initialise a fresh test run state dict."""
    return {
        "status": "running",
        "started_at": time.time(),
        "finished_at": None,
        "suites": {},
        "progress": {
            "total_suites": 0,
            "completed_suites": 0,
            "percent": 0,
            "current_suite": None,
            "message": "Queued",
        },
        "recommendations": [],
        "performance": {},
        "summary": {
            "total": 0,
            "passed": 0,
            "failed": 0,
            "skipped": 0,
            "errors": 0,
        },
    }


def _percentile_ms(samples: list[float], percentile: float) -> float:
    """Return percentile using linear interpolation between ordered samples."""
    if not samples:
        return 0.0
    ordered = sorted(samples)
    if len(ordered) == 1:
        return ordered[0]

    rank = (percentile / 100.0) * (len(ordered) - 1)
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return (ordered[lower] * (1.0 - weight)) + (ordered[upper] * weight)


def _set_progress(
    *,
    total_suites: int,
    completed_suites: int,
    current_suite: str | None,
    message: str,
) -> None:
    """Update progress block for the current diagnostics run."""
    global _current_run
    if _current_run is None:
        return

    percent = 0
    if total_suites > 0:
        percent = max(0, min(100, int(round((completed_suites / total_suites) * 100))))

    _current_run["progress"] = {
        "total_suites": total_suites,
        "completed_suites": completed_suites,
        "percent": percent,
        "current_suite": current_suite,
        "message": message,
    }


def _set_progress_message(message: str) -> None:
    """Mutate only the progress message while keeping percent counters intact."""
    global _current_run
    if _current_run is None:
        return

    progress = _current_run.get("progress")
    if not isinstance(progress, dict):
        return
    progress["message"] = message


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
        math_hits = sum(
            1
            for w in ["pitágoras", "hipotenusa", "catetos", "triángulo"]
            if w in context
        )
        bio_hits = sum(
            1 for w in ["fotosíntesis", "cloroplastos", "clorofila"] if w in context
        )
        passed = math_hits >= bio_hits
        return {
            "name": "Cross-topic isolation (math vs bio)",
            "passed": passed,
            "details": {"math_hits": math_hits, "bio_hits": bio_hits},
        }

    for test_fn in [
        _test_on_topic,
        _test_similarity_quality,
        _test_off_topic_filter,
        _test_cross_topic,
    ]:
        try:
            results.append(test_fn())
        except Exception as exc:
            results.append(
                {
                    "name": test_fn.__name__,
                    "passed": False,
                    "error": str(exc),
                }
            )

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
    results.append(
        {
            "name": "System prompt contains grounding instruction",
            "passed": has_exclusive and has_fabricate,
            "details": {
                "has_exclusively": has_exclusive,
                "has_anti_fabrication": has_fabricate,
            },
        }
    )

    # Test 2: Context injection in all game types
    for game_type in ["quiz", "word-pass", "true_false"]:
        prompt = get_prompt(game_type, context="__CONTEXT_MARKER__")
        test_passed = "__CONTEXT_MARKER__" in prompt
        results.append(
            {
                "name": f"Prompt '{game_type}' injects context",
                "passed": test_passed,
            }
        )

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
    from ai_engine.llm.llama_client import JSON_GRAMMAR, LlamaClient

    results: list[dict[str, Any]] = []

    client = LlamaClient(api_url="http://test:8080")

    results.append(
        {
            "name": "Temperature ≤ 0.4 (hallucination reduction)",
            "passed": client.temperature <= 0.4,
            "details": {"temperature": client.temperature, "max_recommended": 0.4},
        }
    )

    results.append(
        {
            "name": "top_p ≤ 0.95 (controlled generation)",
            "passed": client.top_p <= 0.95,
            "details": {"top_p": client.top_p, "max_recommended": 0.95},
        }
    )

    results.append(
        {
            "name": "JSON Grammar available for structured output",
            "passed": "object" in JSON_GRAMMAR and "string" in JSON_GRAMMAR,
        }
    )

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

    req_easy = GenerateRequest(
        query="fotosíntesis", game_type="quiz", difficulty_percentage=20
    )
    req_hard = GenerateRequest(
        query="fotosíntesis", game_type="quiz", difficulty_percentage=80
    )
    req_same = GenerateRequest(
        query="fotosíntesis", game_type="quiz", difficulty_percentage=20
    )

    key_easy = svc._cache_key(req_easy)
    key_hard = svc._cache_key(req_hard)
    key_same = svc._cache_key(req_same)

    results.append(
        {
            "name": "Different difficulty → different cache key",
            "passed": key_easy != key_hard,
        }
    )

    results.append(
        {
            "name": "Same params → same cache key",
            "passed": key_easy == key_same,
        }
    )

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

    required_keys = [
        "retry_rate",
        "retry_used_count",
        "avg_rag_similarity",
        "avg_rag_context_length_chars",
    ]
    for key in required_keys:
        results.append(
            {
                "name": f"Metric '{key}' present in summary",
                "passed": key in summary,
            }
        )

    passed = sum(1 for r in results if r.get("passed"))
    return {
        "suite": "Metrics Completeness",
        "total": len(results),
        "passed": passed,
        "failed": len(results) - passed,
        "tests": results,
    }


def _run_retrieval_performance_suite(rag_pipeline: Any) -> dict[str, Any]:
    """Measure real retrieval latency with bounded sample queries."""
    if rag_pipeline is None or not hasattr(rag_pipeline, "retrieve"):
        return {
            "suite": "Retrieval Performance",
            "total": 1,
            "passed": 0,
            "failed": 1,
            "tests": [
                {
                    "name": "RAG pipeline availability",
                    "passed": False,
                    "error": "rag_pipeline is not available for diagnostics",
                }
            ],
            "metrics": {
                "avg_latency_ms": 0.0,
                "p95_latency_ms": 0.0,
                "max_latency_ms": 0.0,
            },
        }

    queries = [
        "fotosintesis cloroplastos y ciclo de calvin",
        "revolucion francesa causas y bastilla",
        "teorema de pitagoras hipotenusa",
        "programacion concurrente en python",
    ]

    latencies_ms: list[float] = []
    tests: list[dict[str, Any]] = []

    for index, query in enumerate(queries, start=1):
        _set_progress_message(f"Retrieval performance case {index}/{len(queries)}")
        started = time.perf_counter()
        docs = rag_pipeline.retrieve(query, top_k=5)
        latency_ms = (time.perf_counter() - started) * 1000.0
        latencies_ms.append(latency_ms)

        tests.append(
            {
                "name": f"Retrieval latency case {index}",
                "passed": latency_ms <= _RETRIEVAL_CASE_TARGET_MS,
                "details": {
                    "latency_ms": round(latency_ms, 2),
                    "target_ms": _RETRIEVAL_CASE_TARGET_MS,
                    "docs_returned": len(docs),
                },
            }
        )

    avg_latency = sum(latencies_ms) / len(latencies_ms) if latencies_ms else 0.0
    p95_latency = _percentile_ms(latencies_ms, 95.0)
    max_latency = max(latencies_ms) if latencies_ms else 0.0

    tests.append(
        {
            "name": "Retrieval p95 latency within target",
            "passed": p95_latency <= _RETRIEVAL_P95_TARGET_MS,
            "details": {
                "p95_latency_ms": round(p95_latency, 2),
                "target_ms": _RETRIEVAL_P95_TARGET_MS,
            },
        }
    )

    passed = sum(1 for result in tests if result.get("passed"))
    return {
        "suite": "Retrieval Performance",
        "total": len(tests),
        "passed": passed,
        "failed": len(tests) - passed,
        "tests": tests,
        "metrics": {
            "avg_latency_ms": round(avg_latency, 2),
            "p95_latency_ms": round(p95_latency, 2),
            "max_latency_ms": round(max_latency, 2),
        },
    }


def _run_generation_performance_suite(generator: Any) -> dict[str, Any]:
    """Measure real end-to-end generation latency with bounded workloads."""
    targets = _generation_performance_targets()

    if generator is None or not hasattr(generator, "generate"):
        return {
            "suite": "Generation Performance",
            "total": 1,
            "passed": 0,
            "failed": 1,
            "tests": [
                {
                    "name": "Generator availability",
                    "passed": False,
                    "error": "generator is not available for diagnostics",
                }
            ],
            "metrics": {
                "success_rate": 0.0,
                "avg_latency_ms": 0.0,
                "p95_latency_ms": 0.0,
                "throughput_rps": 0.0,
            },
        }

    cases = [
        {
            "query": "fotosintesis en plantas",
            "game_type": "quiz",
            "difficulty_percentage": 45,
            "num_questions": 1,
            "max_tokens": 128,
            "top_k": 2,
            "letters": "A,B,C,D,E,F,G,H,I,J",
        },
        {
            "query": "revolucion francesa",
            "game_type": "true_false",
            "difficulty_percentage": 55,
            "num_questions": 1,
            "max_tokens": 128,
            "top_k": 2,
            "letters": "A,B,C,D,E,F,G,H,I,J",
        },
        {
            "query": "teorema de pitagoras",
            "game_type": "word-pass",
            "difficulty_percentage": 50,
            "num_questions": 2,
            "max_tokens": 160,
            "top_k": 2,
            "letters": "A,B,C,D,E,F,G,H,I,J",
        },
    ]

    async def _run_cases() -> tuple[list[float], list[dict[str, Any]], int]:
        latencies_ms: list[float] = []
        tests: list[dict[str, Any]] = []
        successful_runs = 0

        for index, case in enumerate(cases, start=1):
            _set_progress_message(
                f"Generation performance case {index}/{len(cases)} ({case['game_type']})"
            )
            started = time.perf_counter()
            timed_out = False
            error_message: str | None = None

            try:
                await asyncio.wait_for(
                    generator.generate(
                        case["query"],
                        game_type=case["game_type"],
                        language="es",
                        difficulty_percentage=case["difficulty_percentage"],
                        num_questions=case["num_questions"],
                        max_tokens=case["max_tokens"],
                        top_k=case["top_k"],
                        letters=case["letters"],
                    ),
                    timeout=targets["timeout_seconds"],
                )
                successful_runs += 1
            except asyncio.TimeoutError:
                timed_out = True
                error_message = (
                    f"generation exceeded {targets['timeout_seconds']:.0f}s timeout"
                )
            except Exception as exc:
                error_message = _format_generation_error(exc, generator)

            latency_ms = (time.perf_counter() - started) * 1000.0
            latencies_ms.append(latency_ms)
            passed_case = (
                error_message is None and latency_ms <= targets["case_target_ms"]
            )

            case_result: dict[str, Any] = {
                "name": f"Generation latency case {index} ({case['game_type']})",
                "passed": passed_case,
                "details": {
                    "latency_ms": round(latency_ms, 2),
                    "target_ms": targets["case_target_ms"],
                    "timeout_s": targets["timeout_seconds"],
                    "num_questions": case["num_questions"],
                    "max_tokens": case["max_tokens"],
                    "top_k": case["top_k"],
                },
            }
            if timed_out:
                case_result["details"]["timed_out"] = True
            if error_message is not None:
                case_result["error"] = error_message
            tests.append(case_result)

        return latencies_ms, tests, successful_runs

    bound_loop = _resolve_generation_event_loop(generator)
    if (
        bound_loop is not None
        and not bound_loop.is_closed()
        and bound_loop.is_running()
    ):
        try:
            future = asyncio.run_coroutine_threadsafe(_run_cases(), bound_loop)
            latencies_ms, tests, successful_runs = future.result()
        except concurrent.futures.CancelledError:
            latencies_ms, tests, successful_runs = [], [], 0
            tests.append(
                {
                    "name": "Generation performance execution",
                    "passed": False,
                    "error": "generation diagnostics coroutine was cancelled",
                }
            )
        except Exception as exc:
            latencies_ms, tests, successful_runs = [], [], 0
            tests.append(
                {
                    "name": "Generation performance execution",
                    "passed": False,
                    "error": _format_generation_error(exc, generator),
                }
            )
    else:
        latencies_ms, tests, successful_runs = asyncio.run(_run_cases())

    avg_latency = sum(latencies_ms) / len(latencies_ms) if latencies_ms else 0.0
    p95_latency = _percentile_ms(latencies_ms, 95.0)
    success_rate = successful_runs / len(cases) if cases else 0.0
    total_elapsed_s = max(sum(latencies_ms) / 1000.0, 0.001)
    throughput_rps = successful_runs / total_elapsed_s

    tests.append(
        {
            "name": "Generation success rate within target",
            "passed": success_rate >= targets["success_rate_target"],
            "details": {
                "success_rate": round(success_rate, 3),
                "target": targets["success_rate_target"],
            },
        }
    )
    tests.append(
        {
            "name": "Generation p95 latency within target",
            "passed": p95_latency <= targets["p95_target_ms"],
            "details": {
                "p95_latency_ms": round(p95_latency, 2),
                "target_ms": targets["p95_target_ms"],
            },
        }
    )

    passed = sum(1 for result in tests if result.get("passed"))
    return {
        "suite": "Generation Performance",
        "total": len(tests),
        "passed": passed,
        "failed": len(tests) - passed,
        "tests": tests,
        "metrics": {
            "success_rate": round(success_rate, 3),
            "avg_latency_ms": round(avg_latency, 2),
            "p95_latency_ms": round(p95_latency, 2),
            "throughput_rps": round(throughput_rps, 3),
        },
    }


def _resolve_generation_event_loop(generator: Any) -> asyncio.AbstractEventLoop | None:
    """Try to locate the event loop bound to the live generation stack."""

    def _semaphore_loop(candidate: Any) -> asyncio.AbstractEventLoop | None:
        semaphore = getattr(candidate, "_api_semaphore", None)
        loop = getattr(semaphore, "_loop", None)
        if isinstance(loop, asyncio.AbstractEventLoop):
            return loop
        return None

    seen: set[int] = set()
    stack = [generator]
    relation_attrs = (
        "_generator",
        "generator",
        "_client",
        "client",
        "llm_client",
        "_llm_client",
    )

    while stack:
        current = stack.pop()
        if current is None:
            continue
        identity = id(current)
        if identity in seen:
            continue
        seen.add(identity)

        loop = _semaphore_loop(current)
        if loop is not None:
            return loop

        for attr in relation_attrs:
            nested = getattr(current, attr, None)
            if nested is not None:
                stack.append(nested)

    return None


def _extract_generation_llama_url(generator: Any) -> str | None:
    """Best-effort extraction of current llama upstream URL from wrapped generators."""
    candidate = generator
    # TrackedGameGenerator keeps the real generator under _generator.
    if hasattr(candidate, "_generator"):
        candidate = getattr(candidate, "_generator")

    llm_client = getattr(candidate, "llm_client", None)
    if llm_client is None:
        return None

    # TrackedLlamaClient keeps the real client under _client.
    if hasattr(llm_client, "_client"):
        llm_client = getattr(llm_client, "_client")

    url = getattr(llm_client, "api_url", None)
    return str(url) if isinstance(url, str) and url.strip() else None


def _format_generation_error(exc: Exception, generator: Any) -> str:
    """Normalize generation errors with actionable llama-connectivity hints."""
    base_message = str(exc).strip() or exc.__class__.__name__
    lowered = base_message.lower()
    connectivity_markers = (
        "all connection attempts failed",
        "connection refused",
        "name or service not known",
        "temporary failure in name resolution",
        "network is unreachable",
        "nodename nor servname provided",
    )

    if any(marker in lowered for marker in connectivity_markers):
        upstream = _extract_generation_llama_url(generator)
        if upstream:
            return (
                f"{base_message}. Llama upstream unreachable ({upstream}); "
                "verify runtime llama target and ai-engine-llama service health."
            )
        return (
            f"{base_message}. Llama upstream unreachable; verify runtime llama "
            "target and ai-engine-llama service health."
        )

    return base_message


def _build_recommendations(run_status: dict[str, Any]) -> list[str]:
    """Generate operator recommendations from measured diagnostics metrics."""
    targets = _generation_performance_targets()

    recommendations: list[str] = []
    suites = run_status.get("suites", {})
    generation_metrics = (
        suites.get("generation_performance", {}).get("metrics", {})
        if isinstance(suites, dict)
        else {}
    )
    generation_tests = (
        suites.get("generation_performance", {}).get("tests", [])
        if isinstance(suites, dict)
        else []
    )
    retrieval_metrics = (
        suites.get("retrieval_performance", {}).get("metrics", {})
        if isinstance(suites, dict)
        else {}
    )

    success_rate = float(generation_metrics.get("success_rate", 0.0) or 0.0)
    generation_p95 = float(generation_metrics.get("p95_latency_ms", 0.0) or 0.0)
    retrieval_p95 = float(retrieval_metrics.get("p95_latency_ms", 0.0) or 0.0)

    if success_rate < targets["success_rate_target"]:
        recommendations.append(
            "Generation stability is below target; review llama runtime logs and increase retry/timeout budgets before peak traffic."
        )
    if isinstance(generation_tests, list) and any(
        isinstance(test, dict)
        and isinstance(test.get("error"), str)
        and "connection attempts failed" in test["error"].lower()
        for test in generation_tests
    ):
        recommendations.append(
            "Generation diagnostics cannot reach llama upstream. Reset the runtime llama target from backoffice and validate ai-engine-llama service DNS/network from the ai-engine-api pod."
        )
    if generation_p95 > targets["p95_target_ms"]:
        recommendations.append(
            "Generation latency is high; reduce max_tokens/top_k for operational flows or scale the inference node."
        )
    if retrieval_p95 > _RETRIEVAL_P95_TARGET_MS:
        recommendations.append(
            "RAG retrieval latency is above target; consider lowering retriever top_k or optimizing embedding/vector index resources."
        )

    summary = run_status.get("summary", {})
    failed = int(summary.get("failed", 0) or 0)
    if failed > 0 and not recommendations:
        recommendations.append(
            "Some diagnostic checks failed. Review suite details and rerun after applying targeted tuning."
        )

    if not recommendations:
        recommendations.append(
            "Performance baseline is healthy. Keep current runtime profile and monitor regressions over time."
        )
    return recommendations


# ---- runner orchestrator ---------------------------------------------------


def _run_all_suites(rag_pipeline: Any, generator: Any | None = None) -> None:
    """Execute all test suites, updating _current_run in place."""
    global _current_run
    if _current_run is None:
        return

    suite_fns: list[tuple[str, str, Callable[[], dict[str, Any]]]] = [
        (
            "retrieval_performance",
            "Retrieval Performance",
            lambda: _run_retrieval_performance_suite(rag_pipeline),
        ),
        (
            "generation_performance",
            "Generation Performance",
            lambda: _run_generation_performance_suite(generator),
        ),
        (
            "rag_retrieval",
            "rag_retrieval",
            lambda: _run_rag_retrieval_suite(rag_pipeline),
        ),
        ("prompt_grounding", "prompt_grounding", _run_prompt_grounding_suite),
        ("generation_params", "generation_params", _run_generation_params_suite),
        ("cache_integrity", "cache_integrity", _run_cache_integrity_suite),
        ("metrics", "metrics", _run_metrics_suite),
    ]

    total_suites = len(suite_fns)
    _set_progress(
        total_suites=total_suites,
        completed_suites=0,
        current_suite=None,
        message="Starting diagnostics",
    )

    for index, (suite_key, suite_label, fn) in enumerate(suite_fns, start=1):
        _set_progress(
            total_suites=total_suites,
            completed_suites=index - 1,
            current_suite=suite_key,
            message=f"Running {suite_label} ({index}/{total_suites})",
        )
        try:
            result = fn()
        except Exception as exc:
            logger.exception("Suite %s failed with error", suite_key)
            result = {
                "suite": suite_key,
                "total": 1,
                "passed": 0,
                "failed": 1,
                "errors": 1,
                "tests": [
                    {
                        "name": suite_key,
                        "passed": False,
                        "error": str(exc),
                    }
                ],
            }

        _current_run["suites"][suite_key] = result
        _current_run["summary"]["total"] += result["total"]
        _current_run["summary"]["passed"] += result["passed"]
        _current_run["summary"]["failed"] += result["failed"]
        _current_run["summary"]["errors"] += int(result.get("errors", 0) or 0)

        _set_progress(
            total_suites=total_suites,
            completed_suites=index,
            current_suite=suite_key,
            message=f"Completed {suite_label}",
        )

    retrieval_metrics = (
        _current_run["suites"].get("retrieval_performance", {}).get("metrics", {})
    )
    generation_metrics = (
        _current_run["suites"].get("generation_performance", {}).get("metrics", {})
    )
    _current_run["performance"] = {
        "retrieval": retrieval_metrics,
        "generation": generation_metrics,
    }
    _current_run["recommendations"] = _build_recommendations(_current_run)

    _current_run["status"] = "completed"
    _current_run["finished_at"] = time.time()
    _set_progress(
        total_suites=total_suites,
        completed_suites=total_suites,
        current_suite=None,
        message="Diagnostics completed",
    )


def start_test_run(rag_pipeline: Any, generator: Any | None = None) -> dict[str, Any]:
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
                _run_all_suites(rag_pipeline, generator=generator)
            except Exception:
                logger.exception("Test run failed")
                if _current_run is not None:
                    _current_run["status"] = "error"
                    _current_run["finished_at"] = time.time()
                    _current_run["recommendations"] = [
                        "Diagnostics run failed unexpectedly. Review server logs and retry the test run."
                    ]
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
            "progress": {
                "total_suites": 0,
                "completed_suites": 0,
                "percent": 0,
                "current_suite": None,
                "message": "Idle",
            },
            "recommendations": [],
            "performance": {},
            "summary": {
                "total": 0,
                "passed": 0,
                "failed": 0,
                "skipped": 0,
                "errors": 0,
            },
        }
    return dict(_current_run)
