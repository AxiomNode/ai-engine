from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace

import pytest

from ai_engine.api import diagnostics
from ai_engine.rag.document import Document


class FakeEmbedder:
    def _vectorize(self, text: str) -> list[float]:
        lowered = text.lower()
        return [
            (
                1.0
                if any(token in lowered for token in ["foto", "cloroplast", "calvin"])
                else 0.0
            ),
            (
                1.0
                if any(
                    token in lowered for token in ["revolución", "bastilla", "rousseau"]
                )
                else 0.0
            ),
            (
                1.0
                if any(
                    token in lowered
                    for token in ["pitágoras", "hipotenusa", "triángulo"]
                )
                else 0.0
            ),
        ]

    def embed_documents(self, documents: list[Document]) -> list[list[float]]:
        return [self._vectorize(document.content) for document in documents]

    def embed_text(self, text: str) -> list[float]:
        return self._vectorize(text)


class ImmediateThread:
    def __init__(self, target, daemon: bool = False):
        self._target = target
        self.daemon = daemon

    def start(self) -> None:
        self._target()


class AsyncFakeGenerator:
    async def generate(self, *args, **kwargs):
        return {"ok": True, "args": args, "kwargs": kwargs}


@pytest.fixture(autouse=True)
def reset_diagnostics_state() -> None:
    diagnostics._current_run = None
    if diagnostics._run_lock.locked():
        diagnostics._run_lock.release()
    yield
    diagnostics._current_run = None
    if diagnostics._run_lock.locked():
        diagnostics._run_lock.release()


@pytest.mark.parametrize(
    ("chunk_count", "expected_level"),
    [
        (0, "empty"),
        (5, "critical"),
        (20, "low"),
        (100, "moderate"),
        (250, "good"),
        (600, "excellent"),
    ],
)
def test_compute_rag_stats_reports_coverage_levels(
    chunk_count: int, expected_level: str
) -> None:
    documents = [
        Document(
            content=f"document content {index}",
            metadata={"source": "source-a" if index % 2 == 0 else "source-b"},
            doc_id=f"doc-{index // 2}",
        )
        for index in range(chunk_count)
    ]
    embeddings = [[0.1, 0.2, 0.3] for _ in range(chunk_count)]
    rag_pipeline = SimpleNamespace(
        vector_store=SimpleNamespace(_documents=documents, _embeddings_list=embeddings),
        retriever=SimpleNamespace(top_k=7, min_score=0.42),
    )

    stats = diagnostics.compute_rag_stats(rag_pipeline)

    assert stats["coverage_level"] == expected_level
    assert stats["retriever_config"] == {"top_k": 7, "min_score": 0.42}
    assert stats["total_chunks"] == chunk_count
    if chunk_count:
        assert stats["embedding_dimensions"] == 3


def test_compute_rag_stats_groups_sources_and_unique_documents() -> None:
    rag_pipeline = SimpleNamespace(
        vector_store=SimpleNamespace(
            _documents=[
                Document(content="alpha", metadata={"source": "docs"}, doc_id="doc-1"),
                Document(
                    content="beta beta", metadata={"source": "docs"}, doc_id="doc-1"
                ),
                Document(content="gamma", metadata={}, doc_id=None),
            ],
            _embeddings_list=[[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
        ),
        retriever=SimpleNamespace(top_k=5, min_score=0.3),
    )

    stats = diagnostics.compute_rag_stats(rag_pipeline)

    assert stats["unique_documents"] == 1
    assert stats["sources"][0]["source"] == "docs"
    assert stats["sources"][0]["chunks"] == 2
    assert stats["sources"][0]["unique_documents"] == 1
    assert stats["sources"][1]["source"] == "unknown"


def test_run_rag_retrieval_suite_returns_passing_results() -> None:
    rag_pipeline = SimpleNamespace(embedder=FakeEmbedder())

    result = diagnostics._run_rag_retrieval_suite(rag_pipeline)

    assert result["suite"] == "RAG Retrieval Quality"
    assert result["failed"] == 0
    assert result["passed"] == result["total"] == 4


def test_retrieval_performance_suite_reports_missing_pipeline() -> None:
    result = diagnostics._run_retrieval_performance_suite(rag_pipeline=None)

    assert result["suite"] == "Retrieval Performance"
    assert result["failed"] == 1
    assert result["tests"][0]["name"] == "RAG pipeline availability"


def test_retrieval_performance_suite_collects_latency_metrics() -> None:
    class FakeRetrievalPipeline:
        def retrieve(self, query: str, top_k: int = 5) -> list[Document]:
            assert top_k == 5
            return [Document(content=f"doc for {query}", metadata={}, doc_id=query)]

    result = diagnostics._run_retrieval_performance_suite(FakeRetrievalPipeline())

    assert result["suite"] == "Retrieval Performance"
    assert result["total"] == 5
    assert result["passed"] + result["failed"] == 5
    assert result["metrics"]["avg_latency_ms"] >= 0.0
    assert result["metrics"]["p95_latency_ms"] >= 0.0
    assert result["metrics"]["max_latency_ms"] >= 0.0


def test_other_diagnostics_suites_return_successful_summaries() -> None:
    prompt_result = diagnostics._run_prompt_grounding_suite()
    generation_result = diagnostics._run_generation_params_suite()
    cache_result = diagnostics._run_cache_integrity_suite()
    metrics_result = diagnostics._run_metrics_suite()

    assert prompt_result["failed"] == 0
    assert generation_result["failed"] == 0
    assert cache_result["failed"] == 0
    assert metrics_result["failed"] == 0


def test_run_all_suites_collects_summary(monkeypatch: pytest.MonkeyPatch) -> None:
    diagnostics._current_run = diagnostics._reset_current_run()
    monkeypatch.setattr(
        diagnostics,
        "_run_retrieval_performance_suite",
        lambda rag_pipeline: {
            "suite": "retrieval_performance",
            "total": 1,
            "passed": 1,
            "failed": 0,
            "tests": [],
            "metrics": {"p95_latency_ms": 200.0},
        },
    )
    monkeypatch.setattr(
        diagnostics,
        "_run_generation_performance_suite",
        lambda generator: {
            "suite": "generation_performance",
            "total": 1,
            "passed": 1,
            "failed": 0,
            "tests": [],
            "metrics": {"p95_latency_ms": 1200.0, "success_rate": 1.0},
        },
    )
    monkeypatch.setattr(
        diagnostics,
        "_run_rag_retrieval_suite",
        lambda rag_pipeline: {
            "suite": "rag",
            "total": 1,
            "passed": 1,
            "failed": 0,
            "tests": [],
        },
    )
    monkeypatch.setattr(
        diagnostics,
        "_run_prompt_grounding_suite",
        lambda: {"suite": "prompt", "total": 2, "passed": 1, "failed": 1, "tests": []},
    )
    monkeypatch.setattr(
        diagnostics,
        "_run_generation_params_suite",
        lambda: {
            "suite": "generation",
            "total": 1,
            "passed": 1,
            "failed": 0,
            "tests": [],
        },
    )
    monkeypatch.setattr(
        diagnostics,
        "_run_cache_integrity_suite",
        lambda: {"suite": "cache", "total": 1, "passed": 1, "failed": 0, "tests": []},
    )
    monkeypatch.setattr(
        diagnostics,
        "_run_metrics_suite",
        lambda: {"suite": "metrics", "total": 1, "passed": 1, "failed": 0, "tests": []},
    )

    diagnostics._run_all_suites(SimpleNamespace(), generator=AsyncFakeGenerator())

    assert diagnostics._current_run is not None
    assert diagnostics._current_run["status"] == "completed"
    assert diagnostics._current_run["summary"] == {
        "total": 8,
        "passed": 7,
        "failed": 1,
        "skipped": 0,
        "errors": 0,
    }
    assert diagnostics._current_run["progress"]["percent"] == 100
    assert diagnostics._current_run["recommendations"]


def test_start_test_run_handles_already_running() -> None:
    acquired = diagnostics._run_lock.acquire(blocking=False)
    assert acquired
    try:
        result = diagnostics.start_test_run(
            SimpleNamespace(), generator=AsyncFakeGenerator()
        )
    finally:
        diagnostics._run_lock.release()

    assert result["status"] == "already_running"


def test_start_test_run_completes_and_status_is_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_run_all_suites(
        rag_pipeline: object, generator: object | None = None
    ) -> None:
        assert diagnostics._current_run is not None
        diagnostics._current_run["suites"]["fake"] = {
            "suite": "fake",
            "total": 1,
            "passed": 1,
            "failed": 0,
            "tests": [],
        }
        diagnostics._current_run["summary"]["total"] = 1
        diagnostics._current_run["summary"]["passed"] = 1
        diagnostics._current_run["status"] = "completed"
        diagnostics._current_run["finished_at"] = 123.0

    monkeypatch.setattr(diagnostics.threading, "Thread", ImmediateThread)
    monkeypatch.setattr(diagnostics, "_run_all_suites", fake_run_all_suites)

    result = diagnostics.start_test_run(
        SimpleNamespace(), generator=AsyncFakeGenerator()
    )
    status = diagnostics.get_test_status()

    assert result["status"] == "started"
    assert status["status"] == "completed"
    assert status["summary"]["passed"] == 1


def test_start_test_run_marks_error_on_worker_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(diagnostics.threading, "Thread", ImmediateThread)

    def fail_run_all_suites(
        rag_pipeline: object, generator: object | None = None
    ) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr(diagnostics, "_run_all_suites", fail_run_all_suites)

    result = diagnostics.start_test_run(
        SimpleNamespace(), generator=AsyncFakeGenerator()
    )
    status = diagnostics.get_test_status()

    assert result["status"] == "started"
    assert status["status"] == "error"
    assert status["finished_at"] is not None


def test_get_test_status_is_idle_without_active_run() -> None:
    status = diagnostics.get_test_status()

    assert status["status"] == "idle"
    assert status["summary"]["total"] == 0
    assert status["progress"]["percent"] == 0


def test_generation_performance_suite_returns_metrics() -> None:
    result = diagnostics._run_generation_performance_suite(AsyncFakeGenerator())

    assert result["suite"] == "Generation Performance"
    assert result["metrics"]["success_rate"] == 1.0
    assert result["total"] >= 5


def test_generation_performance_suite_reports_missing_generator() -> None:
    result = diagnostics._run_generation_performance_suite(generator=None)

    assert result["suite"] == "Generation Performance"
    assert result["failed"] == 1
    assert result["tests"][0]["name"] == "Generator availability"


def test_generation_performance_suite_records_generation_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FailingGenerator:
        async def generate(self, *args, **kwargs):
            raise RuntimeError("simulated generation crash")

    monkeypatch.setattr(
        diagnostics,
        "_generation_performance_targets",
        lambda: {
            "case_target_ms": 35_000.0,
            "p95_target_ms": 40_000.0,
            "success_rate_target": 0.67,
            "timeout_seconds": 0.5,
        },
    )

    result = diagnostics._run_generation_performance_suite(FailingGenerator())

    generation_cases = [
        test
        for test in result["tests"]
        if str(test.get("name", "")).startswith("Generation latency case")
    ]
    assert len(generation_cases) == 3
    assert all(case["passed"] is False for case in generation_cases)
    assert all(
        "simulated generation crash" in case.get("error", "")
        for case in generation_cases
    )


def test_generation_performance_suite_times_out_hanging_cases_quickly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class HangingGenerator:
        async def generate(self, *args, **kwargs):
            await asyncio.sleep(0.2)

    monkeypatch.setattr(
        diagnostics,
        "_generation_performance_targets",
        lambda: {
            "case_target_ms": 35_000.0,
            "p95_target_ms": 40_000.0,
            "success_rate_target": 0.67,
            "timeout_seconds": 0.01,
        },
    )

    started = time.perf_counter()
    result = diagnostics._run_generation_performance_suite(HangingGenerator())
    elapsed_s = time.perf_counter() - started

    assert elapsed_s < 1.0
    assert result["suite"] == "Generation Performance"
    generation_cases = [
        test
        for test in result["tests"]
        if str(test.get("name", "")).startswith("Generation latency case")
    ]
    assert len(generation_cases) == 3
    assert all(case["details"].get("timed_out") is True for case in generation_cases)


def test_generation_performance_suite_reuses_single_event_loop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class LoopBoundGenerator:
        def __init__(self) -> None:
            self._bound_loop = None

        async def generate(self, *args, **kwargs):
            loop = asyncio.get_running_loop()
            if self._bound_loop is None:
                self._bound_loop = loop
            elif self._bound_loop is not loop:
                raise RuntimeError("generator is bound to a different event loop")
            await asyncio.sleep(0)
            return {"ok": True}

    monkeypatch.setattr(
        diagnostics,
        "_generation_performance_targets",
        lambda: {
            "case_target_ms": 35_000.0,
            "p95_target_ms": 40_000.0,
            "success_rate_target": 0.67,
            "timeout_seconds": 1.0,
        },
    )

    result = diagnostics._run_generation_performance_suite(LoopBoundGenerator())

    assert result["suite"] == "Generation Performance"
    generation_cases = [
        test
        for test in result["tests"]
        if str(test.get("name", "")).startswith("Generation latency case")
    ]
    assert len(generation_cases) == 3
    assert all(case.get("passed") is True for case in generation_cases)


def test_run_all_suites_records_suite_exceptions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    diagnostics._current_run = diagnostics._reset_current_run()

    monkeypatch.setattr(
        diagnostics,
        "_run_retrieval_performance_suite",
        lambda rag_pipeline: {
            "suite": "retrieval_performance",
            "total": 1,
            "passed": 1,
            "failed": 0,
            "tests": [],
            "metrics": {"p95_latency_ms": 20.0},
        },
    )
    monkeypatch.setattr(
        diagnostics,
        "_run_generation_performance_suite",
        lambda generator: {
            "suite": "generation_performance",
            "total": 1,
            "passed": 1,
            "failed": 0,
            "tests": [],
            "metrics": {"p95_latency_ms": 20.0, "success_rate": 1.0},
        },
    )
    monkeypatch.setattr(
        diagnostics,
        "_run_rag_retrieval_suite",
        lambda rag_pipeline: {
            "suite": "rag",
            "total": 1,
            "passed": 1,
            "failed": 0,
            "tests": [],
        },
    )
    monkeypatch.setattr(
        diagnostics,
        "_run_prompt_grounding_suite",
        lambda: {"suite": "prompt", "total": 1, "passed": 1, "failed": 0, "tests": []},
    )
    monkeypatch.setattr(
        diagnostics,
        "_run_generation_params_suite",
        lambda: {
            "suite": "generation",
            "total": 1,
            "passed": 1,
            "failed": 0,
            "tests": [],
        },
    )
    monkeypatch.setattr(
        diagnostics,
        "_run_cache_integrity_suite",
        lambda: (_ for _ in ()).throw(RuntimeError("cache suite boom")),
    )
    monkeypatch.setattr(
        diagnostics,
        "_run_metrics_suite",
        lambda: {"suite": "metrics", "total": 1, "passed": 1, "failed": 0, "tests": []},
    )

    diagnostics._run_all_suites(SimpleNamespace(), generator=AsyncFakeGenerator())

    assert diagnostics._current_run is not None
    cache_suite = diagnostics._current_run["suites"]["cache_integrity"]
    assert cache_suite["failed"] == 1
    assert cache_suite["errors"] == 1
    assert "cache suite boom" in cache_suite["tests"][0]["error"]
    assert diagnostics._current_run["summary"]["errors"] == 1


def test_generation_performance_targets_relax_for_stg(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        diagnostics,
        "get_settings",
        lambda: SimpleNamespace(distribution="stg"),
    )

    targets = diagnostics._generation_performance_targets()

    assert targets["case_target_ms"] == 35000.0
    assert targets["p95_target_ms"] == 40000.0
    assert targets["timeout_seconds"] == 40.0


def test_build_recommendations_respects_stg_generation_latency_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        diagnostics,
        "get_settings",
        lambda: SimpleNamespace(distribution="stg"),
    )

    recommendations = diagnostics._build_recommendations(
        {
            "summary": {"failed": 0},
            "suites": {
                "generation_performance": {
                    "metrics": {"success_rate": 1.0, "p95_latency_ms": 20000.0},
                    "tests": [],
                },
                "retrieval_performance": {"metrics": {"p95_latency_ms": 100.0}},
            },
        }
    )

    assert recommendations == [
        "Performance baseline is healthy. Keep current runtime profile and monitor regressions over time."
    ]


def test_build_recommendations_for_slow_metrics() -> None:
    recommendations = diagnostics._build_recommendations(
        {
            "summary": {"failed": 2},
            "suites": {
                "generation_performance": {
                    "metrics": {"success_rate": 0.2, "p95_latency_ms": 12000.0}
                },
                "retrieval_performance": {"metrics": {"p95_latency_ms": 1300.0}},
            },
        }
    )

    assert len(recommendations) >= 2


def test_format_generation_error_adds_llama_connectivity_hint() -> None:
    generator = SimpleNamespace(
        llm_client=SimpleNamespace(api_url="http://llama-server:8080/v1/completions")
    )

    message = diagnostics._format_generation_error(
        RuntimeError("All connection attempts failed"), generator
    )

    assert "Llama upstream unreachable" in message
    assert "http://llama-server:8080/v1/completions" in message


def test_build_recommendations_include_connectivity_actions() -> None:
    recommendations = diagnostics._build_recommendations(
        {
            "summary": {"failed": 1},
            "suites": {
                "generation_performance": {
                    "metrics": {"success_rate": 0.0, "p95_latency_ms": 0.0},
                    "tests": [
                        {
                            "name": "Generation latency case 1 (quiz)",
                            "passed": False,
                            "error": "All connection attempts failed",
                        }
                    ],
                },
                "retrieval_performance": {"metrics": {"p95_latency_ms": 100.0}},
            },
        }
    )

    assert any("cannot reach llama upstream" in item for item in recommendations)
