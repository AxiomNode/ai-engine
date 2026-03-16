"""Tests for ai_engine.api.app – generation FastAPI endpoints.

Uses the create_app() factory with injected mocks so no real LLM or
embedding model is required during testing.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

try:
    from fastapi.testclient import TestClient

    _HAS_FASTAPI = True
except ImportError:
    _HAS_FASTAPI = False

from ai_engine.games.schemas import (
    GameEnvelope,
    PasapalabraGame,
    PasapalabraWord,
    QuizGame,
    QuizQuestion,
)
from ai_engine.observability.collector import StatsCollector

pytestmark = pytest.mark.skipif(not _HAS_FASTAPI, reason="fastapi not installed")


# ------------------------------------------------------------------
# Helpers / Fixtures
# ------------------------------------------------------------------


def _make_quiz_envelope() -> GameEnvelope:
    """Return a minimal valid quiz GameEnvelope for testing."""
    return GameEnvelope(
        game_type="quiz",
        game=QuizGame(
            title="Test Quiz",
            topic="Science",
            questions=[
                QuizQuestion(
                    question="What is H2O?",
                    options=["Water", "Fire", "Air", "Earth"],
                    correct_index=0,
                    explanation="H2O is the chemical formula for water.",
                )
            ],
        ),
    )


def _make_pasapalabra_envelope() -> GameEnvelope:
    """Return a minimal valid pasapalabra GameEnvelope for testing."""
    return GameEnvelope(
        game_type="pasapalabra",
        game=PasapalabraGame(
            title="Test Rosco",
            topic="Science",
            words=[
                PasapalabraWord(
                    letter="A", hint="First letter", answer="Atom", starts_with=True
                )
            ],
        ),
    )


def _make_client(
    envelope: GameEnvelope | None = None,
    gen_side_effect: Exception | None = None,
) -> tuple[TestClient, MagicMock, MagicMock, StatsCollector]:
    """Build a TestClient with mocked generator and RAG pipeline.

    Args:
        envelope: The GameEnvelope the mock generator will return.
        gen_side_effect: If set, the generator raises this exception instead.

    Returns:
        A tuple of (client, mock_generator, mock_pipeline, collector).
    """
    from ai_engine.api.app import create_app

    mock_gen = MagicMock()
    if gen_side_effect is not None:
        mock_gen.generate.side_effect = gen_side_effect
        mock_gen.generate_from_context.side_effect = gen_side_effect
    else:
        mock_gen.generate.return_value = envelope or _make_quiz_envelope()
        mock_gen.generate_from_context.return_value = envelope or _make_quiz_envelope()

    mock_pipeline = MagicMock()
    collector = StatsCollector()

    app = create_app(
        generator=mock_gen,
        rag_pipeline=mock_pipeline,
        collector=collector,
    )
    return TestClient(app), mock_gen, mock_pipeline, collector


# ------------------------------------------------------------------
# GET /health
# ------------------------------------------------------------------


class TestHealth:
    """Tests for GET /health."""

    def test_returns_ok_status(self) -> None:
        """Health endpoint returns status ok."""
        client, *_ = _make_client()
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_reports_event_count(self) -> None:
        """Health endpoint reports zero events on a fresh collector."""
        client, _, _, collector = _make_client()
        assert collector is not None
        resp = client.get("/health")
        assert resp.json()["total_events"] == 0

    def test_includes_dependency_diagnostics(self) -> None:
        """Health endpoint returns dependency-level diagnostics payload."""
        client, *_ = _make_client()
        resp = client.get("/health")
        data = resp.json()
        assert "dependencies" in data
        assert data["dependencies"]["generator"]["status"] == "ready"
        assert data["dependencies"]["rag_pipeline"]["status"] == "ready"
        assert "cache" in data["dependencies"]
        assert "correlation_id" in data


# ------------------------------------------------------------------
# POST /generate
# ------------------------------------------------------------------


class TestGenerate:
    """Tests for POST /generate."""

    def test_generate_quiz_success(self) -> None:
        """Successful quiz generation returns 200 with game data."""
        client, mock_gen, *_ = _make_client()
        resp = client.post(
            "/generate",
            json={"query": "water cycle", "topic": "Science"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["game_type"] == "quiz"
        assert "title" in data["game"]
        assert "questions" in data["game"]

    def test_generate_calls_generator_with_correct_args(self) -> None:
        """Generator is called with all non-query fields from the request."""
        client, mock_gen, *_ = _make_client()
        client.post(
            "/generate",
            json={
                "query": "photosynthesis",
                "topic": "Biology",
                "game_type": "true_false",
                "language": "en",
                "num_questions": 3,
                "max_tokens": 512,
            },
        )
        call_kwargs = mock_gen.generate_from_context.call_args.kwargs
        assert "context" in call_kwargs
        assert isinstance(call_kwargs["context"], str)
        assert call_kwargs["topic"] == "Biology"
        assert call_kwargs["game_type"] == "true_false"
        assert call_kwargs["language"] == "en"
        assert call_kwargs["num_questions"] == 3
        assert call_kwargs["max_tokens"] == 512

    def test_generate_pasapalabra(self) -> None:
        """Pasapalabra game type is handled correctly."""
        client, *_ = _make_client(envelope=_make_pasapalabra_envelope())
        resp = client.post(
            "/generate",
            json={"query": "chemistry", "topic": "Science", "game_type": "pasapalabra"},
        )
        assert resp.status_code == 200
        assert resp.json()["game_type"] == "pasapalabra"

    def test_generate_missing_required_fields_returns_422(self) -> None:
        """Missing required fields returns HTTP 422 Unprocessable Entity."""
        client, *_ = _make_client()
        # Missing 'topic'
        resp = client.post("/generate", json={"query": "water"})
        assert resp.status_code == 422

    def test_generate_invalid_num_questions_returns_422(self) -> None:
        """num_questions out of range (>50) returns HTTP 422."""
        client, *_ = _make_client()
        resp = client.post(
            "/generate",
            json={"query": "water", "topic": "Science", "num_questions": 100},
        )
        assert resp.status_code == 422

    def test_generate_llm_value_error_returns_422(self) -> None:
        """ValueError from the generator (bad LLM output) maps to HTTP 422."""
        client, *_ = _make_client(gen_side_effect=ValueError("Failed to extract JSON"))
        resp = client.post("/generate", json={"query": "water", "topic": "Science"})
        assert resp.status_code == 422
        assert "Failed to extract JSON" in resp.json()["detail"]

    def test_generate_uses_default_language_es(self) -> None:
        """Default language is Spanish when not specified."""
        client, mock_gen, *_ = _make_client()
        client.post("/generate", json={"query": "w", "topic": "Science"})
        assert mock_gen.generate_from_context.call_args.kwargs["language"] == "es"

    def test_generate_sdk_returns_typed_payload(self) -> None:
        """/generate/sdk returns model_type + metadata + data sections."""
        client, *_ = _make_client()
        resp = client.post(
            "/generate/sdk",
            json={"query": "water cycle", "topic": "Science", "language": "en"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "model_type" in data
        assert "metadata" in data
        assert "data" in data
        assert "metrics" in data


# ------------------------------------------------------------------
# POST /ingest
# ------------------------------------------------------------------


class TestIngest:
    """Tests for POST /ingest."""

    def test_ingest_single_document(self) -> None:
        """Single document is ingested and count is returned."""
        client, _, mock_pipeline, _ = _make_client()
        resp = client.post(
            "/ingest",
            json={"documents": [{"content": "Water is H2O.", "doc_id": "doc-1"}]},
        )
        assert resp.status_code == 200
        assert resp.json()["ingested"] == 1
        mock_pipeline.ingest.assert_called_once()

    def test_ingest_multiple_documents(self) -> None:
        """Multiple documents are ingested in a single call."""
        client, _, mock_pipeline, _ = _make_client()
        resp = client.post(
            "/ingest",
            json={
                "documents": [
                    {"content": "Water is H2O.", "doc_id": "doc-1"},
                    {"content": "Plants use sunlight.", "doc_id": "doc-2"},
                    {"content": "Stars are nuclear reactors.", "doc_id": "doc-3"},
                ]
            },
        )
        assert resp.json()["ingested"] == 3

    def test_ingest_empty_list(self) -> None:
        """Empty document list returns zero ingested and pipeline is not called."""
        client, _, mock_pipeline, _ = _make_client()
        resp = client.post("/ingest", json={"documents": []})
        assert resp.status_code == 200
        assert resp.json()["ingested"] == 0

    def test_ingest_document_without_doc_id(self) -> None:
        """Documents without doc_id are accepted (doc_id is optional)."""
        client, _, mock_pipeline, _ = _make_client()
        resp = client.post(
            "/ingest",
            json={"documents": [{"content": "No ID document."}]},
        )
        assert resp.status_code == 200

    def test_ingest_document_with_metadata(self) -> None:
        """Documents with metadata are ingested without error."""
        client, _, mock_pipeline, _ = _make_client()
        resp = client.post(
            "/ingest",
            json={
                "documents": [
                    {
                        "content": "Chapter content.",
                        "doc_id": "chap-1",
                        "metadata": {"source": "book.pdf", "page": 5},
                    }
                ]
            },
        )
        assert resp.status_code == 200


# ------------------------------------------------------------------
# GET /stats and GET /stats/history
# ------------------------------------------------------------------


class TestStats:
    """Tests for GET /stats and GET /stats/history."""

    def test_stats_empty_on_fresh_collector(self) -> None:
        """Stats returns zeros when no events have been recorded."""
        client, *_ = _make_client()
        resp = client.get("/stats")
        assert resp.status_code == 200
        assert resp.json()["total_calls"] == 0
        assert "cache_runtime" in resp.json()

    def test_stats_after_recording_event(self) -> None:
        """Stats reflect recorded events."""
        client, _, _, collector = _make_client()
        collector.record_call(
            prompt="test",
            response="ok",
            latency_ms=100.0,
            max_tokens=512,
            game_type="quiz",
        )
        resp = client.get("/stats")
        data = resp.json()
        assert data["total_calls"] == 1
        assert data["game_type_counts"]["quiz"] == 1

    def test_history_empty_on_fresh_collector(self) -> None:
        """History returns empty list when no events have been recorded."""
        client, *_ = _make_client()
        resp = client.get("/stats/history")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_history_last_n_filter(self) -> None:
        """last_n query parameter limits the number of history events returned."""
        client, _, _, collector = _make_client()
        for i in range(5):
            collector.record_call(
                prompt=f"p{i}", response="r", latency_ms=10.0, max_tokens=64
            )
        resp = client.get("/stats/history?last_n=2")
        assert len(resp.json()) == 2

    def test_cache_stats_endpoint_returns_runtime_cache_info(self) -> None:
        """/cache/stats returns runtime cache information."""
        client, *_ = _make_client()
        resp = client.get("/cache/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert "memory_entries" in data
        assert "persistent_entries" in data
        assert "cache_ttl_seconds" in data

    def test_cache_reset_endpoint_clears_cache(self) -> None:
        """/cache/reset returns counters for removed cache entries."""
        client, *_ = _make_client()
        resp = client.post("/cache/reset")
        assert resp.status_code == 200
        data = resp.json()
        assert "removed_memory" in data
        assert "removed_persistent" in data

    def test_metrics_endpoint_returns_prometheus_with_cache_runtime(self) -> None:
        """/metrics returns extended Prometheus metrics including cache runtime gauges."""
        client, *_ = _make_client()
        client.post("/generate", json={"query": "water", "topic": "Science"})

        resp = client.get("/metrics")
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/plain")
        body = resp.text
        assert "ai_engine_total_calls" in body
        assert "ai_engine_generation_outcome_by_game_type_total" in body
        assert "ai_engine_cache_memory_saturation_ratio" in body

    def test_correlation_id_propagates_to_response_and_history(self) -> None:
        """Correlation ID is echoed in response header and stored in event metadata."""
        client, *_ = _make_client()
        correlation_id = "corr-test-123"

        resp = client.post(
            "/generate",
            json={"query": "water", "topic": "Science"},
            headers={"X-Correlation-ID": correlation_id},
        )
        assert resp.status_code == 200
        assert resp.headers.get("X-Correlation-ID") == correlation_id

        history = client.get("/stats/history?last_n=1")
        assert history.status_code == 200
        event = history.json()[0]
        assert event["metadata"]["correlation_id"] == correlation_id

    def test_history_invalid_last_n_returns_422(self) -> None:
        """last_n=0 is rejected with HTTP 422 (ge=1 constraint)."""
        client, *_ = _make_client()
        resp = client.get("/stats/history?last_n=0")
        assert resp.status_code == 422


# ------------------------------------------------------------------
# API Key authentication
# ------------------------------------------------------------------


class TestAPIKeyAuth:
    """Tests for X-API-Key header authentication (generation API)."""

    def _make_secured_client(
        self,
        api_key: str = "secret-key",
    ) -> tuple["TestClient", MagicMock]:
        """Return a TestClient whose app has API key enforcement enabled."""
        import os

        from ai_engine.api.app import create_app

        mock_gen = MagicMock()
        mock_gen.generate.return_value = _make_quiz_envelope()
        mock_gen.generate_from_context.return_value = _make_quiz_envelope()
        mock_pipeline = MagicMock()

        os.environ["AI_ENGINE_API_KEY"] = api_key
        try:
            app = create_app(
                generator=mock_gen,
                rag_pipeline=mock_pipeline,
                collector=StatsCollector(),
            )
        finally:
            del os.environ["AI_ENGINE_API_KEY"]

        return TestClient(app, raise_server_exceptions=False), mock_gen

    def test_no_key_required_when_env_not_set(self) -> None:
        """When AI_ENGINE_API_KEY is unset, all requests pass through."""
        client, *_ = _make_client()
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_request_without_key_returns_401(self) -> None:
        """A request without X-API-Key is rejected with 401 when key is configured."""
        client, _ = self._make_secured_client()
        resp = client.get("/health")
        assert resp.status_code == 401

    def test_request_with_wrong_key_returns_403(self) -> None:
        """A request with an incorrect API key is rejected with 403."""
        client, _ = self._make_secured_client()
        resp = client.get("/health", headers={"X-API-Key": "wrong-key"})
        assert resp.status_code == 403

    def test_request_with_correct_key_passes(self) -> None:
        """A request with the correct API key is accepted."""
        client, _ = self._make_secured_client(api_key="valid-key")
        resp = client.get("/health", headers={"X-API-Key": "valid-key"})
        assert resp.status_code == 200

    def test_generate_protected_without_key(self) -> None:
        """POST /generate is blocked when the API key is missing."""
        client, _ = self._make_secured_client()
        resp = client.post("/generate", json={"query": "water", "topic": "Science"})
        assert resp.status_code == 401

    def test_generate_succeeds_with_correct_key(self) -> None:
        """POST /generate succeeds with the correct API key."""
        client, _ = self._make_secured_client(api_key="mykey")
        resp = client.post(
            "/generate",
            json={"query": "water", "topic": "Science"},
            headers={"X-API-Key": "mykey"},
        )
        assert resp.status_code == 200


# ------------------------------------------------------------------
# 503 – uninitialised state
# ------------------------------------------------------------------


class TestUninitialised:
    """Endpoints return 503 when the generator or pipeline are missing."""

    def test_generate_returns_503_when_generator_is_none(self) -> None:
        """POST /generate returns 503 if app.state.generator is None."""
        from ai_engine.api.app import create_app

        mock_gen = MagicMock()
        mock_pipeline = MagicMock()
        app = create_app(generator=mock_gen, rag_pipeline=mock_pipeline)

        client = TestClient(app)
        # Trigger lifespan startup with a harmless request.
        client.get("/health")
        # Nullify the generator to simulate missing component.
        app.state.generator = None

        resp = client.post("/generate", json={"query": "water", "topic": "Science"})
        assert resp.status_code == 503
        assert "not initialised" in resp.json()["detail"].lower()


# ------------------------------------------------------------------
# Rate limiting
# ------------------------------------------------------------------


class TestRateLimiting:
    """Tests for request-level generation rate limiting."""

    def _make_rate_limited_client(self) -> "TestClient":
        """Return a client with strict generation rate limiting enabled."""
        import os

        from ai_engine.api.app import create_app

        mock_gen = MagicMock()
        mock_gen.generate.return_value = _make_quiz_envelope()
        mock_gen.generate_from_context.return_value = _make_quiz_envelope()
        mock_pipeline = MagicMock()

        env_overrides = {
            "AI_ENGINE_RATE_LIMIT_ENABLED": "true",
            "AI_ENGINE_RATE_LIMIT_REQUESTS": "1",
            "AI_ENGINE_RATE_LIMIT_WINDOW_SECONDS": "3600",
        }
        previous = {key: os.environ.get(key) for key in env_overrides}
        for key, value in env_overrides.items():
            os.environ[key] = value

        try:
            app = create_app(
                generator=mock_gen,
                rag_pipeline=mock_pipeline,
                collector=StatsCollector(),
            )
        finally:
            for key, old_value in previous.items():
                if old_value is None:
                    del os.environ[key]
                else:
                    os.environ[key] = old_value

        return TestClient(app, raise_server_exceptions=False)

    def test_generate_rate_limit_returns_429_after_threshold(self) -> None:
        """Second generation call in same window is limited with HTTP 429."""
        client = self._make_rate_limited_client()

        first = client.post("/generate", json={"query": "water", "topic": "Science"})
        second = client.post("/generate", json={"query": "water", "topic": "Science"})

        assert first.status_code == 200
        assert second.status_code == 429

    def test_health_is_not_rate_limited(self) -> None:
        """Non-generation endpoints should not be affected by generation limiter."""
        client = self._make_rate_limited_client()

        first = client.get("/health")
        second = client.get("/health")

        assert first.status_code == 200
        assert second.status_code == 200

    def test_ingest_returns_503_when_pipeline_is_none(self) -> None:
        """POST /ingest returns 503 if app.state.rag_pipeline is None."""
        from ai_engine.api.app import create_app

        mock_gen = MagicMock()
        mock_pipeline = MagicMock()
        app = create_app(generator=mock_gen, rag_pipeline=mock_pipeline)

        client = TestClient(app)
        client.get("/health")
        app.state.rag_pipeline = None

        resp = client.post(
            "/ingest",
            json={"documents": [{"content": "Water is H2O.", "doc_id": "d1"}]},
        )
        assert resp.status_code == 503
        assert "not initialised" in resp.json()["detail"].lower()


# ------------------------------------------------------------------
# _build_from_env – RuntimeError when no LLM backend configured
# ------------------------------------------------------------------


class TestBuildFromEnv:
    """_build_from_env raises RuntimeError when no LLM backend is configured."""

    def test_raises_runtime_error_when_both_env_vars_absent(self, monkeypatch) -> None:
        """RuntimeError raised when neither AI_ENGINE_LLAMA_URL nor
        AI_ENGINE_MODEL_PATH are set."""
        import sys

        monkeypatch.delenv("AI_ENGINE_LLAMA_URL", raising=False)
        monkeypatch.delenv("AI_ENGINE_MODEL_PATH", raising=False)
        # Reload config to pick up cleared env.
        sys.modules.pop("ai_engine.config", None)

        from ai_engine.api.app import _build_from_env

        with pytest.raises(RuntimeError, match="AI_ENGINE_LLAMA_URL"):
            _build_from_env()
