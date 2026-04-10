"""Tests for ai_engine.api.app – generation FastAPI endpoints.

Uses the create_app() factory with injected mocks so no real LLM or
embedding model is required during testing.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

try:
    from fastapi.testclient import TestClient

    _HAS_FASTAPI = True
except ImportError:
    _HAS_FASTAPI = False

from ai_engine.games.schemas import (
    GameEnvelope,
    WordPassGame,
    WordPassWord,
    QuizGame,
    QuizQuestion,
)
from ai_engine.observability.collector import StatsCollector

pytestmark = pytest.mark.skipif(not _HAS_FASTAPI, reason="fastapi not installed")


def _clear_all_settings_caches() -> None:
    """Clear **all** get_settings LRU caches across duplicate module imports."""
    import sys

    seen_ids: set[int] = set()
    for _name, mod in list(sys.modules.items()):
        try:
            gs = getattr(mod, "get_settings", None)
        except Exception:
            continue
        if gs is None:
            continue
        try:
            has_clear = hasattr(gs, "cache_clear")
        except Exception:
            continue
        if has_clear and id(gs) not in seen_ids:
            seen_ids.add(id(gs))
            gs.cache_clear()


# ------------------------------------------------------------------
# Helpers / Fixtures
# ------------------------------------------------------------------


def _make_quiz_envelope() -> GameEnvelope:
    """Return a minimal valid quiz GameEnvelope for testing."""
    return GameEnvelope(
        game_type="quiz",
        game=QuizGame(
            title="Test Quiz",
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


def _make_word_pass_envelope() -> GameEnvelope:
    """Return a minimal valid word-pass GameEnvelope for testing."""
    return GameEnvelope(
        game_type="word-pass",
        game=WordPassGame(
            title="Test Rosco",
            words=[
                WordPassWord(
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
        mock_gen.generate = AsyncMock(side_effect=gen_side_effect)
        mock_gen.generate_from_context = AsyncMock(side_effect=gen_side_effect)
    else:
        mock_gen.generate = AsyncMock(return_value=envelope or _make_quiz_envelope())
        mock_gen.generate_from_context = AsyncMock(return_value=envelope or _make_quiz_envelope())

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
        assert resp.json()["startup"]["status"] == "ready"
        assert resp.headers.get("X-Distribution-Version")

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
        assert "distribution_version" in data


class TestReadiness:
    """Tests for GET /ready."""

    def test_returns_ready_when_dependencies_are_injected(self) -> None:
        """Readiness returns 200 once the app bootstrap has completed."""
        client, *_ = _make_client()
        resp = client.get("/ready")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ready"
        assert resp.json()["startup"]["status"] == "ready"

    def test_returns_not_ready_when_startup_is_in_progress(self) -> None:
        """Readiness returns 503 while async bootstrap is still in progress."""
        client, *_ = _make_client()
        client.app.state.startup_status = "initializing"
        resp = client.get("/ready")
        assert resp.status_code == 503
        assert resp.json()["status"] == "not_ready"
        assert resp.json()["startup"]["status"] == "initializing"


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
            json={"query": "water cycle"},
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
                "game_type": "true_false",
                "language": "en",
                "num_questions": 3,
                "max_tokens": 512,
            },
        )
        call_kwargs = mock_gen.generate_from_context.call_args.kwargs
        assert "context" in call_kwargs
        assert isinstance(call_kwargs["context"], str)
        assert call_kwargs["game_type"] == "true_false"
        assert call_kwargs["language"] == "en"
        assert call_kwargs["num_questions"] == 3
        assert call_kwargs["max_tokens"] == 512

    def test_generate_word_pass(self) -> None:
        """WordPass game type is handled correctly."""
        client, *_ = _make_client(envelope=_make_word_pass_envelope())
        resp = client.post(
            "/generate",
            json={"query": "chemistry", "game_type": "word-pass"},
        )
        assert resp.status_code == 200
        assert resp.json()["game_type"] == "word-pass"

    def test_generate_missing_required_fields_returns_422(self) -> None:
        """Missing required fields returns HTTP 422 Unprocessable Entity."""
        client, *_ = _make_client()
        # Missing 'query'
        resp = client.post("/generate", json={"game_type": "quiz"})
        assert resp.status_code == 422

    def test_generate_invalid_num_questions_returns_422(self) -> None:
        """num_questions out of range (>50) returns HTTP 422."""
        client, *_ = _make_client()
        resp = client.post(
            "/generate",
            json={"query": "water", "num_questions": 100},
        )
        assert resp.status_code == 422

    def test_generate_llm_value_error_returns_422(self) -> None:
        """ValueError from the generator (bad LLM output) maps to HTTP 422."""
        client, *_ = _make_client(gen_side_effect=ValueError("Failed to extract JSON"))
        resp = client.post("/generate", json={"query": "water"})
        assert resp.status_code == 422
        assert "Failed to extract JSON" in resp.json()["detail"]

    def test_generate_uses_default_language_es(self) -> None:
        """Default language is Spanish when not specified."""
        client, mock_gen, *_ = _make_client()
        client.post("/generate", json={"query": "w"})
        assert mock_gen.generate_from_context.call_args.kwargs["language"] == "es"

    def test_generate_sdk_returns_typed_payload(self) -> None:
        """/generate/sdk returns model_type + metadata + data sections."""
        client, *_ = _make_client()
        resp = client.post(
            "/generate/sdk",
            json={"query": "water cycle", "language": "en"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "model_type" in data
        assert "metadata" in data
        assert "data" in data
        assert "metrics" in data

    def test_generate_applies_language_and_difficulty_headers(self) -> None:
        """Header overrides must be propagated to the generator call."""
        client, mock_gen, *_ = _make_client()
        resp = client.post(
            "/generate",
            json={"query": "water cycle", "language": "es"},
            headers={
                "X-Game-Language": "en",
                "X-Difficulty-Percentage": "80",
            },
        )
        assert resp.status_code == 200
        call_kwargs = mock_gen.generate_from_context.call_args.kwargs
        assert call_kwargs["language"] == "en"
        assert call_kwargs["difficulty_percentage"] == 80

    def test_generate_quiz_model_specific_endpoint(self) -> None:
        """/generate/quiz should build a quiz request from query params + headers."""
        client, mock_gen, *_ = _make_client()
        resp = client.post(
            "/generate/quiz",
            params={"query": "water cycle", "num_questions": 4},
            headers={"X-Game-Language": "en", "X-Difficulty-Percentage": "65"},
        )
        assert resp.status_code == 200
        call_kwargs = mock_gen.generate_from_context.call_args.kwargs
        assert call_kwargs["game_type"] == "quiz"
        assert call_kwargs["num_questions"] == 4
        assert call_kwargs["language"] == "en"
        assert call_kwargs["difficulty_percentage"] == 65

    def test_generate_word_pass_model_specific_endpoint(self) -> None:
        """/generate/word-pass should pass letters and force word-pass game type."""
        client, mock_gen, *_ = _make_client(envelope=_make_word_pass_envelope())
        resp = client.post(
            "/generate/word-pass",
            params={"query": "chemistry", "letters": "A,B,C"},
        )
        assert resp.status_code == 200
        call_kwargs = mock_gen.generate_from_context.call_args.kwargs
        assert call_kwargs["game_type"] == "word-pass"
        assert call_kwargs["letters"] == "A,B,C"


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

    def test_ingest_supports_source_header_override(self) -> None:
        """Header source should override query source for ingest telemetry metadata."""
        client, _, _, collector = _make_client()
        resp = client.post(
            "/ingest",
            params={"source": "query-source"},
            headers={"X-Ingest-Source": "header-source"},
            json={"documents": [{"content": "Water is H2O.", "doc_id": "doc-1"}]},
        )
        assert resp.status_code == 200
        events = collector.history(last_n=1)
        assert events[-1]["metadata"].get("ingest_source") == "header-source"

    def test_ingest_model_specific_endpoint(self) -> None:
        """/ingest/quiz should ingest content and tag ingest_model in metadata."""
        client, _, _, collector = _make_client()
        resp = client.post(
            "/ingest/quiz",
            json={"documents": [{"content": "Quiz source document."}]},
        )
        assert resp.status_code == 200
        events = collector.history(last_n=1)
        assert events[-1]["metadata"].get("ingest_model") == "quiz"


# ------------------------------------------------------------------
# Monitoring endpoints moved to ai-stats
# ------------------------------------------------------------------


class TestMonitoringOwnership:
    """Tests that monitoring endpoints are not exposed by ai-api."""

    def test_stats_endpoint_not_exposed_in_main_api(self) -> None:
        """GET /stats must be served by ai-stats, not ai-api."""
        client, *_ = _make_client()
        resp = client.get("/stats")
        assert resp.status_code == 404

    def test_history_endpoint_not_exposed_in_main_api(self) -> None:
        """GET /stats/history must be served by ai-stats, not ai-api."""
        client, *_ = _make_client()
        resp = client.get("/stats/history")
        assert resp.status_code == 404

    def test_cache_endpoints_not_exposed_in_main_api(self) -> None:
        """Cache runtime endpoints are removed from ai-api."""
        client, *_ = _make_client()
        stats_resp = client.get("/cache/stats")
        reset_resp = client.post("/cache/reset")
        assert stats_resp.status_code == 404
        assert reset_resp.status_code == 404

    def test_metrics_endpoint_not_exposed_in_main_api(self) -> None:
        """Prometheus endpoint must be served by ai-stats only."""
        client, *_ = _make_client()
        resp = client.get("/metrics")
        assert resp.status_code == 404

    def test_correlation_id_is_stored_in_internal_event_metadata(self) -> None:
        """Correlation ID is echoed in response and stored in collector metadata."""
        client, _, _, collector = _make_client()
        correlation_id = "corr-test-123"

        resp = client.post(
            "/generate",
            json={"query": "water"},
            headers={"X-Correlation-ID": correlation_id},
        )
        assert resp.status_code == 200
        assert resp.headers.get("X-Correlation-ID") == correlation_id

        history = collector.history(last_n=1)
        assert len(history) == 1
        assert history[0]["metadata"]["correlation_id"] == correlation_id


# ------------------------------------------------------------------
# API Key authentication
# ------------------------------------------------------------------


class TestAPIKeyAuth:
    """Tests for X-API-Key header authentication (generation API)."""

    def _make_secured_client(
        self,
        games_api_key: str = "games-secret",
        bridge_api_key: str = "bridge-secret",
    ) -> tuple["TestClient", MagicMock]:
        """Return a TestClient whose app has API key enforcement enabled."""
        import os

        from ai_engine.api.app import create_app

        mock_gen = MagicMock()
        mock_gen.generate = AsyncMock(return_value=_make_quiz_envelope())
        mock_gen.generate_from_context = AsyncMock(return_value=_make_quiz_envelope())
        mock_pipeline = MagicMock()

        os.environ["AI_ENGINE_GAMES_API_KEY"] = games_api_key
        os.environ["AI_ENGINE_BRIDGE_API_KEY"] = bridge_api_key
        os.environ["AI_ENGINE_GENERATION_API_URL"] = ""
        _clear_all_settings_caches()
        try:
            app = create_app(
                generator=mock_gen,
                rag_pipeline=mock_pipeline,
                collector=StatsCollector(),
            )
            app.state.generation_api_url = ""
        finally:
            del os.environ["AI_ENGINE_GAMES_API_KEY"]
            del os.environ["AI_ENGINE_BRIDGE_API_KEY"]
            del os.environ["AI_ENGINE_GENERATION_API_URL"]
            _clear_all_settings_caches()

        return TestClient(app, raise_server_exceptions=False), mock_gen

    def test_no_key_required_when_env_not_set(self) -> None:
        """When AI_ENGINE_API_KEY is unset, all requests pass through."""
        client, *_ = _make_client()
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_is_public_even_with_keys(self) -> None:
        """Health remains public to keep container liveness probes working."""
        client, _ = self._make_secured_client()
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_generate_wrong_key_returns_403(self) -> None:
        """Generation endpoints reject an invalid key when games key is set."""
        client, _ = self._make_secured_client()
        resp = client.post(
            "/generate",
            json={"query": "water"},
            headers={"X-API-Key": "wrong-key"},
        )
        assert resp.status_code == 403

    def test_generate_requires_games_key(self) -> None:
        """Generation endpoints require the games key."""
        client, _ = self._make_secured_client(games_api_key="games-key")
        resp = client.post(
            "/generate",
            json={"query": "water"},
            headers={"X-API-Key": "games-key"},
        )
        assert resp.status_code == 200

    def test_generate_protected_without_key(self) -> None:
        """POST /generate is blocked when the API key is missing."""
        client, _ = self._make_secured_client()
        resp = client.post("/generate", json={"query": "water"})
        assert resp.status_code == 401

    def test_generate_succeeds_with_correct_key(self) -> None:
        """POST /generate succeeds with the configured games API key."""
        client, _ = self._make_secured_client(games_api_key="mykey")
        resp = client.post(
            "/generate",
            json={"query": "water"},
            headers={"X-API-Key": "mykey"},
        )
        assert resp.status_code == 200

    def test_ingest_requires_bridge_key(self) -> None:
        """Ingest endpoints require the bridge API key."""
        client, _ = self._make_secured_client(
            games_api_key="games-key",
            bridge_api_key="bridge-key",
        )
        wrong = client.post(
            "/ingest",
            json={"documents": [{"content": "Water is H2O."}]},
            headers={"X-API-Key": "games-key"},
        )
        ok = client.post(
            "/ingest",
            json={"documents": [{"content": "Water is H2O."}]},
            headers={"X-API-Key": "bridge-key"},
        )
        assert wrong.status_code == 403
        assert ok.status_code == 200


# ------------------------------------------------------------------
# 503 – uninitialised state
# ------------------------------------------------------------------


class TestUninitialised:
    """Endpoints return 503 when the generator or pipeline are missing."""

    def test_generate_returns_503_when_generator_is_none(self) -> None:
        """POST /generate returns 503 if app.state.generator is None."""
        from ai_engine.api.app import create_app

        mock_gen = MagicMock()
        mock_gen.generate = AsyncMock(return_value=_make_quiz_envelope())
        mock_gen.generate_from_context = AsyncMock(return_value=_make_quiz_envelope())
        mock_pipeline = MagicMock()
        app = create_app(generator=mock_gen, rag_pipeline=mock_pipeline)

        client = TestClient(app)
        # Trigger lifespan startup with a harmless request.
        client.get("/health")
        # Nullify the generator to simulate missing component.
        app.state.generator = None

        resp = client.post("/generate", json={"query": "water"})
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
        mock_gen.generate = AsyncMock(return_value=_make_quiz_envelope())
        mock_gen.generate_from_context = AsyncMock(return_value=_make_quiz_envelope())
        mock_pipeline = MagicMock()

        env_overrides = {
            "AI_ENGINE_RATE_LIMIT_ENABLED": "true",
            "AI_ENGINE_RATE_LIMIT_REQUESTS": "1",
            "AI_ENGINE_RATE_LIMIT_WINDOW_SECONDS": "3600",
        }
        previous = {key: os.environ.get(key) for key in env_overrides}
        for key, value in env_overrides.items():
            os.environ[key] = value
        _clear_all_settings_caches()

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
            _clear_all_settings_caches()

        return TestClient(app, raise_server_exceptions=False)

    def test_generate_rate_limit_returns_429_after_threshold(self) -> None:
        """Second generation call in same window is limited with HTTP 429."""
        client = self._make_rate_limited_client()

        first = client.post("/generate", json={"query": "water"})
        second = client.post("/generate", json={"query": "water"})

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
        mock_gen.generate = AsyncMock(return_value=_make_quiz_envelope())
        mock_gen.generate_from_context = AsyncMock(return_value=_make_quiz_envelope())
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
