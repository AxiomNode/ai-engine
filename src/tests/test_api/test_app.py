"""Tests for ai_engine.api.app – generation FastAPI endpoints.

Uses the create_app() factory with injected mocks so no real LLM or
embedding model is required during testing.
"""

from __future__ import annotations

import asyncio
import sys
import types
import uuid
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

try:
    from fastapi.testclient import TestClient

    _HAS_FASTAPI = True
except ImportError:
    _HAS_FASTAPI = False

from ai_engine.games.schemas import (
    GameEnvelope,
    QuizGame,
    QuizQuestion,
    WordPassGame,
    WordPassWord,
)
from ai_engine.observability.collector import StatsCollector

pytestmark = pytest.mark.skipif(not _HAS_FASTAPI, reason="fastapi not installed")


# ------------------------------------------------------------------
# Helpers / Fixtures
# ------------------------------------------------------------------


def _clear_all_settings_caches() -> None:
    """Clear all get_settings LRU caches across duplicate module imports."""
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
    *,
    raise_server_exceptions: bool = True,
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
        mock_gen.generate_from_context = AsyncMock(
            return_value=envelope or _make_quiz_envelope()
        )

    mock_pipeline = MagicMock()
    collector = StatsCollector()

    app = create_app(
        generator=mock_gen,
        rag_pipeline=mock_pipeline,
        collector=collector,
    )
    return (
        TestClient(app, raise_server_exceptions=raise_server_exceptions),
        mock_gen,
        mock_pipeline,
        collector,
    )


def test_llama_url_helpers_and_target_payload() -> None:
    from ai_engine.api.app import (
        _build_llama_url,
        _get_llama_target_payload,
        _normalize_llama_host,
        _parse_llama_url,
    )

    assert _normalize_llama_host(" https://llama.local:7002/v1 ") == "llama.local"
    with pytest.raises(ValueError, match="valid hostname"):
        _normalize_llama_host("bad host!")

    assert (
        _build_llama_url("https", "llama.local", 8443)
        == "https://llama.local:8443/v1/completions"
    )
    assert _parse_llama_url("https://llama.local/v1/completions") == (
        "llama.local",
        "https",
        443,
    )
    assert _parse_llama_url("not a url") == (None, None, None)

    app = MagicMock()
    app.state = types.SimpleNamespace(
        llama_url="http://llama.local:7002/v1/completions",
        llama_env_url="http://env.local:7002/v1/completions",
        llama_target_override=types.SimpleNamespace(
            label="desk",
            updated_at="2026-04-22T12:00:00Z",
        ),
    )

    assert _get_llama_target_payload(app) == {
        "source": "override",
        "label": "desk",
        "host": "llama.local",
        "protocol": "http",
        "port": 7002,
        "llamaBaseUrl": "http://llama.local:7002/v1/completions",
        "envLlamaBaseUrl": "http://env.local:7002/v1/completions",
        "updatedAt": "2026-04-22T12:00:00Z",
    }


def test_apply_runtime_llama_url_updates_state_and_validates_runtime_support() -> None:
    from ai_engine.api.app import _apply_runtime_llama_url

    mutable_client = types.SimpleNamespace(set_api_url=AsyncMock())
    app = MagicMock()
    app.state = types.SimpleNamespace(
        generator=types.SimpleNamespace(
            _generator=types.SimpleNamespace(llm_client=mutable_client)
        ),
        llama_url=None,
    )

    asyncio.run(_apply_runtime_llama_url(app, "http://llama.local:7002/v1/completions"))

    mutable_client.set_api_url.assert_awaited_once_with(
        "http://llama.local:7002/v1/completions"
    )
    assert app.state.llama_url == "http://llama.local:7002/v1/completions"

    app.state.generator = object()
    with pytest.raises(RuntimeError, match="unavailable"):
        asyncio.run(_apply_runtime_llama_url(app, None))


def test_generation_request_helpers_normalize_metadata_and_headers() -> None:
    from ai_engine.api.app import (
        _apply_generate_headers,
        _build_model_generate_request,
        _generation_failure_metadata,
        _resolve_category_name,
        _resolve_effective_max_tokens,
    )
    from ai_engine.api.schemas import GenerateRequest

    req = GenerateRequest(query="biology", game_type="quiz", max_tokens=512)
    updated = _apply_generate_headers(
        req,
        language_header=" EN ",
        difficulty_header=140,
    )

    assert updated.language == "en"
    assert updated.difficulty_percentage == 100
    assert _resolve_category_name("17", "Custom") == "Science & Nature"
    assert _resolve_category_name(None, "  Custom  ") == "Custom"

    built = _build_model_generate_request(
        game_type="word-pass",
        query_text=None,
        language_header="ES",
        difficulty_header=-5,
        item_count=2,
        category_id="17",
        category_name="ignored",
        letters="A,B",
        max_tokens=256,
        use_cache=False,
        force_refresh=True,
    )
    assert built.category_name == "Science & Nature"
    assert built.language == "es"
    assert built.difficulty_percentage == 0
    assert built.force_refresh is True

    metadata = _generation_failure_metadata(
        built,
        correlation_id="corr-1",
        distribution_version="dev-v1",
        effective_max_tokens=_resolve_effective_max_tokens(built),
        extra_metadata={"upstream_service": "llama"},
    )
    assert metadata["correlation_id"] == "corr-1"
    assert metadata["distribution_version"] == "dev-v1"
    assert metadata["upstream_service"] == "llama"
    assert metadata["effective_max_tokens"] >= 0


def test_rate_limit_and_request_identity_helpers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ai_engine.api.app import (
        _FixedWindowRateLimiter,
        _GenerationCapacityLimiter,
        _get_correlation_id,
        _resolve_generation_caller_tier,
        _resolve_rate_limit_identity,
    )

    limiter = _FixedWindowRateLimiter(max_requests=2, window_seconds=10)
    time_values = iter([0.0, 1.0, 2.0, 20.0])
    monkeypatch.setattr("ai_engine.api.app.time.time", lambda: next(time_values))

    assert limiter.allow("client-1") is True
    assert limiter.allow("client-1") is True
    assert limiter.allow("client-1") is False
    assert limiter.allow("client-1") is True

    request = MagicMock()
    request.headers = {"X-API-Key": "secret", "X-Correlation-ID": "corr-header"}
    request.client = types.SimpleNamespace(host="127.0.0.1")
    request.state = types.SimpleNamespace(auth_scope="games", correlation_id="")

    assert _resolve_generation_caller_tier(request) == "background"
    assert _resolve_rate_limit_identity(request) == "api_key:secret"
    assert _get_correlation_id(request) == "corr-header"

    request.headers = {}
    request.state = types.SimpleNamespace(auth_scope="api", correlation_id="corr-state")
    assert _resolve_generation_caller_tier(request) == "interactive"
    assert _resolve_rate_limit_identity(request) == "ip:127.0.0.1"
    assert _get_correlation_id(request) == "corr-state"

    monkeypatch.setattr(
        uuid, "uuid4", lambda: types.SimpleNamespace(hex="generated-corr")
    )
    request.state = types.SimpleNamespace(auth_scope="api", correlation_id="")
    request.client = None
    assert _get_correlation_id(request) == "generated-corr"

    capacity = _GenerationCapacityLimiter(max_in_flight=1, max_queue_size=1)
    assert capacity.stats()["max_in_flight"] == 1


def test_warmup_cache_tracks_generated_cached_and_failed_paths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ai_engine.api.app as app_module

    class _Optimizer:
        def __init__(self) -> None:
            self.calls: list[tuple[str, str, str]] = []

        async def generate(self, req, correlation_id: str | None = None):
            self.calls.append((req.game_type, req.language, req.category_name))
            if len(self.calls) == 1:
                return types.SimpleNamespace(metrics={"cache_hit": False})
            if len(self.calls) == 2:
                return types.SimpleNamespace(metrics={"cache_hit": True})
            raise RuntimeError("boom")

    sleep_calls: list[float] = []
    monkeypatch.setattr(app_module, "_WARMUP_GAME_TYPES", ["quiz"])
    monkeypatch.setattr(app_module, "_WARMUP_LANGUAGES", ["es"])
    monkeypatch.setattr(
        app_module,
        "_WARMUP_CATEGORIES",
        [("17", "Science"), ("23", "History"), ("27", "Animals")],
    )
    monkeypatch.setattr(
        app_module.asyncio,
        "sleep",
        AsyncMock(side_effect=lambda delay: sleep_calls.append(delay)),
    )

    optimizer = _Optimizer()
    asyncio.run(app_module._warmup_cache(optimizer))

    assert optimizer.calls == [
        ("quiz", "es", "Science"),
        ("quiz", "es", "History"),
        ("quiz", "es", "Animals"),
    ]
    assert sleep_calls == [0.5, 0.5, 0.5]


def test_prime_runtime_content_ingests_examples_and_runs_optional_warmup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ai_engine.api.app as app_module

    docs = [
        types.SimpleNamespace(content="water"),
        types.SimpleNamespace(content="plants"),
    ]
    rag_pipeline = MagicMock()
    optimizer = MagicMock()
    warmup_calls: list[object] = []

    class _FakeExampleInjector:
        @staticmethod
        def _corpus_to_documents(corpus):
            assert corpus == ["corpus"]
            return docs

    async def fake_to_thread(func, *args, **kwargs):
        return func(*args, **kwargs)

    async def fake_warmup(passed_optimizer):
        warmup_calls.append(passed_optimizer)

    monkeypatch.setitem(
        sys.modules,
        "ai_engine.examples",
        types.SimpleNamespace(ExampleInjector=_FakeExampleInjector),
    )
    monkeypatch.setattr(app_module, "get_full_corpus", lambda: ["corpus"])
    monkeypatch.setattr(app_module.asyncio, "to_thread", fake_to_thread)
    monkeypatch.setattr(app_module, "_warmup_cache", fake_warmup)

    asyncio.run(
        app_module._prime_runtime_content(
            rag_pipeline,
            optimizer,
            cache_warmup_enabled=True,
        )
    )

    rag_pipeline.ingest.assert_called_once_with(docs)
    optimizer.on_ingest.assert_called_once_with(docs)
    assert warmup_calls == [optimizer]


def test_prime_runtime_content_stops_after_bootstrap_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ai_engine.api.app as app_module

    rag_pipeline = MagicMock()
    optimizer = MagicMock()
    warmup_calls: list[object] = []

    class _BrokenExampleInjector:
        @staticmethod
        def _corpus_to_documents(corpus):
            raise RuntimeError("inject failed")

    async def fake_warmup(passed_optimizer):
        warmup_calls.append(passed_optimizer)

    monkeypatch.setitem(
        sys.modules,
        "ai_engine.examples",
        types.SimpleNamespace(ExampleInjector=_BrokenExampleInjector),
    )
    monkeypatch.setattr(app_module, "get_full_corpus", lambda: ["corpus"])
    monkeypatch.setattr(app_module, "_warmup_cache", fake_warmup)

    asyncio.run(
        app_module._prime_runtime_content(
            rag_pipeline,
            optimizer,
            cache_warmup_enabled=True,
        )
    )

    rag_pipeline.ingest.assert_not_called()
    optimizer.on_ingest.assert_not_called()
    assert warmup_calls == []


def test_install_api_key_openapi_marks_only_protected_routes() -> None:
    from fastapi import FastAPI

    from ai_engine.api.app import _install_api_key_openapi

    app = FastAPI(title="demo", version="1.0.0")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/generate")
    def generate() -> dict[str, str]:
        return {"status": "ok"}

    _install_api_key_openapi(app, public_paths={"/health"})

    schema = app.openapi()

    assert schema["components"]["securitySchemes"]["ApiKeyAuth"]["name"] == "X-API-Key"
    assert "security" not in schema["paths"]["/health"]["get"]
    assert schema["paths"]["/generate"]["post"]["security"] == [{"ApiKeyAuth": []}]
    assert app.openapi() is schema


def test_publish_and_record_observability_event_bridge(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ai_engine.api.app as app_module

    posted: list[dict[str, object]] = []

    class _FakeAsyncClient:
        def __init__(self, timeout: float) -> None:
            assert timeout == 2.0

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb) -> None:
            return None

        async def post(
            self, url: str, json: dict[str, object], headers: dict[str, str]
        ) -> None:
            posted.append({"url": url, "json": json, "headers": headers})

    monkeypatch.setattr(app_module.httpx, "AsyncClient", _FakeAsyncClient)

    request = MagicMock()
    request.app.state = types.SimpleNamespace(
        stats_url="http://stats.local",
        stats_api_key="stats-secret",
        collector=StatsCollector(),
    )

    asyncio.run(
        app_module._record_observability_event(
            request,
            prompt="prompt",
            response="response",
            latency_ms=12.5,
            max_tokens=256,
            json_mode=True,
            success=False,
            game_type="quiz",
            metadata={"correlation_id": "corr-1"},
            error="boom",
        )
    )

    history = request.app.state.collector.history(last_n=1)
    assert len(history) == 1
    assert history[0]["error"] == "boom"
    assert posted == [
        {
            "url": "http://stats.local/events",
            "json": {
                "prompt": "prompt",
                "response": "response",
                "latency_ms": 12.5,
                "max_tokens": 256,
                "json_mode": True,
                "success": False,
                "game_type": "quiz",
                "error": "boom",
                "metadata": {"correlation_id": "corr-1"},
            },
            "headers": {"X-API-Key": "stats-secret"},
        }
    ]


def test_publish_event_to_stats_noops_without_url() -> None:
    import ai_engine.api.app as app_module

    request = MagicMock()
    request.app.state = types.SimpleNamespace(stats_url=" ", stats_api_key=None)

    asyncio.run(
        app_module._publish_event_to_stats(request, {"event_type": "generation"})
    )


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
        assert call_kwargs["max_tokens"] == 256

    def test_generate_preserves_lower_requested_max_tokens(self) -> None:
        client, mock_gen, *_ = _make_client()
        client.post(
            "/generate",
            json={
                "query": "photosynthesis",
                "game_type": "quiz",
                "language": "en",
                "num_questions": 8,
                "max_tokens": 128,
            },
        )
        call_kwargs = mock_gen.generate_from_context.call_args.kwargs
        assert call_kwargs["max_tokens"] == 128

    def test_generate_resolves_category_name_from_catalog(self) -> None:
        client, _, mock_pipeline, _ = _make_client()
        resp = client.post(
            "/generate",
            json={
                "query": "biology",
                "game_type": "quiz",
                "category_id": "17",
            },
        )
        assert resp.status_code == 200
        retrieve_kwargs = mock_pipeline.retrieve.call_args.kwargs
        assert retrieve_kwargs["metadata_preferences"]["category"] == "Science & Nature"

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

    def test_generate_runtime_error_records_observability_and_returns_500(self) -> None:
        """Unexpected generator failures should emit a failed event with metadata."""
        client, _, _, collector = _make_client(
            gen_side_effect=RuntimeError("llama read timeout"),
            raise_server_exceptions=False,
        )
        resp = client.post("/generate", json={"query": "water", "max_tokens": 512})
        assert resp.status_code == 500

        events = collector.history(last_n=1)
        assert len(events) == 1
        event = events[0]
        assert event["success"] is False
        assert event["error"] == "llama read timeout"
        assert event["metadata"]["event_type"] == "generation"
        assert event["metadata"]["correlation_id"]
        assert event["metadata"]["requested_max_tokens"] == 512
        assert event["metadata"]["effective_max_tokens"] == 512
        assert event["metadata"]["query_chars"] == len("water")

    def test_generate_timeout_returns_504_and_records_upstream_metadata(self) -> None:
        """Upstream llama timeouts should surface as HTTP 504, not raw 500s."""
        client, _, _, collector = _make_client(
            gen_side_effect=httpx.ReadTimeout("llama request timed out"),
            raise_server_exceptions=False,
        )
        resp = client.post("/generate", json={"query": "water", "max_tokens": 512})
        assert resp.status_code == 504
        assert resp.json()["detail"] == "Upstream LLM request timed out."

        events = collector.history(last_n=1)
        assert len(events) == 1
        event = events[0]
        assert event["success"] is False
        assert event["error"] == "llama request timed out"
        assert event["metadata"]["upstream_service"] == "llama"
        assert event["metadata"]["error_type"] == "ReadTimeout"

    def test_generate_connect_error_returns_503_and_records_upstream_metadata(
        self,
    ) -> None:
        """Upstream connection failures should not leak as raw 500s."""
        client, _, _, collector = _make_client(
            gen_side_effect=httpx.ConnectError("llama unavailable"),
            raise_server_exceptions=False,
        )
        resp = client.post("/generate", json={"query": "water", "max_tokens": 512})
        assert resp.status_code == 503
        assert resp.json()["detail"] == "Upstream LLM request failed."

        events = collector.history(last_n=1)
        assert len(events) == 1
        event = events[0]
        assert event["success"] is False
        assert event["error"] == "llama unavailable"
        assert event["metadata"]["upstream_service"] == "llama"
        assert event["metadata"]["error_type"] == "ConnectError"

    def test_generate_returns_503_when_capacity_queue_is_full(self) -> None:
        from ai_engine.api.app import _GenerationCapacityLimiter

        client, *_ = _make_client(raise_server_exceptions=False)

        class RejectingLimiter(_GenerationCapacityLimiter):
            async def acquire(self, caller_tier: str = "interactive") -> bool:
                return False

        client.app.state.generation_capacity_limiter = RejectingLimiter(1, 0)
        resp = client.post("/generate", json={"query": "water"})
        assert resp.status_code == 503
        assert (
            resp.json()["detail"] == "Generation service is busy. Please retry shortly."
        )

    def test_background_generation_rejects_when_slot_is_busy(self) -> None:
        from ai_engine.api.app import _GenerationCapacityLimiter

        limiter = _GenerationCapacityLimiter(max_in_flight=1, max_queue_size=1)

        async def _scenario() -> None:
            assert await limiter.acquire("interactive") is True
            assert await limiter.acquire("background") is False
            limiter.release()

        asyncio.run(_scenario())

    def test_background_generation_rejects_when_interactive_queue_exists(self) -> None:
        from ai_engine.api.app import _GenerationCapacityLimiter

        limiter = _GenerationCapacityLimiter(max_in_flight=1, max_queue_size=1)

        async def _occupy_slot() -> None:
            acquired = await limiter.acquire("interactive")
            assert acquired is True
            await asyncio.sleep(0.05)
            limiter.release()

        async def _queued_interactive() -> bool:
            acquired = await limiter.acquire("interactive")
            if acquired:
                limiter.release()
            return acquired

        async def _scenario() -> None:
            holder = asyncio.create_task(_occupy_slot())
            await asyncio.sleep(0.01)
            queued = asyncio.create_task(_queued_interactive())
            await asyncio.sleep(0.01)
            assert await limiter.acquire("background") is False
            await holder
            assert await queued is True

        asyncio.run(_scenario())

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

    def test_generate_sdk_timeout_returns_504(self) -> None:
        """SDK endpoint should also map upstream llama timeouts to HTTP 504."""
        client, _, _, collector = _make_client(
            gen_side_effect=httpx.ReadTimeout("sdk llama request timed out"),
            raise_server_exceptions=False,
        )
        resp = client.post(
            "/generate/sdk", json={"query": "water cycle", "language": "en"}
        )
        assert resp.status_code == 504
        assert resp.json()["detail"] == "Upstream LLM request timed out."

        events = collector.history(last_n=1)
        assert len(events) == 1
        event = events[0]
        assert event["error"] == "sdk llama request timed out"
        assert event["metadata"]["upstream_service"] == "llama"

    def test_generate_sdk_connect_error_returns_503(self) -> None:
        """SDK endpoint should also map upstream connection failures to HTTP 503."""
        client, _, _, collector = _make_client(
            gen_side_effect=httpx.ConnectError("sdk llama unavailable"),
            raise_server_exceptions=False,
        )
        resp = client.post(
            "/generate/sdk", json={"query": "water cycle", "language": "en"}
        )
        assert resp.status_code == 503
        assert resp.json()["detail"] == "Upstream LLM request failed."

        events = collector.history(last_n=1)
        assert len(events) == 1
        event = events[0]
        assert event["error"] == "sdk llama unavailable"
        assert event["metadata"]["upstream_service"] == "llama"
        assert event["metadata"]["error_type"] == "ConnectError"

    def test_generate_sdk_returns_503_when_capacity_queue_is_full(self) -> None:
        from ai_engine.api.app import _GenerationCapacityLimiter

        client, *_ = _make_client(raise_server_exceptions=False)

        class RejectingLimiter(_GenerationCapacityLimiter):
            async def acquire(self, caller_tier: str = "interactive") -> bool:
                return False

        client.app.state.generation_capacity_limiter = RejectingLimiter(1, 0)
        resp = client.post(
            "/generate/sdk",
            json={"query": "water cycle", "language": "en"},
        )
        assert resp.status_code == 503
        assert (
            resp.json()["detail"] == "Generation service is busy. Please retry shortly."
        )

    def test_health_reports_generation_capacity(self) -> None:
        client, *_ = _make_client()
        resp = client.get("/health")
        assert resp.status_code == 200
        capacity = resp.json()["dependencies"]["generation_capacity"]
        assert capacity["status"] == "ready"
        assert capacity["max_in_flight"] == 2
        assert capacity["max_queue_size"] == 2
        assert capacity["active"] == 0
        assert capacity["queued"] == 0

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
            params={"query": "water cycle", "item_count": 4},
            headers={"X-Game-Language": "en", "X-Difficulty-Percentage": "65"},
        )
        assert resp.status_code == 200
        call_kwargs = mock_gen.generate_from_context.call_args.kwargs
        assert call_kwargs["game_type"] == "quiz"
        assert call_kwargs["num_questions"] == 4
        assert call_kwargs["language"] == "en"
        assert call_kwargs["difficulty_percentage"] == 65

    def test_generate_quiz_model_specific_endpoint_accepts_category(self) -> None:
        client, _, mock_pipeline, _ = _make_client()
        resp = client.post(
            "/generate/quiz",
            params={"query": "cells", "category_id": "17"},
        )
        assert resp.status_code == 200
        retrieve_kwargs = mock_pipeline.retrieve.call_args.kwargs
        assert retrieve_kwargs["metadata_preferences"]["category"] == "Science & Nature"

    def test_generate_word_pass_model_specific_endpoint(self) -> None:
        """/generate/word-pass should use item_count and force word-pass game type."""
        client, mock_gen, *_ = _make_client(envelope=_make_word_pass_envelope())
        resp = client.post(
            "/generate/word-pass",
            params={"query": "chemistry", "item_count": 3},
        )
        assert resp.status_code == 200
        call_kwargs = mock_gen.generate_from_context.call_args.kwargs
        assert call_kwargs["game_type"] == "word-pass"
        assert call_kwargs["num_questions"] == 3

    def test_generate_quiz_language_and_difficulty_as_query_params(self) -> None:
        """language and difficulty_percentage as query params should override header defaults."""
        client, mock_gen, *_ = _make_client()
        resp = client.post(
            "/generate/quiz",
            params={"query": "sports", "language": "de", "difficulty_percentage": "75"},
        )
        assert resp.status_code == 200
        call_kwargs = mock_gen.generate_from_context.call_args.kwargs
        assert call_kwargs["language"] == "de"
        assert call_kwargs["difficulty_percentage"] == 75

    def test_generate_quiz_query_params_override_headers(self) -> None:
        """Query params should take precedence over headers for language/difficulty."""
        client, mock_gen, *_ = _make_client()
        resp = client.post(
            "/generate/quiz",
            params={"query": "math", "language": "fr", "difficulty_percentage": "90"},
            headers={"X-Game-Language": "it", "X-Difficulty-Percentage": "10"},
        )
        assert resp.status_code == 200
        call_kwargs = mock_gen.generate_from_context.call_args.kwargs
        assert call_kwargs["language"] == "fr"
        assert call_kwargs["difficulty_percentage"] == 90

    def test_generate_word_pass_language_and_difficulty_as_query_params(self) -> None:
        """word-pass should accept language and difficulty_percentage as query params."""
        client, mock_gen, *_ = _make_client(envelope=_make_word_pass_envelope())
        resp = client.post(
            "/generate/word-pass",
            params={
                "query": "science",
                "item_count": 3,
                "language": "en",
                "difficulty_percentage": "60",
            },
        )
        assert resp.status_code == 200
        call_kwargs = mock_gen.generate_from_context.call_args.kwargs
        assert call_kwargs["language"] == "en"
        assert call_kwargs["difficulty_percentage"] == 60

    def test_generate_word_pass_accepts_category_without_query(self) -> None:
        client, mock_gen, *_ = _make_client(envelope=_make_word_pass_envelope())
        resp = client.post(
            "/generate/word-pass",
            params={"category_id": "17", "item_count": 2},
        )

        assert resp.status_code == 200
        call_kwargs = mock_gen.generate_from_context.call_args.kwargs
        assert call_kwargs["topic"] == "Science & Nature"
        assert call_kwargs["num_questions"] == 2


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

    def test_rag_stats_cache_is_reused_and_invalidated_after_ingest(
        self, monkeypatch
    ) -> None:
        """RAG stats should reuse short-lived cache and invalidate it after ingest."""
        import ai_engine.api.app as app_module

        client, *_ = _make_client()
        call_count = 0

        def fake_compute_rag_stats(_pipeline):
            nonlocal call_count
            call_count += 1
            return {
                "total_chunks": call_count,
                "total_chars": 100,
                "unique_documents": 1,
                "embedding_dimensions": 3,
                "avg_chunk_chars": 100,
                "coverage_level": "good",
                "coverage_message": "ok",
                "retriever_config": {},
                "sources": [],
            }

        monkeypatch.setattr(app_module, "compute_rag_stats", fake_compute_rag_stats)

        first = client.get("/diagnostics/rag/stats")
        second = client.get("/diagnostics/rag/stats")

        assert first.status_code == 200
        assert second.status_code == 200
        assert first.json()["total_chunks"] == 1
        assert second.json()["total_chunks"] == 1
        assert call_count == 1

        ingest = client.post(
            "/ingest",
            json={
                "documents": [
                    {"content": "Invalidate diagnostics cache.", "doc_id": "doc-1"}
                ]
            },
        )
        assert ingest.status_code == 200

        third = client.get("/diagnostics/rag/stats")
        assert third.status_code == 200
        assert third.json()["total_chunks"] == 2
        assert call_count == 2


# ------------------------------------------------------------------
# Internal cache endpoints and monitoring separation
# ------------------------------------------------------------------


class TestMonitoringSeparation:
    """Monitoring endpoints are delegated to ai-stats service."""

    def test_public_monitoring_endpoints_are_not_exposed(self) -> None:
        """ai-api should not expose public /stats, /metrics or /cache endpoints."""
        client, *_ = _make_client()
        assert client.get("/stats").status_code == 404
        assert client.get("/stats/history").status_code == 404
        assert client.get("/cache/stats").status_code == 404
        assert client.post("/cache/reset").status_code == 404
        assert client.get("/metrics").status_code == 404

    def test_internal_cache_stats_is_available(self) -> None:
        """ai-stats should consume cache stats through this internal endpoint."""
        client, *_ = _make_client()
        resp = client.get("/internal/cache/stats")
        assert resp.status_code == 200
        payload = resp.json()
        assert "memory_entries" in payload
        assert "persistent_entries" in payload

    def test_internal_cache_reset_is_available(self) -> None:
        """ai-stats should be able to trigger cache invalidation internally."""
        client, *_ = _make_client()
        resp = client.post("/internal/cache/reset")
        assert resp.status_code == 200
        payload = resp.json()
        assert "removed_memory" in payload
        assert "removed_persistent" in payload

    def test_correlation_id_propagates_in_generate_response(self) -> None:
        """Generation response should preserve incoming correlation id."""
        client, *_ = _make_client()
        correlation_id = "corr-test-123"
        resp = client.post(
            "/generate",
            json={"query": "water"},
            headers={"X-Correlation-ID": correlation_id},
        )
        assert resp.status_code == 200
        assert resp.headers.get("X-Correlation-ID") == correlation_id


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
        _clear_all_settings_caches()
        try:
            app = create_app(
                generator=mock_gen,
                rag_pipeline=mock_pipeline,
                collector=StatsCollector(),
            )
        finally:
            del os.environ["AI_ENGINE_GAMES_API_KEY"]
            del os.environ["AI_ENGINE_BRIDGE_API_KEY"]
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

    def test_ready_is_public_even_with_keys(self) -> None:
        """Readiness remains public so Kubernetes probes can mark pods Ready."""
        client, _ = self._make_secured_client()
        resp = client.get("/ready")
        assert resp.status_code == 200

    def test_ready_with_prefixed_path_stays_public(self) -> None:
        """Public probe routes should stay exempt even when a path prefix is present."""
        from ai_engine.api.middleware import APIKeyMiddleware

        middleware = APIKeyMiddleware(
            lambda scope, receive, send: None,
            api_key="secret",
            games_api_key="games-secret",
            service="generation",
            public_paths={"/health", "/ready"},
        )

        assert middleware._is_public_path("/internal/ready") is True
        assert middleware._is_public_path("/internal/health") is True
        assert middleware._is_public_path("/internal/generate") is False

    def test_generation_scope_resolves_bridge_for_backoffice_calls(self) -> None:
        from ai_engine.api.middleware import APIKeyMiddleware

        middleware = APIKeyMiddleware(
            lambda scope, receive, send: None,
            api_key="secret",
            games_api_key="games-secret",
            bridge_api_key="bridge-secret",
            service="generation",
        )

        assert middleware._resolve_auth_scope("bridge-secret") == "bridge"
        assert middleware._resolve_auth_scope("games-secret") == "games"

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
        """Generation endpoints accept the games key for microservice callers."""
        client, _ = self._make_secured_client(games_api_key="games-key")
        resp = client.post(
            "/generate",
            json={"query": "water"},
            headers={"X-API-Key": "games-key"},
        )
        assert resp.status_code == 200

    def test_generate_accepts_bridge_key_for_backoffice_calls(self) -> None:
        """Generation endpoints also accept the bridge key for backoffice workflows."""
        client, _ = self._make_secured_client(
            games_api_key="games-key",
            bridge_api_key="bridge-key",
        )
        resp = client.post(
            "/generate",
            json={"query": "water"},
            headers={"X-API-Key": "bridge-key"},
        )
        assert resp.status_code == 200

    def test_generate_protected_without_key(self) -> None:
        """POST /generate is blocked when the API key is missing."""
        client, _ = self._make_secured_client()
        resp = client.post("/generate", json={"query": "water"})
        assert resp.status_code == 401

    def test_generate_succeeds_with_correct_key(self) -> None:
        """POST /generate succeeds with an allowed generation API key."""
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
