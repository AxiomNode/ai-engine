"""Tests for the FastAPI observability API endpoints."""

from __future__ import annotations

import pytest

try:
    from fastapi.testclient import TestClient

    _HAS_FASTAPI = True
except ImportError:
    _HAS_FASTAPI = False

from ai_engine.observability.collector import StatsCollector

pytestmark = pytest.mark.skipif(not _HAS_FASTAPI, reason="fastapi not installed")


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


def _make_client() -> tuple["TestClient", StatsCollector]:
    """Create a fresh TestClient + collector pair."""
    from ai_engine.observability.api import create_app

    collector = StatsCollector()
    app = create_app(collector)
    return TestClient(app), collector


class TestHealthEndpoint:
    """Tests for GET /health."""

    def test_health_returns_ok(self) -> None:
        """Health check returns status ok."""
        client, _ = _make_client()
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "uptime_seconds" in data
        assert "distribution_version" in data
        assert resp.headers.get("X-Distribution-Version")

    def test_health_event_count(self) -> None:
        """Health reports event count."""
        client, collector = _make_client()
        collector.record_call(prompt="p", response="r", latency_ms=1.0, max_tokens=64)
        resp = client.get("/health")
        assert resp.json()["total_events"] == 1


class TestStatsEndpoint:
    """Tests for GET /stats."""

    def test_empty_stats(self) -> None:
        """Returns zeros when nothing has been recorded."""
        client, _ = _make_client()
        resp = client.get("/stats")
        assert resp.status_code == 200
        assert resp.json()["total_calls"] == 0

    def test_stats_after_recording(self) -> None:
        """Returns proper aggregates after recording events."""
        client, collector = _make_client()
        collector.record_call(
            prompt="p", response="r", latency_ms=10.0, max_tokens=64, game_type="quiz"
        )
        collector.record_call(
            prompt="p", response="r", latency_ms=20.0, max_tokens=64, game_type="quiz"
        )
        resp = client.get("/stats")
        data = resp.json()
        assert data["total_calls"] == 2
        assert data["avg_latency_ms"] == 15.0
        assert data["game_type_counts"]["quiz"] == 2


class TestHistoryEndpoint:
    """Tests for GET /stats/history."""

    def test_empty_history(self) -> None:
        """Returns empty list when no events exist."""
        client, _ = _make_client()
        resp = client.get("/stats/history")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_history_returns_events(self) -> None:
        """Returns recorded events."""
        client, collector = _make_client()
        collector.record_call(prompt="p", response="r", latency_ms=5.0, max_tokens=64)
        resp = client.get("/stats/history")
        data = resp.json()
        assert len(data) == 1
        assert data[0]["latency_ms"] == 5.0

    def test_history_last_n(self) -> None:
        """last_n query parameter limits results."""
        client, collector = _make_client()
        for i in range(5):
            collector.record_call(
                prompt="p", response="r", latency_ms=float(i), max_tokens=64
            )
        resp = client.get("/stats/history", params={"last_n": 2})
        data = resp.json()
        assert len(data) == 2
        assert data[0]["latency_ms"] == 3.0


class TestResetEndpoint:
    """Tests for POST /stats/reset."""

    def test_reset_clears_stats(self) -> None:
        """Reset endpoint clears all events."""
        client, collector = _make_client()
        collector.record_call(prompt="p", response="r", latency_ms=1.0, max_tokens=64)
        resp = client.post("/stats/reset")
        assert resp.status_code == 200
        assert len(collector) == 0


class TestMetricsEndpoint:
    """Tests for GET /metrics."""

    def test_metrics_returns_prometheus_text(self) -> None:
        """Metrics endpoint returns text/plain payload with metric names."""
        client, collector = _make_client()
        collector.record_call(
            prompt="p",
            response="r",
            latency_ms=10.0,
            max_tokens=64,
            metadata={
                "event_type": "generation",
                "cache_hit": True,
                "cache_layer": "memory",
            },
            game_type="quiz",
        )

        resp = client.get("/metrics")
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/plain")
        body = resp.text
        assert "ai_engine_total_calls" in body
        assert "ai_engine_cache_hit_rate" in body
        assert 'ai_engine_game_type_calls{game_type="quiz"}' in body
        assert "ai_engine_generation_outcome_by_game_type_total" in body
        assert "ai_engine_distribution_version_info" in body


class TestCacheMonitoringBridge:
    """Tests for cache monitoring endpoints delegated to ai-api."""

    @staticmethod
    def _patch_async_client(monkeypatch, fake_response):
        """Replace httpx.AsyncClient with a fake that returns *fake_response*."""
        from ai_engine.observability import api as obs_api

        class _FakeAsyncClient:
            async def __aenter__(self):
                return self

            async def __aexit__(self, *args):
                pass

            async def get(self, *a, **kw):
                return fake_response

            async def post(self, *a, **kw):
                return fake_response

        monkeypatch.setattr(
            obs_api.httpx, "AsyncClient", lambda **kw: _FakeAsyncClient()
        )

    def test_cache_stats_returns_payload_from_generation_api(self, monkeypatch) -> None:
        """GET /cache/stats should proxy runtime cache payload from ai-api."""

        class _FakeResponse:
            status_code = 200

            @staticmethod
            def raise_for_status() -> None:
                return None

            @staticmethod
            def json() -> dict[str, object]:
                return {
                    "memory_entries": 2,
                    "memory_max_entries": 10,
                    "memory_saturation_ratio": 0.2,
                    "persistent_entries": 5,
                    "memory_enabled": True,
                    "persistent_enabled": True,
                    "persistent_backend": "tinydb",
                    "cache_namespace": "dev-v1",
                    "distribution_version": "dev-v1",
                    "cache_ttl_seconds": 900,
                    "persistent_backend_errors": {
                        "read": 0,
                        "write": 0,
                        "delete": 0,
                        "stats": 0,
                    },
                }

        self._patch_async_client(monkeypatch, _FakeResponse())

        client, _ = _make_client()
        resp = client.get("/cache/stats")
        assert resp.status_code == 200
        payload = resp.json()
        assert payload["memory_entries"] == 2
        assert payload["persistent_entries"] == 5

    def test_cache_reset_proxies_to_generation_api(self, monkeypatch) -> None:
        """POST /cache/reset should return reset counters from ai-api."""

        class _FakeResponse:
            status_code = 200

            @staticmethod
            def raise_for_status() -> None:
                return None

            @staticmethod
            def json() -> dict[str, int]:
                return {"removed_memory": 3, "removed_persistent": 7}

        self._patch_async_client(monkeypatch, _FakeResponse())

        client, _ = _make_client()
        resp = client.post("/cache/reset")
        assert resp.status_code == 200
        payload = resp.json()
        assert payload["removed_memory"] == 3
        assert payload["removed_persistent"] == 7

    def test_stats_includes_cache_runtime(self, monkeypatch) -> None:
        """GET /stats should include cache_runtime object for backoffice dashboards."""

        class _FakeResponse:
            status_code = 200

            @staticmethod
            def raise_for_status() -> None:
                return None

            @staticmethod
            def json() -> dict[str, object]:
                return {
                    "memory_entries": 1,
                    "memory_max_entries": 10,
                    "memory_saturation_ratio": 0.1,
                    "persistent_entries": 4,
                    "memory_enabled": True,
                    "persistent_enabled": True,
                    "persistent_backend": "tinydb",
                    "cache_namespace": "dev-v1",
                    "distribution_version": "dev-v1",
                    "cache_ttl_seconds": 900,
                    "persistent_backend_errors": {
                        "read": 0,
                        "write": 0,
                        "delete": 0,
                        "stats": 0,
                    },
                }

        self._patch_async_client(monkeypatch, _FakeResponse())

        client, _ = _make_client()
        resp = client.get("/stats")
        assert resp.status_code == 200
        assert "cache_runtime" in resp.json()

    def test_bridge_reuses_single_async_client_instance(self, monkeypatch) -> None:
        """Cache bridge should reuse one AsyncClient across multiple proxy calls."""
        from ai_engine.observability import api as obs_api

        created_clients: list[object] = []

        class _FakeResponse:
            status_code = 200

            @staticmethod
            def raise_for_status() -> None:
                return None

            @staticmethod
            def json() -> dict[str, object]:
                return {
                    "memory_entries": 1,
                    "memory_max_entries": 10,
                    "memory_saturation_ratio": 0.1,
                    "persistent_entries": 0,
                    "memory_enabled": True,
                    "persistent_enabled": False,
                    "persistent_backend": "none",
                    "cache_namespace": "dev-v1",
                    "distribution_version": "dev-v1",
                    "cache_ttl_seconds": 900,
                    "persistent_backend_errors": {
                        "read": 0,
                        "write": 0,
                        "delete": 0,
                        "stats": 0,
                    },
                    "removed_memory": 2,
                    "removed_persistent": 0,
                }

        class _FakeAsyncClient:
            def __init__(self, *args, **kwargs) -> None:
                created_clients.append(self)

            async def get(self, *args, **kwargs):
                return _FakeResponse()

            async def post(self, *args, **kwargs):
                return _FakeResponse()

            async def aclose(self) -> None:
                return None

        monkeypatch.setattr(obs_api.httpx, "AsyncClient", _FakeAsyncClient)

        client, _ = _make_client()
        stats_response = client.get("/cache/stats")
        reset_response = client.post("/cache/reset")

        assert stats_response.status_code == 200
        assert reset_response.status_code == 200
        assert len(created_clients) == 1

    def test_cache_stats_without_generation_url_returns_empty_runtime(self) -> None:
        """GET /cache/stats should fall back to an empty payload when bridge URL is absent."""
        client, _ = _make_client()
        client.app.state.generation_api_url = "  "
        client.app.state.distribution_version = None

        resp = client.get("/cache/stats")

        assert resp.status_code == 200
        assert resp.json() == {
            "memory_entries": 0,
            "memory_max_entries": 0,
            "memory_saturation_ratio": 0.0,
            "persistent_entries": 0,
            "memory_enabled": False,
            "persistent_enabled": False,
            "persistent_backend": "none",
            "cache_namespace": "v1",
            "distribution_version": "unknown-v0",
            "cache_ttl_seconds": 0,
            "persistent_backend_errors": {
                "read": 0,
                "write": 0,
                "delete": 0,
                "stats": 0,
            },
        }

    def test_cache_stats_invalid_payload_falls_back_to_empty_runtime(
        self, monkeypatch
    ) -> None:
        """Non-dict proxy payloads should be treated as invalid and replaced by fallback stats."""

        class _FakeResponse:
            status_code = 200

            @staticmethod
            def raise_for_status() -> None:
                return None

            @staticmethod
            def json() -> list[int]:
                return [1, 2, 3]

        self._patch_async_client(monkeypatch, _FakeResponse())

        client, _ = _make_client()
        resp = client.get("/cache/stats")

        assert resp.status_code == 200
        assert resp.json()["persistent_backend"] == "none"
        assert resp.json()["memory_entries"] == 0

    def test_cache_reset_without_generation_url_returns_503(self) -> None:
        """POST /cache/reset should reject requests when bridge URL is not configured."""
        client, _ = _make_client()
        client.app.state.generation_api_url = ""

        resp = client.post("/cache/reset")

        assert resp.status_code == 503
        assert (
            resp.json()["detail"]
            == "Generation API URL not configured for cache reset."
        )

    def test_cache_reset_invalid_payload_returns_503(self, monkeypatch) -> None:
        """Unexpected proxy responses should surface as 503 to the monitoring caller."""

        class _FakeResponse:
            status_code = 200

            @staticmethod
            def raise_for_status() -> None:
                return None

            @staticmethod
            def json() -> list[int]:
                return [3, 7]

        self._patch_async_client(monkeypatch, _FakeResponse())

        client, _ = _make_client()
        resp = client.post("/cache/reset")

        assert resp.status_code == 503
        assert (
            resp.json()["detail"]
            == "Unexpected cache-reset response from generation API."
        )

    def test_cache_reset_forwards_query_params_and_auth_header(
        self, monkeypatch
    ) -> None:
        """Cache reset should forward namespace flags and monitor auth header to ai-api."""
        from ai_engine.observability import api as obs_api

        captured: dict[str, object] = {}

        class _FakeResponse:
            status_code = 200

            @staticmethod
            def raise_for_status() -> None:
                return None

            @staticmethod
            def json() -> dict[str, int]:
                return {"removed_memory": 1, "removed_persistent": 4}

        class _FakeAsyncClient:
            async def get(self, *args, **kwargs):
                return _FakeResponse()

            async def post(self, *args, **kwargs):
                captured["args"] = args
                captured["kwargs"] = kwargs
                return _FakeResponse()

            async def aclose(self) -> None:
                return None

        monkeypatch.setattr(
            obs_api.httpx, "AsyncClient", lambda **kw: _FakeAsyncClient()
        )

        client, _ = _make_client()
        client.app.state.generation_monitor_api_key = "bridge-secret"

        resp = client.post(
            "/cache/reset",
            params={"namespace": "quiz", "all_namespaces": True},
        )

        assert resp.status_code == 200
        assert captured["kwargs"] == {
            "headers": {"X-API-Key": "bridge-secret"},
            "params": {"namespace": "quiz", "all_namespaces": "true"},
            "timeout": 5.0,
        }


class TestOpenAPISecurity:
    """Tests for generated OpenAPI security metadata."""

    def test_openapi_marks_only_protected_routes_as_requiring_api_key(self) -> None:
        """Public health/docs routes stay unsecured while protected routes expose ApiKeyAuth."""
        client, _ = _make_client()

        schema = client.app.openapi()

        assert (
            schema["components"]["securitySchemes"]["ApiKeyAuth"]["name"] == "X-API-Key"
        )
        assert "security" not in schema["paths"]["/health"]["get"]
        assert schema["paths"]["/stats"]["get"]["security"] == [{"ApiKeyAuth": []}]
        assert schema["paths"]["/events"]["post"]["security"] == [{"ApiKeyAuth": []}]

    def test_openapi_result_is_cached_between_calls(self) -> None:
        """The custom OpenAPI builder should memoize the generated schema on app state."""
        client, _ = _make_client()

        schema = client.app.openapi()

        assert client.app.openapi() is schema


class TestLifecycle:
    """Tests for application lifecycle hooks."""

    def test_shutdown_closes_generation_api_client(self) -> None:
        """App shutdown should close the reusable AsyncClient stored on state."""
        from ai_engine.observability.api import create_app

        closed: list[bool] = []

        class _FakeAsyncClient:
            async def get(self, *args, **kwargs):
                raise AssertionError("unexpected get call")

            async def post(self, *args, **kwargs):
                raise AssertionError("unexpected post call")

            async def aclose(self) -> None:
                closed.append(True)

        app = create_app(StatsCollector())
        app.state.generation_api_client = _FakeAsyncClient()

        with TestClient(app):
            pass

        assert closed == [True]


# ------------------------------------------------------------------
# API Key authentication (observability API)
# ------------------------------------------------------------------


class TestObsAPIKeyAuth:
    """Tests for X-API-Key authentication on the observability API."""

    def _make_secured_client(
        self, bridge_api_key: str = "bridge-secret"
    ) -> "TestClient":
        """Return a TestClient with bridge API key enforcement enabled."""
        from ai_engine.api.middleware import APIKeyMiddleware
        from ai_engine.observability.api import create_app

        app = create_app(StatsCollector())
        app.state.generation_api_url = ""
        app.add_middleware(
            APIKeyMiddleware,
            bridge_api_key=bridge_api_key,
            service="observability",
            public_paths={
                "/health",
                "/docs",
                "/openapi.json",
                "/docs/oauth2-redirect",
                "/redoc",
            },
        )
        return TestClient(app, raise_server_exceptions=False)

    def test_no_key_required_when_env_not_set(self) -> None:
        """All requests pass through when AI_ENGINE_API_KEY is not set."""
        client, _ = _make_client()
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_is_public_even_with_key_configured(self) -> None:
        """Health remains public for liveness probes."""
        client = self._make_secured_client()
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_stats_without_key_returns_401(self) -> None:
        """Stats endpoint requires API key when bridge key is configured."""
        client = self._make_secured_client()
        resp = client.get("/stats")
        assert resp.status_code == 401

    def test_stats_with_wrong_key_returns_403(self) -> None:
        """Wrong X-API-Key returns 403 on protected observability endpoints."""
        client = self._make_secured_client()
        resp = client.get("/stats", headers={"X-API-Key": "nope"})
        assert resp.status_code == 403

    def test_request_with_correct_key_passes(self) -> None:
        """Correct X-API-Key allows the request through."""
        client = self._make_secured_client(bridge_api_key="obs-secret")
        resp = client.get("/stats", headers={"X-API-Key": "obs-secret"})
        assert resp.status_code == 200
