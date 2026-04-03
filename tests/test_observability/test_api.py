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


class TestEventsIngestionEndpoint:
    """Tests for POST /events."""

    def test_ingests_single_event(self) -> None:
        """Posting an event records it in the collector."""
        client, collector = _make_client()
        resp = client.post(
            "/events",
            json={
                "prompt": "p",
                "response": "r",
                "latency_ms": 12.5,
                "max_tokens": 128,
                "json_mode": True,
                "success": True,
                "game_type": "quiz",
                "metadata": {"source": "ai-api"},
            },
        )

        assert resp.status_code == 200
        assert len(collector) == 1
        history = collector.history(last_n=1)
        assert history[0]["latency_ms"] == 12.5
        assert history[0]["metadata"]["source"] == "ai-api"

    def test_rejects_negative_latency(self) -> None:
        """Validation rejects malformed event payloads."""
        client, _ = _make_client()
        resp = client.post(
            "/events",
            json={
                "prompt": "p",
                "response": "r",
                "latency_ms": -1,
                "max_tokens": 64,
            },
        )
        assert resp.status_code == 422


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


# ------------------------------------------------------------------
# API Key authentication (observability API)
# ------------------------------------------------------------------


class TestObsAPIKeyAuth:
    """Tests for X-API-Key authentication on the observability API."""

    def _make_secured_client(
        self, bridge_api_key: str = "bridge-secret"
    ) -> "TestClient":
        """Return a TestClient with bridge API key enforcement enabled."""
        import os

        from ai_engine.observability.api import create_app

        os.environ["AI_ENGINE_BRIDGE_API_KEY"] = bridge_api_key
        os.environ["AI_ENGINE_GENERATION_API_URL"] = ""
        _clear_all_settings_caches()
        try:
            app = create_app(StatsCollector())
            app.state.generation_api_url = ""
        finally:
            del os.environ["AI_ENGINE_BRIDGE_API_KEY"]
            os.environ.pop("AI_ENGINE_GENERATION_API_URL", None)
            _clear_all_settings_caches()
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
