"""Focused unit tests for API key middleware helpers."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
from starlette.requests import Request
from starlette.responses import JSONResponse

from ai_engine.api.middleware import APIKeyMiddleware, add_api_key_middleware


def _run(coro):
    return asyncio.run(coro)


def _request(path: str, headers: list[tuple[bytes, bytes]] | None = None) -> Request:
    scope = {
        "type": "http",
        "method": "GET",
        "path": path,
        "headers": headers or [],
        "query_string": b"",
        "client": ("127.0.0.1", 1234),
        "server": ("testserver", 80),
        "scheme": "http",
        "http_version": "1.1",
    }
    return Request(scope)


def test_normalize_path_handles_blank_and_trailing_slash() -> None:
    assert APIKeyMiddleware._normalize_path("") == "/"
    assert APIKeyMiddleware._normalize_path(" /internal/ready/ ") == "/internal/ready"


def test_normalized_keys_strips_and_deduplicates() -> None:
    assert APIKeyMiddleware._normalized_keys(" secret ", None, "secret", "", "games") == (
        "secret",
        "games",
    )


def test_resolve_allowed_keys_for_observability_routes() -> None:
    middleware = APIKeyMiddleware(
        lambda scope, receive, send: None,
        api_key="api-secret",
        bridge_api_key="bridge-secret",
        stats_api_key="stats-secret",
        service="observability",
    )

    assert middleware._resolve_allowed_keys("/events") == (
        "stats-secret",
        "bridge-secret",
        "api-secret",
    )
    assert middleware._resolve_allowed_keys("/metrics") == (
        "bridge-secret",
        "api-secret",
    )


def test_resolve_auth_scope_covers_stats_api_and_unknown() -> None:
    middleware = APIKeyMiddleware(
        lambda scope, receive, send: None,
        api_key="api-secret",
        bridge_api_key="bridge-secret",
        stats_api_key="stats-secret",
        service="observability",
    )

    assert middleware._resolve_auth_scope("stats-secret") == "stats"
    assert middleware._resolve_auth_scope("api-secret") == "api"
    assert middleware._resolve_auth_scope("unknown") == "anonymous"
    assert middleware._resolve_auth_scope(None) == "anonymous"


def test_dispatch_passes_through_when_no_keys_are_configured() -> None:
    middleware = APIKeyMiddleware(lambda scope, receive, send: None)
    request = _request("/generate")

    async def call_next(req: Request) -> JSONResponse:
        assert req.state.auth_scope == "anonymous"
        return JSONResponse({"ok": True})

    response = _run(middleware.dispatch(request, call_next))

    assert response.status_code == 200
    assert response.body == b'{"ok":true}'


def test_dispatch_rejects_missing_api_key_when_protected() -> None:
    middleware = APIKeyMiddleware(lambda scope, receive, send: None, api_key="secret")
    request = _request("/generate")

    response = _run(middleware.dispatch(request, lambda request: JSONResponse({"ok": True})))

    assert response.status_code == 401


def test_add_api_key_middleware_skips_when_settings_have_no_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[object, dict[str, object]]] = []

    class _App:
        def add_middleware(self, middleware_class, **kwargs):
            calls.append((middleware_class, kwargs))

    monkeypatch.setattr(
        "ai_engine.api.middleware.get_settings",
        lambda: SimpleNamespace(
            api_key=None,
            games_api_key=None,
            bridge_api_key=None,
            stats_api_key=None,
        ),
    )

    add_api_key_middleware(_App(), public_paths={"/health"})

    assert calls == []


def test_add_api_key_middleware_registers_when_any_key_exists(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[object, dict[str, object]]] = []

    class _App:
        def add_middleware(self, middleware_class, **kwargs):
            calls.append((middleware_class, kwargs))

    monkeypatch.setattr(
        "ai_engine.api.middleware.get_settings",
        lambda: SimpleNamespace(
            api_key="api-secret",
            games_api_key="games-secret",
            bridge_api_key="bridge-secret",
            stats_api_key="stats-secret",
        ),
    )

    add_api_key_middleware(_App(), service="observability", public_paths={"/health/"})

    assert calls == [
        (
            APIKeyMiddleware,
            {
                "api_key": "api-secret",
                "games_api_key": "games-secret",
                "bridge_api_key": "bridge-secret",
                "stats_api_key": "stats-secret",
                "service": "observability",
                "public_paths": {"/health/"},
            },
        )
    ]