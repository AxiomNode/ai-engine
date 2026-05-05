"""FastAPI application for serving observability statistics.

Provides a factory function :func:`create_app` that returns a fully
configured :class:`~fastapi.FastAPI` instance with the following
endpoints:

- ``GET /health`` – Liveness check.
- ``POST /events`` – Ingest one observability event.
- ``GET /stats`` – Aggregate statistics from the collector.
- ``GET /stats/history`` – Recent event log.
- ``POST /stats/reset`` – Clear all recorded events.
- ``GET /cache/stats`` – Runtime generation-cache counters (proxied from ai-api).
- ``POST /cache/reset`` – Cache invalidation operation (proxied to ai-api).
- ``GET /metrics`` – Prometheus text exposition format.

The app can be run standalone via::

    uvicorn ai_engine.observability.api:app --reload

or embedded inside another FastAPI application as a sub-application.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import httpx

try:
    from fastapi import FastAPI, HTTPException, Query, Request
    from fastapi.openapi.utils import get_openapi
    from fastapi.responses import PlainTextResponse
    from pydantic import BaseModel, Field
except ImportError as _imp_err:  # pragma: no cover
    raise ImportError(
        "FastAPI is required for the observability API.  "
        "Install it with:  pip install ai-engine[api]"
    ) from _imp_err

from ai_engine.config import get_settings
from ai_engine.observability.collector import StatsCollector, summary_to_prometheus

try:
    from ai_engine.api.middleware import add_api_key_middleware
except ImportError:  # pragma: no cover
    add_api_key_middleware = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

_start_time: float = time.time()


def _build_default_collector() -> StatsCollector:
    """Create the module-level collector with configured history persistence."""
    settings = get_settings()
    return StatsCollector(persistence_path=settings.observability_history_path)


# Module-level collector used when running ``uvicorn … api:app``.
_default_collector = _build_default_collector()


class EventIngestRequest(BaseModel):
    """Payload accepted by the observability event ingestion endpoint."""

    prompt: str = ""
    response: str = ""
    latency_ms: float = Field(default=0.0, ge=0.0)
    max_tokens: int = Field(default=0, ge=0)
    json_mode: bool = False
    success: bool = True
    game_type: str = "unknown"
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


def _get_collector(request: Request) -> StatsCollector:
    """Retrieve the :class:`StatsCollector` from app state.

    Args:
        request: The incoming FastAPI request.

    Returns:
        The collector attached to ``app.state``.
    """
    return request.app.state.collector  # type: ignore[no-any-return]


def _get_distribution_version(request: Request) -> str:
    """Return distribution-version tag stored on the application state."""
    value = getattr(request.app.state, "distribution_version", None)
    if isinstance(value, str) and value.strip():
        return value.strip()
    return "unknown-v0"


async def _summary_with_distribution(request: Request) -> dict[str, Any]:
    """Return collector summary enriched with distribution-version tag."""
    summary = _get_collector(request).summary()
    summary["cache_runtime"] = await _fetch_generation_cache_stats(request)
    summary["distribution_version"] = _get_distribution_version(request)
    return summary


def _get_generation_api_url(request: Request) -> str | None:
    """Return generation API base URL configured for monitoring bridge."""
    value = getattr(request.app.state, "generation_api_url", None)
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _build_generation_api_headers(request: Request) -> dict[str, str]:
    """Build auth headers for internal ai-stats -> ai-api monitoring calls."""
    headers: dict[str, str] = {}
    api_key = getattr(request.app.state, "generation_monitor_api_key", None)
    if isinstance(api_key, str) and api_key.strip():
        headers["X-API-Key"] = api_key.strip()
    return headers


def _get_generation_api_client(request: Request) -> httpx.AsyncClient:
    """Return a reusable AsyncClient stored on the FastAPI app state."""
    client = getattr(request.app.state, "generation_api_client", None)
    if client is not None and all(
        hasattr(client, attribute) for attribute in ("get", "post", "aclose")
    ):
        return client

    client = httpx.AsyncClient()
    request.app.state.generation_api_client = client
    return client


def _empty_cache_runtime(request: Request) -> dict[str, Any]:
    """Return fallback cache runtime payload when generation API is unavailable."""
    return {
        "memory_entries": 0,
        "memory_max_entries": 0,
        "memory_saturation_ratio": 0.0,
        "persistent_entries": 0,
        "memory_enabled": False,
        "persistent_enabled": False,
        "persistent_backend": "none",
        "cache_namespace": "v1",
        "distribution_version": _get_distribution_version(request),
        "cache_ttl_seconds": 0,
        "persistent_backend_errors": {
            "read": 0,
            "write": 0,
            "delete": 0,
            "stats": 0,
        },
    }


async def _fetch_generation_cache_stats(request: Request) -> dict[str, Any]:
    """Fetch cache stats from ai-api internal monitoring endpoint."""
    base_url = _get_generation_api_url(request)
    if not base_url:
        return _empty_cache_runtime(request)

    endpoint = f"{base_url.rstrip('/')}/internal/cache/stats"
    try:
        client = _get_generation_api_client(request)
        response = await client.get(
            endpoint,
            headers=_build_generation_api_headers(request),
            timeout=2.0,
        )
        response.raise_for_status()
        payload = response.json()
        if isinstance(payload, dict):
            return payload
    except Exception:
        logger.warning("Failed to read cache stats from generation API", exc_info=True)

    return _empty_cache_runtime(request)


async def _reset_generation_cache(
    request: Request,
    *,
    namespace: str | None,
    all_namespaces: bool,
) -> dict[str, int]:
    """Proxy cache reset operation to ai-api internal endpoint."""
    base_url = _get_generation_api_url(request)
    if not base_url:
        raise HTTPException(
            status_code=503,
            detail="Generation API URL not configured for cache reset.",
        )

    endpoint = f"{base_url.rstrip('/')}/internal/cache/reset"
    try:
        client = _get_generation_api_client(request)
        response = await client.post(
            endpoint,
            headers=_build_generation_api_headers(request),
            params={
                "namespace": namespace,
                "all_namespaces": str(all_namespaces).lower(),
            },
            timeout=5.0,
        )
        response.raise_for_status()
        payload = response.json()
        if isinstance(payload, dict):
            return {
                "removed_memory": int(payload.get("removed_memory", 0) or 0),
                "removed_persistent": int(payload.get("removed_persistent", 0) or 0),
            }
    except HTTPException:
        raise
    except Exception as exc:
        logger.warning("Failed to reset cache via generation API", exc_info=True)
        raise HTTPException(
            status_code=503,
            detail="Failed to reset generation cache via monitoring bridge.",
        ) from exc

    raise HTTPException(
        status_code=503,
        detail="Unexpected cache-reset response from generation API.",
    )


def _install_api_key_openapi(
    app: FastAPI,
    *,
    public_paths: set[str] | None = None,
) -> None:
    """Inject API key security schema so `/docs` shows Authorize + X-API-Key."""

    def custom_openapi() -> dict[str, Any]:
        if app.openapi_schema:
            return app.openapi_schema  # type: ignore[return-value]

        schema = get_openapi(
            title=app.title,
            version=app.version,
            description=app.description,
            routes=app.routes,
        )

        components = schema.setdefault("components", {})
        security_schemes = components.setdefault("securitySchemes", {})
        security_schemes["ApiKeyAuth"] = {
            "type": "apiKey",
            "in": "header",
            "name": "X-API-Key",
            "description": "Use a valid scoped key. Public `/health` is excluded.",
        }

        public = public_paths or set()
        paths = schema.get("paths", {})
        for path, path_item in paths.items():
            if path in public or not isinstance(path_item, dict):
                continue
            for method, operation in path_item.items():
                if method.lower() not in {
                    "get",
                    "post",
                    "put",
                    "patch",
                    "delete",
                    "options",
                    "head",
                    "trace",
                }:
                    continue
                if isinstance(operation, dict):
                    operation.setdefault("security", [{"ApiKeyAuth": []}])

        app.openapi_schema = schema
        return schema  # type: ignore[return-value]

    setattr(app, "openapi", custom_openapi)


def create_app(collector: StatsCollector | None = None) -> FastAPI:
    """Create and return a configured FastAPI application.

    Args:
        collector: The :class:`StatsCollector` to expose.  If ``None``,
            a module-level default collector is used.

    Returns:
        A :class:`~fastapi.FastAPI` instance ready to be served.
    """
    stats = collector if collector is not None else _default_collector
    settings = get_settings()
    distribution_version = settings.distribution_version_tag
    openapi_version = f"0.1.0+{distribution_version}"
    openapi_tags = [
        {
            "name": "monitoring",
            "description": (
                "Backoffice monitoring endpoints for health, stats, cache runtime, "
                "cache reset, and metrics."
            ),
        },
        {
            "name": "ingestion",
            "description": (
                "Internal event-ingestion endpoint used by ai-api and bridge services."
            ),
        },
    ]

    app = FastAPI(
        title=f"ai-engine monitoring API ({distribution_version})",
        description=(
            "Dedicated monitoring and operational API for backoffice systems. "
            "This service centralizes health, statistics, cache operations, "
            f"and Prometheus metrics. Active deployment version: {distribution_version}."
        ),
        version=openapi_version,
        openapi_tags=openapi_tags,
    )

    # Store collector on app state — endpoints read from here via the
    # ``Request`` object so each app instance uses its own collector.
    app.state.collector = stats
    app.state.distribution_version = distribution_version
    app.state.generation_api_url = settings.generation_api_url
    app.state.generation_monitor_api_key = settings.bridge_api_key or settings.api_key
    app.state.generation_api_client = httpx.AsyncClient()

    @app.on_event("shutdown")
    async def close_generation_api_client() -> None:
        client = getattr(app.state, "generation_api_client", None)
        if client is not None and hasattr(client, "aclose"):
            await client.aclose()

    # Attach route-scoped API key middleware for bridge/internal integrations.
    if add_api_key_middleware is not None:
        add_api_key_middleware(
            app,
            service="observability",
            public_paths={
                "/health",
                "/docs",
                "/openapi.json",
                "/docs/oauth2-redirect",
                "/redoc",
            },
        )

    @app.middleware("http")
    async def distribution_header_middleware(request: Request, call_next):  # type: ignore[no-untyped-def]
        """Attach distribution-version tag to every response."""
        response = await call_next(request)
        response.headers["X-Distribution-Version"] = _get_distribution_version(request)
        return response

    # ------------------------------------------------------------------
    # Endpoints
    # ------------------------------------------------------------------

    @app.get("/health", tags=["monitoring"])
    def health(request: Request) -> dict[str, Any]:
        """Return basic health and uptime information.

        Returns:
            A dictionary with ``status``, ``uptime_seconds``, and
            ``total_events`` keys.
        """
        coll = _get_collector(request)
        return {
            "status": "ok",
            "uptime_seconds": round(time.time() - _start_time, 2),
            "total_events": len(coll),
            "distribution_version": _get_distribution_version(request),
        }

    @app.get("/stats", tags=["monitoring"])
    async def get_stats(request: Request) -> dict[str, Any]:
        """Return aggregate statistics over all recorded events.

        Returns:
            A summary dictionary produced by
            :meth:`StatsCollector.summary`.
        """
        return await _summary_with_distribution(request)

    @app.post("/events", tags=["ingestion"])
    def ingest_event(request: Request, payload: EventIngestRequest) -> dict[str, str]:
        """Ingest one observability event emitted by another service."""
        _get_collector(request).record_call(
            prompt=payload.prompt,
            response=payload.response,
            latency_ms=payload.latency_ms,
            max_tokens=payload.max_tokens,
            json_mode=payload.json_mode,
            success=payload.success,
            game_type=payload.game_type,
            error=payload.error,
            metadata=payload.metadata,
        )
        return {"message": "Event recorded."}

    @app.get("/stats/history", tags=["monitoring"])
    def get_history(
        request: Request,
        last_n: int | None = Query(
            default=None,
            ge=1,
            description="Return only the N most recent events.",
        ),
    ) -> list[dict[str, Any]]:
        """Return the event log as a list of dictionaries.

        Args:
            last_n: If provided, only the *last_n* most recent events
                are included.

        Returns:
            A list of event dictionaries.
        """
        return _get_collector(request).history(last_n=last_n)

    @app.post("/stats/reset", tags=["monitoring"])
    def reset_stats(request: Request) -> dict[str, str]:
        """Clear all recorded events.

        Returns:
            A confirmation message.
        """
        _get_collector(request).reset()
        return {"message": "Stats cleared."}

    @app.get("/cache/stats", tags=["monitoring"])
    async def get_cache_stats(request: Request) -> dict[str, Any]:
        """Return runtime cache counters retrieved from generation API."""
        return await _fetch_generation_cache_stats(request)

    @app.post("/cache/reset", tags=["monitoring"])
    async def reset_cache(
        request: Request,
        namespace: str | None = Query(default=None),
        all_namespaces: bool = Query(default=False),
    ) -> dict[str, int]:
        """Reset generation cache via ai-api internal monitoring endpoint."""
        return await _reset_generation_cache(
            request,
            namespace=namespace,
            all_namespaces=all_namespaces,
        )

    @app.get("/metrics", tags=["monitoring"], response_class=PlainTextResponse)
    async def metrics(request: Request) -> str:
        """Return Prometheus scrape-compatible metrics text."""
        return summary_to_prometheus(await _summary_with_distribution(request))

    _install_api_key_openapi(
        app,
        public_paths={
            "/health",
            "/docs",
            "/openapi.json",
            "/docs/oauth2-redirect",
            "/redoc",
        },
    )

    return app


# Default app instance for ``uvicorn ai_engine.observability.api:app``.
app: FastAPI = create_app()
