"""FastAPI application for serving observability statistics.

Provides a factory function :func:`create_app` that returns a fully
configured :class:`~fastapi.FastAPI` instance with the following
endpoints:

- ``GET /health`` – Liveness check.
- ``POST /events`` – Ingest one observability event.
- ``GET /stats`` – Aggregate statistics from the collector.
- ``GET /stats/history`` – Recent event log.
- ``POST /stats/reset`` – Clear all recorded events.
- ``GET /metrics`` – Prometheus text exposition format.

The app can be run standalone via::

    uvicorn ai_engine.observability.api:app --reload

or embedded inside another FastAPI application as a sub-application.
"""

from __future__ import annotations

import time
from typing import Any

try:
    from fastapi import FastAPI, Query, Request
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

# Module-level collector used when running ``uvicorn … api:app``.
_default_collector = StatsCollector()

_start_time: float = time.time()


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


def _summary_with_distribution(request: Request) -> dict[str, Any]:
    """Return collector summary enriched with distribution-version tag."""
    summary = _get_collector(request).summary()
    summary["distribution_version"] = _get_distribution_version(request)
    return summary


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

    app = FastAPI(
        title="ai-engine observability",
        description="Statistics and monitoring API for ai-engine LLM usage.",
        version="0.1.0",
    )

    # Store collector on app state — endpoints read from here via the
    # ``Request`` object so each app instance uses its own collector.
    app.state.collector = stats
    app.state.distribution_version = distribution_version

    # Attach route-scoped API key middleware for bridge/internal integrations.
    if add_api_key_middleware is not None:
        add_api_key_middleware(
            app,
            service="observability",
            public_paths={"/health"},
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
    def get_stats(request: Request) -> dict[str, Any]:
        """Return aggregate statistics over all recorded events.

        Returns:
            A summary dictionary produced by
            :meth:`StatsCollector.summary`.
        """
        return _summary_with_distribution(request)

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

    @app.get("/metrics", tags=["monitoring"], response_class=PlainTextResponse)
    def metrics(request: Request) -> str:
        """Return Prometheus scrape-compatible metrics text."""
        return summary_to_prometheus(_summary_with_distribution(request))

    return app


# Default app instance for ``uvicorn ai_engine.observability.api:app``.
app: FastAPI = create_app()
