"""FastAPI application for serving observability statistics.

Provides a factory function :func:`create_app` that returns a fully
configured :class:`~fastapi.FastAPI` instance with the following
endpoints:

- ``GET /health`` – Liveness check.
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


def _get_collector(request: Request) -> StatsCollector:
    """Retrieve the :class:`StatsCollector` from app state.

    Args:
        request: The incoming FastAPI request.

    Returns:
        The collector attached to ``app.state``.
    """
    return request.app.state.collector  # type: ignore[no-any-return]


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

    # Attach API key middleware when AI_ENGINE_API_KEY is set in the environment.
    if add_api_key_middleware is not None:
        add_api_key_middleware(app)

    @app.middleware("http")
    async def distribution_header_middleware(request: Request, call_next):  # type: ignore[no-untyped-def]
        """Attach distribution-version tag to every response."""
        response = await call_next(request)
        response.headers["X-Distribution-Version"] = str(
            getattr(request.app.state, "distribution_version", "unknown-v0")
        )
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
            "distribution_version": str(app.state.distribution_version),
        }

    @app.get("/stats", tags=["monitoring"])
    def get_stats(request: Request) -> dict[str, Any]:
        """Return aggregate statistics over all recorded events.

        Returns:
            A summary dictionary produced by
            :meth:`StatsCollector.summary`.
        """
        summary = _get_collector(request).summary()
        summary["distribution_version"] = str(app.state.distribution_version)
        return summary

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
        summary = _get_collector(request).summary()
        summary["distribution_version"] = str(app.state.distribution_version)
        return summary_to_prometheus(summary)

    return app


# Default app instance for ``uvicorn ai_engine.observability.api:app``.
app: FastAPI = create_app()
