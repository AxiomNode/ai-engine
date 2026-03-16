"""FastAPI application for the ai-engine content generation service.

Provides a factory function :func:`create_app` that returns a fully
configured :class:`~fastapi.FastAPI` instance with the following endpoints:

- ``GET  /health``           – Liveness check.
- ``POST /generate``         – Generate a structured educational game via RAG + LLM.
- ``POST /ingest``           – Ingest documents into the RAG pipeline.
- ``GET  /stats``            – Aggregate generation statistics.
- ``GET  /stats/history``    – Recent generation event log.

Environment variables (used when ``generator`` / ``rag_pipeline`` are not
injected directly):

- ``AI_ENGINE_LLAMA_URL``       – URL of a running llama.cpp HTTP server
  (e.g. ``http://llama-server:8080/completion``).  Takes priority over local.
- ``AI_ENGINE_MODEL_PATH``      – Path to a local GGUF model file (fallback
  when ``AI_ENGINE_LLAMA_URL`` is not set).
- ``AI_ENGINE_EMBEDDING_MODEL`` – sentence-transformers model name
  (default: ``all-MiniLM-L6-v2``).
- ``AI_ENGINE_GENERATION_CACHE_PATH`` – persistent cache file path for
    optimized generation responses.
- ``AI_ENGINE_GENERATION_CACHE_BACKEND`` – persistent cache backend
    (``tinydb`` or ``redis``).
- ``AI_ENGINE_GENERATION_CACHE_REDIS_URL`` – Redis URL used when backend is
    ``redis``.
- ``AI_ENGINE_GENERATION_CACHE_REDIS_PREFIX`` – key prefix used for Redis
    cache entries.
- ``AI_ENGINE_GENERATION_CACHE_NAMESPACE`` – namespace/version used for
    cache keying and selective invalidation.
- ``AI_ENGINE_API_KEY``         – When set, every request must include an
  ``X-API-Key`` header with this value.  Absent or incorrect keys return
  HTTP 401 / 403 respectively.  Unset means no authentication required.

Run standalone::

    uvicorn ai_engine.api.app:app --host 0.0.0.0 --port 8001 --reload
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

logger = logging.getLogger(__name__)

try:
    from fastapi import FastAPI, HTTPException, Query, Request
    from fastapi.responses import PlainTextResponse
except ImportError as _imp_err:  # pragma: no cover
    raise ImportError(
        "FastAPI is required for the generation API.  "
        "Install it with:  pip install ai-engine[api]"
    ) from _imp_err

from ai_engine.api.middleware import add_api_key_middleware  # noqa: E402
from ai_engine.api.optimization import GenerationOptimizationService  # noqa: E402
from ai_engine.api.schemas import (  # noqa: E402
    GenerateRequest,
    GenerateSDKResponse,
    IngestRequest,
)
from ai_engine.config import get_settings  # noqa: E402
from ai_engine.observability.collector import (  # noqa: E402
    StatsCollector,
    summary_to_prometheus,
)


def _build_from_env() -> tuple[Any, Any]:
    """Initialise components from environment variables.

    Reads ``AI_ENGINE_LLAMA_URL``, ``AI_ENGINE_MODEL_PATH``, and
    ``AI_ENGINE_EMBEDDING_MODEL`` to wire up the LLM client, RAG pipeline,
    and game generator.

    Returns:
        A ``(TrackedGameGenerator, RAGPipeline)`` tuple ready to attach to
        the FastAPI app state.

    Raises:
        RuntimeError: If neither ``AI_ENGINE_LLAMA_URL`` nor
            ``AI_ENGINE_MODEL_PATH`` is set.
    """
    from ai_engine.games.generator import GameGenerator
    from ai_engine.llm.llama_client import LlamaClient
    from ai_engine.observability.middleware import TrackedGameGenerator
    from ai_engine.rag.embedders.sentence_transformers import (
        SentenceTransformersEmbedder,
    )
    from ai_engine.rag.pipeline import RAGPipeline
    from ai_engine.rag.vector_store import InMemoryVectorStore

    settings = get_settings()
    api_url = settings.llama_url
    model_path = settings.model_path

    if api_url is None and model_path is None:
        raise RuntimeError(
            "Set AI_ENGINE_LLAMA_URL (HTTP server) or "
            "AI_ENGINE_MODEL_PATH (local GGUF) before starting the API."
        )

    embedding_model = settings.embedding_model
    logger.info("Loading embedding model: %s", embedding_model)
    embedder = SentenceTransformersEmbedder(model_name=embedding_model)
    pipeline = RAGPipeline(embedder=embedder, vector_store=InMemoryVectorStore())

    logger.info(
        "Connecting to LLM: %s",
        api_url if api_url else model_path,
    )
    llm = LlamaClient(api_url=api_url, model_path=model_path, json_mode=True)
    raw_gen = GameGenerator(rag_pipeline=pipeline, llm_client=llm)

    # Wrap for automatic stats collection
    collector = StatsCollector()
    tracked = TrackedGameGenerator(raw_gen, collector)

    return tracked, pipeline


def _get_generator(request: Request) -> Any:
    """Retrieve the generator from app state.

    Args:
        request: The incoming FastAPI request.

    Returns:
        The :class:`~ai_engine.observability.middleware.TrackedGameGenerator`
        attached to ``app.state``.
    """
    return request.app.state.generator


def _get_pipeline(request: Request) -> Any:
    """Retrieve the RAG pipeline from app state.

    Args:
        request: The incoming FastAPI request.

    Returns:
        The :class:`~ai_engine.rag.pipeline.RAGPipeline` attached to
        ``app.state``.
    """
    return request.app.state.rag_pipeline


def _get_collector(request: Request) -> StatsCollector:
    """Retrieve the StatsCollector from app state.

    Args:
        request: The incoming FastAPI request.

    Returns:
        The :class:`~ai_engine.observability.collector.StatsCollector`
        attached to ``app.state``.
    """
    return request.app.state.collector  # type: ignore[no-any-return]


def _get_optimizer(request: Request) -> Any:
    """Retrieve the generation optimization service from app state."""
    return request.app.state.optimizer


def _get_distribution_version(request: Request) -> str:
    """Return distribution-version tag associated with this app instance."""
    value = getattr(request.app.state, "distribution_version", None)
    if isinstance(value, str) and value.strip():
        return value.strip()
    return "unknown-v0"


class _FixedWindowRateLimiter:
    """Thread-safe fixed-window limiter keyed by client identity."""

    def __init__(self, max_requests: int, window_seconds: int) -> None:
        self._max_requests = max_requests
        self._window_seconds = window_seconds
        self._lock = threading.Lock()
        self._buckets: dict[str, tuple[float, int]] = {}

    def allow(self, identity: str) -> bool:
        """Return True when request is allowed, else False."""
        now = time.time()
        with self._lock:
            window_start, count = self._buckets.get(identity, (now, 0))
            if now - window_start >= self._window_seconds:
                self._buckets[identity] = (now, 1)
                return True

            if count >= self._max_requests:
                return False

            self._buckets[identity] = (window_start, count + 1)
            return True


def _get_rate_limiter(request: Request) -> _FixedWindowRateLimiter | None:
    """Return generation rate limiter from app state when enabled."""
    limiter = getattr(request.app.state, "rate_limiter", None)
    return limiter if isinstance(limiter, _FixedWindowRateLimiter) else None


def _resolve_rate_limit_identity(request: Request) -> str:
    """Build a stable limiter identity using API key first, then client IP."""
    api_key = request.headers.get("X-API-Key")
    if api_key:
        return f"api_key:{api_key}"
    client_host = request.client.host if request.client is not None else "unknown"
    return f"ip:{client_host}"


def _enforce_generation_rate_limit(request: Request) -> None:
    """Raise HTTP 429 when generation rate limit is exceeded."""
    limiter = _get_rate_limiter(request)
    if limiter is None:
        return

    identity = _resolve_rate_limit_identity(request)
    if not limiter.allow(identity):
        raise HTTPException(status_code=429, detail="Rate limit exceeded.")


def _get_correlation_id(request: Request) -> str:
    """Return request correlation ID from state or fallback header/value."""
    state_value = getattr(request.state, "correlation_id", None)
    if isinstance(state_value, str) and state_value.strip():
        return state_value
    header_value = request.headers.get("X-Correlation-ID")
    if header_value and header_value.strip():
        return header_value.strip()
    return uuid.uuid4().hex


def _unwrap_generator(generator: Any) -> Any:
    """Return wrapped raw generator instance when middleware wrappers are used."""
    return getattr(generator, "_generator", generator)


def _health_dependencies(request: Request) -> dict[str, Any]:
    """Build dependency diagnostics for health endpoint responses."""
    generator = _get_generator(request)
    pipeline = _get_pipeline(request)
    optimizer = _get_optimizer(request)

    raw_generator = _unwrap_generator(generator) if generator is not None else None
    llm_client = getattr(raw_generator, "llm_client", None)

    if llm_client is None:
        llm_status = {"status": "unavailable", "mode": "none", "target": None}
    elif getattr(llm_client, "api_url", None):
        llm_status = {
            "status": "ready",
            "mode": "api",
            "target": str(getattr(llm_client, "api_url", "")),
        }
    elif getattr(llm_client, "model_path", None):
        llm_status = {
            "status": "ready",
            "mode": "local",
            "target": str(getattr(llm_client, "model_path", "")),
        }
    else:
        llm_status = {"status": "unknown", "mode": "unknown", "target": None}

    embedder = getattr(pipeline, "embedder", None) if pipeline is not None else None
    vector_store = (
        getattr(pipeline, "vector_store", None) if pipeline is not None else None
    )

    cache_status: dict[str, Any] = {
        "status": "unavailable",
        "backend": "none",
        "memory_enabled": False,
        "namespace": None,
    }
    if optimizer is not None:
        try:
            cache_runtime = optimizer.cache_stats()
            cache_status = {
                "status": "ready",
                "backend": str(cache_runtime.get("persistent_backend", "none")),
                "memory_enabled": bool(cache_runtime.get("memory_enabled", False)),
                "namespace": cache_runtime.get("cache_namespace"),
            }
        except Exception:
            cache_status = {
                "status": "degraded",
                "backend": "unknown",
                "memory_enabled": False,
                "namespace": None,
            }

    return {
        "generator": {
            "status": "ready" if generator is not None else "unavailable",
            "type": type(raw_generator).__name__ if raw_generator is not None else None,
        },
        "rag_pipeline": {
            "status": "ready" if pipeline is not None else "unavailable",
            "embedder": type(embedder).__name__ if embedder is not None else None,
            "vector_store": (
                type(vector_store).__name__ if vector_store is not None else None
            ),
        },
        "llm": llm_status,
        "cache": cache_status,
    }


def create_app(
    generator: Any = None,
    rag_pipeline: Any = None,
    collector: StatsCollector | None = None,
) -> FastAPI:
    """Create and return a configured FastAPI generation application.

    When ``generator`` and ``rag_pipeline`` are provided they are used
    directly (useful for testing and sub-app mounting).  When omitted,
    the components are initialised from environment variables during the
    application lifespan startup event.

    Args:
        generator: A :class:`~ai_engine.observability.middleware.TrackedGameGenerator`
            (or duck-type compatible) to use for content generation.
        rag_pipeline: A :class:`~ai_engine.rag.pipeline.RAGPipeline` to use
            for document ingestion.
        collector: A :class:`~ai_engine.observability.collector.StatsCollector`
            to record generation events.  Creates a new one if ``None``.

    Returns:
        A :class:`~fastapi.FastAPI` instance ready to be served.
    """
    _collector = collector if collector is not None else StatsCollector()
    settings = get_settings()
    distribution_version = settings.distribution_version_tag

    # Store mutable references so lifespan can replace them if needed.
    _state: dict[str, Any] = {
        "generator": generator,
        "rag_pipeline": rag_pipeline,
        "optimizer": None,
    }

    @asynccontextmanager
    async def lifespan(app: FastAPI):  # type: ignore[type-arg]
        """Initialise and tear down application-level resources."""
        if app.state.generator is None:
            # Build from environment variables when no explicit injection.
            _state["generator"], _state["rag_pipeline"] = _build_from_env()
            app.state.generator = _state["generator"]
            app.state.rag_pipeline = _state["rag_pipeline"]

        if (
            app.state.optimizer is None
            and app.state.generator is not None
            and app.state.rag_pipeline is not None
        ):
            app.state.optimizer = GenerationOptimizationService(
                app.state.generator,
                app.state.rag_pipeline,
                cache_backend=settings.generation_cache_backend,
                cache_namespace=settings.generation_cache_namespace,
                distribution_version=distribution_version,
                persistent_cache_path=settings.generation_cache_path,
                redis_url=settings.generation_cache_redis_url,
                redis_prefix=settings.generation_cache_redis_prefix,
            )

        logger.info(
            "ai-engine generation API started distribution_version=%s",
            distribution_version,
        )
        yield
        logger.info(
            "ai-engine generation API shutting down distribution_version=%s",
            distribution_version,
        )

    app = FastAPI(
        title="ai-engine generation API",
        description=(
            "Content generation service for AxiomNode educational games. "
            "Exposes RAG-augmented LLM generation as a single REST endpoint."
        ),
        version="0.1.0",
        lifespan=lifespan,
    )

    # Set state immediately so the app is usable without triggering the lifespan
    # (e.g. when using TestClient without context manager, or when dependencies
    # are injected directly).  Lifespan will fill in None values from env vars.
    app.state.generator = generator
    app.state.rag_pipeline = rag_pipeline
    app.state.collector = _collector
    app.state.distribution_version = distribution_version
    app.state.rate_limiter = (
        _FixedWindowRateLimiter(
            max_requests=settings.rate_limit_requests,
            window_seconds=settings.rate_limit_window_seconds,
        )
        if settings.rate_limit_enabled
        else None
    )
    app.state.optimizer = (
        GenerationOptimizationService(
            generator,
            rag_pipeline,
            cache_max_entries=0,
            distribution_version=distribution_version,
            persistent_cache_path=None,
        )
        if generator is not None and rag_pipeline is not None
        else None
    )

    # Attach API key middleware when AI_ENGINE_API_KEY is set in the environment.
    add_api_key_middleware(app)

    @app.middleware("http")
    async def correlation_id_middleware(request: Request, call_next):  # type: ignore[no-untyped-def]
        """Attach a correlation ID to every request and response."""
        correlation_id = request.headers.get("X-Correlation-ID") or uuid.uuid4().hex
        request.state.correlation_id = correlation_id
        response = await call_next(request)
        response.headers["X-Correlation-ID"] = correlation_id
        response.headers["X-Distribution-Version"] = _get_distribution_version(request)
        return response

    # ------------------------------------------------------------------
    # Endpoints
    # ------------------------------------------------------------------

    @app.get("/health", tags=["monitoring"])
    def health(request: Request) -> dict[str, Any]:
        """Return basic health and event-count information.

        Returns:
            A dictionary with ``status`` and ``total_events`` keys.
        """
        return {
            "status": "ok",
            "total_events": len(_get_collector(request)),
            "correlation_id": _get_correlation_id(request),
            "distribution_version": _get_distribution_version(request),
            "dependencies": _health_dependencies(request),
        }

    @app.post("/generate", tags=["generation"])
    def generate(req: GenerateRequest, request: Request) -> dict[str, Any]:
        """Generate a structured educational game using RAG + LLM.

        The pipeline:
        1. Retrieves relevant context from the ingested knowledge base.
        2. Builds a game-type-specific prompt.
        3. Sends it to the LLM with JSON grammar constraint.
        4. Validates and returns the parsed game as JSON.

        Args:
            req: Generation parameters (topic, game_type, language, …).

        Returns:
            A dictionary with ``game_type`` and ``game`` keys matching the
            :class:`~ai_engine.games.schemas.GameEnvelope` structure.

        Raises:
            HTTPException 422: If JSON extraction or schema validation fails.
            HTTPException 429: If generation rate limit is exceeded.
            HTTPException 503: If the generator is not available.
        """
        _enforce_generation_rate_limit(request)

        gen = _get_generator(request)
        if gen is None:
            raise HTTPException(status_code=503, detail="Generator not initialised.")

        optimizer = _get_optimizer(request)
        if optimizer is None:
            raise HTTPException(status_code=503, detail="Generator not initialised.")

        if req.force_refresh:
            req = req.model_copy(update={"use_cache": False})

        correlation_id = _get_correlation_id(request)
        distribution_version = _get_distribution_version(request)
        logger.info(
            "generate request correlation_id=%s distribution_version=%s",
            correlation_id,
            distribution_version,
        )

        try:
            result = optimizer.generate(req, correlation_id=correlation_id)
            _get_collector(request).record_call(
                prompt=req.query,
                response=str(result.payload),
                latency_ms=float(result.metrics.get("total_latency_ms", 0.0)),
                max_tokens=req.max_tokens,
                json_mode=True,
                success=True,
                game_type=req.game_type,
                metadata=result.metrics,
            )
            return result.payload
        except ValueError as exc:
            _get_collector(request).record_call(
                prompt=req.query,
                response="",
                latency_ms=0.0,
                max_tokens=req.max_tokens,
                json_mode=True,
                success=False,
                game_type=req.game_type,
                error=str(exc),
                metadata={
                    "event_type": "generation",
                    "cache_hit": False,
                    "cache_layer": "none",
                    "language": req.language,
                    "correlation_id": correlation_id,
                    "distribution_version": distribution_version,
                },
            )
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/generate/sdk", tags=["generation"], response_model=GenerateSDKResponse)
    def generate_sdk(req: GenerateRequest, request: Request) -> GenerateSDKResponse:
        """Generate and return SDK-shaped game objects for microservice consumers."""
        _enforce_generation_rate_limit(request)

        gen = _get_generator(request)
        if gen is None:
            raise HTTPException(status_code=503, detail="Generator not initialised.")

        optimizer = _get_optimizer(request)
        if optimizer is None:
            raise HTTPException(status_code=503, detail="Generator not initialised.")

        if req.force_refresh:
            req = req.model_copy(update={"use_cache": False})

        correlation_id = _get_correlation_id(request)
        distribution_version = _get_distribution_version(request)
        logger.info(
            "generate_sdk request correlation_id=%s distribution_version=%s",
            correlation_id,
            distribution_version,
        )

        started = time.perf_counter()
        try:
            result = optimizer.generate(req, correlation_id=correlation_id)
            sdk_payload = result.sdk_payload
            _get_collector(request).record_call(
                prompt=req.query,
                response=str(sdk_payload),
                latency_ms=float(result.metrics.get("total_latency_ms", 0.0)),
                max_tokens=req.max_tokens,
                json_mode=True,
                success=True,
                game_type=req.game_type,
                metadata=result.metrics,
            )
            return GenerateSDKResponse(
                model_type=str(sdk_payload.get("model_type", req.game_type)),
                metadata=dict(sdk_payload.get("metadata", {})),
                data={
                    key: value
                    for key, value in sdk_payload.items()
                    if key not in {"model_type", "metadata"}
                },
                metrics=result.metrics,
            )
        except ValueError as exc:
            elapsed_ms = (time.perf_counter() - started) * 1000
            _get_collector(request).record_call(
                prompt=req.query,
                response="",
                latency_ms=elapsed_ms,
                max_tokens=req.max_tokens,
                json_mode=True,
                success=False,
                game_type=req.game_type,
                error=str(exc),
                metadata={
                    "event_type": "generation",
                    "cache_hit": False,
                    "cache_layer": "none",
                    "language": req.language,
                    "correlation_id": correlation_id,
                    "distribution_version": distribution_version,
                },
            )
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.post("/ingest", tags=["rag"])
    def ingest(req: IngestRequest, request: Request) -> dict[str, int]:
        """Ingest documents into the RAG knowledge base.

        Documents are chunked, embedded, and stored in the in-memory vector
        store.  Call this endpoint before ``/generate`` to provide the LLM
        with relevant context for a specific topic.

        Args:
            req: List of documents to ingest.

        Returns:
            A dictionary with ``ingested`` key reporting how many documents
            were processed.

        Raises:
            HTTPException 503: If the RAG pipeline is not available.
        """
        from ai_engine.rag.document import Document

        pipeline = _get_pipeline(request)
        if pipeline is None:
            raise HTTPException(status_code=503, detail="RAG pipeline not initialised.")

        started = time.perf_counter()
        docs = [
            Document(
                content=d.content,
                doc_id=d.doc_id,
                metadata=d.metadata,
            )
            for d in req.documents
        ]
        pipeline.ingest(docs)

        optimizer = _get_optimizer(request)
        if optimizer is not None:
            optimizer.on_ingest(docs)

        elapsed_ms = (time.perf_counter() - started) * 1000
        correlation_id = _get_correlation_id(request)
        distribution_version = _get_distribution_version(request)
        logger.info(
            "ingest request correlation_id=%s distribution_version=%s",
            correlation_id,
            distribution_version,
        )
        _get_collector(request).record_call(
            prompt="ingest",
            response=f"ingested={len(docs)}",
            latency_ms=elapsed_ms,
            max_tokens=0,
            json_mode=False,
            success=True,
            game_type="ingest",
            metadata={
                "event_type": "ingest",
                "rag_latency_ms": round(elapsed_ms, 2),
                "cache_hit": False,
                "cache_layer": "none",
                "language": "n/a",
                "db_writes": len(docs),
                "kbd_hits": len(docs),
                "correlation_id": correlation_id,
                "distribution_version": distribution_version,
            },
        )
        return {"ingested": len(docs)}

    @app.get("/stats", tags=["monitoring"])
    def get_stats(request: Request) -> dict[str, Any]:
        """Return aggregate statistics over all recorded generation events.

        Returns:
            A summary dictionary produced by
            :meth:`~ai_engine.observability.collector.StatsCollector.summary`.
        """
        summary = _get_collector(request).summary()
        summary["distribution_version"] = _get_distribution_version(request)
        optimizer = _get_optimizer(request)
        if optimizer is not None:
            summary["cache_runtime"] = optimizer.cache_stats()
        return summary

    @app.get("/cache/stats", tags=["monitoring"])
    def get_cache_stats(request: Request) -> dict[str, Any]:
        """Return runtime cache metrics for generation optimization layers."""
        optimizer = _get_optimizer(request)
        if optimizer is None:
            raise HTTPException(status_code=503, detail="Generator not initialised.")
        data = optimizer.cache_stats()
        data["distribution_version"] = _get_distribution_version(request)
        return data

    @app.post("/cache/reset", tags=["monitoring"])
    def reset_cache(
        request: Request,
        namespace: str | None = Query(
            default=None,
            description="Optional cache namespace/version to invalidate.",
        ),
        all_namespaces: bool = Query(
            default=False,
            description="When true, invalidate all namespaces in persistent cache.",
        ),
    ) -> dict[str, int]:
        """Clear in-memory and persistent generation cache entries."""
        optimizer = _get_optimizer(request)
        if optimizer is None:
            raise HTTPException(status_code=503, detail="Generator not initialised.")
        return optimizer.reset_cache(
            namespace=namespace,
            all_namespaces=all_namespaces,
        )

    @app.get("/stats/history", tags=["monitoring"])
    def get_history(
        request: Request,
        last_n: int | None = Query(
            default=None,
            ge=1,
            description="Return only the N most recent events.",
        ),
    ) -> list[dict[str, Any]]:
        """Return the raw generation event log.

        Args:
            last_n: If provided, only the *last_n* most recent events are
                included.

        Returns:
            A list of event dictionaries ordered oldest-first.
        """
        return _get_collector(request).history(last_n=last_n)

    @app.get("/metrics", tags=["monitoring"], response_class=PlainTextResponse)
    def get_metrics(request: Request) -> str:
        """Return Prometheus scrape-compatible metrics with cache runtime gauges."""
        summary = _get_collector(request).summary()
        summary["distribution_version"] = _get_distribution_version(request)
        optimizer = _get_optimizer(request)
        if optimizer is not None:
            summary["cache_runtime"] = optimizer.cache_stats()
        return summary_to_prometheus(summary)

    return app


# Module-level app instance — used by:
#   uvicorn ai_engine.api.app:app --port 8001
app: FastAPI = create_app()
