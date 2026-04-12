"""FastAPI application for the ai-engine content generation service.

Provides a factory function :func:`create_app` that returns a fully
configured :class:`~fastapi.FastAPI` instance with the following endpoints:

- ``GET  /health``           – Liveness check.
- ``POST /generate``         – Generate a structured educational game via RAG + LLM.
- ``POST /ingest``           – Ingest documents into the RAG pipeline.

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

import asyncio
import logging
import threading
import time
import uuid
from contextlib import asynccontextmanager, suppress
from typing import Any

import httpx

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(name)s  %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_WORD_PASS_LETTERS = "A,B,C,D,E,F,G,H,I,J,L,M,N,O,P,R,S,T,V,Z"

SUPPORTED_LANGUAGES_CATALOG: list[dict[str, str]] = [
    {"code": "es", "name": "spanish"},
    {"code": "en", "name": "english"},
    {"code": "fr", "name": "french"},
    {"code": "de", "name": "german"},
    {"code": "it", "name": "italian"},
]

GAME_CATEGORIES_CATALOG: list[dict[str, str]] = [
    {"id": "9", "name": "General Knowledge"},
    {"id": "10", "name": "Entertainment: Books"},
    {"id": "11", "name": "Entertainment: Film"},
    {"id": "12", "name": "Entertainment: Music"},
    {"id": "13", "name": "Entertainment: Musicals & Theatres"},
    {"id": "14", "name": "Entertainment: Television"},
    {"id": "15", "name": "Entertainment: Video Games"},
    {"id": "16", "name": "Entertainment: Board Games"},
    {"id": "17", "name": "Science & Nature"},
    {"id": "18", "name": "Science: Computers"},
    {"id": "19", "name": "Science: Mathematics"},
    {"id": "20", "name": "Mythology"},
    {"id": "21", "name": "Sports"},
    {"id": "22", "name": "Geography"},
    {"id": "23", "name": "History"},
    {"id": "24", "name": "Politics"},
    {"id": "25", "name": "Art"},
    {"id": "26", "name": "Celebrities"},
    {"id": "27", "name": "Animals"},
    {"id": "28", "name": "Vehicles"},
    {"id": "29", "name": "Entertainment: Comics"},
    {"id": "30", "name": "Science: Gadgets"},
    {"id": "31", "name": "Entertainment: Japanese Anime & Manga"},
    {"id": "32", "name": "Entertainment: Cartoon & Animations"},
]

try:
    from fastapi import FastAPI, Header, HTTPException, Query, Request, Response
    from fastapi.openapi.utils import get_openapi
except ImportError as _imp_err:  # pragma: no cover
    raise ImportError(
        "FastAPI is required for the generation API.  "
        "Install it with:  pip install ai-engine[api]"
    ) from _imp_err

from ai_engine.api.diagnostics import (  # noqa: E402
    compute_rag_stats,
    get_test_status,
    start_test_run,
)
from ai_engine.api.middleware import add_api_key_middleware  # noqa: E402
from ai_engine.api.optimization import GenerationOptimizationService  # noqa: E402
from ai_engine.api.schemas import (  # noqa: E402
    GenerateRequest,
    GenerateSDKResponse,
    IngestRequest,
)
from ai_engine.config import get_settings  # noqa: E402
from ai_engine.observability.collector import StatsCollector  # noqa: E402


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
    llm = LlamaClient(
        api_url=api_url,
        model_path=model_path,
        json_mode=True,
        request_timeout_seconds=settings.llama_timeout_seconds,
    )
    raw_gen = GameGenerator(rag_pipeline=pipeline, llm_client=llm)

    # Wrap for automatic stats collection
    collector = StatsCollector()
    tracked = TrackedGameGenerator(raw_gen, collector)

    return tracked, pipeline


# ── Cache warm-up ─────────────────────────────────────────────────────

_WARMUP_CATEGORIES = [
    ("9", "General Knowledge"),
    ("17", "Science & Nature"),
    ("21", "Sports"),
    ("22", "Geography"),
    ("23", "History"),
    ("25", "Art"),
    ("27", "Animals"),
]

_WARMUP_GAME_TYPES = ["quiz", "word-pass", "true_false"]
_WARMUP_LANGUAGES = ["es", "en"]


async def _warmup_cache(optimizer: GenerationOptimizationService) -> None:
    """Pre-generate popular game combinations in the background.

    Runs after the server is up so incoming requests are accepted
    immediately.  Each combo that already exists in cache is skipped
    (``use_cache=True``), so only cold entries trigger LLM calls.
    """
    from ai_engine.api.schemas import GenerateRequest

    total = len(_WARMUP_GAME_TYPES) * len(_WARMUP_LANGUAGES) * len(_WARMUP_CATEGORIES)
    logger.info("cache-warmup: starting %d combinations", total)
    ok, skipped, failed = 0, 0, 0

    for gt in _WARMUP_GAME_TYPES:
        for lang in _WARMUP_LANGUAGES:
            for cid, cname in _WARMUP_CATEGORIES:
                req = GenerateRequest(
                    query=cname,
                    game_type=gt,
                    language=lang,
                    difficulty_percentage=50,
                    use_cache=True,
                )
                try:
                    result = await optimizer.generate(req, correlation_id="warmup")
                    if result.metrics.get("cache_hit"):
                        skipped += 1
                    else:
                        ok += 1
                    logger.debug(
                        "cache-warmup: %s|%s|%s → %s",
                        gt,
                        lang,
                        cname,
                        "hit" if result.metrics.get("cache_hit") else "generated",
                    )
                except Exception:
                    failed += 1
                    logger.warning(
                        "cache-warmup: %s|%s|%s failed",
                        gt,
                        lang,
                        cname,
                        exc_info=True,
                    )
                # Small pause between LLM calls to avoid overloading.
                await asyncio.sleep(0.5)

    logger.info(
        "cache-warmup: done — generated=%d cached=%d failed=%d",
        ok,
        skipped,
        failed,
    )


async def _prime_runtime_content(
    rag_pipeline: Any | None,
    optimizer: GenerationOptimizationService | None,
    *,
    cache_warmup_enabled: bool,
) -> None:
    """Seed curated examples and optionally warm cache after readiness."""
    if rag_pipeline is not None:
        try:
            from ai_engine.examples import ExampleInjector

            await asyncio.to_thread(ExampleInjector(rag_pipeline).inject_all)
        except Exception:
            logger.exception("example-bootstrap: failed to inject curated corpus")
            return

    if cache_warmup_enabled and optimizer is not None:
        await _warmup_cache(optimizer)


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


def _generation_failure_metadata(
    req: GenerateRequest,
    *,
    correlation_id: str,
    distribution_version: str,
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build normalized metadata for failed generation events."""
    metadata: dict[str, Any] = {
        "event_type": "generation",
        "cache_hit": False,
        "cache_layer": "none",
        "game_type": req.game_type,
        "language": req.language,
        "requested_max_tokens": req.max_tokens,
        "difficulty_percentage": req.difficulty_percentage,
        "use_cache": req.use_cache,
        "force_refresh": req.force_refresh,
        "query_chars": len(req.query),
        "correlation_id": correlation_id,
        "distribution_version": distribution_version,
    }
    if extra_metadata:
        metadata.update(extra_metadata)
    return metadata


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

    app.openapi = custom_openapi


def _apply_generate_headers(
    req: GenerateRequest,
    *,
    language_header: str | None,
    difficulty_header: int | None,
) -> GenerateRequest:
    """Apply optional generation overrides coming from HTTP headers."""
    updates: dict[str, Any] = {}
    if isinstance(language_header, str) and language_header.strip():
        updates["language"] = language_header.strip().lower()
    if difficulty_header is not None:
        updates["difficulty_percentage"] = max(0, min(100, int(difficulty_header)))
    return req.model_copy(update=updates) if updates else req


def _build_model_generate_request(
    *,
    game_type: str,
    query_text: str,
    language_header: str | None,
    difficulty_header: int | None,
    num_questions: int,
    letters: str,
    max_tokens: int,
    use_cache: bool,
    force_refresh: bool,
) -> GenerateRequest:
    """Build a validated GenerateRequest from query params and headers."""
    req = GenerateRequest(
        query=query_text,
        game_type=game_type,
        language=(language_header or "es").strip().lower(),
        difficulty_percentage=max(0, min(100, int(difficulty_header or 50))),
        num_questions=num_questions,
        letters=letters,
        max_tokens=max_tokens,
        use_cache=use_cache,
        force_refresh=force_refresh,
    )
    return req


async def _publish_event_to_stats(request: Request, payload: dict[str, Any]) -> None:
    """Push observability event to ai-stats without breaking generation flow."""
    stats_url = getattr(request.app.state, "stats_url", None)
    if not isinstance(stats_url, str) or not stats_url.strip():
        return

    endpoint = f"{stats_url.rstrip('/')}/events"
    headers: dict[str, str] = {}
    stats_api_key = getattr(request.app.state, "stats_api_key", None)
    if isinstance(stats_api_key, str) and stats_api_key.strip():
        headers["X-API-Key"] = stats_api_key.strip()

    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            await client.post(endpoint, json=payload, headers=headers)
    except Exception:
        logger.warning("Failed to push observability event to ai-stats", exc_info=True)


async def _record_observability_event(
    request: Request,
    *,
    prompt: str,
    response: str,
    latency_ms: float,
    max_tokens: int,
    json_mode: bool,
    success: bool,
    game_type: str,
    metadata: dict[str, Any],
    error: str | None = None,
) -> None:
    """Record event locally and forward it to ai-stats."""
    _get_collector(request).record_call(
        prompt=prompt,
        response=response,
        latency_ms=latency_ms,
        max_tokens=max_tokens,
        json_mode=json_mode,
        success=success,
        game_type=game_type,
        error=error,
        metadata=metadata,
    )
    await _publish_event_to_stats(
        request,
        {
            "prompt": prompt,
            "response": response,
            "latency_ms": latency_ms,
            "max_tokens": max_tokens,
            "json_mode": json_mode,
            "success": success,
            "game_type": game_type,
            "error": error,
            "metadata": metadata,
        },
    )


class _FixedWindowRateLimiter:
    """Thread-safe fixed-window limiter keyed by client identity."""

    _CLEANUP_THRESHOLD = 10_000

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
                self._maybe_cleanup(now)
                return True

            if count >= self._max_requests:
                return False

            self._buckets[identity] = (window_start, count + 1)
            return True

    def _maybe_cleanup(self, now: float) -> None:
        """Evict expired buckets when the map grows too large (called under lock)."""
        if len(self._buckets) < self._CLEANUP_THRESHOLD:
            return
        expired = [
            k
            for k, (ws, _) in self._buckets.items()
            if now - ws >= self._window_seconds
        ]
        for k in expired:
            del self._buckets[k]


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


def _startup_status(request: Request) -> dict[str, Any]:
    """Return app bootstrap state for liveness and readiness endpoints."""
    status = getattr(request.app.state, "startup_status", "unknown")
    error = getattr(request.app.state, "startup_error", None)
    return {
        "status": status,
        "error": error,
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
    openapi_version = f"0.1.0+{distribution_version}"
    openapi_tags = [
        {
            "name": "service",
            "description": (
                "Service-level endpoints for liveness and dependency readiness."
            ),
        },
        {
            "name": "generation",
            "description": (
                "Game-generation endpoints consumed by game microservices."
            ),
        },
        {
            "name": "rag",
            "description": "RAG ingestion endpoints for knowledge synchronization.",
        },
        {
            "name": "catalog",
            "description": (
                "Canonical language and category catalogs for downstream clients."
            ),
        },
        {
            "name": "internal",
            "description": (
                "Internal-only endpoints used by ai-stats monitoring bridge."
            ),
        },
        {
            "name": "diagnostics",
            "description": (
                "AI diagnostics: test runner, hallucination benchmarks, and RAG health metrics."
            ),
        },
    ]

    # Store mutable references so lifespan can replace them if needed.
    _state: dict[str, Any] = {
        "generator": generator,
        "rag_pipeline": rag_pipeline,
        "optimizer": None,
    }

    @asynccontextmanager
    async def lifespan(app: FastAPI):  # type: ignore[type-arg]
        """Initialise and tear down application-level resources."""

        async def initialise_runtime_state(load_from_env: bool) -> None:
            app.state.startup_status = "initializing"
            app.state.startup_error = None

            try:
                if load_from_env:
                    # Build from environment variables without blocking socket bind.
                    _state["generator"], _state["rag_pipeline"] = (
                        await asyncio.to_thread(_build_from_env)
                    )
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

                app.state.startup_status = "ready"
                logger.info(
                    "ai-engine generation API started distribution_version=%s",
                    distribution_version,
                )

                if (
                    app.state.rag_pipeline is not None
                    or app.state.optimizer is not None
                ):
                    app.state.warmup_task = asyncio.create_task(
                        _prime_runtime_content(
                            app.state.rag_pipeline,
                            app.state.optimizer,
                            cache_warmup_enabled=settings.cache_warmup_enabled,
                        )
                    )
            except Exception as exc:
                app.state.startup_status = "failed"
                app.state.startup_error = str(exc)
                logger.exception(
                    "ai-engine generation API failed to initialise distribution_version=%s",
                    distribution_version,
                )

        if app.state.generator is None:
            app.state.bootstrap_task = asyncio.create_task(
                initialise_runtime_state(load_from_env=True)
            )
        else:
            await initialise_runtime_state(load_from_env=False)

        yield

        bootstrap_task = getattr(app.state, "bootstrap_task", None)
        if bootstrap_task is not None and not bootstrap_task.done():
            bootstrap_task.cancel()
            with suppress(asyncio.CancelledError):
                await bootstrap_task

        warmup_task = getattr(app.state, "warmup_task", None)
        if warmup_task is not None and not warmup_task.done():
            warmup_task.cancel()
            with suppress(asyncio.CancelledError):
                await warmup_task

        # Clean up async resources on shutdown
        raw_gen = (
            _unwrap_generator(app.state.generator) if app.state.generator else None
        )
        llm_client = getattr(raw_gen, "llm_client", None)
        if llm_client is not None and hasattr(llm_client, "close"):
            await llm_client.close()
        logger.info(
            "ai-engine generation API shutting down distribution_version=%s",
            distribution_version,
        )

    app = FastAPI(
        title=f"ai-engine game & RAG API ({distribution_version})",
        description=(
            "Dedicated API for game generation and RAG ingestion workflows. "
            "Monitoring, metrics, and cache management are exposed by ai-stats. "
            f"Active deployment version: {distribution_version}."
        ),
        version=openapi_version,
        lifespan=lifespan,
        openapi_tags=openapi_tags,
    )

    # Set state immediately so the app is usable without triggering the lifespan
    # (e.g. when using TestClient without context manager, or when dependencies
    # are injected directly).  Lifespan will fill in None values from env vars.
    app.state.generator = generator
    app.state.rag_pipeline = rag_pipeline
    app.state.collector = _collector
    app.state.distribution_version = distribution_version
    app.state.startup_status = (
        "ready" if generator is not None and rag_pipeline is not None else "pending"
    )
    app.state.startup_error = None
    app.state.bootstrap_task = None
    app.state.warmup_task = None
    app.state.stats_url = settings.stats_url
    app.state.stats_api_key = (
        settings.stats_api_key or settings.bridge_api_key or settings.api_key
    )
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

    # Attach route-scoped API key middleware for service-to-service integrations.
    add_api_key_middleware(
        app,
        service="generation",
        public_paths={
            "/health",
            "/ready",
            "/docs",
            "/openapi.json",
            "/docs/oauth2-redirect",
            "/redoc",
        },
    )

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

    @app.get("/health", tags=["service"])
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
            "startup": _startup_status(request),
            "dependencies": _health_dependencies(request),
        }

    @app.get("/ready", tags=["service"])
    def ready(request: Request, response: Response) -> dict[str, Any]:
        """Return readiness based on async bootstrap completion."""
        startup = _startup_status(request)
        if startup["status"] != "ready":
            response.status_code = 503

        return {
            "status": "ready" if startup["status"] == "ready" else "not_ready",
            "correlation_id": _get_correlation_id(request),
            "distribution_version": _get_distribution_version(request),
            "startup": startup,
            "dependencies": _health_dependencies(request),
        }

    @app.get("/catalogs", tags=["catalog"])
    def get_catalogs(request: Request) -> dict[str, Any]:
        """Return canonical language/category catalogs for microservice discovery."""
        return {
            "distribution_version": _get_distribution_version(request),
            "languages": SUPPORTED_LANGUAGES_CATALOG,
            "categories": GAME_CATEGORIES_CATALOG,
        }

    @app.get("/catalogs/languages", tags=["catalog"])
    def get_catalog_languages(request: Request) -> dict[str, Any]:
        """Return canonical language catalog."""
        return {
            "distribution_version": _get_distribution_version(request),
            "languages": SUPPORTED_LANGUAGES_CATALOG,
        }

    @app.get("/catalogs/categories", tags=["catalog"])
    def get_catalog_categories(request: Request) -> dict[str, Any]:
        """Return canonical category catalog."""
        return {
            "distribution_version": _get_distribution_version(request),
            "categories": GAME_CATEGORIES_CATALOG,
        }

    @app.get("/internal/cache/stats", tags=["internal"], include_in_schema=False)
    def get_internal_cache_stats(request: Request) -> dict[str, Any]:
        """Return cache runtime counters for internal monitoring consumers."""
        optimizer = _get_optimizer(request)
        if optimizer is None:
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
        return optimizer.cache_stats()

    @app.post("/internal/cache/reset", tags=["internal"], include_in_schema=False)
    def reset_internal_cache(
        request: Request,
        namespace: str | None = Query(default=None),
        all_namespaces: bool = Query(default=False),
    ) -> dict[str, int]:
        """Invalidate cache entries for one namespace or all namespaces."""
        optimizer = _get_optimizer(request)
        if optimizer is None:
            return {"removed_memory": 0, "removed_persistent": 0}
        return optimizer.reset_cache(
            namespace=namespace,
            all_namespaces=all_namespaces,
        )

    async def _execute_generate(
        req: GenerateRequest, request: Request
    ) -> dict[str, Any]:
        """Execute generation request and return normalized payload."""
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
            result = await optimizer.generate(req, correlation_id=correlation_id)
            await _record_observability_event(
                request,
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
            await _record_observability_event(
                request,
                prompt=req.query,
                response="",
                latency_ms=0.0,
                max_tokens=req.max_tokens,
                json_mode=True,
                success=False,
                game_type=req.game_type,
                error=str(exc),
                metadata=_generation_failure_metadata(
                    req,
                    correlation_id=correlation_id,
                    distribution_version=distribution_version,
                ),
            )
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:
            failure_metadata = _generation_failure_metadata(
                req,
                correlation_id=correlation_id,
                distribution_version=distribution_version,
                extra_metadata=getattr(exc, "generation_metrics", None),
            )
            logger.exception(
                "generate request failed correlation_id=%s game_type=%s max_tokens=%s",
                correlation_id,
                req.game_type,
                req.max_tokens,
            )
            await _record_observability_event(
                request,
                prompt=req.query,
                response="",
                latency_ms=float(failure_metadata.get("total_latency_ms", 0.0)),
                max_tokens=req.max_tokens,
                json_mode=True,
                success=False,
                game_type=req.game_type,
                error=str(exc),
                metadata=failure_metadata,
            )
            raise

    async def _execute_generate_sdk(
        req: GenerateRequest,
        request: Request,
    ) -> GenerateSDKResponse:
        """Execute generation request and return SDK-shaped response."""
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
            result = await optimizer.generate(req, correlation_id=correlation_id)
            sdk_payload = result.sdk_payload
            await _record_observability_event(
                request,
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
            await _record_observability_event(
                request,
                prompt=req.query,
                response="",
                latency_ms=elapsed_ms,
                max_tokens=req.max_tokens,
                json_mode=True,
                success=False,
                game_type=req.game_type,
                error=str(exc),
                metadata=_generation_failure_metadata(
                    req,
                    correlation_id=correlation_id,
                    distribution_version=distribution_version,
                ),
            )
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except Exception as exc:
            failure_metadata = _generation_failure_metadata(
                req,
                correlation_id=correlation_id,
                distribution_version=distribution_version,
                extra_metadata=getattr(exc, "generation_metrics", None),
            )
            logger.exception(
                "generate_sdk request failed correlation_id=%s game_type=%s max_tokens=%s",
                correlation_id,
                req.game_type,
                req.max_tokens,
            )
            await _record_observability_event(
                request,
                prompt=req.query,
                response="",
                latency_ms=float(failure_metadata.get("total_latency_ms", 0.0)),
                max_tokens=req.max_tokens,
                json_mode=True,
                success=False,
                game_type=req.game_type,
                error=str(exc),
                metadata=failure_metadata,
            )
            raise

    async def _execute_ingest(
        req: IngestRequest,
        request: Request,
        *,
        ingest_model: str,
        ingest_source: str,
    ) -> dict[str, int]:
        """Execute ingestion request and emit normalized observability event."""
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
        await _record_observability_event(
            request,
            prompt="ingest",
            response=f"ingested={len(docs)}",
            latency_ms=elapsed_ms,
            max_tokens=0,
            json_mode=False,
            success=True,
            game_type="ingest",
            metadata={
                "event_type": "ingest",
                "ingest_model": ingest_model,
                "ingest_source": ingest_source,
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

    @app.post("/generate", tags=["generation"])
    async def generate(
        req: GenerateRequest,
        request: Request,
        x_game_language: str | None = Header(default=None, alias="X-Game-Language"),
        x_difficulty_percentage: int | None = Header(
            default=None,
            alias="X-Difficulty-Percentage",
        ),
    ) -> dict[str, Any]:
        """Generate a structured educational game using RAG + LLM.

        The pipeline:
        1. Retrieves relevant context from the ingested knowledge base.
        2. Builds a game-type-specific prompt.
        3. Sends it to the LLM with JSON grammar constraint.
        4. Validates and returns the parsed game as JSON.

        Args:
            req: Generation parameters (game_type, language, …).

        Returns:
            A dictionary with ``game_type`` and ``game`` keys matching the
            :class:`~ai_engine.games.schemas.GameEnvelope` structure.

        Raises:
            HTTPException 422: If JSON extraction or schema validation fails.
            HTTPException 429: If generation rate limit is exceeded.
            HTTPException 503: If the generator is not available.
        """
        resolved_req = _apply_generate_headers(
            req,
            language_header=x_game_language,
            difficulty_header=x_difficulty_percentage,
        )
        return await _execute_generate(resolved_req, request)

    @app.post("/generate/sdk", tags=["generation"], response_model=GenerateSDKResponse)
    async def generate_sdk(
        req: GenerateRequest,
        request: Request,
        x_game_language: str | None = Header(default=None, alias="X-Game-Language"),
        x_difficulty_percentage: int | None = Header(
            default=None,
            alias="X-Difficulty-Percentage",
        ),
    ) -> GenerateSDKResponse:
        """Generate and return SDK-shaped game objects for microservice consumers."""
        resolved_req = _apply_generate_headers(
            req,
            language_header=x_game_language,
            difficulty_header=x_difficulty_percentage,
        )
        return await _execute_generate_sdk(resolved_req, request)

    @app.post("/generate/quiz", tags=["generation"])
    async def generate_quiz(
        request: Request,
        query_text: str = Query(..., alias="query"),
        num_questions: int = Query(default=5, ge=1, le=50),
        max_tokens: int = Query(default=1024, ge=64, le=4096),
        use_cache: bool = Query(default=True),
        force_refresh: bool = Query(default=False),
        language: str | None = Query(default=None),
        difficulty_percentage: int | None = Query(default=None),
        x_game_language: str = Header(default="es", alias="X-Game-Language"),
        x_difficulty_percentage: int = Header(
            default=50,
            alias="X-Difficulty-Percentage",
        ),
    ) -> dict[str, Any]:
        """Generate quiz with model-specific endpoint and header-driven settings."""
        resolved_language = language or x_game_language
        resolved_difficulty = (
            difficulty_percentage
            if difficulty_percentage is not None
            else x_difficulty_percentage
        )
        req = _build_model_generate_request(
            game_type="quiz",
            query_text=query_text,
            language_header=resolved_language,
            difficulty_header=resolved_difficulty,
            num_questions=num_questions,
            letters=DEFAULT_WORD_PASS_LETTERS,
            max_tokens=max_tokens,
            use_cache=use_cache,
            force_refresh=force_refresh,
        )
        return await _execute_generate(req, request)

    @app.post("/generate/word-pass", tags=["generation"])
    async def generate_word_pass(
        request: Request,
        query_text: str = Query(..., alias="query"),
        letters: str = Query(default=DEFAULT_WORD_PASS_LETTERS),
        num_questions: int = Query(default=5, ge=1, le=50),
        max_tokens: int = Query(default=1024, ge=64, le=4096),
        use_cache: bool = Query(default=True),
        force_refresh: bool = Query(default=False),
        language: str | None = Query(default=None),
        difficulty_percentage: int | None = Query(default=None),
        x_game_language: str = Header(default="es", alias="X-Game-Language"),
        x_difficulty_percentage: int = Header(
            default=50,
            alias="X-Difficulty-Percentage",
        ),
    ) -> dict[str, Any]:
        """Generate word-pass with model-specific endpoint and settings headers."""
        resolved_language = language or x_game_language
        resolved_difficulty = (
            difficulty_percentage
            if difficulty_percentage is not None
            else x_difficulty_percentage
        )
        req = _build_model_generate_request(
            game_type="word-pass",
            query_text=query_text,
            language_header=resolved_language,
            difficulty_header=resolved_difficulty,
            num_questions=num_questions,
            letters=letters,
            max_tokens=max_tokens,
            use_cache=use_cache,
            force_refresh=force_refresh,
        )
        return await _execute_generate(req, request)

    @app.post("/generate/true-false", tags=["generation"])
    async def generate_true_false(
        request: Request,
        query_text: str = Query(..., alias="query"),
        num_questions: int = Query(default=5, ge=1, le=50),
        max_tokens: int = Query(default=1024, ge=64, le=4096),
        use_cache: bool = Query(default=True),
        force_refresh: bool = Query(default=False),
        language: str | None = Query(default=None),
        difficulty_percentage: int | None = Query(default=None),
        x_game_language: str = Header(default="es", alias="X-Game-Language"),
        x_difficulty_percentage: int = Header(
            default=50,
            alias="X-Difficulty-Percentage",
        ),
    ) -> dict[str, Any]:
        """Generate true/false with model-specific endpoint and settings headers."""
        resolved_language = language or x_game_language
        resolved_difficulty = (
            difficulty_percentage
            if difficulty_percentage is not None
            else x_difficulty_percentage
        )
        req = _build_model_generate_request(
            game_type="true_false",
            query_text=query_text,
            language_header=resolved_language,
            difficulty_header=resolved_difficulty,
            num_questions=num_questions,
            letters=DEFAULT_WORD_PASS_LETTERS,
            max_tokens=max_tokens,
            use_cache=use_cache,
            force_refresh=force_refresh,
        )
        return await _execute_generate(req, request)

    @app.post("/ingest", tags=["rag"])
    async def ingest(
        req: IngestRequest,
        request: Request,
        source: str = Query(default="default"),
        x_ingest_source: str | None = Header(default=None, alias="X-Ingest-Source"),
    ) -> dict[str, int]:
        """Ingest documents into the RAG knowledge base.

        Documents are chunked, embedded, and stored in the in-memory vector
        store.  Call this endpoint before ``/generate`` to provide the LLM
        with relevant context for game generation.

        Args:
            req: List of documents to ingest.

        Returns:
            A dictionary with ``ingested`` key reporting how many documents
            were processed.

        Raises:
            HTTPException 503: If the RAG pipeline is not available.
        """
        ingest_source = x_ingest_source.strip() if x_ingest_source else source
        return await _execute_ingest(
            req,
            request,
            ingest_model="generic",
            ingest_source=ingest_source,
        )

    @app.post("/ingest/quiz", tags=["rag"])
    async def ingest_quiz(
        req: IngestRequest,
        request: Request,
        source: str = Query(default="quiz"),
        x_ingest_source: str | None = Header(default=None, alias="X-Ingest-Source"),
    ) -> dict[str, int]:
        """Ingest quiz-oriented content using model-specific ingest endpoint."""
        ingest_source = x_ingest_source.strip() if x_ingest_source else source
        return await _execute_ingest(
            req,
            request,
            ingest_model="quiz",
            ingest_source=ingest_source,
        )

    @app.post("/ingest/word-pass", tags=["rag"])
    async def ingest_word_pass(
        req: IngestRequest,
        request: Request,
        source: str = Query(default="word-pass"),
        x_ingest_source: str | None = Header(default=None, alias="X-Ingest-Source"),
    ) -> dict[str, int]:
        """Ingest word-pass-oriented content using model-specific endpoint."""
        ingest_source = x_ingest_source.strip() if x_ingest_source else source
        return await _execute_ingest(
            req,
            request,
            ingest_model="word-pass",
            ingest_source=ingest_source,
        )

    @app.post("/ingest/true-false", tags=["rag"])
    async def ingest_true_false(
        req: IngestRequest,
        request: Request,
        source: str = Query(default="true_false"),
        x_ingest_source: str | None = Header(default=None, alias="X-Ingest-Source"),
    ) -> dict[str, int]:
        """Ingest true/false-oriented content using model-specific endpoint."""
        ingest_source = x_ingest_source.strip() if x_ingest_source else source
        return await _execute_ingest(
            req,
            request,
            ingest_model="true_false",
            ingest_source=ingest_source,
        )

    # ------------------------------------------------------------------
    # Diagnostics endpoints
    # ------------------------------------------------------------------

    @app.get("/diagnostics/rag/stats", tags=["diagnostics"])
    def get_rag_stats(request: Request) -> dict[str, Any]:
        """Return RAG vector store health and coverage metrics."""
        pipeline = _get_pipeline(request)
        if pipeline is None:
            return {
                "total_chunks": 0,
                "total_chars": 0,
                "unique_documents": 0,
                "embedding_dimensions": 0,
                "avg_chunk_chars": 0,
                "coverage_level": "empty",
                "coverage_message": "RAG pipeline no inicializado.",
                "retriever_config": {},
                "sources": [],
            }
        return compute_rag_stats(pipeline)

    @app.post("/diagnostics/tests/run", tags=["diagnostics"])
    def run_diagnostics_tests(request: Request) -> dict[str, Any]:
        """Start a hallucination & quality diagnostic test run.

        Tests execute in a background thread. Poll ``GET /diagnostics/tests/status``
        for real-time progress.
        """
        pipeline = _get_pipeline(request)
        return start_test_run(pipeline)

    @app.get("/diagnostics/tests/status", tags=["diagnostics"])
    def get_diagnostics_tests_status(request: Request) -> dict[str, Any]:
        """Return current diagnostic test run status and results."""
        return get_test_status()

    _install_api_key_openapi(
        app,
        public_paths={
            "/health",
            "/ready",
            "/docs",
            "/openapi.json",
            "/docs/oauth2-redirect",
            "/redoc",
        },
    )

    return app


# Module-level app instance — used by:
#   uvicorn ai_engine.api.app:app --port 8001
app: FastAPI = create_app()
