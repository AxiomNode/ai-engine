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
    (default: ``paraphrase-multilingual-MiniLM-L12-v2``).
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
import time
import uuid
from contextlib import asynccontextmanager, suppress
from typing import Any

import httpx
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(name)s  %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_WORD_PASS_LETTERS = "A,B,C,D,E,F,G,H,I,J,L,M,N,O,P,R,S,T,V,Z"

SUPPORTED_LANGUAGES_CATALOG: list[dict[str, str]] = [
    {"code": "en", "name": "english"},
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
    from fastapi.middleware.cors import CORSMiddleware
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
from ai_engine.api.events import (  # noqa: E402
    _generation_failure_metadata,
    _get_collector,
    _handle_generation_failure,
    _record_observability_event,
)
from ai_engine.api.limiters import (  # noqa: E402
    _acquire_generation_capacity,
    _enforce_generation_rate_limit,
    _FixedWindowRateLimiter,
    _GenerationCapacityLimiter,
    get_capacity_limiter_stats,
)
from ai_engine.api.llama_target_store import (  # noqa: E402
    LlamaTargetStore,
    PersistedLlamaTarget,
)
from ai_engine.api.middleware import add_api_key_middleware  # noqa: E402
from ai_engine.api.optimization import GenerationOptimizationService  # noqa: E402
from ai_engine.api.schemas import (  # noqa: E402
    GenerateRequest,
    GenerateSDKResponse,
    IngestRequest,
)
from ai_engine.config import get_settings  # noqa: E402
from ai_engine.examples import get_corpus_signature  # noqa: E402
from ai_engine.examples.corpus import get_full_corpus  # noqa: E402
from ai_engine.games.catalog import estimate_effective_max_tokens  # noqa: E402
from ai_engine.games.prompts import get_prompt_version  # noqa: E402
from ai_engine.observability.collector import StatsCollector  # noqa: E402
from ai_engine.safety import (  # noqa: E402
    SafetyDecision,
    evaluate_user_prompt,
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
    embedder = SentenceTransformersEmbedder(
        model_name=embedding_model,
        device=settings.embedding_device,
        batch_size=settings.embedding_batch_size,
    )
    vector_store = _build_vector_store_from_settings(settings)
    reranker = _build_reranker_from_settings(settings)
    pipeline = RAGPipeline(
        embedder=embedder,
        vector_store=vector_store,
        context_char_limit=settings.rag_context_char_limit,
        query_embedding_cache_max_entries=settings.query_embedding_cache_max_entries,
        retrieval_result_cache_max_entries=settings.retrieval_result_cache_max_entries,
        candidate_multiplier=settings.retriever_candidate_multiplier,
        metadata_match_boost=settings.retriever_metadata_match_boost,
        lexical_content_match_boost=settings.retriever_lexical_content_match_boost,
        lexical_metadata_match_boost=settings.retriever_lexical_metadata_match_boost,
        reranker=reranker,
        rerank_candidate_count=settings.retriever_rerank_candidate_count,
        rerank_score_weight=settings.retriever_rerank_score_weight,
    )

    logger.info(
        "Connecting to LLM: %s",
        api_url if api_url else model_path,
    )
    llm = LlamaClient(
        api_url=api_url,
        model_path=model_path,
        json_mode=True,
        request_timeout_seconds=settings.llama_timeout_seconds,
        max_concurrent_requests=settings.llama_max_concurrent_requests,
    )
    raw_gen = GameGenerator(rag_pipeline=pipeline, llm_client=llm)

    # Wrap for automatic stats collection
    collector = StatsCollector()
    tracked = TrackedGameGenerator(raw_gen, collector)

    return tracked, pipeline


def _build_vector_store_from_settings(settings: Any) -> Any:
    backend = str(getattr(settings, "vector_store_backend", "memory") or "memory")
    normalized_backend = backend.strip().lower()

    if normalized_backend == "memory":
        from ai_engine.rag.vector_store import InMemoryVectorStore

        logger.info("Using in-memory vector store backend")
        return InMemoryVectorStore()

    if normalized_backend == "chroma":
        from ai_engine.rag.vectorstores.chroma import ChromaVectorStore

        collection_name = (
            str(
                getattr(settings, "vector_store_collection", "ai_engine_default")
                or "ai_engine_default"
            ).strip()
            or "ai_engine_default"
        )
        path = str(getattr(settings, "vector_store_path", "data/chroma") or "").strip()
        logger.info(
            "Using Chroma vector store backend collection=%s path=%s",
            collection_name,
            path,
        )
        return ChromaVectorStore(collection_name=collection_name, path=path or None)

    raise RuntimeError(
        "Unsupported AI_ENGINE_VECTOR_STORE_BACKEND: "
        f"{settings.vector_store_backend}"
    )


def _build_reranker_from_settings(settings: Any) -> Any | None:
    backend = str(getattr(settings, "retriever_reranker_backend", "none") or "none")
    normalized_backend = backend.strip().lower()

    if normalized_backend == "none":
        logger.info("Retriever second-stage reranker disabled")
        return None

    if normalized_backend == "lexical":
        from ai_engine.rag.reranker import LexicalReranker

        logger.info("Using lexical second-stage reranker")
        return LexicalReranker()

    raise RuntimeError(
        "Unsupported AI_ENGINE_RETRIEVER_RERANKER_BACKEND: "
        f"{settings.retriever_reranker_backend}"
    )


class LlamaTargetUpdateRequest(BaseModel):
    host: str = Field(min_length=1, max_length=255)
    protocol: str = Field(default="http", pattern="^(http|https)$")
    port: int = Field(default=7002, ge=1, le=65535)
    label: str | None = Field(default=None, max_length=80)


def _normalize_llama_host(raw: str) -> str:
    trimmed = raw.strip().replace("http://", "").replace("https://", "").rstrip("/")
    host = (trimmed.split("/", 1)[0] or "").split(":", 1)[0]
    if not host or not all(ch.isalnum() or ch in ".-" for ch in host):
        raise ValueError("host must be a valid hostname or IPv4 address")
    return host


def _build_llama_url(protocol: str, host: str, port: int) -> str:
    return f"{protocol}://{host}:{port}/v1/completions"


def _parse_llama_url(url: str | None) -> tuple[str | None, str | None, int | None]:
    if not url:
        return None, None, None
    try:
        parsed = httpx.URL(url)
    except Exception:
        return None, None, None
    protocol = parsed.scheme if parsed.scheme in {"http", "https"} else None
    port = parsed.port
    if port is None and protocol == "http":
        port = 80
    if port is None and protocol == "https":
        port = 443
    return parsed.host or None, protocol, port


def _unwrap_llama_client(generator: Any) -> Any | None:
    raw_generator = _unwrap_generator(generator) if generator is not None else None
    return getattr(raw_generator, "llm_client", None)


async def _apply_runtime_llama_url(app: FastAPI, url: str | None) -> None:
    llm_client = _unwrap_llama_client(app.state.generator)
    if llm_client is None or not hasattr(llm_client, "set_api_url"):
        raise RuntimeError("Runtime llama target update is unavailable")
    await llm_client.set_api_url(url)
    app.state.llama_url = url


async def _get_plugin_capabilities_payload(app: FastAPI) -> dict[str, Any]:
    """Return the current thin-plugin capability view for the LLM runtime."""
    target = _get_llama_target_payload(app)
    llm_client = _unwrap_llama_client(app.state.generator)
    base_payload: dict[str, Any] = {
        "pluginType": "thin-llm-runtime",
        "target": target,
    }

    if llm_client is None or not hasattr(llm_client, "plugin_capabilities"):
        return {
            **base_payload,
            "status": "unavailable",
            "reason": "llm client does not expose plugin capabilities",
        }

    try:
        capabilities = await llm_client.plugin_capabilities()
    except httpx.HTTPError as exc:
        return {
            **base_payload,
            "status": "unreachable",
            "error": str(exc),
        }
    except Exception as exc:  # noqa: BLE001
        return {
            **base_payload,
            "status": "error",
            "error": str(exc),
        }

    status = capabilities.get("status") if isinstance(capabilities, dict) else None
    return {
        **base_payload,
        "status": status or "unknown",
        "runtime": capabilities,
    }


def _get_llama_target_payload(app: FastAPI) -> dict[str, Any]:
    current_url = getattr(app.state, "llama_url", None)
    env_url = getattr(app.state, "llama_env_url", None)
    override = getattr(app.state, "llama_target_override", None)
    host, protocol, port = _parse_llama_url(current_url)
    return {
        "source": "override" if override is not None else "env",
        "label": override.label if override is not None else None,
        "host": host,
        "protocol": protocol,
        "port": port,
        "llamaBaseUrl": current_url,
        "envLlamaBaseUrl": env_url,
        "updatedAt": override.updated_at if override is not None else None,
    }


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

CATEGORY_NAME_BY_ID = {entry["id"]: entry["name"] for entry in GAME_CATEGORIES_CATALOG}

_WARMUP_GAME_TYPES = ["quiz", "word-pass", "true_false"]
_WARMUP_LANGUAGES = ["en"]


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
        for cid, cname in _WARMUP_CATEGORIES:
            req = GenerateRequest(
                query=cname,
                game_type=gt,
                difficulty_percentage=50,
                category_id=cid,
                category_name=cname,
                use_cache=True,
            )
            try:
                result = await optimizer.generate(req, correlation_id="warmup")
                if result.metrics.get("cache_hit"):
                    skipped += 1
                else:
                    ok += 1
                logger.debug(
                    "cache-warmup: %s|%s → %s",
                    gt,
                    cname,
                    "hit" if result.metrics.get("cache_hit") else "generated",
                )
            except Exception:
                failed += 1
                logger.warning(
                    "cache-warmup: %s|%s failed",
                    gt,
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

            docs = ExampleInjector._corpus_to_documents(get_full_corpus())
            if docs:
                _normalize_rag_document_metadata(
                    docs,
                    default_source="ai-engine-curated-corpus",
                    ingest_model="bootstrap",
                )
                await asyncio.to_thread(rag_pipeline.ingest, docs)
                if optimizer is not None:
                    await asyncio.to_thread(optimizer.on_ingest, docs)
        except Exception:
            logger.exception("example-bootstrap: failed to inject curated corpus")
            return

    if rag_pipeline is not None and optimizer is not None:
        try:
            rebuilt = await asyncio.to_thread(optimizer.rebuild_kbd_from_rag_store)
            logger.info("kbd-bootstrap: indexed %d RAG documents", rebuilt)
        except Exception:
            logger.exception("kbd-bootstrap: failed to rebuild KBD from RAG store")

    if cache_warmup_enabled and optimizer is not None:
        await _warmup_cache(optimizer)


_UNKNOWN_RAG_SOURCE_VALUES = {"", "unknown", "none", "null"}


def _metadata_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    return normalized or None


def _normalize_rag_document_metadata(
    documents: list[Any],
    *,
    default_source: str,
    ingest_model: str,
) -> None:
    """Ensure RAG documents carry consistent operational metadata."""
    safe_source = default_source.strip() or ingest_model or "rag-ingest"
    normalized_model = ingest_model.strip() or "generic"
    game_type = (
        None if normalized_model in {"generic", "bootstrap"} else normalized_model
    )

    for doc in documents:
        metadata = getattr(doc, "metadata", None)
        if not isinstance(metadata, dict):
            metadata = {}
            setattr(doc, "metadata", metadata)

        source = _metadata_text(metadata.get("source"))
        if source is None or source.lower() in _UNKNOWN_RAG_SOURCE_VALUES:
            metadata["source"] = safe_source

        if not _metadata_text(metadata.get("language")):
            metadata["language"] = "en"

        if game_type and not _metadata_text(metadata.get("game_type")):
            metadata["game_type"] = game_type

        if not _metadata_text(metadata.get("ingest_model")):
            metadata["ingest_model"] = normalized_model


def _get_generator(request: Request) -> Any:
    """Retrieve the generator from app state."""
    return request.app.state.generator


def _get_pipeline(request: Request) -> Any:
    """Retrieve the RAG pipeline from app state."""
    return request.app.state.rag_pipeline


def _get_optimizer(request: Request) -> Any:
    """Retrieve the generation optimization service from app state."""
    return request.app.state.optimizer


def _invalidate_rag_stats_cache(app: FastAPI) -> None:
    """Drop cached RAG diagnostics so future reads recompute fresh stats."""
    app.state.rag_stats_cache = None


def _get_cached_rag_stats(request: Request, pipeline: Any) -> dict[str, Any]:
    """Return cached RAG stats when still fresh, otherwise recompute them."""
    ttl_ms = max(0, get_settings().diagnostics_cache_ttl_ms)
    now = time.monotonic()
    cached = getattr(request.app.state, "rag_stats_cache", None)
    if (
        ttl_ms > 0
        and isinstance(cached, dict)
        and cached.get("expires_at", -1.0) > now
        and isinstance(cached.get("payload"), dict)
    ):
        return cached["payload"]

    payload = compute_rag_stats(pipeline)
    if ttl_ms > 0:
        request.app.state.rag_stats_cache = {
            "payload": payload,
            "expires_at": now + (ttl_ms / 1000),
        }
    else:
        request.app.state.rag_stats_cache = None

    return payload


def _get_distribution_version(request: Request) -> str:
    """Return distribution-version tag associated with this app instance."""
    value = getattr(request.app.state, "distribution_version", None)
    if isinstance(value, str) and value.strip():
        return value.strip()
    return "unknown-v0"


def _resolve_effective_max_tokens(req: GenerateRequest) -> int:
    """Clamp token budgets to realistic per-game envelopes for staging throughput."""
    return estimate_effective_max_tokens(
        req.game_type,
        req.max_tokens,
        item_count=req.item_count,
        letters=req.letters,
    )


async def _enforce_prompt_safety(
    request: Request,
    req: GenerateRequest,
    *,
    correlation_id: str,
    distribution_version: str,
) -> SafetyDecision:
    """Reject prompts that match jailbreak heuristics with HTTP 422.

    Returns the :class:`SafetyDecision` so callers can attach the
    ``safety_*`` metadata to downstream observability events. When the
    decision is ``block`` the function records a failed generation event
    with ``safety_block_reason`` and raises :class:`HTTPException`.
    """
    decision = evaluate_user_prompt(req.resolved_topic)
    if not decision.blocked:
        return decision

    metadata = _generation_failure_metadata(
        req,
        correlation_id=correlation_id,
        distribution_version=distribution_version,
        extra_metadata={
            "safety_block_reason": decision.primary_reason,
            "safety_score": decision.score,
            "safety_categories": list(decision.matched_categories),
            "safety_patterns": list(decision.matched_patterns),
            "error_type": "prompt_policy_violation",
        },
    )
    await _record_observability_event(
        request,
        prompt=req.resolved_topic,
        response="",
        latency_ms=0.0,
        max_tokens=req.max_tokens,
        json_mode=True,
        success=False,
        game_type=req.game_type,
        error="prompt-policy-violation",
        metadata=metadata,
    )
    logger.warning(
        "generate prompt rejected by safety policy "
        "correlation_id=%s reason=%s score=%.2f",
        correlation_id,
        decision.primary_reason,
        decision.score,
    )
    raise HTTPException(
        status_code=422,
        detail=("Prompt rejected by safety policy " f"({decision.primary_reason})."),
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


def _apply_generate_headers(
    req: GenerateRequest,
    *,
    language_header: str | None,
    difficulty_header: int | None,
) -> GenerateRequest:
    """Apply optional generation overrides coming from HTTP headers."""
    updates: dict[str, Any] = {}
    if difficulty_header is not None:
        updates["difficulty_percentage"] = max(0, min(100, int(difficulty_header)))
    return req.model_copy(update=updates) if updates else req


def _build_model_generate_request(
    *,
    game_type: str,
    query_text: str | None,
    language_header: str | None,
    difficulty_header: int | None,
    item_count: int,
    category_id: str | None,
    category_name: str | None,
    letters: str | None,
    max_tokens: int,
    use_cache: bool,
    force_refresh: bool,
) -> GenerateRequest:
    """Build a validated GenerateRequest from query params and headers."""
    resolved_category_id = (
        category_id.strip()
        if isinstance(category_id, str) and category_id.strip()
        else None
    )
    resolved_category_name = _resolve_category_name(
        resolved_category_id,
        category_name,
    )
    req = GenerateRequest(
        query=query_text,
        game_type=game_type,
        difficulty_percentage=max(0, min(100, int(difficulty_header or 50))),
        category_id=resolved_category_id,
        category_name=resolved_category_name,
        item_count=item_count,
        letters=letters,
        max_tokens=max_tokens,
        use_cache=use_cache,
        force_refresh=force_refresh,
    )
    return req


def _resolve_category_name(
    category_id: str | None,
    category_name: str | None,
) -> str | None:
    if category_id and category_id in CATEGORY_NAME_BY_ID:
        return CATEGORY_NAME_BY_ID[category_id]
    if isinstance(category_name, str) and category_name.strip():
        return category_name.strip()
    return None


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
        "generation_capacity": get_capacity_limiter_stats(request),
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
                "Game-generation endpoints consumed by backoffice workflows and game microservices."
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

                override = await app.state.llama_target_store.load()
                app.state.llama_target_override = override
                effective_llama_url = (
                    override.url if override is not None else app.state.llama_env_url
                )
                await _apply_runtime_llama_url(app, effective_llama_url)

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
                        embedding_model=settings.embedding_model,
                        corpus_signature=get_corpus_signature(),
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
    app.state.llama_env_url = settings.llama_url
    app.state.llama_url = settings.llama_url
    app.state.llama_target_store = LlamaTargetStore(settings.llama_target_state_file)
    app.state.llama_target_override = None
    app.state.stats_api_key = (
        settings.stats_api_key or settings.bridge_api_key or settings.api_key
    )
    app.state.rag_stats_cache = None
    app.state.rate_limiter = (
        _FixedWindowRateLimiter(
            max_requests=settings.rate_limit_requests,
            window_seconds=settings.rate_limit_window_seconds,
        )
        if settings.rate_limit_enabled
        else None
    )
    app.state.generation_capacity_limiter = _GenerationCapacityLimiter(
        max_in_flight=settings.generation_max_in_flight,
        max_queue_size=settings.generation_max_queue_size,
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

    # Sentry error tracking — opt-in via AI_ENGINE_SENTRY_DSN.
    if settings.sentry_dsn:
        try:
            import sentry_sdk  # type: ignore[import-untyped]

            sentry_sdk.init(dsn=settings.sentry_dsn, traces_sample_rate=0.1)
            logger.info("Sentry initialised")
        except ImportError:
            logger.warning(
                "sentry_dsn is set but sentry-sdk is not installed; "
                "install it with: pip install sentry-sdk[fastapi]"
            )

    # CORS — allow origins from AI_ENGINE_CORS_ALLOWED_ORIGINS (default: open).
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_allowed_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["X-API-Key", "X-Correlation-ID", "Content-Type"],
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

    @app.get("/internal/admin/llama-target", tags=["internal"], include_in_schema=False)
    def get_internal_llama_target(request: Request) -> dict[str, Any]:
        return _get_llama_target_payload(request.app)

    @app.get(
        "/internal/plugin/capabilities", tags=["internal"], include_in_schema=False
    )
    async def get_internal_plugin_capabilities(request: Request) -> dict[str, Any]:
        """Return the effective thin-plugin runtime capabilities."""
        return await _get_plugin_capabilities_payload(request.app)

    @app.put("/internal/admin/llama-target", tags=["internal"], include_in_schema=False)
    async def put_internal_llama_target(
        request: Request, payload: LlamaTargetUpdateRequest
    ) -> dict[str, Any]:
        host = _normalize_llama_host(payload.host)
        url = _build_llama_url(payload.protocol, host, payload.port)
        override = PersistedLlamaTarget(
            url=url,
            label=(
                payload.label.strip()
                if isinstance(payload.label, str) and payload.label.strip()
                else None
            ),
            updated_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )
        await _apply_runtime_llama_url(request.app, override.url)
        await request.app.state.llama_target_store.save(override)
        request.app.state.llama_target_override = override
        return _get_llama_target_payload(request.app)

    @app.delete(
        "/internal/admin/llama-target", tags=["internal"], include_in_schema=False
    )
    async def delete_internal_llama_target(request: Request) -> dict[str, Any]:
        await request.app.state.llama_target_store.reset()
        request.app.state.llama_target_override = None
        await _apply_runtime_llama_url(request.app, request.app.state.llama_env_url)
        return _get_llama_target_payload(request.app)

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
        await _enforce_prompt_safety(
            request,
            req,
            correlation_id=_get_correlation_id(request),
            distribution_version=_get_distribution_version(request),
        )

        if _get_generator(request) is None:
            raise HTTPException(status_code=503, detail="Generator not initialised.")
        optimizer = _get_optimizer(request)
        if optimizer is None:
            raise HTTPException(status_code=503, detail="Generator not initialised.")

        capacity_limiter = await _acquire_generation_capacity(request)
        try:
            if req.force_refresh:
                req = req.model_copy(update={"use_cache": False})
            effective_max_tokens = _resolve_effective_max_tokens(req)
            effective_req = req.model_copy(update={"max_tokens": effective_max_tokens})
            correlation_id = _get_correlation_id(request)
            distribution_version = _get_distribution_version(request)
            logger.info(
                "generate request correlation_id=%s distribution_version=%s",
                correlation_id,
                distribution_version,
            )
            pv = get_prompt_version(req.game_type)
            started = time.perf_counter()
            try:
                result = await optimizer.generate(
                    effective_req, correlation_id=correlation_id
                )
                result.metrics["requested_max_tokens"] = req.max_tokens
                result.metrics["effective_max_tokens"] = effective_max_tokens
                result.metrics["prompt_version"] = pv
                await _record_observability_event(
                    request,
                    prompt=req.resolved_topic,
                    response=str(result.payload),
                    latency_ms=float(result.metrics.get("total_latency_ms", 0.0)),
                    max_tokens=effective_max_tokens,
                    json_mode=True,
                    success=True,
                    game_type=req.game_type,
                    metadata=result.metrics,
                    prompt_version=pv,
                )
                return result.payload
            except Exception as exc:
                await _handle_generation_failure(
                    exc,
                    request=request,
                    req=req,
                    correlation_id=correlation_id,
                    distribution_version=distribution_version,
                    effective_max_tokens=effective_max_tokens,
                    elapsed_ms=(time.perf_counter() - started) * 1000,
                    log_label="generate",
                )
        finally:
            if capacity_limiter is not None:
                capacity_limiter.release()

    async def _execute_generate_sdk(
        req: GenerateRequest,
        request: Request,
    ) -> GenerateSDKResponse:
        """Execute generation request and return SDK-shaped response."""
        _enforce_generation_rate_limit(request)
        await _enforce_prompt_safety(
            request,
            req,
            correlation_id=_get_correlation_id(request),
            distribution_version=_get_distribution_version(request),
        )

        if _get_generator(request) is None:
            raise HTTPException(status_code=503, detail="Generator not initialised.")
        optimizer = _get_optimizer(request)
        if optimizer is None:
            raise HTTPException(status_code=503, detail="Generator not initialised.")

        capacity_limiter = await _acquire_generation_capacity(request)
        try:
            if req.force_refresh:
                req = req.model_copy(update={"use_cache": False})
            effective_max_tokens = _resolve_effective_max_tokens(req)
            effective_req = req.model_copy(update={"max_tokens": effective_max_tokens})
            correlation_id = _get_correlation_id(request)
            distribution_version = _get_distribution_version(request)
            logger.info(
                "generate_sdk request correlation_id=%s distribution_version=%s",
                correlation_id,
                distribution_version,
            )
            pv = get_prompt_version(req.game_type)
            started = time.perf_counter()
            try:
                result = await optimizer.generate(
                    effective_req, correlation_id=correlation_id
                )
                result.metrics["requested_max_tokens"] = req.max_tokens
                result.metrics["effective_max_tokens"] = effective_max_tokens
                result.metrics["prompt_version"] = pv
                sdk_payload = result.sdk_payload
                await _record_observability_event(
                    request,
                    prompt=req.resolved_topic,
                    response=str(sdk_payload),
                    latency_ms=float(result.metrics.get("total_latency_ms", 0.0)),
                    max_tokens=effective_max_tokens,
                    json_mode=True,
                    success=True,
                    game_type=req.game_type,
                    metadata=result.metrics,
                    prompt_version=pv,
                )
                sdk_metadata = sdk_payload.get("metadata", {})
                return GenerateSDKResponse(
                    model_type=str(sdk_payload.get("model_type", req.game_type)),
                    metadata=sdk_metadata if isinstance(sdk_metadata, dict) else {},
                    data={
                        key: value
                        for key, value in sdk_payload.items()
                        if key not in {"model_type", "metadata"}
                    },
                    metrics=result.metrics,
                )
            except Exception as exc:
                await _handle_generation_failure(
                    exc,
                    request=request,
                    req=req,
                    correlation_id=correlation_id,
                    distribution_version=distribution_version,
                    effective_max_tokens=effective_max_tokens,
                    elapsed_ms=(time.perf_counter() - started) * 1000,
                    log_label="generate_sdk",
                )
        finally:
            if capacity_limiter is not None:
                capacity_limiter.release()

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
        _normalize_rag_document_metadata(
            docs,
            default_source=ingest_source,
            ingest_model=ingest_model,
        )
        pipeline.ingest(docs)
        _invalidate_rag_stats_cache(request.app)

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
        resolved_req = resolved_req.model_copy(
            update={
                "category_name": _resolve_category_name(
                    resolved_req.category_id,
                    resolved_req.category_name,
                )
            }
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
        resolved_req = resolved_req.model_copy(
            update={
                "category_name": _resolve_category_name(
                    resolved_req.category_id,
                    resolved_req.category_name,
                )
            }
        )
        return await _execute_generate_sdk(resolved_req, request)

    @app.post("/generate/quiz", tags=["generation"])
    async def generate_quiz(
        request: Request,
        query_text: str | None = Query(default=None, alias="query"),
        item_count: int | None = Query(default=None, ge=1, le=50),
        num_questions: int | None = Query(
            default=None, ge=1, le=50, include_in_schema=False
        ),
        max_tokens: int = Query(default=1024, ge=64, le=4096),
        use_cache: bool = Query(default=True),
        force_refresh: bool = Query(default=False),
        category_id: str | None = Query(default=None),
        category_name: str | None = Query(default=None),
        language: str | None = Query(default=None),
        difficulty_percentage: int | None = Query(default=None),
        x_game_language: str = Header(default="en", alias="X-Game-Language"),
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
            item_count=item_count if item_count is not None else (num_questions or 5),
            category_id=category_id,
            category_name=category_name,
            letters=None,
            max_tokens=max_tokens,
            use_cache=use_cache,
            force_refresh=force_refresh,
        )
        return await _execute_generate(req, request)

    @app.post("/generate/word-pass", tags=["generation"])
    async def generate_word_pass(
        request: Request,
        query_text: str | None = Query(default=None, alias="query"),
        item_count: int | None = Query(default=None, ge=1, le=50),
        num_questions: int | None = Query(
            default=None, ge=1, le=50, include_in_schema=False
        ),
        max_tokens: int = Query(default=1024, ge=64, le=4096),
        use_cache: bool = Query(default=True),
        force_refresh: bool = Query(default=False),
        category_id: str | None = Query(default=None),
        category_name: str | None = Query(default=None),
        language: str | None = Query(default=None),
        difficulty_percentage: int | None = Query(default=None),
        x_game_language: str = Header(default="en", alias="X-Game-Language"),
        x_difficulty_percentage: int = Header(
            default=50,
            alias="X-Difficulty-Percentage",
        ),
    ) -> dict[str, Any]:
        """Generate standalone WordPass entries with model-specific settings headers."""
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
            item_count=item_count if item_count is not None else (num_questions or 5),
            category_id=category_id,
            category_name=category_name,
            letters=None,
            max_tokens=max_tokens,
            use_cache=use_cache,
            force_refresh=force_refresh,
        )
        return await _execute_generate(req, request)

    @app.post("/generate/true-false", tags=["generation"])
    async def generate_true_false(
        request: Request,
        query_text: str | None = Query(default=None, alias="query"),
        item_count: int | None = Query(default=None, ge=1, le=50),
        num_questions: int | None = Query(
            default=None, ge=1, le=50, include_in_schema=False
        ),
        max_tokens: int = Query(default=1024, ge=64, le=4096),
        use_cache: bool = Query(default=True),
        force_refresh: bool = Query(default=False),
        category_id: str | None = Query(default=None),
        category_name: str | None = Query(default=None),
        language: str | None = Query(default=None),
        difficulty_percentage: int | None = Query(default=None),
        x_game_language: str = Header(default="en", alias="X-Game-Language"),
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
            item_count=item_count if item_count is not None else (num_questions or 5),
            category_id=category_id,
            category_name=category_name,
            letters=None,
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
        return _get_cached_rag_stats(request, pipeline)

    @app.post("/diagnostics/tests/run", tags=["diagnostics"])
    def run_diagnostics_tests(request: Request) -> dict[str, Any]:
        """Start a hallucination & quality diagnostic test run.

        Tests execute in a background thread. Poll ``GET /diagnostics/tests/status``
        for real-time progress.
        """
        pipeline = _get_pipeline(request)
        generator = _get_generator(request)
        return start_test_run(pipeline, generator=generator)

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
