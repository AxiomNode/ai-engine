"""Centralised configuration for AI-Engine using pydantic-settings.

All environment variables consumed by the application are declared here as
typed fields on :class:`AIEngineSettings`.  Callers should use
:func:`get_settings` to obtain a fully-populated settings instance rather than
calling ``os.environ.get`` directly.

Environment variables
---------------------
``AI_ENGINE_LLAMA_URL``
    HTTP base URL of a running ``llama.cpp`` server (e.g.
    ``http://localhost:8080``).  Mutually exclusive with
    ``AI_ENGINE_MODEL_PATH``.

``AI_ENGINE_MODEL_PATH``
    Absolute path to a local GGUF model file for in-process inference.
    Mutually exclusive with ``AI_ENGINE_LLAMA_URL``.

``AI_ENGINE_EMBEDDING_MODEL``
    Name of the sentence-transformers model used for RAG embeddings.
    Defaults to ``"all-MiniLM-L6-v2"``.

``AI_ENGINE_EMBEDDING_DEVICE``
    Device used by sentence-transformers for embedding inference (e.g. ``cpu`` or ``cuda``).

``AI_ENGINE_EMBEDDING_BATCH_SIZE``
    Batch size used when encoding multiple RAG documents.

``AI_ENGINE_LLAMA_TIMEOUT_SECONDS``
    Timeout (seconds) for upstream HTTP calls from ai-api to llama.cpp.

``AI_ENGINE_LLAMA_MAX_CONCURRENT_REQUESTS``
    Maximum simultaneous upstream HTTP calls from ai-api to llama.cpp.

``AI_ENGINE_API_KEY``
    Shared secret propagated to ``X-API-Key`` header validation.  When
    absent, authentication is disabled.

``AI_ENGINE_GAMES_API_KEY``
    API key used for game microservices consuming generation endpoints
    (``/generate*``).

``AI_ENGINE_BRIDGE_API_KEY``
    API key used for bridge/backoffice integration on generation, ingest and
    observability endpoints.

``AI_ENGINE_STATS_API_KEY``
    Dedicated API key used by ``ai-api`` when pushing internal
    observability events to ``ai-stats``.

``AI_ENGINE_STATS_URL``
    Base URL for the observability service receiving event ingestion
    from the main API (e.g. ``http://ai-stats:8000``).

``AI_ENGINE_GENERATION_API_URL``
    Base URL for the generation API used by the observability service
    to query monitoring/cache runtime data (e.g. ``http://ai-api:8001``).

``AI_ENGINE_MODELS_DIR``
    Directory where GGUF model files are stored.  Defaults to the
    ``src/models/`` folder used by the tracked deployment distributions.

``AI_ENGINE_GENERATION_CACHE_PATH``
    File path used by the generation optimizer for persistent cache storage.
    Defaults to ``"data/generation_cache.json"``.

``AI_ENGINE_GENERATION_CACHE_BACKEND``
    Persistent cache backend for generation optimization.  Supported values:
    ``"tinydb"`` (default) and ``"redis"``.

``AI_ENGINE_GENERATION_CACHE_REDIS_URL``
    Redis connection URL used when backend is ``"redis"``.

``AI_ENGINE_GENERATION_CACHE_REDIS_PREFIX``
    Redis key prefix for generation cache data structures.

``AI_ENGINE_GENERATION_CACHE_NAMESPACE``
    Cache namespace/version for key-versioning and selective invalidation.

``AI_ENGINE_RATE_LIMIT_ENABLED``
    Enable request-level rate limiting on generation endpoints.

``AI_ENGINE_RATE_LIMIT_REQUESTS``
    Maximum allowed requests per client identity in one rate-limit window.

``AI_ENGINE_RATE_LIMIT_WINDOW_SECONDS``
    Fixed time window in seconds used for generation request limiting.

``AI_ENGINE_GENERATION_MAX_IN_FLIGHT``
    Maximum number of generation requests actively executing at once.

``AI_ENGINE_GENERATION_MAX_QUEUE_SIZE``
    Maximum number of additional generation requests allowed to wait for capacity.

``AI_ENGINE_QUERY_EMBEDDING_CACHE_MAX_ENTRIES``
    Maximum number of cached query embeddings kept in-process.

``AI_ENGINE_RETRIEVAL_RESULT_CACHE_MAX_ENTRIES``
    Maximum number of cached retrieval result sets kept in-process.

``AI_ENGINE_RETRIEVER_CANDIDATE_MULTIPLIER``
    Multiplier used to over-fetch vector results before metadata reranking.

``AI_ENGINE_RETRIEVER_METADATA_MATCH_BOOST``
    Score bonus added for matching retrieval metadata preferences.

``AI_ENGINE_RAG_CONTEXT_CHAR_LIMIT``
    Default context length budget used when building prompts from retrieved chunks.

``AI_ENGINE_DIAGNOSTICS_CACHE_TTL_MS``
    Time-to-live in milliseconds for short-lived diagnostics endpoint caches.

``AI_ENGINE_DISTRIBUTION``
    Deployment distribution label (e.g. ``dev``, ``stg``, ``pro``).

``AI_ENGINE_RELEASE_VERSION``
    Deployment release version tag (e.g. ``v1``, ``2026.03.16``).

Examples:
    Basic usage::

        from ai_engine.config import get_settings

        settings = get_settings()
        print(settings.llama_url)
        print(settings.embedding_model)
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Default models directory: <project_root>/src/models/
_DEFAULT_MODELS_DIR = str(Path(__file__).resolve().parent.parent / "models")


class AIEngineSettings(BaseSettings):
    """Application-wide settings loaded from environment variables.

    All fields are optional with sensible defaults.  ``pydantic-settings``
    automatically reads values from the process environment.

    Attributes:
        llama_url: HTTP base URL of a llama.cpp server.
        model_path: Path to a local GGUF model file.
        embedding_model: Sentence-transformers model name for RAG.
        embedding_device: Device used by sentence-transformers inference.
        embedding_batch_size: Batch size for multi-document embedding generation.
        llama_timeout_seconds: Timeout for upstream llama HTTP calls.
        llama_max_concurrent_requests: Maximum simultaneous upstream llama HTTP calls.
        api_key: Shared secret for ``X-API-Key`` header auth.
        games_api_key: API key for game microservice generation routes.
        bridge_api_key: API key for bridge/backoffice routes.
        stats_api_key: API key for internal ai-api -> ai-stats events.
        stats_url: Base URL for observability event ingestion.
        generation_api_url: Base URL for generation API monitoring queries.
        models_dir: Directory where GGUF model files are stored.
        generation_cache_path: Path to persistent generation cache file.
        generation_cache_backend: Persistent cache backend (tinydb/redis).
        generation_cache_redis_url: Redis URL used by redis cache backend.
        generation_cache_redis_prefix: Key prefix for redis cache backend.
        generation_cache_namespace: Cache namespace/version tag.
        rate_limit_enabled: Whether generation rate limiting is enabled.
        rate_limit_requests: Maximum requests allowed per window.
        rate_limit_window_seconds: Window size in seconds for limiting.
        generation_max_in_flight: Maximum concurrently executing generation requests.
        generation_max_queue_size: Maximum queued generation requests waiting for capacity.
        query_embedding_cache_max_entries: Maximum cached query embeddings.
        retrieval_result_cache_max_entries: Maximum cached retrieval result sets.
        retriever_candidate_multiplier: Over-fetch factor used before metadata reranking.
        retriever_metadata_match_boost: Score bonus for metadata preference matches.
        rag_context_char_limit: Default prompt context character budget.
        diagnostics_cache_ttl_ms: Short-lived diagnostics cache TTL in milliseconds.
        distribution: Deployment distribution label.
        release_version: Deployment release version label.
    """

    model_config = SettingsConfigDict(
        env_prefix="",  # Variables use their full names (e.g. AI_ENGINE_LLAMA_URL)
        extra="ignore",  # Silently ignore unrecognised env vars
    )

    llama_url: str | None = Field(
        default=None,
        alias="AI_ENGINE_LLAMA_URL",
        validation_alias="AI_ENGINE_LLAMA_URL",
    )
    model_path: str | None = Field(
        default=None,
        alias="AI_ENGINE_MODEL_PATH",
        validation_alias="AI_ENGINE_MODEL_PATH",
    )
    embedding_model: str = Field(
        default="paraphrase-multilingual-MiniLM-L12-v2",
        alias="AI_ENGINE_EMBEDDING_MODEL",
        validation_alias="AI_ENGINE_EMBEDDING_MODEL",
    )
    embedding_device: str = Field(
        default="cpu",
        alias="AI_ENGINE_EMBEDDING_DEVICE",
        validation_alias="AI_ENGINE_EMBEDDING_DEVICE",
    )
    embedding_batch_size: int = Field(
        default=64,
        ge=1,
        alias="AI_ENGINE_EMBEDDING_BATCH_SIZE",
        validation_alias="AI_ENGINE_EMBEDDING_BATCH_SIZE",
    )
    llama_timeout_seconds: float = Field(
        default=600.0,
        ge=1.0,
        alias="AI_ENGINE_LLAMA_TIMEOUT_SECONDS",
        validation_alias="AI_ENGINE_LLAMA_TIMEOUT_SECONDS",
    )
    llama_max_concurrent_requests: int = Field(
        default=1,
        ge=1,
        alias="AI_ENGINE_LLAMA_MAX_CONCURRENT_REQUESTS",
        validation_alias="AI_ENGINE_LLAMA_MAX_CONCURRENT_REQUESTS",
    )
    api_key: str | None = Field(
        default=None,
        alias="AI_ENGINE_API_KEY",
        validation_alias="AI_ENGINE_API_KEY",
    )
    games_api_key: str | None = Field(
        default=None,
        alias="AI_ENGINE_GAMES_API_KEY",
        validation_alias="AI_ENGINE_GAMES_API_KEY",
    )
    bridge_api_key: str | None = Field(
        default=None,
        alias="AI_ENGINE_BRIDGE_API_KEY",
        validation_alias="AI_ENGINE_BRIDGE_API_KEY",
    )
    stats_api_key: str | None = Field(
        default=None,
        alias="AI_ENGINE_STATS_API_KEY",
        validation_alias="AI_ENGINE_STATS_API_KEY",
    )
    stats_url: str | None = Field(
        default=None,
        alias="AI_ENGINE_STATS_URL",
        validation_alias="AI_ENGINE_STATS_URL",
    )
    llama_target_state_file: str = Field(
        default="data/llama-target-state.json",
        alias="AI_ENGINE_LLAMA_TARGET_STATE_FILE",
        validation_alias="AI_ENGINE_LLAMA_TARGET_STATE_FILE",
    )
    generation_api_url: str = Field(
        default="http://ai-api:8001",
        alias="AI_ENGINE_GENERATION_API_URL",
        validation_alias="AI_ENGINE_GENERATION_API_URL",
    )
    models_dir: str = Field(
        default=_DEFAULT_MODELS_DIR,
        alias="AI_ENGINE_MODELS_DIR",
        validation_alias="AI_ENGINE_MODELS_DIR",
    )
    generation_cache_path: str = Field(
        default="data/generation_cache.json",
        alias="AI_ENGINE_GENERATION_CACHE_PATH",
        validation_alias="AI_ENGINE_GENERATION_CACHE_PATH",
    )
    generation_cache_backend: str = Field(
        default="tinydb",
        alias="AI_ENGINE_GENERATION_CACHE_BACKEND",
        validation_alias="AI_ENGINE_GENERATION_CACHE_BACKEND",
    )
    generation_cache_redis_url: str | None = Field(
        default=None,
        alias="AI_ENGINE_GENERATION_CACHE_REDIS_URL",
        validation_alias="AI_ENGINE_GENERATION_CACHE_REDIS_URL",
    )
    generation_cache_redis_prefix: str = Field(
        default="ai-engine:generation-cache",
        alias="AI_ENGINE_GENERATION_CACHE_REDIS_PREFIX",
        validation_alias="AI_ENGINE_GENERATION_CACHE_REDIS_PREFIX",
    )
    generation_cache_namespace: str = Field(
        default="v1",
        alias="AI_ENGINE_GENERATION_CACHE_NAMESPACE",
        validation_alias="AI_ENGINE_GENERATION_CACHE_NAMESPACE",
    )
    rate_limit_enabled: bool = Field(
        default=False,
        alias="AI_ENGINE_RATE_LIMIT_ENABLED",
        validation_alias="AI_ENGINE_RATE_LIMIT_ENABLED",
    )
    rate_limit_requests: int = Field(
        default=60,
        ge=1,
        alias="AI_ENGINE_RATE_LIMIT_REQUESTS",
        validation_alias="AI_ENGINE_RATE_LIMIT_REQUESTS",
    )
    rate_limit_window_seconds: int = Field(
        default=60,
        ge=1,
        alias="AI_ENGINE_RATE_LIMIT_WINDOW_SECONDS",
        validation_alias="AI_ENGINE_RATE_LIMIT_WINDOW_SECONDS",
    )
    generation_max_in_flight: int = Field(
        default=2,
        ge=1,
        alias="AI_ENGINE_GENERATION_MAX_IN_FLIGHT",
        validation_alias="AI_ENGINE_GENERATION_MAX_IN_FLIGHT",
    )
    generation_max_queue_size: int = Field(
        default=2,
        ge=0,
        alias="AI_ENGINE_GENERATION_MAX_QUEUE_SIZE",
        validation_alias="AI_ENGINE_GENERATION_MAX_QUEUE_SIZE",
    )
    query_embedding_cache_max_entries: int = Field(
        default=2048,
        ge=0,
        alias="AI_ENGINE_QUERY_EMBEDDING_CACHE_MAX_ENTRIES",
        validation_alias="AI_ENGINE_QUERY_EMBEDDING_CACHE_MAX_ENTRIES",
    )
    retrieval_result_cache_max_entries: int = Field(
        default=1024,
        ge=0,
        alias="AI_ENGINE_RETRIEVAL_RESULT_CACHE_MAX_ENTRIES",
        validation_alias="AI_ENGINE_RETRIEVAL_RESULT_CACHE_MAX_ENTRIES",
    )
    retriever_candidate_multiplier: int = Field(
        default=4,
        ge=1,
        alias="AI_ENGINE_RETRIEVER_CANDIDATE_MULTIPLIER",
        validation_alias="AI_ENGINE_RETRIEVER_CANDIDATE_MULTIPLIER",
    )
    retriever_metadata_match_boost: float = Field(
        default=0.08,
        ge=0.0,
        alias="AI_ENGINE_RETRIEVER_METADATA_MATCH_BOOST",
        validation_alias="AI_ENGINE_RETRIEVER_METADATA_MATCH_BOOST",
    )
    rag_context_char_limit: int = Field(
        default=3500,
        ge=512,
        alias="AI_ENGINE_RAG_CONTEXT_CHAR_LIMIT",
        validation_alias="AI_ENGINE_RAG_CONTEXT_CHAR_LIMIT",
    )
    diagnostics_cache_ttl_ms: int = Field(
        default=2000,
        ge=0,
        alias="AI_ENGINE_DIAGNOSTICS_CACHE_TTL_MS",
        validation_alias="AI_ENGINE_DIAGNOSTICS_CACHE_TTL_MS",
    )
    cache_warmup_enabled: bool = Field(
        default=True,
        alias="AI_ENGINE_CACHE_WARMUP_ENABLED",
        validation_alias="AI_ENGINE_CACHE_WARMUP_ENABLED",
    )
    distribution: str = Field(
        default="dev",
        alias="AI_ENGINE_DISTRIBUTION",
        validation_alias="AI_ENGINE_DISTRIBUTION",
    )
    release_version: str = Field(
        default="v1",
        alias="AI_ENGINE_RELEASE_VERSION",
        validation_alias="AI_ENGINE_RELEASE_VERSION",
    )

    @property
    def distribution_version_tag(self) -> str:
        """Return canonical `distribution-version` tag used across channels."""
        distribution = self.distribution.strip() or "unknown"
        release_version = self.release_version.strip() or "v0"
        return f"{distribution}-{release_version}"


@lru_cache(maxsize=1)
def get_settings() -> AIEngineSettings:
    """Return a cached :class:`AIEngineSettings` instance.

    The settings are read once from the process environment and then
    cached for the lifetime of the process.  This avoids the overhead of
    re-parsing env vars and re-validating on every call.

    Returns:
        A fully-populated :class:`AIEngineSettings` object.

    Examples:
        >>> from ai_engine.config import get_settings
        >>> s = get_settings()
        >>> isinstance(s, AIEngineSettings)
        True
    """
    return AIEngineSettings()


def get_distribution_version_tag() -> str:
    """Return deployment tag formatted as `<distribution>-<version>`."""
    return get_settings().distribution_version_tag
