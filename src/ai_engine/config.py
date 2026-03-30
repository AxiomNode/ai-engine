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

``AI_ENGINE_LLAMA_TIMEOUT_SECONDS``
    Timeout (seconds) for upstream HTTP calls from ai-api to llama.cpp.

``AI_ENGINE_API_KEY``
    Shared secret propagated to ``X-API-Key`` header validation.  When
    absent, authentication is disabled.

``AI_ENGINE_GAMES_API_KEY``
    API key used for game microservices consuming generation endpoints
    (``/generate*``).

``AI_ENGINE_BRIDGE_API_KEY``
    API key used for bridge microservice integration on ingest and
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
    ``models/`` folder at the project root.

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

# Default models directory: <project_root>/models/
_DEFAULT_MODELS_DIR = str(Path(__file__).resolve().parent.parent.parent / "models")


class AIEngineSettings(BaseSettings):
    """Application-wide settings loaded from environment variables.

    All fields are optional with sensible defaults.  ``pydantic-settings``
    automatically reads values from the process environment.

    Attributes:
        llama_url: HTTP base URL of a llama.cpp server.
        model_path: Path to a local GGUF model file.
        embedding_model: Sentence-transformers model name for RAG.
        llama_timeout_seconds: Timeout for upstream llama HTTP calls.
        api_key: Shared secret for ``X-API-Key`` header auth.
        games_api_key: API key for game microservice generation routes.
        bridge_api_key: API key for bridge microservice routes.
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
        default="all-MiniLM-L6-v2",
        alias="AI_ENGINE_EMBEDDING_MODEL",
        validation_alias="AI_ENGINE_EMBEDDING_MODEL",
    )
    llama_timeout_seconds: float = Field(
        default=600.0,
        ge=1.0,
        alias="AI_ENGINE_LLAMA_TIMEOUT_SECONDS",
        validation_alias="AI_ENGINE_LLAMA_TIMEOUT_SECONDS",
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
