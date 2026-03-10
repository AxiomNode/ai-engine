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

``AI_ENGINE_API_KEY``
    Shared secret propagated to ``X-API-Key`` header validation.  When
    absent, authentication is disabled.

``AI_ENGINE_MODELS_DIR``
    Directory where GGUF model files are stored.  Defaults to the
    ``models/`` folder at the project root.

Examples:
    Basic usage::

        from ai_engine.config import get_settings

        settings = get_settings()
        print(settings.llama_url)
        print(settings.embedding_model)
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Default models directory: <project_root>/models/
_DEFAULT_MODELS_DIR = str(
    Path(__file__).resolve().parent.parent.parent / "models"
)


class AIEngineSettings(BaseSettings):
    """Application-wide settings loaded from environment variables.

    All fields are optional with sensible defaults.  ``pydantic-settings``
    automatically reads values from the process environment.

    Attributes:
        llama_url: HTTP base URL of a llama.cpp server.
        model_path: Path to a local GGUF model file.
        embedding_model: Sentence-transformers model name for RAG.
        api_key: Shared secret for ``X-API-Key`` header auth.
        models_dir: Directory where GGUF model files are stored.
    """

    model_config = SettingsConfigDict(
        env_prefix="",      # Variables use their full names (e.g. AI_ENGINE_LLAMA_URL)
        extra="ignore",     # Silently ignore unrecognised env vars
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
    api_key: str | None = Field(
        default=None,
        alias="AI_ENGINE_API_KEY",
        validation_alias="AI_ENGINE_API_KEY",
    )
    models_dir: str = Field(
        default=_DEFAULT_MODELS_DIR,
        alias="AI_ENGINE_MODELS_DIR",
        validation_alias="AI_ENGINE_MODELS_DIR",
    )


def get_settings() -> AIEngineSettings:
    """Create and return a fresh :class:`AIEngineSettings` instance.

    Each call reads the current process environment, so settings always
    reflect the latest values.  For production use, you may cache the
    result; for tests, a fresh call ensures isolation.

    Returns:
        A fully-populated :class:`AIEngineSettings` object.

    Examples:
        >>> from ai_engine.config import get_settings
        >>> s = get_settings()
        >>> isinstance(s, AIEngineSettings)
        True
    """
    return AIEngineSettings()
