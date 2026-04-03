"""Tests for centralised pydantic-settings configuration.

Tests cover default values, env-var override, and the ``get_settings``
factory.  Uses monkeypatching to isolate environment between tests.
"""

from __future__ import annotations

import sys

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _reload_config(monkeypatch):
    """Re-import config module after env changes so Settings re-reads env."""
    # Remove cached module so each test gets a fresh instance.
    sys.modules.pop("ai_engine.config", None)
    import ai_engine.config as cfg  # noqa: WPS433

    return cfg


# ---------------------------------------------------------------------------
# Default values
# ---------------------------------------------------------------------------


class TestAIEngineSettingsDefaults:
    """AIEngineSettings uses sensible defaults when no env vars are set."""

    def test_llama_url_default_is_none(self, monkeypatch):
        monkeypatch.delenv("AI_ENGINE_LLAMA_URL", raising=False)
        cfg = _reload_config(monkeypatch)
        settings = cfg.AIEngineSettings()
        assert settings.llama_url is None

    def test_model_path_default_is_none(self, monkeypatch):
        monkeypatch.delenv("AI_ENGINE_MODEL_PATH", raising=False)
        cfg = _reload_config(monkeypatch)
        settings = cfg.AIEngineSettings()
        assert settings.model_path is None

    def test_embedding_model_default(self, monkeypatch):
        monkeypatch.delenv("AI_ENGINE_EMBEDDING_MODEL", raising=False)
        cfg = _reload_config(monkeypatch)
        settings = cfg.AIEngineSettings()
        assert settings.embedding_model == "all-MiniLM-L6-v2"

    def test_api_key_default_is_none(self, monkeypatch):
        monkeypatch.delenv("AI_ENGINE_API_KEY", raising=False)
        cfg = _reload_config(monkeypatch)
        settings = cfg.AIEngineSettings()
        assert settings.api_key is None

    def test_games_api_key_default_is_none(self, monkeypatch):
        monkeypatch.delenv("AI_ENGINE_GAMES_API_KEY", raising=False)
        cfg = _reload_config(monkeypatch)
        settings = cfg.AIEngineSettings()
        assert settings.games_api_key is None

    def test_bridge_api_key_default_is_none(self, monkeypatch):
        monkeypatch.delenv("AI_ENGINE_BRIDGE_API_KEY", raising=False)
        cfg = _reload_config(monkeypatch)
        settings = cfg.AIEngineSettings()
        assert settings.bridge_api_key is None

    def test_stats_api_key_default_is_none(self, monkeypatch):
        monkeypatch.delenv("AI_ENGINE_STATS_API_KEY", raising=False)
        cfg = _reload_config(monkeypatch)
        settings = cfg.AIEngineSettings()
        assert settings.stats_api_key is None

    def test_models_dir_default_is_not_none(self, monkeypatch):
        monkeypatch.delenv("AI_ENGINE_MODELS_DIR", raising=False)
        cfg = _reload_config(monkeypatch)
        settings = cfg.AIEngineSettings()
        assert settings.models_dir is not None

    def test_generation_cache_path_default(self, monkeypatch):
        monkeypatch.delenv("AI_ENGINE_GENERATION_CACHE_PATH", raising=False)
        cfg = _reload_config(monkeypatch)
        settings = cfg.AIEngineSettings()
        assert settings.generation_cache_path == "data/generation_cache.json"

    def test_generation_cache_backend_default(self, monkeypatch):
        monkeypatch.delenv("AI_ENGINE_GENERATION_CACHE_BACKEND", raising=False)
        cfg = _reload_config(monkeypatch)
        settings = cfg.AIEngineSettings()
        assert settings.generation_cache_backend == "tinydb"

    def test_generation_cache_namespace_default(self, monkeypatch):
        monkeypatch.delenv("AI_ENGINE_GENERATION_CACHE_NAMESPACE", raising=False)
        cfg = _reload_config(monkeypatch)
        settings = cfg.AIEngineSettings()
        assert settings.generation_cache_namespace == "v1"

    def test_distribution_default(self, monkeypatch):
        monkeypatch.delenv("AI_ENGINE_DISTRIBUTION", raising=False)
        cfg = _reload_config(monkeypatch)
        settings = cfg.AIEngineSettings()
        assert settings.distribution == "dev"

    def test_release_version_default(self, monkeypatch):
        monkeypatch.delenv("AI_ENGINE_RELEASE_VERSION", raising=False)
        cfg = _reload_config(monkeypatch)
        settings = cfg.AIEngineSettings()
        assert settings.release_version == "v1"

    def test_distribution_version_tag_default(self, monkeypatch):
        cfg = _reload_config(monkeypatch)
        settings = cfg.AIEngineSettings()
        assert settings.distribution_version_tag == "dev-v1"


# ---------------------------------------------------------------------------
# Env-var overrides
# ---------------------------------------------------------------------------


class TestAIEngineSettingsFromEnv:
    """AIEngineSettings reads values from environment variables correctly."""

    def test_llama_url_from_env(self, monkeypatch):
        monkeypatch.setenv("AI_ENGINE_LLAMA_URL", "http://localhost:8080")
        cfg = _reload_config(monkeypatch)
        settings = cfg.AIEngineSettings()
        assert settings.llama_url == "http://localhost:8080"

    def test_model_path_from_env(self, monkeypatch):
        monkeypatch.setenv("AI_ENGINE_MODEL_PATH", "/models/model.gguf")
        cfg = _reload_config(monkeypatch)
        settings = cfg.AIEngineSettings()
        assert settings.model_path == "/models/model.gguf"

    def test_embedding_model_from_env(self, monkeypatch):
        monkeypatch.setenv(
            "AI_ENGINE_EMBEDDING_MODEL", "paraphrase-multilingual-MiniLM-L12-v2"
        )
        cfg = _reload_config(monkeypatch)
        settings = cfg.AIEngineSettings()
        assert settings.embedding_model == "paraphrase-multilingual-MiniLM-L12-v2"

    def test_api_key_from_env(self, monkeypatch):
        monkeypatch.setenv("AI_ENGINE_API_KEY", "secret-key-123")
        cfg = _reload_config(monkeypatch)
        settings = cfg.AIEngineSettings()
        assert settings.api_key == "secret-key-123"

    def test_games_api_key_from_env(self, monkeypatch):
        monkeypatch.setenv("AI_ENGINE_GAMES_API_KEY", "games-key-123")
        cfg = _reload_config(monkeypatch)
        settings = cfg.AIEngineSettings()
        assert settings.games_api_key == "games-key-123"

    def test_bridge_api_key_from_env(self, monkeypatch):
        monkeypatch.setenv("AI_ENGINE_BRIDGE_API_KEY", "bridge-key-123")
        cfg = _reload_config(monkeypatch)
        settings = cfg.AIEngineSettings()
        assert settings.bridge_api_key == "bridge-key-123"

    def test_stats_api_key_from_env(self, monkeypatch):
        monkeypatch.setenv("AI_ENGINE_STATS_API_KEY", "stats-key-123")
        cfg = _reload_config(monkeypatch)
        settings = cfg.AIEngineSettings()
        assert settings.stats_api_key == "stats-key-123"

    def test_models_dir_from_env(self, monkeypatch):
        monkeypatch.setenv("AI_ENGINE_MODELS_DIR", "/custom/models")
        cfg = _reload_config(monkeypatch)
        settings = cfg.AIEngineSettings()
        assert settings.models_dir == "/custom/models"

    def test_generation_cache_path_from_env(self, monkeypatch):
        monkeypatch.setenv(
            "AI_ENGINE_GENERATION_CACHE_PATH", "/tmp/ai-engine/cache.json"
        )
        cfg = _reload_config(monkeypatch)
        settings = cfg.AIEngineSettings()
        assert settings.generation_cache_path == "/tmp/ai-engine/cache.json"

    def test_generation_cache_backend_from_env(self, monkeypatch):
        monkeypatch.setenv("AI_ENGINE_GENERATION_CACHE_BACKEND", "redis")
        monkeypatch.setenv(
            "AI_ENGINE_GENERATION_CACHE_REDIS_URL", "redis://localhost:6379/0"
        )
        monkeypatch.setenv(
            "AI_ENGINE_GENERATION_CACHE_REDIS_PREFIX", "ai-engine:test-cache"
        )
        cfg = _reload_config(monkeypatch)
        settings = cfg.AIEngineSettings()
        assert settings.generation_cache_backend == "redis"
        assert settings.generation_cache_redis_url == "redis://localhost:6379/0"
        assert settings.generation_cache_redis_prefix == "ai-engine:test-cache"

    def test_generation_cache_namespace_from_env(self, monkeypatch):
        monkeypatch.setenv("AI_ENGINE_GENERATION_CACHE_NAMESPACE", "v2")
        cfg = _reload_config(monkeypatch)
        settings = cfg.AIEngineSettings()
        assert settings.generation_cache_namespace == "v2"

    def test_distribution_version_from_env(self, monkeypatch):
        monkeypatch.setenv("AI_ENGINE_DISTRIBUTION", "stg")
        monkeypatch.setenv("AI_ENGINE_RELEASE_VERSION", "2026.03")
        cfg = _reload_config(monkeypatch)
        settings = cfg.AIEngineSettings()
        assert settings.distribution == "stg"
        assert settings.release_version == "2026.03"
        assert settings.distribution_version_tag == "stg-2026.03"


# ---------------------------------------------------------------------------
# get_settings factory
# ---------------------------------------------------------------------------


class TestGetSettings:
    """get_settings returns an AIEngineSettings instance."""

    def test_returns_settings_instance(self, monkeypatch):
        cfg = _reload_config(monkeypatch)
        result = cfg.get_settings()
        assert isinstance(result, cfg.AIEngineSettings)

    def test_returns_cached_instance_on_repeated_calls(self, monkeypatch):
        cfg = _reload_config(monkeypatch)
        s1 = cfg.get_settings()
        s2 = cfg.get_settings()
        # lru_cache returns the same instance on repeated calls
        assert s1 is s2

    def test_get_settings_respects_env(self, monkeypatch):
        monkeypatch.setenv("AI_ENGINE_API_KEY", "test-key")
        cfg = _reload_config(monkeypatch)
        settings = cfg.get_settings()
        assert settings.api_key == "test-key"

    def test_get_distribution_version_tag(self, monkeypatch):
        monkeypatch.setenv("AI_ENGINE_DISTRIBUTION", "pro")
        monkeypatch.setenv("AI_ENGINE_RELEASE_VERSION", "v9")
        cfg = _reload_config(monkeypatch)
        assert cfg.get_distribution_version_tag() == "pro-v9"
