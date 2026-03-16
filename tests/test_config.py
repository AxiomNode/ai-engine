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

    def test_models_dir_default_is_not_none(self, monkeypatch):
        monkeypatch.delenv("AI_ENGINE_MODELS_DIR", raising=False)
        cfg = _reload_config(monkeypatch)
        settings = cfg.AIEngineSettings()
        assert settings.models_dir is not None


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

    def test_models_dir_from_env(self, monkeypatch):
        monkeypatch.setenv("AI_ENGINE_MODELS_DIR", "/custom/models")
        cfg = _reload_config(monkeypatch)
        settings = cfg.AIEngineSettings()
        assert settings.models_dir == "/custom/models"


# ---------------------------------------------------------------------------
# get_settings factory
# ---------------------------------------------------------------------------


class TestGetSettings:
    """get_settings returns an AIEngineSettings instance."""

    def test_returns_settings_instance(self, monkeypatch):
        cfg = _reload_config(monkeypatch)
        result = cfg.get_settings()
        assert isinstance(result, cfg.AIEngineSettings)

    def test_returns_fresh_instance_each_call(self, monkeypatch):
        cfg = _reload_config(monkeypatch)
        s1 = cfg.get_settings()
        s2 = cfg.get_settings()
        # Each call returns a new instance (not a stale cache)
        assert s1 is not s2

    def test_get_settings_respects_env(self, monkeypatch):
        monkeypatch.setenv("AI_ENGINE_API_KEY", "test-key")
        cfg = _reload_config(monkeypatch)
        settings = cfg.get_settings()
        assert settings.api_key == "test-key"
