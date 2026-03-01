"""Tests for ai_engine.llm.model_manager – model download & registry."""

import pytest

from ai_engine.llm.model_manager import (
    DEFAULT_MODEL,
    MODELS,
    get_models_dir,
    list_models,
    model_path,
)


class TestModelRegistry:

    def test_default_model_exists(self):
        assert DEFAULT_MODEL in MODELS

    def test_all_models_have_required_keys(self):
        required = {"filename", "url", "size_mb", "description", "n_ctx"}
        for name, info in MODELS.items():
            missing = required - set(info.keys())
            assert not missing, f"Model {name!r} missing keys: {missing}"

    def test_all_urls_are_https(self):
        for name, info in MODELS.items():
            assert info["url"].startswith("https://"), (
                f"Model {name!r} URL should use HTTPS"
            )


class TestGetModelsDir:

    def test_returns_path(self, tmp_path, monkeypatch):
        monkeypatch.setenv("AI_ENGINE_MODELS_DIR", str(tmp_path / "mdl"))
        d = get_models_dir()
        assert d.exists()
        assert d == tmp_path / "mdl"


class TestModelPath:

    def test_unknown_model_raises(self):
        with pytest.raises(ValueError, match="Unknown model"):
            model_path("nonexistent-model")

    def test_missing_file_raises(self, tmp_path, monkeypatch):
        monkeypatch.setenv("AI_ENGINE_MODELS_DIR", str(tmp_path))
        with pytest.raises(FileNotFoundError, match="Model file not found"):
            model_path(DEFAULT_MODEL)

    def test_returns_path_when_file_exists(self, tmp_path, monkeypatch):
        monkeypatch.setenv("AI_ENGINE_MODELS_DIR", str(tmp_path))
        info = MODELS[DEFAULT_MODEL]
        (tmp_path / info["filename"]).write_bytes(b"fake")
        p = model_path(DEFAULT_MODEL)
        assert p.exists()


class TestListModels:

    def test_returns_list(self):
        result = list_models()
        assert isinstance(result, list)
        assert len(result) == len(MODELS)

    def test_entries_have_downloaded_field(self):
        for entry in list_models():
            assert "downloaded" in entry
            assert isinstance(entry["downloaded"], bool)
