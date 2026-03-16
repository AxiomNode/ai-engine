"""Tests for ai_engine.llm.model_manager – model download & registry."""

import hashlib
import sys
from unittest.mock import MagicMock, patch

import pytest

from ai_engine.llm.model_manager import (
    DEFAULT_MODEL,
    MODELS,
    download_model,
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
            assert info["url"].startswith(
                "https://"
            ), f"Model {name!r} URL should use HTTPS"


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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_response(content: bytes, status_code: int = 200) -> MagicMock:
    """Build a mock requests.Response that streams *content* in one chunk."""
    mock_resp = MagicMock()
    mock_resp.status_code = status_code
    mock_resp.headers = {"content-length": str(len(content))}
    mock_resp.iter_content = MagicMock(return_value=iter([content]))
    mock_resp.__enter__ = MagicMock(return_value=mock_resp)
    mock_resp.__exit__ = MagicMock(return_value=False)
    mock_resp.raise_for_status = MagicMock()
    return mock_resp


class TestDownloadModel:

    def test_unknown_model_raises(self, tmp_path, monkeypatch):
        monkeypatch.setenv("AI_ENGINE_MODELS_DIR", str(tmp_path))
        with pytest.raises(ValueError, match="Unknown model"):
            download_model("nonexistent-model")

    def test_already_exists_skips_download(self, tmp_path, monkeypatch):
        """If dest file exists and force=False, download is skipped."""
        monkeypatch.setenv("AI_ENGINE_MODELS_DIR", str(tmp_path))
        info = MODELS[DEFAULT_MODEL]
        dest = tmp_path / info["filename"]
        dest.write_bytes(b"already_here")

        with patch("requests.get") as mock_get:
            result = download_model(DEFAULT_MODEL, force=False)
            mock_get.assert_not_called()

        assert result == dest

    def test_force_redownloads_existing_file(self, tmp_path, monkeypatch):
        """force=True re-downloads even when the file already exists."""
        monkeypatch.setenv("AI_ENGINE_MODELS_DIR", str(tmp_path))
        info = MODELS[DEFAULT_MODEL]
        dest = tmp_path / info["filename"]
        dest.write_bytes(b"old_content")

        new_content = b"new_model_bytes"
        mock_resp = _make_mock_response(new_content)

        with patch("requests.get", return_value=mock_resp):
            result = download_model(DEFAULT_MODEL, force=True)

        assert result.read_bytes() == new_content

    def test_successful_download_creates_file(self, tmp_path, monkeypatch):
        """A successful HTTP response writes the file to disk."""
        monkeypatch.setenv("AI_ENGINE_MODELS_DIR", str(tmp_path))
        content = b"fake_gguf_bytes"
        mock_resp = _make_mock_response(content)

        with patch("requests.get", return_value=mock_resp):
            result = download_model(DEFAULT_MODEL)

        assert result.exists()
        assert result.read_bytes() == content

    def test_http_error_cleans_up_part_file(self, tmp_path, monkeypatch):
        """If the HTTP request raises, the .part temp file is removed."""
        monkeypatch.setenv("AI_ENGINE_MODELS_DIR", str(tmp_path))

        with patch("requests.get", side_effect=OSError("network error")):
            with pytest.raises(OSError, match="network error"):
                download_model(DEFAULT_MODEL)

        # No leftover .part files
        leftover = list(tmp_path.glob("*.part"))
        assert not leftover

    def test_sha256_mismatch_raises_and_cleans_up(self, tmp_path, monkeypatch):
        """When sha256 is set and doesn't match, RuntimeError is raised."""
        monkeypatch.setenv("AI_ENGINE_MODELS_DIR", str(tmp_path))

        # Temporarily inject a sha256 that won't match
        bad_sha = "a" * 64
        original_info = MODELS[DEFAULT_MODEL].copy()
        MODELS[DEFAULT_MODEL]["sha256"] = bad_sha

        content = b"model_content"
        mock_resp = _make_mock_response(content)

        try:
            with patch("requests.get", return_value=mock_resp):
                with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
                    download_model(DEFAULT_MODEL, force=True)
        finally:
            MODELS[DEFAULT_MODEL]["sha256"] = original_info["sha256"]

        # dest file must not exist after a bad hash
        dest = tmp_path / MODELS[DEFAULT_MODEL]["filename"]
        assert not dest.exists()

    def test_sha256_match_succeeds(self, tmp_path, monkeypatch):
        """When sha256 matches, the file is saved successfully."""
        monkeypatch.setenv("AI_ENGINE_MODELS_DIR", str(tmp_path))

        content = b"verified_model"
        good_sha = hashlib.sha256(content).hexdigest()
        original_sha = MODELS[DEFAULT_MODEL]["sha256"]
        MODELS[DEFAULT_MODEL]["sha256"] = good_sha

        mock_resp = _make_mock_response(content)

        try:
            with patch("requests.get", return_value=mock_resp):
                result = download_model(DEFAULT_MODEL)
        finally:
            MODELS[DEFAULT_MODEL]["sha256"] = original_sha

        assert result.exists()
        assert result.read_bytes() == content


class TestCli:

    def test_list_command_prints_models(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setenv("AI_ENGINE_MODELS_DIR", str(tmp_path))
        monkeypatch.setattr(sys, "argv", ["model_manager", "list"])

        from ai_engine.llm.model_manager import _cli

        _cli()

        out = capsys.readouterr().out
        for name in MODELS:
            assert name in out

    def test_path_command_prints_path(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setenv("AI_ENGINE_MODELS_DIR", str(tmp_path))
        info = MODELS[DEFAULT_MODEL]
        (tmp_path / info["filename"]).write_bytes(b"x")
        monkeypatch.setattr(sys, "argv", ["model_manager", "path", DEFAULT_MODEL])

        from ai_engine.llm.model_manager import _cli

        _cli()

        out = capsys.readouterr().out.strip()
        assert info["filename"] in out

    def test_download_command(self, tmp_path, monkeypatch):
        monkeypatch.setenv("AI_ENGINE_MODELS_DIR", str(tmp_path))
        monkeypatch.setattr(sys, "argv", ["model_manager", "download", DEFAULT_MODEL])

        content = b"cli_download"
        mock_resp = _make_mock_response(content)

        from ai_engine.llm.model_manager import _cli

        with patch("requests.get", return_value=mock_resp):
            _cli()

        dest = tmp_path / MODELS[DEFAULT_MODEL]["filename"]
        assert dest.exists()

    def test_unknown_command_exits(self, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["model_manager", "explode"])

        from ai_engine.llm.model_manager import _cli

        with pytest.raises(SystemExit) as exc_info:
            _cli()

        assert exc_info.value.code == 1

    def test_help_flag_returns_without_error(self, monkeypatch, capsys):
        monkeypatch.setattr(sys, "argv", ["model_manager", "--help"])

        from ai_engine.llm.model_manager import _cli

        _cli()  # should not raise

        out = capsys.readouterr().out
        assert "Usage" in out
