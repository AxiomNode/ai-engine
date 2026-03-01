"""Tests for ai_engine.llm.llama_client – LlamaClient."""

import pytest

from ai_engine.llm.llama_client import LlamaClient


class TestLlamaClientInit:
    """LlamaClient constructor tests."""

    def test_init_with_api_url(self):
        client = LlamaClient(api_url="http://localhost:8080/completion")
        assert client.api_url == "http://localhost:8080/completion"
        assert client._local_model is None

    def test_init_with_model_path(self):
        """Passing a model_path stores it but does NOT load the model eagerly."""
        client = LlamaClient(model_path="/fake/model.gguf")
        assert client.model_path == "/fake/model.gguf"
        assert client._local_model is None  # lazy loading

    def test_init_requires_at_least_one_backend(self):
        with pytest.raises(ValueError, match="api_url.*model_path"):
            LlamaClient()

    def test_default_generation_params(self):
        client = LlamaClient(api_url="http://localhost:8080/completion")
        assert client.default_max_tokens == 256
        assert client.temperature >= 0


class TestLlamaClientGenerateAPI:
    """Tests for generate() via HTTP API using a monkeypatched requests."""

    def test_generate_calls_api(self, monkeypatch):
        """Verify generate() sends a POST and returns the completion text."""

        class FakeResponse:
            status_code = 200

            def json(self):
                return {"content": "Hello from API"}

            def raise_for_status(self):
                pass

        def fake_post(url, json=None, timeout=None):
            return FakeResponse()

        import ai_engine.llm.llama_client as mod

        monkeypatch.setattr(mod.requests, "post", fake_post)

        client = LlamaClient(api_url="http://localhost:8080/completion")
        result = client.generate("Say hi")
        assert result == "Hello from API"

    def test_generate_respects_max_tokens(self, monkeypatch):
        captured = {}

        class FakeResponse:
            status_code = 200

            def json(self):
                return {"content": "ok"}

            def raise_for_status(self):
                pass

        def fake_post(url, json=None, timeout=None):
            captured.update(json or {})
            return FakeResponse()

        import ai_engine.llm.llama_client as mod

        monkeypatch.setattr(mod.requests, "post", fake_post)

        client = LlamaClient(api_url="http://localhost:8080/completion")
        client.generate("Say hi", max_tokens=128)
        assert captured.get("n_predict") == 128

    def test_generate_raises_on_http_error(self, monkeypatch):
        class FakeResponse:
            status_code = 500

            def raise_for_status(self):
                raise Exception("Server Error")

        def fake_post(url, json=None, timeout=None):
            return FakeResponse()

        import ai_engine.llm.llama_client as mod

        monkeypatch.setattr(mod.requests, "post", fake_post)

        client = LlamaClient(api_url="http://localhost:8080/completion")
        with pytest.raises(Exception, match="Server Error"):
            client.generate("Say hi")


class TestLlamaClientProtocol:
    """Ensure LlamaClient satisfies the protocol expected by RAGPipeline."""

    def test_has_generate_method(self):
        client = LlamaClient(api_url="http://localhost:8080/completion")
        assert callable(getattr(client, "generate", None))
