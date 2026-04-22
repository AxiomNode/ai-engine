"""Tests for ai_engine.llm.llama_client – LlamaClient."""

import asyncio

import pytest

from ai_engine.llm.llama_client import JSON_GRAMMAR, LlamaClient


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
        assert client.default_max_tokens == 512
        assert client.temperature >= 0
        assert client.top_p == 0.9
        assert client.repeat_penalty == 1.1

    def test_json_mode_off_by_default(self):
        client = LlamaClient(api_url="http://localhost:8080/completion")
        assert client.json_mode is False

    def test_json_mode_on(self):
        client = LlamaClient(api_url="http://x", json_mode=True)
        assert client.json_mode is True

    def test_default_max_concurrent_requests_is_one(self):
        client = LlamaClient(api_url="http://localhost:8080/completion")
        assert client.max_concurrent_requests == 1


class TestLlamaClientGenerateAPI:
    """Tests for generate() via HTTP API using a monkeypatched httpx.AsyncClient."""

    @staticmethod
    def _make_mock_client(response_json, captured=None):
        """Create a mock httpx.AsyncClient whose post() returns response_json."""

        class _MockResponse:
            status_code = 200

            def json(self):
                return response_json

            def raise_for_status(self):
                pass

        class _MockAsyncClient:
            is_closed = False

            async def post(self, url, *, json=None, **kw):
                if captured is not None:
                    captured.update(json or {})
                return _MockResponse()

            async def aclose(self):
                self.is_closed = True

        return _MockAsyncClient()

    def test_generate_calls_api(self, monkeypatch):
        """Verify generate() sends a POST and returns the completion text."""
        client = LlamaClient(api_url="http://localhost:8080/completion")
        mock = self._make_mock_client({"content": "Hello from API"})
        client._http_client = mock

        result = asyncio.run(client.generate("Say hi"))
        assert result == "Hello from API"

    def test_generate_reads_legacy_text_payload(self):
        client = LlamaClient(api_url="http://localhost:8080/completion")
        client._http_client = self._make_mock_client({"text": "legacy text"})

        result = asyncio.run(client.generate("Say hi"))

        assert result == "legacy text"

    def test_generate_reads_openai_message_content_payload(self):
        client = LlamaClient(api_url="http://localhost:8080/v1/completions")
        client._http_client = self._make_mock_client(
            {"choices": [{"message": {"content": "chat text"}}]}
        )

        result = asyncio.run(client.generate("Say hi"))

        assert result == "chat text"

    def test_generate_returns_empty_string_for_unrecognized_payload(self):
        client = LlamaClient(api_url="http://localhost:8080/completion")
        client._http_client = self._make_mock_client({"choices": ["bad"]})

        result = asyncio.run(client.generate("Say hi"))

        assert result == ""

    def test_generate_respects_max_tokens(self, monkeypatch):
        captured = {}
        client = LlamaClient(api_url="http://localhost:8080/completion")
        mock = self._make_mock_client({"content": "ok"}, captured)
        client._http_client = mock

        asyncio.run(client.generate("Say hi", max_tokens=128))
        assert captured.get("n_predict") == 128

    def test_generate_uses_openai_max_tokens_for_v1_completions(self, monkeypatch):
        captured = {}
        client = LlamaClient(api_url="http://localhost:8080/v1/completions")
        mock = self._make_mock_client({"choices": [{"text": "ok"}]}, captured)
        client._http_client = mock

        asyncio.run(client.generate("Say hi", max_tokens=128))
        assert captured.get("max_tokens") == 128
        assert "n_predict" not in captured

    def test_json_mode_sends_grammar(self, monkeypatch):
        """When json_mode=True, the grammar should be sent to the API."""
        captured = {}
        client = LlamaClient(api_url="http://localhost:8080/completion", json_mode=True)
        mock = self._make_mock_client({"content": '{"a":1}'}, captured)
        client._http_client = mock

        asyncio.run(client.generate("Give me JSON"))
        assert "grammar" in captured
        assert "root" in captured["grammar"]

    def test_json_mode_override_per_call(self, monkeypatch):
        """json_mode can be overridden per generate() call."""
        captured = {}
        client = LlamaClient(
            api_url="http://localhost:8080/completion", json_mode=False
        )
        mock = self._make_mock_client({"content": "ok"}, captured)
        client._http_client = mock

        asyncio.run(client.generate("Give me JSON", json_mode=True))
        assert "grammar" in captured

    def test_generate_raises_on_http_error(self, monkeypatch):
        import httpx

        class _ErrorResponse:
            status_code = 500

            def raise_for_status(self):
                raise httpx.HTTPStatusError(
                    "Server Error",
                    request=httpx.Request("POST", "http://x"),
                    response=httpx.Response(500),
                )

        class _ErrorClient:
            is_closed = False

            async def post(self, url, *, json=None, **kw):
                return _ErrorResponse()

            async def aclose(self):
                pass

        client = LlamaClient(api_url="http://localhost:8080/completion")
        client._http_client = _ErrorClient()

        with pytest.raises(httpx.HTTPStatusError):
            asyncio.run(client.generate("Say hi"))

    def test_generate_retries_once_on_transient_http_error(self, monkeypatch):
        class _RetryResponse:
            def __init__(self, status_code, payload=None):
                self.status_code = status_code
                self._payload = payload or {}
                self.text = ""

            def json(self):
                return self._payload

            def raise_for_status(self):
                if self.status_code >= 400:
                    import httpx

                    raise httpx.HTTPStatusError(
                        "Server Error",
                        request=httpx.Request("POST", "http://x"),
                        response=httpx.Response(self.status_code),
                    )

        class _RetryClient:
            is_closed = False

            def __init__(self) -> None:
                self.calls = 0

            async def post(self, url, *, json=None, **kw):
                self.calls += 1
                if self.calls == 1:
                    return _RetryResponse(500)
                return _RetryResponse(200, {"content": "ok after retry"})

            async def aclose(self):
                pass

        client = LlamaClient(api_url="http://localhost:8080/completion")
        client._http_client = _RetryClient()

        result = asyncio.run(client.generate("Say hi"))

        assert result == "ok after retry"
        assert client._http_client.calls == 2

    def test_generate_reduces_max_tokens_when_kv_budget_is_exceeded(self, monkeypatch):
        class _RetryResponse:
            def __init__(self, status_code, payload=None, text=""):
                self.status_code = status_code
                self._payload = payload or {}
                self.text = text

            def json(self):
                return self._payload

            def raise_for_status(self):
                if self.status_code >= 400:
                    import httpx

                    raise httpx.HTTPStatusError(
                        "Server Error",
                        request=httpx.Request("POST", "http://x"),
                        response=httpx.Response(self.status_code),
                    )

        class _RetryClient:
            is_closed = False

            def __init__(self) -> None:
                self.max_tokens_seen: list[int] = []

            async def post(self, url, *, json=None, **kw):
                requested = int((json or {}).get("max_tokens", 0))
                self.max_tokens_seen.append(requested)
                if requested > 192:
                    return _RetryResponse(
                        500,
                        text="Input prompt is too big compared to KV size.",
                    )
                return _RetryResponse(200, {"choices": [{"text": "ok after reduction"}]})

            async def aclose(self):
                pass

        client = LlamaClient(api_url="http://localhost:8080/v1/completions")
        client._http_client = _RetryClient()

        result = asyncio.run(client.generate("Say hi", max_tokens=504))

        assert result == "ok after reduction"
        assert client._http_client.max_tokens_seen == [504, 504, 252, 252, 126]

    def test_generate_reduces_max_tokens_when_upstream_returns_budget_400(self, monkeypatch):
        class _RetryResponse:
            def __init__(self, status_code, payload=None, text=""):
                self.status_code = status_code
                self._payload = payload or {}
                self.text = text

            def json(self):
                return self._payload

            def raise_for_status(self):
                if self.status_code >= 400:
                    import httpx

                    raise httpx.HTTPStatusError(
                        "Bad Request",
                        request=httpx.Request("POST", "http://x"),
                        response=httpx.Response(self.status_code),
                    )

        class _RetryClient:
            is_closed = False

            def __init__(self) -> None:
                self.max_tokens_seen: list[int] = []

            async def post(self, url, *, json=None, **kw):
                requested = int((json or {}).get("max_tokens", 0))
                self.max_tokens_seen.append(requested)
                if requested > 192:
                    return _RetryResponse(400, text="")
                return _RetryResponse(200, {"choices": [{"text": "ok after reduction"}]})

            async def aclose(self):
                pass

        client = LlamaClient(api_url="http://localhost:8080/v1/completions")
        client._http_client = _RetryClient()

        result = asyncio.run(client.generate("Say hi", max_tokens=504))

        assert result == "ok after reduction"
        assert client._http_client.max_tokens_seen == [504, 252, 126]

    def test_seed_sent_when_non_negative(self, monkeypatch):
        captured = {}
        client = LlamaClient(api_url="http://x", seed=42)
        mock = self._make_mock_client({"content": "ok"}, captured)
        client._http_client = mock

        asyncio.run(client.generate("test"))
        assert captured.get("seed") == 42

    def test_http_client_limits_follow_max_concurrent_requests(self, monkeypatch):
        captured = {}

        class _FakeAsyncClient:
            def __init__(self, **kwargs):
                captured.update(kwargs)
                self.is_closed = False

            async def aclose(self):
                self.is_closed = True

        monkeypatch.setattr("ai_engine.llm.llama_client.httpx.AsyncClient", _FakeAsyncClient)
        client = LlamaClient(
            api_url="http://localhost:8080/completion",
            max_concurrent_requests=3,
        )
        client._get_http_client()
        limits = captured["limits"]
        assert limits.max_connections == 3
        assert limits.max_keepalive_connections == 3

    def test_get_api_semaphore_is_reused(self):
        client = LlamaClient(api_url="http://localhost:8080/completion", max_concurrent_requests=4)

        semaphore = client._get_api_semaphore()

        assert semaphore is client._get_api_semaphore()
        assert getattr(semaphore, "_value", None) == 4

    def test_close_resets_http_client(self):
        class _ClosableClient:
            def __init__(self) -> None:
                self.is_closed = False
                self.closed = False

            async def aclose(self):
                self.is_closed = True
                self.closed = True

        client = LlamaClient(api_url="http://localhost:8080/completion")
        closable = _ClosableClient()
        client._http_client = closable

        asyncio.run(client.close())

        assert closable.closed is True
        assert client._http_client is None

    def test_set_api_url_closes_existing_http_client(self):
        class _ClosableClient:
            def __init__(self) -> None:
                self.is_closed = False
                self.closed = False

            async def aclose(self):
                self.is_closed = True
                self.closed = True

        client = LlamaClient(api_url="http://localhost:8080/completion")
        closable = _ClosableClient()
        client._http_client = closable

        asyncio.run(client.set_api_url("http://localhost:8081/completion"))

        assert closable.closed is True
        assert client.api_url == "http://localhost:8081/completion"
        assert client._http_client is None

    def test_should_retry_with_smaller_budget_rejects_non_budget_cases(self):
        class _Response:
            def __init__(self, status_code, text):
                self.status_code = status_code
                self.text = text

        client = LlamaClient(api_url="http://localhost:8080/completion")

        assert client._should_retry_with_smaller_budget(_Response(503, "retryable but not budget"), requested_tokens=256) is False
        assert client._should_retry_with_smaller_budget(_Response(500, "different failure"), requested_tokens=256) is False
        assert client._should_retry_with_smaller_budget(_Response(500, "KV cache is full"), requested_tokens=16) is False

    def test_generate_sync_uses_local_backend_directly(self, monkeypatch):
        client = LlamaClient(model_path="/fake/model.gguf")
        monkeypatch.setattr(client, "_generate_local", lambda prompt, tokens, use_json: f"{prompt}-{tokens}-{use_json}")

        result = client.generate_sync("hello", max_tokens=64, json_mode=True)

        assert result == "hello-64-True"


class TestLlamaClientProtocol:
    """Ensure LlamaClient satisfies the protocol expected by RAGPipeline."""

    def test_has_generate_method(self):
        client = LlamaClient(api_url="http://localhost:8080/completion")
        assert callable(getattr(client, "generate", None))


class TestLlamaClientGenerateLocal:
    """Tests for generate() via local GGUF model using monkeypatching."""

    def _fake_model(self, output_text: str = "local output"):
        """Return a callable that mimics llama-cpp-python Llama output."""

        def _call(prompt, **kwargs):
            return {"choices": [{"text": output_text}]}

        return _call

    def test_generate_local_returns_text(self, monkeypatch):
        """generate() delegates to _generate_local when api_url is None."""
        client = LlamaClient(model_path="/fake/model.gguf")
        fake_model = self._fake_model("hello local")
        monkeypatch.setattr(client, "_get_or_load_model", lambda: fake_model)
        result = asyncio.run(client.generate("Say hello"))
        assert result == "hello local"

    def test_generate_local_passes_max_tokens(self, monkeypatch):
        """max_tokens is forwarded to the model call."""
        captured: dict = {}
        client = LlamaClient(model_path="/fake/model.gguf")

        def fake_model(prompt, **kwargs):
            captured.update(kwargs)
            return {"choices": [{"text": "ok"}]}

        monkeypatch.setattr(client, "_get_or_load_model", lambda: fake_model)
        asyncio.run(client.generate("test", max_tokens=64))
        assert captured.get("max_tokens") == 64

    def test_generate_local_with_seed(self, monkeypatch):
        """A non-negative seed is forwarded to the local model."""
        captured: dict = {}
        client = LlamaClient(model_path="/fake/model.gguf", seed=7)

        def fake_model(prompt, **kwargs):
            captured.update(kwargs)
            return {"choices": [{"text": "ok"}]}

        monkeypatch.setattr(client, "_get_or_load_model", lambda: fake_model)
        asyncio.run(client.generate("test"))
        assert captured.get("seed") == 7

    def test_generate_local_json_mode_grammar_import_error(self, monkeypatch):
        """When llama_cpp.LlamaGrammar is unavailable, generation still works."""
        import builtins

        real_import = builtins.__import__

        def _fail_llama_grammar(name, *args, **kwargs):
            if name == "llama_cpp":
                raise ImportError("llama_cpp not available")
            return real_import(name, *args, **kwargs)

        client = LlamaClient(model_path="/fake/model.gguf", json_mode=True)
        fake_model = self._fake_model("{}")
        monkeypatch.setattr(client, "_get_or_load_model", lambda: fake_model)
        monkeypatch.setattr(builtins, "__import__", _fail_llama_grammar)

        result = asyncio.run(client.generate("test"))
        assert result == "{}"

    def test_get_or_load_model_raises_if_llama_cpp_missing(self, monkeypatch):
        """_get_or_load_model raises ImportError when llama-cpp-python is absent."""
        import builtins

        real_import = builtins.__import__

        def _fail_llama(name, *args, **kwargs):
            if name == "llama_cpp":
                raise ImportError("not installed")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _fail_llama)

        client = LlamaClient(model_path="/fake/model.gguf")
        client._local_model = None  # ensure not cached
        with pytest.raises(ImportError, match="llama-cpp-python"):
            client._get_or_load_model()

    def test_generate_local_empty_choices(self, monkeypatch):
        """An empty choices list returns an empty string gracefully."""
        client = LlamaClient(model_path="/fake/model.gguf")

        def fake_model(prompt, **kwargs):
            return {"choices": []}

        monkeypatch.setattr(client, "_get_or_load_model", lambda: fake_model)
        result = asyncio.run(client.generate("test"))
        assert result == ""

    def test_get_or_load_model_requires_model_path_when_local_backend_is_used(self, monkeypatch):
        import builtins
        import types

        real_import = builtins.__import__

        def _fake_import(name, *args, **kwargs):
            if name == "llama_cpp":
                return types.SimpleNamespace(Llama=object)
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _fake_import)

        client = LlamaClient(api_url="http://localhost:8080/completion")
        client.api_url = None

        with pytest.raises(ValueError, match="model_path is required"):
            client._get_or_load_model()


class TestJsonGrammar:
    """Verify the JSON GBNF grammar string is well-formed."""

    def test_grammar_contains_root_rule(self):
        assert "root" in JSON_GRAMMAR

    def test_grammar_contains_object_and_array(self):
        assert "object" in JSON_GRAMMAR
        assert "array" in JSON_GRAMMAR
