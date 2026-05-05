"""Client for llama.cpp-based LLM inference.

Supports two backends:

- **HTTP API**: Connects to a running ``llama.cpp`` server (or any
  compatible OpenAI-like endpoint) via ``requests``.
- **Local model** *(lazy)*: Loads a GGUF model file via
  ``llama-cpp-python`` on first call to :meth:`generate`.

Both backends support optional **JSON grammar** constraining so the
model output is guaranteed to be valid JSON.

At least one of ``api_url`` or ``model_path`` must be provided.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

import httpx

logger = logging.getLogger(__name__)

_RETRYABLE_STATUS_CODES = {500, 502, 503, 504}
_TRANSIENT_API_RETRIES = 1
_TRANSIENT_API_RETRY_DELAY_SECONDS = 0.35
_KV_CACHE_RETRY_MARKERS = (
    "Input prompt is too big compared to KV size",
    "KV cache is full",
)
_KV_CACHE_RETRY_ATTEMPTS = 5
_MIN_KV_CACHE_RETRY_TOKENS = 16

# Formal GBNF grammar that constrains output to valid JSON.
# This is sent to llama.cpp so the sampler only emits tokens that form
# syntactically correct JSON objects or arrays.
JSON_GRAMMAR = r"""
root   ::= object | array
value  ::= object | array | string | number | "true" | "false" | "null"

object ::= "{" ws ( pair ( "," ws pair )* )? "}" ws
pair   ::= string ":" ws value

array  ::= "[" ws ( value ( "," ws value )* )? "]" ws

string ::= "\"" ([^"\\] | "\\" .)* "\"" ws

number ::= ("-"? ([0-9] | [1-9][0-9]*)) ("." [0-9]+)? ([eE][-+]?[0-9]+)? ws

ws     ::= ([ \t\n\r])*
""".strip()


class LlamaClient:
    """Thin wrapper around a llama.cpp backend for text generation.

    Args:
        api_url: URL of a running llama.cpp HTTP server
            (e.g. ``http://localhost:8080/completion``).
        model_path: Path to a local GGUF model file.  The model is
            loaded lazily on the first call to :meth:`generate`.
        default_max_tokens: Default maximum tokens for generation.
        temperature: Sampling temperature (0 = greedy).
        top_p: Nucleus sampling probability mass.
        repeat_penalty: Penalty applied to repeated tokens.
        n_ctx: Context size for the local model (ignored when using the API).
        n_gpu_layers: Number of layers to offload to GPU (``0`` = CPU-only).
        seed: Random seed for reproducible output (``-1`` = random).
        json_mode: When ``True``, constrain generation to valid JSON via
            the built-in GBNF grammar.
    """

    def __init__(
        self,
        api_url: str | None = None,
        model_path: str | None = None,
        default_max_tokens: int = 512,
        temperature: float = 0.15,
        top_p: float = 0.9,
        repeat_penalty: float = 1.1,
        n_ctx: int = 4096,
        n_gpu_layers: int = 0,
        seed: int = -1,
        json_mode: bool = False,
        request_timeout_seconds: float = 600.0,
        max_concurrent_requests: int = 1,
    ) -> None:
        if api_url is None and model_path is None:
            raise ValueError("At least one of api_url or model_path must be provided")

        self.api_url = api_url
        self.model_path = model_path
        self.default_max_tokens = default_max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.repeat_penalty = repeat_penalty
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        self.seed = seed
        self.json_mode = json_mode
        self.request_timeout_seconds = max(1.0, float(request_timeout_seconds))
        self.max_concurrent_requests = max(1, int(max_concurrent_requests))

        # Lazy-loaded local model
        self._local_model: Any = None

        # Reusable async HTTP client with connection pooling
        self._http_client: httpx.AsyncClient | None = None
        self._api_semaphore: asyncio.Semaphore | None = None

    # ------------------------------------------------------------------
    # Async HTTP client lifecycle
    # ------------------------------------------------------------------

    def _get_http_client(self) -> httpx.AsyncClient:
        """Return a reusable async HTTP client (created lazily)."""
        if self._http_client is None or self._http_client.is_closed:
            self._http_client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.request_timeout_seconds, connect=10.0),
                limits=httpx.Limits(
                    max_connections=self.max_concurrent_requests,
                    max_keepalive_connections=self.max_concurrent_requests,
                    keepalive_expiry=30.0,
                ),
            )
        return self._http_client

    def _get_api_semaphore(self) -> asyncio.Semaphore:
        """Return the shared concurrency gate for upstream API requests."""
        if self._api_semaphore is None:
            self._api_semaphore = asyncio.Semaphore(self.max_concurrent_requests)
        return self._api_semaphore

    async def close(self) -> None:
        """Close the underlying HTTP client (call on shutdown)."""
        if self._http_client is not None and not self._http_client.is_closed:
            await self._http_client.aclose()
            self._http_client = None

    async def set_api_url(self, api_url: str | None) -> None:
        """Update the upstream llama.cpp URL without rebuilding the client object."""
        self.api_url = api_url
        if self._http_client is not None and not self._http_client.is_closed:
            await self._http_client.aclose()
            self._http_client = None

    async def plugin_capabilities(self) -> dict[str, Any]:
        """Return the effective thin-plugin capability snapshot.

        For HTTP-backed runtimes this probes the OpenAI-compatible models
        endpoint derived from ``api_url``. For local GGUF execution it reports
        the local model path without forcing model load.
        """
        if self.api_url is not None:
            models_url = _models_url_from_generation_url(self.api_url)
            client = self._get_http_client()
            response = await client.get(models_url)
            response.raise_for_status()
            payload = response.json()
            return {
                "status": "ready",
                "pluginMode": "thin-llm-server",
                "generationUrl": self.api_url,
                "modelsUrl": models_url,
                "models": _extract_model_ids(payload),
                "rawModelsPayload": payload,
                "capabilities": self._static_generation_capabilities(),
            }

        return {
            "status": "ready" if self.model_path else "unavailable",
            "pluginMode": "local-gguf-model",
            "generationUrl": None,
            "modelsUrl": None,
            "models": [Path(self.model_path).name] if self.model_path else [],
            "rawModelsPayload": None,
            "capabilities": self._static_generation_capabilities(),
        }

    def _static_generation_capabilities(self) -> dict[str, Any]:
        return {
            "jsonModeRequested": self.json_mode,
            "defaultMaxTokens": self.default_max_tokens,
            "temperature": self.temperature,
            "topP": self.top_p,
            "repeatPenalty": self.repeat_penalty,
            "contextSize": self.n_ctx,
            "gpuLayers": self.n_gpu_layers,
            "maxConcurrentRequests": self.max_concurrent_requests,
            "timeoutSeconds": self.request_timeout_seconds,
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def generate(
        self,
        prompt: str,
        max_tokens: int | None = None,
        *,
        json_mode: bool | None = None,
    ) -> str:
        """Generate a text completion for *prompt* (async).

        Uses the HTTP API backend when ``api_url`` is set, otherwise
        falls back to the local GGUF model (run in a thread to avoid
        blocking the event loop).

        Args:
            prompt: The input prompt.
            max_tokens: Override the default maximum number of tokens.
            json_mode: Override the instance-level ``json_mode`` flag for
                this single call.

        Returns:
            The generated text.
        """
        tokens = max_tokens or self.default_max_tokens
        use_json = json_mode if json_mode is not None else self.json_mode

        if self.api_url is not None:
            return await self._generate_via_api(prompt, tokens, use_json)
        return await asyncio.to_thread(self._generate_local, prompt, tokens, use_json)

    def generate_sync(
        self,
        prompt: str,
        max_tokens: int | None = None,
        *,
        json_mode: bool | None = None,
    ) -> str:
        """Synchronous wrapper kept for backward compatibility and tests."""
        tokens = max_tokens or self.default_max_tokens
        use_json = json_mode if json_mode is not None else self.json_mode

        if self.api_url is not None:
            return asyncio.get_event_loop().run_until_complete(
                self._generate_via_api(prompt, tokens, use_json)
            )
        return self._generate_local(prompt, tokens, use_json)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    async def _generate_via_api(
        self, prompt: str, max_tokens: int, json_mode: bool
    ) -> str:
        """Send a completion request to the llama.cpp HTTP server (async)."""
        api_url = (self.api_url or "").rstrip("/")
        is_openai_completions = api_url.endswith("/v1/completions")

        payload: dict[str, Any] = {
            "prompt": prompt,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "repeat_penalty": self.repeat_penalty,
        }
        if self.seed >= 0:
            payload["seed"] = self.seed
        if json_mode:
            payload["grammar"] = JSON_GRAMMAR

        client = self._get_http_client()
        requested_tokens = max(1, int(max_tokens))
        async with self._get_api_semaphore():
            response: httpx.Response | None = None
            for reduction_attempt in range(_KV_CACHE_RETRY_ATTEMPTS + 1):
                if is_openai_completions:
                    payload["max_tokens"] = requested_tokens
                else:
                    payload["n_predict"] = requested_tokens

                for attempt in range(_TRANSIENT_API_RETRIES + 1):
                    response = await client.post(
                        self.api_url,  # type: ignore[arg-type]
                        json=payload,
                    )
                    if response.status_code not in _RETRYABLE_STATUS_CODES:
                        break
                    if attempt >= _TRANSIENT_API_RETRIES:
                        break
                    logger.warning(
                        "Transient llama upstream error %s; retrying request once.",
                        response.status_code,
                    )
                    await asyncio.sleep(_TRANSIENT_API_RETRY_DELAY_SECONDS)

                if response is None or not self._should_retry_with_smaller_budget(
                    response,
                    requested_tokens=requested_tokens,
                ):
                    break

                reduced_tokens = max(
                    _MIN_KV_CACHE_RETRY_TOKENS,
                    requested_tokens // 2,
                )
                if (
                    reduction_attempt >= _KV_CACHE_RETRY_ATTEMPTS
                    or reduced_tokens >= requested_tokens
                ):
                    break

                logger.warning(
                    "Llama upstream rejected prompt for KV budget with max_tokens=%s; retrying with reduced max_tokens=%s.",
                    requested_tokens,
                    reduced_tokens,
                )
                requested_tokens = reduced_tokens
        assert response is not None
        response.raise_for_status()
        data = response.json()

        # llama.cpp may return either legacy {"content"|"text"} payloads or
        # OpenAI-compatible payloads with choices[0].text.
        if isinstance(data, dict):
            if "content" in data:
                return str(data.get("content", ""))
            if "text" in data:
                return str(data.get("text", ""))

            choices = data.get("choices")
            if isinstance(choices, list) and choices:
                first = choices[0]
                if isinstance(first, dict):
                    if "text" in first:
                        return str(first.get("text", ""))
                    message = first.get("message")
                    if isinstance(message, dict) and "content" in message:
                        return str(message.get("content", ""))

        return ""

    def _should_retry_with_smaller_budget(
        self,
        response: httpx.Response,
        *,
        requested_tokens: int,
    ) -> bool:
        if (
            response.status_code not in {400, 500}
            or requested_tokens <= _MIN_KV_CACHE_RETRY_TOKENS
        ):
            return False
        message = str(getattr(response, "text", "") or "")
        return not message or any(
            marker in message for marker in _KV_CACHE_RETRY_MARKERS
        )

    def _generate_local(self, prompt: str, max_tokens: int, json_mode: bool) -> str:
        """Generate using a locally loaded GGUF model via llama-cpp-python."""
        model = self._get_or_load_model()

        kwargs: dict[str, Any] = {
            "max_tokens": max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "repeat_penalty": self.repeat_penalty,
        }
        if self.seed >= 0:
            kwargs["seed"] = self.seed

        if json_mode:
            try:
                from llama_cpp import LlamaGrammar  # type: ignore[import-untyped]

                kwargs["grammar"] = LlamaGrammar.from_string(JSON_GRAMMAR)
            except (ImportError, Exception) as exc:  # noqa: BLE001
                logger.warning(
                    "Could not apply JSON grammar (%s). "
                    "Generating without grammar constraint.",
                    exc,
                )

        output = model(prompt, **kwargs)
        # llama-cpp-python returns a dict with "choices"
        choices = output.get("choices", [{}])
        return choices[0].get("text", "") if choices else ""

    def _get_or_load_model(self) -> Any:
        """Lazily load the local GGUF model."""
        if self._local_model is None:
            try:
                from llama_cpp import Llama  # type: ignore[import-untyped]
            except ImportError as exc:
                raise ImportError(
                    "llama-cpp-python is required for local model inference. "
                    "Install it with: pip install llama-cpp-python"
                ) from exc

            if self.model_path is None:
                raise ValueError("model_path is required for local model inference")

            logger.info("Loading GGUF model from %s …", self.model_path)
            self._local_model = Llama(
                model_path=self.model_path,
                n_ctx=self.n_ctx,
                n_gpu_layers=self.n_gpu_layers,
                seed=self.seed if self.seed >= 0 else -1,
                verbose=False,
            )
            logger.info("Model loaded successfully.")
        return self._local_model


def _models_url_from_generation_url(api_url: str) -> str:
    """Derive the OpenAI-compatible models endpoint from a generation URL."""
    parsed = httpx.URL(api_url)
    path = parsed.path.rstrip("/")
    if path.endswith("/v1/completions"):
        path = path[: -len("/completions")]
    elif path.endswith("/v1/chat/completions"):
        path = path[: -len("/chat/completions")]
    elif not path.endswith("/v1"):
        path = f"{path}/v1" if path else "/v1"
    return str(parsed.copy_with(path=f"{path}/models", query=None, fragment=None))


def _extract_model_ids(payload: Any) -> list[str]:
    """Extract model ids from common llama.cpp/OpenAI model-list payloads."""
    if not isinstance(payload, dict):
        return []
    data = payload.get("data")
    if isinstance(data, list):
        ids: list[str] = []
        for item in data:
            if isinstance(item, dict):
                model_id = item.get("id") or item.get("name")
                if isinstance(model_id, str) and model_id.strip():
                    ids.append(model_id.strip())
        return ids
    models = payload.get("models")
    if isinstance(models, list):
        return [str(model).strip() for model in models if str(model).strip()]
    return []
