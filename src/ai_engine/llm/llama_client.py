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

import logging
from typing import Any

import requests

logger = logging.getLogger(__name__)

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
        default_max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.95,
        repeat_penalty: float = 1.1,
        n_ctx: int = 4096,
        n_gpu_layers: int = 0,
        seed: int = -1,
        json_mode: bool = False,
    ) -> None:
        if api_url is None and model_path is None:
            raise ValueError(
                "At least one of api_url or model_path must be provided"
            )

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

        # Lazy-loaded local model
        self._local_model: Any = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        max_tokens: int | None = None,
        *,
        json_mode: bool | None = None,
    ) -> str:
        """Generate a text completion for *prompt*.

        Uses the HTTP API backend when ``api_url`` is set, otherwise
        falls back to the local GGUF model.

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
            return self._generate_via_api(prompt, tokens, use_json)
        return self._generate_local(prompt, tokens, use_json)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _generate_via_api(
        self, prompt: str, max_tokens: int, json_mode: bool
    ) -> str:
        """Send a completion request to the llama.cpp HTTP server."""
        payload: dict[str, Any] = {
            "prompt": prompt,
            "n_predict": max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "repeat_penalty": self.repeat_penalty,
        }
        if self.seed >= 0:
            payload["seed"] = self.seed
        if json_mode:
            payload["grammar"] = JSON_GRAMMAR

        response = requests.post(
            self.api_url,  # type: ignore[arg-type]
            json=payload,
            timeout=180,
        )
        response.raise_for_status()
        data = response.json()
        return data.get("content", data.get("text", ""))

    def _generate_local(
        self, prompt: str, max_tokens: int, json_mode: bool
    ) -> str:
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
