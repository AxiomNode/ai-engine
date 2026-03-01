"""Simple Llama client using `llama-cpp-python`.

This client expects a local GGML/gguf model file (path provided in config).
"""

from __future__ import annotations

import json
from typing import Any

try:
    from llama_cpp import Llama
except Exception:  # pragma: no cover - optional dependency
    Llama = None


class LlamaClient:
    """Lightweight wrapper around `llama-cpp-python`.

    Args:
        model_path: Path to the local gguf/ggml model file.
        n_ctx: Context window size.
    """

    def __init__(self, model_path: str, n_ctx: int = 2048, **kwargs: Any) -> None:
        if Llama is None:
            raise RuntimeError(
                "llama-cpp-python is not installed. Install extras 'llm' or 'pip install llama-cpp-python'."
            )
        self.model_path = model_path
        self._client = Llama(model_path=model_path, n_ctx=n_ctx, **kwargs)

    def generate(self, prompt: str, max_tokens: int = 256, temperature: float = 0.2) -> str:
        """Generate text for a prompt and return the raw completion string.

        Returns only the generated text (not metadata).
        """
        resp = self._client.create(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=0.95,
        )
        # resp may contain 'choices' with 'text'
        if isinstance(resp, dict):
            choices = resp.get("choices") or []
            if choices:
                return choices[0].get("text", "").strip()
            # fallback for other llama-cpp-python return formats
            return json.dumps(resp)
        return str(resp)
