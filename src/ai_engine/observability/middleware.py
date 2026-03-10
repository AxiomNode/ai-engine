"""Instrumentation middleware for LLM clients and game generators.

Provides :class:`TrackedLlamaClient` and :class:`TrackedGameGenerator`,
thin wrappers that record every generation call to a
:class:`~ai_engine.observability.collector.StatsCollector`.

Using these wrappers is entirely optional — they are a convenience for
automatic stats collection without modifying existing code.

Example::

    from ai_engine.llm import LlamaClient
    from ai_engine.observability import StatsCollector
    from ai_engine.observability.middleware import TrackedLlamaClient

    collector = StatsCollector()
    raw_llm = LlamaClient(api_url="http://localhost:8080/completion")
    llm = TrackedLlamaClient(raw_llm, collector)

    response = llm.generate("Hello!")  # automatically recorded
"""

from __future__ import annotations

import logging
import time
from typing import Any

from ai_engine.observability.collector import StatsCollector

logger = logging.getLogger(__name__)


class TrackedLlamaClient:
    """Wrapper around any LLM client that records calls to a collector.

    The wrapped object must expose a ``generate(prompt, max_tokens, **kw)``
    method (same interface as :class:`~ai_engine.llm.LlamaClient`).

    All attribute access that is not overridden is forwarded to the
    inner client, so the wrapper is a transparent drop-in replacement.

    Args:
        client: The underlying LLM client.
        collector: A :class:`StatsCollector` to record events.
    """

    def __init__(self, client: Any, collector: StatsCollector) -> None:
        self._client = client
        self._collector = collector

    def generate(
        self,
        prompt: str,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> str:
        """Call the inner client's ``generate`` and record the event.

        All positional and keyword arguments are forwarded to the
        underlying client.

        Args:
            prompt: The input prompt.
            max_tokens: Token budget.
            **kwargs: Extra keyword arguments forwarded to the LLM.

        Returns:
            The generated text from the underlying client.

        Raises:
            Exception: Re-raises any exception from the LLM after
                recording the failure event.
        """
        _fallback: int = int(getattr(self._client, "default_max_tokens", 256))
        tokens: int = max_tokens if max_tokens is not None else _fallback
        json_mode = kwargs.get("json_mode", getattr(self._client, "json_mode", False))

        start = time.perf_counter()
        try:
            response = self._client.generate(prompt, max_tokens=max_tokens, **kwargs)
            elapsed_ms = (time.perf_counter() - start) * 1000
            self._collector.record_call(
                prompt=prompt,
                response=response,
                latency_ms=elapsed_ms,
                max_tokens=tokens,
                json_mode=json_mode,
                success=True,
            )
            return response
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - start) * 1000
            self._collector.record_call(
                prompt=prompt,
                response="",
                latency_ms=elapsed_ms,
                max_tokens=tokens,
                json_mode=json_mode,
                success=False,
                error=str(exc),
            )
            raise

    def __getattr__(self, name: str) -> Any:
        """Forward attribute access to the inner client."""
        return getattr(self._client, name)


class TrackedGameGenerator:
    """Wrapper around :class:`~ai_engine.games.generator.GameGenerator`
    that records each generation call to a :class:`StatsCollector`.

    The wrapper intercepts ``generate`` and ``generate_raw`` calls,
    measures latency, and records successes/failures.

    Args:
        generator: The underlying game generator.
        collector: A :class:`StatsCollector` to record events.
    """

    def __init__(self, generator: Any, collector: StatsCollector) -> None:
        self._generator = generator
        self._collector = collector

    def generate(self, *args: Any, **kwargs: Any) -> Any:
        """Wrap :meth:`GameGenerator.generate` with event recording.

        Args:
            *args: Positional arguments forwarded to the generator.
            **kwargs: Keyword arguments forwarded to the generator.

        Returns:
            A :class:`~ai_engine.games.schemas.GameEnvelope`.

        Raises:
            Exception: Re-raises any exception after recording the
                failure event.
        """
        game_type = kwargs.get("game_type", "quiz")
        tokens: int = int(kwargs.get("max_tokens") or self._generator.default_max_tokens)
        query = args[0] if args else kwargs.get("query", "")

        start = time.perf_counter()
        try:
            result = self._generator.generate(*args, **kwargs)
            elapsed_ms = (time.perf_counter() - start) * 1000
            self._collector.record_call(
                prompt=str(query),
                response=str(result),
                latency_ms=elapsed_ms,
                max_tokens=tokens,
                json_mode=True,
                success=True,
                game_type=game_type,
            )
            return result
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - start) * 1000
            self._collector.record_call(
                prompt=str(query),
                response="",
                latency_ms=elapsed_ms,
                max_tokens=tokens,
                json_mode=True,
                success=False,
                game_type=game_type,
                error=str(exc),
            )
            raise

    def generate_raw(self, *args: Any, **kwargs: Any) -> Any:
        """Wrap :meth:`GameGenerator.generate_raw` with event recording.

        Args:
            *args: Positional arguments forwarded to the generator.
            **kwargs: Keyword arguments forwarded to the generator.

        Returns:
            A raw ``dict`` with the parsed game data.

        Raises:
            Exception: Re-raises any exception after recording the
                failure event.
        """
        game_type = kwargs.get("game_type", "quiz")
        tokens: int = int(kwargs.get("max_tokens") or self._generator.default_max_tokens)
        query = args[0] if args else kwargs.get("query", "")

        start = time.perf_counter()
        try:
            result = self._generator.generate_raw(*args, **kwargs)
            elapsed_ms = (time.perf_counter() - start) * 1000
            self._collector.record_call(
                prompt=str(query),
                response=str(result),
                latency_ms=elapsed_ms,
                max_tokens=tokens,
                json_mode=True,
                success=True,
                game_type=game_type,
            )
            return result
        except Exception as exc:
            elapsed_ms = (time.perf_counter() - start) * 1000
            self._collector.record_call(
                prompt=str(query),
                response="",
                latency_ms=elapsed_ms,
                max_tokens=tokens,
                json_mode=True,
                success=False,
                game_type=game_type,
                error=str(exc),
            )
            raise

    def __getattr__(self, name: str) -> Any:
        """Forward attribute access to the inner generator."""
        return getattr(self._generator, name)
