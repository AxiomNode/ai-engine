"""Thread-safe event collector and in-memory statistics store.

This module provides :class:`GenerationEvent` — a lightweight data-class that
represents a single LLM generation call — and :class:`StatsCollector`, which
accumulates events and computes aggregate statistics on demand.

All public methods on :class:`StatsCollector` are thread-safe, so the
collector can be shared across ASGI workers or background threads safely.
"""

from __future__ import annotations

import statistics
import threading
import time
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class GenerationEvent:
    """Immutable record of a single LLM generation call.

    Attributes:
        timestamp: Unix epoch seconds when the event was recorded.
        prompt_chars: Number of characters in the prompt.
        response_chars: Number of characters in the response.
        latency_ms: Wall-clock latency in milliseconds.
        max_tokens: Token budget passed to the LLM.
        json_mode: Whether JSON grammar was active.
        success: Whether the call completed without error.
        game_type: Game type if triggered by :class:`GameGenerator` (optional).
        error: Error message if ``success`` is ``False``.
    """

    timestamp: float
    prompt_chars: int
    response_chars: int
    latency_ms: float
    max_tokens: int
    json_mode: bool
    success: bool
    game_type: str | None = None
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the event to a plain dictionary."""
        return asdict(self)


@dataclass
class StatsCollector:
    """Accumulates :class:`GenerationEvent` records and provides aggregates.

    This class is the central piece of the observability module.  Instrument
    your LLM client or game generator with :meth:`record` and then query
    :meth:`summary` or :meth:`history` to inspect performance.

    Attributes:
        max_history: Maximum number of events to retain.  When exceeded the
            oldest events are discarded.
    """

    max_history: int = 10_000
    _events: list[GenerationEvent] = field(default_factory=list, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record(self, event: GenerationEvent) -> None:
        """Append an event to the internal buffer (thread-safe).

        If the buffer exceeds *max_history*, the oldest event is removed.

        Args:
            event: The generation event to store.
        """
        with self._lock:
            self._events.append(event)
            if len(self._events) > self.max_history:
                self._events = self._events[-self.max_history :]

    # ------------------------------------------------------------------
    # Convenience builder
    # ------------------------------------------------------------------

    def record_call(
        self,
        prompt: str,
        response: str,
        latency_ms: float,
        max_tokens: int,
        json_mode: bool = False,
        success: bool = True,
        game_type: str | None = None,
        error: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> GenerationEvent:
        """Build a :class:`GenerationEvent` from raw values and record it.

        This is a shortcut for creating the event manually and calling
        :meth:`record`.

        Args:
            prompt: The prompt string sent to the LLM.
            response: The text returned by the LLM.
            latency_ms: Wall-clock latency in milliseconds.
            max_tokens: Token budget for this call.
            json_mode: Whether JSON grammar was active.
            success: Whether the call succeeded.
            game_type: Optional game type identifier.
            error: Optional error message.

        Returns:
            The newly created :class:`GenerationEvent`.
        """
        event = GenerationEvent(
            timestamp=time.time(),
            prompt_chars=len(prompt),
            response_chars=len(response),
            latency_ms=round(latency_ms, 2),
            max_tokens=max_tokens,
            json_mode=json_mode,
            success=success,
            game_type=game_type,
            error=error,
            metadata=dict(metadata or {}),
        )
        self.record(event)
        return event

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def history(self, last_n: int | None = None) -> list[dict[str, Any]]:
        """Return recorded events as a list of dictionaries.

        Args:
            last_n: If provided, only the *last_n* most recent events are
                returned.

        Returns:
            A list of event dictionaries ordered oldest-first.
        """
        with self._lock:
            events = list(self._events)
        if last_n is not None:
            events = events[-last_n:]
        return [e.to_dict() for e in events]

    def summary(self) -> dict[str, Any]:
        """Compute aggregate statistics over all recorded events.

        Returns:
            A dictionary with keys such as ``total_calls``,
            ``success_rate``, ``avg_latency_ms``, etc.  Returns zeros
            when no events have been recorded.
        """
        with self._lock:
            events = list(self._events)

        if not events:
            return {
                "total_calls": 0,
                "successful_calls": 0,
                "failed_calls": 0,
                "success_rate": 0.0,
                "avg_latency_ms": 0.0,
                "p50_latency_ms": 0.0,
                "p95_latency_ms": 0.0,
                "p99_latency_ms": 0.0,
                "max_latency_ms": 0.0,
                "avg_prompt_chars": 0.0,
                "avg_response_chars": 0.0,
                "total_prompt_chars": 0,
                "total_response_chars": 0,
                "json_mode_calls": 0,
                "game_type_counts": {},
                "cache_hits": 0,
                "cache_misses": 0,
                "cache_hit_rate": 0.0,
                "cache_layer_counts": {},
                "avg_rag_latency_ms": 0.0,
                "avg_llm_latency_ms": 0.0,
                "avg_parse_latency_ms": 0.0,
                "avg_total_latency_ms": 0.0,
                "kbd_hits_total": 0,
                "db_reads_total": 0,
                "db_writes_total": 0,
                "language_counts": {},
            }

        total = len(events)
        successes = sum(1 for e in events if e.success)
        latencies = sorted(e.latency_ms for e in events)

        game_counts: dict[str, int] = {}
        cache_hits = 0
        cache_misses = 0
        cache_layer_counts: dict[str, int] = {}
        rag_latencies: list[float] = []
        llm_latencies: list[float] = []
        parse_latencies: list[float] = []
        total_latencies: list[float] = []
        kbd_hits_total = 0
        db_reads_total = 0
        db_writes_total = 0
        language_counts: dict[str, int] = {}

        for e in events:
            if e.game_type:
                game_counts[e.game_type] = game_counts.get(e.game_type, 0) + 1

            meta = e.metadata or {}
            if bool(meta.get("cache_hit", False)):
                cache_hits += 1
            else:
                cache_misses += 1

            layer = str(meta.get("cache_layer", "none"))
            cache_layer_counts[layer] = cache_layer_counts.get(layer, 0) + 1

            rag_ms = meta.get("rag_latency_ms")
            if isinstance(rag_ms, (int, float)):
                rag_latencies.append(float(rag_ms))

            llm_ms = meta.get("llm_latency_ms")
            if isinstance(llm_ms, (int, float)):
                llm_latencies.append(float(llm_ms))

            parse_ms = meta.get("parse_latency_ms")
            if isinstance(parse_ms, (int, float)):
                parse_latencies.append(float(parse_ms))

            total_ms = meta.get("total_latency_ms")
            if isinstance(total_ms, (int, float)):
                total_latencies.append(float(total_ms))

            kbd_hits_total += int(meta.get("kbd_hits", 0) or 0)
            db_reads_total += int(meta.get("db_reads", 0) or 0)
            db_writes_total += int(meta.get("db_writes", 0) or 0)

            language = str(meta.get("language", "unknown"))
            language_counts[language] = language_counts.get(language, 0) + 1

        return {
            "total_calls": total,
            "successful_calls": successes,
            "failed_calls": total - successes,
            "success_rate": round(successes / total, 4),
            "avg_latency_ms": round(statistics.mean(latencies), 2),
            "p50_latency_ms": round(_percentile(latencies, 50), 2),
            "p95_latency_ms": round(_percentile(latencies, 95), 2),
            "p99_latency_ms": round(_percentile(latencies, 99), 2),
            "max_latency_ms": round(max(latencies), 2),
            "avg_prompt_chars": round(
                statistics.mean(e.prompt_chars for e in events), 2
            ),
            "avg_response_chars": round(
                statistics.mean(e.response_chars for e in events), 2
            ),
            "total_prompt_chars": sum(e.prompt_chars for e in events),
            "total_response_chars": sum(e.response_chars for e in events),
            "json_mode_calls": sum(1 for e in events if e.json_mode),
            "game_type_counts": game_counts,
            "cache_hits": cache_hits,
            "cache_misses": cache_misses,
            "cache_hit_rate": round(cache_hits / total, 4),
            "cache_layer_counts": cache_layer_counts,
            "avg_rag_latency_ms": (
                round(statistics.mean(rag_latencies), 2) if rag_latencies else 0.0
            ),
            "avg_llm_latency_ms": (
                round(statistics.mean(llm_latencies), 2) if llm_latencies else 0.0
            ),
            "avg_parse_latency_ms": (
                round(statistics.mean(parse_latencies), 2) if parse_latencies else 0.0
            ),
            "avg_total_latency_ms": (
                round(statistics.mean(total_latencies), 2) if total_latencies else 0.0
            ),
            "kbd_hits_total": kbd_hits_total,
            "db_reads_total": db_reads_total,
            "db_writes_total": db_writes_total,
            "language_counts": language_counts,
        }

    def reset(self) -> None:
        """Remove all recorded events."""
        with self._lock:
            self._events.clear()

    def __len__(self) -> int:
        """Return the number of recorded events."""
        with self._lock:
            return len(self._events)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _percentile(sorted_data: list[float], pct: float) -> float:
    """Compute the *pct*-th percentile of already-sorted data.

    Uses linear interpolation between closest ranks.

    Args:
        sorted_data: A sorted list of numeric values (must not be empty).
        pct: Percentile to compute (0–100).

    Returns:
        The interpolated percentile value.
    """
    if not sorted_data:
        return 0.0
    k = (pct / 100) * (len(sorted_data) - 1)
    f = int(k)
    c = f + 1
    if c >= len(sorted_data):
        return sorted_data[-1]
    return sorted_data[f] + (k - f) * (sorted_data[c] - sorted_data[f])
