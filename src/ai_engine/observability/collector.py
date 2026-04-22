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
    _summary_cache: dict[str, Any] | None = field(default=None, repr=False)
    _summary_cache_event_count: int = field(default=0, repr=False)
    _summary_cache_ts: float = field(default=0.0, repr=False)
    _summary_cache_ttl: float = 5.0

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
            self._summary_cache = None

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
            now = time.time()
            if (
                self._summary_cache is not None
                and self._summary_cache_event_count == len(self._events)
                and (now - self._summary_cache_ts) < self._summary_cache_ttl
            ):
                return self._summary_cache
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
                "event_type_counts": {},
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
                "retry_used_count": 0,
                "retry_rate": 0.0,
                "avg_rag_similarity": 0.0,
                "avg_rag_context_length_chars": 0.0,
                "language_counts": {},
                "generation_outcome_by_game_type": {},
                "generation_outcome_by_language": {},
                "persistent_backend_counts": {},
                "persistent_fallback_total": 0,
                "persistent_error_counts": {},
                "correlation_id_counts": {},
                "distribution_version_counts": {},
            }

        total = len(events)
        successes = sum(1 for e in events if e.success)
        latencies = sorted(e.latency_ms for e in events)

        game_counts: dict[str, int] = {}
        event_type_counts: dict[str, int] = {}
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
        retry_used_count = 0
        generation_count = 0
        rag_similarity_scores: list[float] = []
        rag_context_lengths: list[int] = []
        language_counts: dict[str, int] = {}
        generation_outcome_by_game_type: dict[str, dict[str, int]] = {}
        generation_outcome_by_language: dict[str, dict[str, int]] = {}
        persistent_backend_counts: dict[str, int] = {}
        persistent_fallback_total = 0
        persistent_error_counts: dict[str, int] = {}
        correlation_id_counts: dict[str, int] = {}
        distribution_version_counts: dict[str, int] = {}

        for e in events:
            if e.game_type:
                game_counts[e.game_type] = game_counts.get(e.game_type, 0) + 1

            meta = e.metadata or {}
            event_type = str(
                meta.get(
                    "event_type", "generation" if e.game_type != "ingest" else "ingest"
                )
            )
            event_type_counts[event_type] = event_type_counts.get(event_type, 0) + 1

            if event_type == "generation":
                if bool(meta.get("cache_hit", False)):
                    cache_hits += 1
                else:
                    cache_misses += 1

                layer = str(meta.get("cache_layer", "none"))
                cache_layer_counts[layer] = cache_layer_counts.get(layer, 0) + 1

                outcome = "success" if e.success else "failure"
                game_label = str(meta.get("game_type", e.game_type or "unknown"))
                language_label = str(meta.get("language", "unknown"))

                game_bucket = generation_outcome_by_game_type.setdefault(
                    game_label,
                    {"success": 0, "failure": 0},
                )
                game_bucket[outcome] += 1

                language_bucket = generation_outcome_by_language.setdefault(
                    language_label,
                    {"success": 0, "failure": 0},
                )
                language_bucket[outcome] += 1

                backend = str(meta.get("persistent_backend", "none"))
                persistent_backend_counts[backend] = (
                    persistent_backend_counts.get(backend, 0) + 1
                )

                if bool(meta.get("persistent_fallback_used", False)):
                    persistent_fallback_total += 1

                persistent_error = str(meta.get("persistent_error", "")).strip()
                if persistent_error:
                    persistent_error_counts[persistent_error] = (
                        persistent_error_counts.get(persistent_error, 0) + 1
                    )

                correlation_id = str(meta.get("correlation_id", "")).strip()
                if correlation_id:
                    correlation_id_counts[correlation_id] = (
                        correlation_id_counts.get(correlation_id, 0) + 1
                    )

                distribution_version = str(meta.get("distribution_version", "")).strip()
                if distribution_version:
                    distribution_version_counts[distribution_version] = (
                        distribution_version_counts.get(distribution_version, 0) + 1
                    )

                generation_count += 1
                if bool(meta.get("retry_used", False)):
                    retry_used_count += 1

                avg_sim = meta.get("avg_rag_similarity")
                if isinstance(avg_sim, (int, float)) and avg_sim > 0:
                    rag_similarity_scores.append(float(avg_sim))

                ctx_len = meta.get("rag_context_length_chars")
                if isinstance(ctx_len, (int, float)):
                    rag_context_lengths.append(int(ctx_len))

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

        result = {
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
            "event_type_counts": event_type_counts,
            "cache_hits": cache_hits,
            "cache_misses": cache_misses,
            "cache_hit_rate": (
                round(cache_hits / (cache_hits + cache_misses), 4)
                if (cache_hits + cache_misses) > 0
                else 0.0
            ),
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
            "retry_used_count": retry_used_count,
            "retry_rate": (
                round(retry_used_count / generation_count, 4)
                if generation_count > 0
                else 0.0
            ),
            "avg_rag_similarity": (
                round(statistics.mean(rag_similarity_scores), 4)
                if rag_similarity_scores
                else 0.0
            ),
            "avg_rag_context_length_chars": (
                round(statistics.mean(rag_context_lengths), 2)
                if rag_context_lengths
                else 0.0
            ),
            "language_counts": language_counts,
            "generation_outcome_by_game_type": generation_outcome_by_game_type,
            "generation_outcome_by_language": generation_outcome_by_language,
            "persistent_backend_counts": persistent_backend_counts,
            "persistent_fallback_total": persistent_fallback_total,
            "persistent_error_counts": persistent_error_counts,
            "correlation_id_counts": correlation_id_counts,
            "distribution_version_counts": distribution_version_counts,
        }

        with self._lock:
            self._summary_cache = result
            self._summary_cache_event_count = len(self._events)
            self._summary_cache_ts = time.time()

        return result

    def reset(self) -> None:
        """Remove all recorded events."""
        with self._lock:
            self._events.clear()
            self._summary_cache = None

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


def summary_to_prometheus(summary: dict[str, Any]) -> str:
    """Render a summary dictionary in Prometheus text exposition format."""

    def _metric(name: str, value: float | int) -> str:
        return f"ai_engine_{name} {value}"

    lines = [
        "# HELP ai_engine_total_calls Total recorded events.",
        "# TYPE ai_engine_total_calls counter",
        _metric("total_calls", int(summary.get("total_calls", 0))),
        "# HELP ai_engine_successful_calls Successful events.",
        "# TYPE ai_engine_successful_calls counter",
        _metric("successful_calls", int(summary.get("successful_calls", 0))),
        "# HELP ai_engine_failed_calls Failed events.",
        "# TYPE ai_engine_failed_calls counter",
        _metric("failed_calls", int(summary.get("failed_calls", 0))),
        "# HELP ai_engine_success_rate Success ratio across events.",
        "# TYPE ai_engine_success_rate gauge",
        _metric("success_rate", float(summary.get("success_rate", 0.0))),
        "# HELP ai_engine_cache_hit_rate Cache hit ratio for generation events.",
        "# TYPE ai_engine_cache_hit_rate gauge",
        _metric("cache_hit_rate", float(summary.get("cache_hit_rate", 0.0))),
        "# HELP ai_engine_avg_latency_ms Average latency in milliseconds.",
        "# TYPE ai_engine_avg_latency_ms gauge",
        _metric("avg_latency_ms", float(summary.get("avg_latency_ms", 0.0))),
        "# HELP ai_engine_avg_rag_latency_ms Average RAG latency in milliseconds.",
        "# TYPE ai_engine_avg_rag_latency_ms gauge",
        _metric("avg_rag_latency_ms", float(summary.get("avg_rag_latency_ms", 0.0))),
        "# HELP ai_engine_avg_llm_latency_ms Average LLM latency in milliseconds.",
        "# TYPE ai_engine_avg_llm_latency_ms gauge",
        _metric("avg_llm_latency_ms", float(summary.get("avg_llm_latency_ms", 0.0))),
        "# HELP ai_engine_avg_parse_latency_ms Average parse latency in milliseconds.",
        "# TYPE ai_engine_avg_parse_latency_ms gauge",
        _metric(
            "avg_parse_latency_ms", float(summary.get("avg_parse_latency_ms", 0.0))
        ),
        "# HELP ai_engine_kbd_hits_total Total KBD hits.",
        "# TYPE ai_engine_kbd_hits_total counter",
        _metric("kbd_hits_total", int(summary.get("kbd_hits_total", 0))),
        "# HELP ai_engine_db_reads_total Total DB reads.",
        "# TYPE ai_engine_db_reads_total counter",
        _metric("db_reads_total", int(summary.get("db_reads_total", 0))),
        "# HELP ai_engine_db_writes_total Total DB writes.",
        "# TYPE ai_engine_db_writes_total counter",
        _metric("db_writes_total", int(summary.get("db_writes_total", 0))),
        "# HELP ai_engine_retry_used_count Generations that needed JSON retry.",
        "# TYPE ai_engine_retry_used_count counter",
        _metric("retry_used_count", int(summary.get("retry_used_count", 0))),
        "# HELP ai_engine_retry_rate Ratio of generations requiring retry.",
        "# TYPE ai_engine_retry_rate gauge",
        _metric("retry_rate", float(summary.get("retry_rate", 0.0))),
        "# HELP ai_engine_avg_rag_similarity Average RAG retrieval similarity score.",
        "# TYPE ai_engine_avg_rag_similarity gauge",
        _metric("avg_rag_similarity", float(summary.get("avg_rag_similarity", 0.0))),
        "# HELP ai_engine_avg_rag_context_length_chars Average RAG context length.",
        "# TYPE ai_engine_avg_rag_context_length_chars gauge",
        _metric(
            "avg_rag_context_length_chars",
            float(summary.get("avg_rag_context_length_chars", 0.0)),
        ),
    ]

    for label, count in sorted((summary.get("game_type_counts") or {}).items()):
        lines.append(f'ai_engine_game_type_calls{{game_type="{label}"}} {int(count)}')

    for label, count in sorted((summary.get("language_counts") or {}).items()):
        lines.append(f'ai_engine_language_calls{{language="{label}"}} {int(count)}')

    for label, count in sorted((summary.get("event_type_counts") or {}).items()):
        lines.append(f'ai_engine_event_type_calls{{event_type="{label}"}} {int(count)}')

    for label, count in sorted((summary.get("cache_layer_counts") or {}).items()):
        lines.append(
            f'ai_engine_cache_layer_calls{{cache_layer="{label}"}} {int(count)}'
        )

    for game_type, outcomes in sorted(
        (summary.get("generation_outcome_by_game_type") or {}).items()
    ):
        for outcome, count in sorted((outcomes or {}).items()):
            lines.append(
                "ai_engine_generation_outcome_by_game_type_total"
                f'{{game_type="{game_type}",outcome="{outcome}"}} {int(count)}'
            )

    for language, outcomes in sorted(
        (summary.get("generation_outcome_by_language") or {}).items()
    ):
        for outcome, count in sorted((outcomes or {}).items()):
            lines.append(
                "ai_engine_generation_outcome_by_language_total"
                f'{{language="{language}",outcome="{outcome}"}} {int(count)}'
            )

    for backend, count in sorted(
        (summary.get("persistent_backend_counts") or {}).items()
    ):
        lines.append(
            f'ai_engine_persistent_backend_calls{{backend="{backend}"}} {int(count)}'
        )

    lines.append(
        _metric(
            "persistent_fallback_total",
            int(summary.get("persistent_fallback_total", 0)),
        )
    )

    for error_type, count in sorted(
        (summary.get("persistent_error_counts") or {}).items()
    ):
        lines.append(
            "ai_engine_persistent_error_total"
            f'{{error_type="{error_type}"}} {int(count)}'
        )

    for correlation_id, count in sorted(
        (summary.get("correlation_id_counts") or {}).items()
    ):
        lines.append(
            "ai_engine_correlation_id_calls"
            f'{{correlation_id="{correlation_id}"}} {int(count)}'
        )

    for distribution_version, count in sorted(
        (summary.get("distribution_version_counts") or {}).items()
    ):
        lines.append(
            "ai_engine_distribution_version_calls"
            f'{{distribution_version="{distribution_version}"}} {int(count)}'
        )

    summary_distribution_version = str(summary.get("distribution_version", "")).strip()
    if summary_distribution_version:
        lines.append(
            "ai_engine_distribution_version_info"
            f'{{distribution_version="{summary_distribution_version}"}} 1'
        )

    cache_runtime = summary.get("cache_runtime") if isinstance(summary, dict) else None
    if isinstance(cache_runtime, dict):
        memory_entries = int(cache_runtime.get("memory_entries", 0) or 0)
        memory_capacity = int(cache_runtime.get("memory_max_entries", 0) or 0)
        persistent_entries = int(cache_runtime.get("persistent_entries", 0) or 0)
        saturation_ratio = float(
            cache_runtime.get("memory_saturation_ratio", 0.0) or 0.0
        )

        lines.append(_metric("cache_memory_entries", memory_entries))
        lines.append(_metric("cache_memory_capacity", memory_capacity))
        lines.append(_metric("cache_persistent_entries", persistent_entries))
        lines.append(_metric("cache_memory_saturation_ratio", saturation_ratio))

    # ------------------------------------------------------------------
    # LLMOps metrics (ADR 0009 step 3)
    # ------------------------------------------------------------------
    # Stable, ADR-mandated metric names for LLM runtime observability.
    # Token counts are character-derived approximations until the LLM
    # client surfaces real token usage on each call.

    avg_llm_latency_ms = float(summary.get("avg_llm_latency_ms", 0.0) or 0.0)
    p95_latency_ms = float(summary.get("p95_latency_ms", 0.0) or 0.0)
    failed_calls = int(summary.get("failed_calls", 0) or 0)
    persistent_fallback_total = int(summary.get("persistent_fallback_total", 0) or 0)
    total_prompt_chars = int(summary.get("total_prompt_chars", 0) or 0)
    total_response_chars = int(summary.get("total_response_chars", 0) or 0)

    # Approximate tokens as chars / 4 (industry rule of thumb for English/Spanish).
    approx_tokens_in = total_prompt_chars // 4
    approx_tokens_out = total_response_chars // 4

    lines.extend(
        [
            "# HELP ai_engine_llm_latency_seconds Average LLM call latency in seconds.",
            "# TYPE ai_engine_llm_latency_seconds gauge",
            _metric("llm_latency_seconds", round(avg_llm_latency_ms / 1000.0, 6)),
            "# HELP ai_engine_llm_latency_p95_seconds p95 LLM latency in seconds.",
            "# TYPE ai_engine_llm_latency_p95_seconds gauge",
            _metric("llm_latency_p95_seconds", round(p95_latency_ms / 1000.0, 6)),
            "# HELP ai_engine_llm_fallback_total Total fallback events across runtimes.",
            "# TYPE ai_engine_llm_fallback_total counter",
            _metric("llm_fallback_total", persistent_fallback_total),
            "# HELP ai_engine_llm_tokens_total Approximate token volume by direction.",
            "# TYPE ai_engine_llm_tokens_total counter",
            f'ai_engine_llm_tokens_total{{direction="in"}} {approx_tokens_in}',
            f'ai_engine_llm_tokens_total{{direction="out"}} {approx_tokens_out}',
            "# HELP ai_engine_llm_errors_total LLM call failures grouped by kind.",
            "# TYPE ai_engine_llm_errors_total counter",
            f'ai_engine_llm_errors_total{{kind="generation"}} {failed_calls}',
        ]
    )

    return "\n".join(lines) + "\n"
