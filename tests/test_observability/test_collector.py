"""Tests for the observability stats collector."""

from __future__ import annotations

import threading

from ai_engine.observability.collector import (
    GenerationEvent,
    StatsCollector,
    _percentile,
)

# ------------------------------------------------------------------
# GenerationEvent
# ------------------------------------------------------------------


class TestGenerationEvent:
    """Tests for the GenerationEvent dataclass."""

    def test_create_event(self) -> None:
        """An event can be created with required fields."""
        evt = GenerationEvent(
            timestamp=1_000_000.0,
            prompt_chars=100,
            response_chars=50,
            latency_ms=120.5,
            max_tokens=256,
            json_mode=False,
            success=True,
        )
        assert evt.prompt_chars == 100
        assert evt.success is True
        assert evt.game_type is None

    def test_to_dict(self) -> None:
        """to_dict returns a plain dictionary with all fields."""
        evt = GenerationEvent(
            timestamp=1.0,
            prompt_chars=10,
            response_chars=5,
            latency_ms=1.0,
            max_tokens=64,
            json_mode=True,
            success=True,
            game_type="quiz",
        )
        d = evt.to_dict()
        assert isinstance(d, dict)
        assert d["game_type"] == "quiz"
        assert d["json_mode"] is True

    def test_event_is_immutable(self) -> None:
        """Events are frozen dataclasses."""
        evt = GenerationEvent(
            timestamp=1.0,
            prompt_chars=10,
            response_chars=5,
            latency_ms=1.0,
            max_tokens=64,
            json_mode=False,
            success=True,
        )
        try:
            evt.success = False  # type: ignore[misc]
            assert False, "Should have raised"
        except AttributeError:
            pass


# ------------------------------------------------------------------
# StatsCollector
# ------------------------------------------------------------------


class TestStatsCollector:
    """Tests for the StatsCollector class."""

    def test_empty_summary(self) -> None:
        """Summary on an empty collector returns zeros."""
        c = StatsCollector()
        s = c.summary()
        assert s["total_calls"] == 0
        assert s["success_rate"] == 0.0

    def test_record_and_summary(self) -> None:
        """Record an event and verify the summary."""
        c = StatsCollector()
        c.record_call(
            prompt="hello",
            response="world",
            latency_ms=50.0,
            max_tokens=128,
            json_mode=True,
            success=True,
            game_type="quiz",
        )
        s = c.summary()
        assert s["total_calls"] == 1
        assert s["successful_calls"] == 1
        assert s["json_mode_calls"] == 1
        assert s["game_type_counts"] == {"quiz": 1}
        assert s["avg_latency_ms"] == 50.0

    def test_record_call_returns_event(self) -> None:
        """record_call returns the created event."""
        c = StatsCollector()
        evt = c.record_call(
            prompt="p",
            response="r",
            latency_ms=10.0,
            max_tokens=64,
        )
        assert isinstance(evt, GenerationEvent)
        assert evt.prompt_chars == 1

    def test_history(self) -> None:
        """History returns events in chronological order."""
        c = StatsCollector()
        for i in range(5):
            c.record_call(
                prompt=f"p{i}",
                response=f"r{i}",
                latency_ms=float(i),
                max_tokens=64,
            )
        h = c.history()
        assert len(h) == 5
        assert h[0]["latency_ms"] == 0.0

    def test_history_last_n(self) -> None:
        """History with last_n returns only recent events."""
        c = StatsCollector()
        for i in range(10):
            c.record_call(prompt="p", response="r", latency_ms=float(i), max_tokens=64)
        h = c.history(last_n=3)
        assert len(h) == 3
        assert h[0]["latency_ms"] == 7.0

    def test_max_history_eviction(self) -> None:
        """Collector evicts oldest events when max_history is exceeded."""
        c = StatsCollector(max_history=5)
        for i in range(10):
            c.record_call(prompt="p", response="r", latency_ms=float(i), max_tokens=64)
        assert len(c) == 5
        h = c.history()
        assert h[0]["latency_ms"] == 5.0

    def test_reset(self) -> None:
        """reset() clears all events."""
        c = StatsCollector()
        c.record_call(prompt="p", response="r", latency_ms=1.0, max_tokens=64)
        assert len(c) == 1
        c.reset()
        assert len(c) == 0

    def test_success_rate(self) -> None:
        """Success rate is computed correctly."""
        c = StatsCollector()
        c.record_call(
            prompt="p", response="r", latency_ms=1.0, max_tokens=64, success=True
        )
        c.record_call(
            prompt="p",
            response="r",
            latency_ms=1.0,
            max_tokens=64,
            success=False,
            error="boom",
        )
        s = c.summary()
        assert s["success_rate"] == 0.5
        assert s["failed_calls"] == 1

    def test_percentile_latencies(self) -> None:
        """Percentiles are computed for latency distribution."""
        c = StatsCollector()
        for i in range(100):
            c.record_call(prompt="p", response="r", latency_ms=float(i), max_tokens=64)
        s = c.summary()
        assert s["p50_latency_ms"] >= 49.0
        assert s["p95_latency_ms"] >= 94.0
        assert s["max_latency_ms"] == 99.0

    def test_summary_includes_cache_and_phase_metrics(self) -> None:
        """Advanced metadata fields are aggregated in summary output."""
        c = StatsCollector()
        c.record_call(
            prompt="p1",
            response="r1",
            latency_ms=10.0,
            max_tokens=64,
            metadata={
                "cache_hit": True,
                "cache_layer": "memory",
                "rag_latency_ms": 3.0,
                "llm_latency_ms": 5.0,
                "parse_latency_ms": 1.0,
                "total_latency_ms": 10.0,
                "kbd_hits": 2,
                "db_reads": 1,
                "db_writes": 0,
                "language": "en",
            },
        )
        c.record_call(
            prompt="p2",
            response="r2",
            latency_ms=20.0,
            max_tokens=64,
            metadata={
                "cache_hit": False,
                "cache_layer": "none",
                "rag_latency_ms": 6.0,
                "llm_latency_ms": 10.0,
                "parse_latency_ms": 2.0,
                "total_latency_ms": 20.0,
                "kbd_hits": 1,
                "db_reads": 2,
                "db_writes": 1,
                "language": "es",
            },
        )

        s = c.summary()
        assert s["cache_hits"] == 1
        assert s["cache_misses"] == 1
        assert s["cache_hit_rate"] == 0.5
        assert s["cache_layer_counts"]["memory"] == 1
        assert s["avg_rag_latency_ms"] == 4.5
        assert s["avg_llm_latency_ms"] == 7.5
        assert s["avg_parse_latency_ms"] == 1.5
        assert s["avg_total_latency_ms"] == 15.0
        assert s["kbd_hits_total"] == 3
        assert s["db_reads_total"] == 3
        assert s["db_writes_total"] == 1
        assert s["language_counts"]["en"] == 1
        assert s["language_counts"]["es"] == 1

    def test_cache_metrics_ignore_non_generation_events(self) -> None:
        """cache_hit_rate only considers generation-classified events."""
        c = StatsCollector()
        c.record_call(
            prompt="gen-hit",
            response="ok",
            latency_ms=10.0,
            max_tokens=64,
            metadata={
                "event_type": "generation",
                "cache_hit": True,
                "cache_layer": "memory",
            },
        )
        c.record_call(
            prompt="gen-miss",
            response="ok",
            latency_ms=10.0,
            max_tokens=64,
            metadata={
                "event_type": "generation",
                "cache_hit": False,
                "cache_layer": "none",
            },
        )
        c.record_call(
            prompt="ingest",
            response="ingested=2",
            latency_ms=5.0,
            max_tokens=0,
            metadata={
                "event_type": "ingest",
                "cache_hit": False,
                "cache_layer": "none",
            },
            game_type="ingest",
            json_mode=False,
        )

        s = c.summary()
        assert s["event_type_counts"]["generation"] == 2
        assert s["event_type_counts"]["ingest"] == 1
        assert s["cache_hits"] == 1
        assert s["cache_misses"] == 1
        assert s["cache_hit_rate"] == 0.5

    def test_thread_safety(self) -> None:
        """Concurrent recording does not lose events."""
        c = StatsCollector()
        n_threads = 4
        n_per_thread = 250

        def worker() -> None:
            for _ in range(n_per_thread):
                c.record_call(prompt="p", response="r", latency_ms=1.0, max_tokens=64)

        threads = [threading.Thread(target=worker) for _ in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(c) == n_threads * n_per_thread


# ------------------------------------------------------------------
# _percentile helper
# ------------------------------------------------------------------


class TestPercentileHelper:
    """Tests for the _percentile utility function."""

    def test_empty(self) -> None:
        """Returns 0 for empty data."""
        assert _percentile([], 50) == 0.0

    def test_single_value(self) -> None:
        """Single value is always the percentile."""
        assert _percentile([42.0], 99) == 42.0

    def test_linear_interpolation(self) -> None:
        """Interpolation between two values."""
        data = [0.0, 10.0]
        assert _percentile(data, 50) == 5.0
