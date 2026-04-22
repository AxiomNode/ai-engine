"""Tests for ADR 0009 LLMOps Prometheus metrics emitted by summary_to_prometheus."""

from __future__ import annotations

from ai_engine.observability.collector import (
    StatsCollector,
    summary_to_prometheus,
)


def _build_collector() -> StatsCollector:
    collector = StatsCollector()
    collector.record_call(
        prompt="hello world prompt",
        response="hello world response",
        latency_ms=120.0,
        max_tokens=128,
        success=True,
        game_type="quiz",
        metadata={"event_type": "generation", "llm_latency_ms": 100.0},
    )
    collector.record_call(
        prompt="another prompt",
        response="",
        latency_ms=80.0,
        max_tokens=128,
        success=False,
        game_type="quiz",
        error="boom",
        metadata={"event_type": "generation", "llm_latency_ms": 80.0},
    )
    return collector


def test_llmops_metrics_present() -> None:
    summary = _build_collector().summary()
    summary["persistent_fallback_total"] = 2
    text = summary_to_prometheus(summary)

    assert "ai_engine_llm_latency_seconds " in text
    assert "ai_engine_llm_latency_p95_seconds " in text
    assert "ai_engine_llm_fallback_total 2" in text
    assert 'ai_engine_llm_tokens_total{direction="in"}' in text
    assert 'ai_engine_llm_tokens_total{direction="out"}' in text
    assert 'ai_engine_llm_errors_total{kind="generation"} 1' in text


def test_llmops_metrics_zero_when_empty() -> None:
    text = summary_to_prometheus(StatsCollector().summary())

    assert "ai_engine_llm_latency_seconds 0" in text
    assert "ai_engine_llm_fallback_total 0" in text
    assert 'ai_engine_llm_tokens_total{direction="in"} 0' in text
    assert 'ai_engine_llm_tokens_total{direction="out"} 0' in text
    assert 'ai_engine_llm_errors_total{kind="generation"} 0' in text
