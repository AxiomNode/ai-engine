"""Micro-benchmark for generation latency paths.

Runs deterministic, local-only benchmarks for the optimization service:
- Cache miss path
- Memory cache hit path
- Persistent fallback write-error path

Usage:
    python scripts/benchmark_generation_paths.py
"""

from __future__ import annotations

import json
import statistics
import time
from dataclasses import dataclass
from fnmatch import fnmatch
from pathlib import Path
from typing import Any

import ai_engine.api.optimization as optimization_module
from ai_engine.api.optimization import GenerationOptimizationService
from ai_engine.api.schemas import GenerateRequest
from ai_engine.games.schemas import GameEnvelope, QuizGame, QuizQuestion


@dataclass
class _Doc:
    content: str


class _StubRAGPipeline:
    """Deterministic retrieval pipeline for benchmarks."""

    def retrieve(self, query: str) -> list[_Doc]:
        _ = query
        return [_Doc(content="Benchmark context")]


class _StubGenerator:
    """Deterministic generator for stable timing runs."""

    def __init__(self) -> None:
        self.last_run_metrics = {
            "llm_total_ms": 2.0,
            "parse_total_ms": 0.5,
            "retry_used": False,
        }

    def generate_from_context(
        self,
        context: str,
        topic: str,
        game_type: str = "quiz",
        *,
        language: str | None = None,
        num_questions: int = 5,
        letters: str = "A,B,C",
        max_tokens: int | None = None,
    ) -> GameEnvelope:
        _ = (context, topic, game_type, language, num_questions, letters, max_tokens)
        return GameEnvelope(
            game_type="quiz",
            game=QuizGame(
                title="Benchmark Quiz",
                topic="Science",
                questions=[
                    QuizQuestion(
                        question="What is H2O?",
                        options=["Water", "Fire", "Air", "Earth"],
                        correct_index=0,
                        explanation="H2O is water.",
                    )
                ],
            ),
        )


class _FakeRedis:
    """Simple in-memory Redis fake used to trigger fallback errors."""

    fail_setex = False

    def __init__(self) -> None:
        self.kv: dict[str, str] = {}
        self.sets: dict[str, set[str]] = {}

    @classmethod
    def from_url(cls, url: str, decode_responses: bool = True) -> _FakeRedis:
        _ = (url, decode_responses)
        return cls()

    def ping(self) -> bool:
        return True

    def setex(self, key: str, ttl: int, value: str) -> None:
        _ = ttl
        if self.fail_setex:
            raise RuntimeError("setex failed")
        self.kv[key] = value

    def get(self, key: str) -> str | None:
        return self.kv.get(key)

    def sadd(self, key: str, member: str) -> None:
        self.sets.setdefault(key, set()).add(member)

    def smembers(self, key: str) -> set[str]:
        return set(self.sets.get(key, set()))

    def scard(self, key: str) -> int:
        return len(self.sets.get(key, set()))

    def scan_iter(self, match: str) -> list[str]:
        return [key for key in self.sets if fnmatch(key, match)]

    def delete(self, *keys: str) -> int:
        removed = 0
        for key in keys:
            if key in self.kv:
                del self.kv[key]
                removed += 1
            if key in self.sets:
                del self.sets[key]
                removed += 1
        return removed


def _run_times(fn: Any, runs: int = 30) -> list[float]:
    values: list[float] = []
    for _ in range(runs):
        start = time.perf_counter()
        fn()
        values.append((time.perf_counter() - start) * 1000)
    return values


def _summarize(values: list[float]) -> dict[str, float]:
    sorted_vals = sorted(values)
    p95_index = int(len(sorted_vals) * 0.95) - 1
    p95_index = max(0, min(p95_index, len(sorted_vals) - 1))
    return {
        "avg_ms": round(statistics.mean(values), 3),
        "p95_ms": round(sorted_vals[p95_index], 3),
        "max_ms": round(max(values), 3),
    }


def main() -> int:
    req = GenerateRequest(query="water", topic="Science")

    # Scenario 1: cache miss
    miss_service = GenerationOptimizationService(
        generator=_StubGenerator(),
        rag_pipeline=_StubRAGPipeline(),
        cache_max_entries=256,
        persistent_cache_path=None,
    )

    miss_values = _run_times(
        lambda: miss_service.generate(
            req.model_copy(update={"query": f"water-{time.time_ns()}"})
        )
    )

    # Scenario 2: memory hit
    hit_service = GenerationOptimizationService(
        generator=_StubGenerator(),
        rag_pipeline=_StubRAGPipeline(),
        cache_max_entries=256,
        persistent_cache_path=None,
    )
    hit_service.generate(req)
    hit_values = _run_times(lambda: hit_service.generate(req))

    # Scenario 3: persistent fallback on write error (redis)
    original_redis_client = optimization_module._RedisClient
    optimization_module._RedisClient = _FakeRedis
    try:
        fallback_service = GenerationOptimizationService(
            generator=_StubGenerator(),
            rag_pipeline=_StubRAGPipeline(),
            cache_max_entries=0,
            cache_backend="redis",
            redis_url="redis://local/0",
            persistent_cache_path=None,
        )
        assert fallback_service._redis_cache is not None
        fallback_service._redis_cache.fail_setex = True
        fallback_values = _run_times(lambda: fallback_service.generate(req))
    finally:
        optimization_module._RedisClient = original_redis_client

    report = {
        "cache_miss": _summarize(miss_values),
        "memory_hit": _summarize(hit_values),
        "fallback_write_error": _summarize(fallback_values),
        "runs": len(miss_values),
    }

    print(json.dumps(report, indent=2))

    out_path = Path("docs") / "generation-benchmark-baseline.json"
    out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Baseline written to: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
