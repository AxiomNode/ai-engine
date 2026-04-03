"""Tests for generation optimization service internals."""

from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass
from fnmatch import fnmatch

from ai_engine.api.optimization import GenerationOptimizationService
from ai_engine.api.schemas import GenerateRequest
from ai_engine.games.schemas import GameEnvelope, QuizGame, QuizQuestion


def _run(coro):
    """Run a coroutine synchronously for test convenience."""
    return asyncio.run(coro)


@dataclass
class _Doc:
    """Simple retrieval document stub."""

    content: str


class _StubRAGPipeline:
    """Minimal RAG stub for optimization tests."""

    def retrieve(self, query: str) -> list[_Doc]:
        _ = query
        return [_Doc(content="Water cycle context")]


class _StubGenerator:
    """Minimal game generator stub exposing generate_from_context."""

    def __init__(self) -> None:
        self.last_run_metrics = {
            "llm_total_ms": 5.0,
            "parse_total_ms": 1.0,
            "retry_used": False,
        }

    async def generate_from_context(
        self,
        context: str,
        game_type: str = "quiz",
        *,
        language: str | None = None,
        difficulty_percentage: int | None = None,
        num_questions: int = 5,
        letters: str = "A,B,C",
        max_tokens: int | None = None,
    ) -> GameEnvelope:
        _ = (
            context,
            game_type,
            language,
            difficulty_percentage,
            num_questions,
            letters,
            max_tokens,
        )
        return GameEnvelope(
            game_type="quiz",
            game=QuizGame(
                title="Test Quiz",
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
    """In-memory Redis test double supporting required operations."""

    fail_get = False
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
        if self.fail_get:
            raise RuntimeError("get failed")
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


def test_persistent_cache_stats_and_reset_use_cache_index(tmp_path) -> None:
    """Persistent cache stats/reset should track only cache entries efficiently."""
    cache_file = tmp_path / "generation_cache.json"
    service = GenerationOptimizationService(
        generator=_StubGenerator(),
        rag_pipeline=_StubRAGPipeline(),
        cache_max_entries=0,
        persistent_cache_path=str(cache_file),
    )

    req = GenerateRequest(query="water")
    _run(service.generate(req))

    stats_before = service.cache_stats()
    assert stats_before["persistent_enabled"] is True
    assert stats_before["persistent_entries"] == 1

    removed = service.reset_cache()
    assert removed["removed_persistent"] == 1

    stats_after = service.cache_stats()
    assert stats_after["persistent_entries"] == 0


def test_redis_backend_falls_back_to_tinydb_when_unavailable(tmp_path) -> None:
    """Redis backend selection should gracefully fallback to TinyDB."""
    cache_file = tmp_path / "generation_cache.json"
    service = GenerationOptimizationService(
        generator=_StubGenerator(),
        rag_pipeline=_StubRAGPipeline(),
        cache_max_entries=0,
        cache_backend="redis",
        redis_url=None,
        persistent_cache_path=str(cache_file),
    )

    req = GenerateRequest(query="water")
    _run(service.generate(req))

    stats = service.cache_stats()
    assert stats["persistent_enabled"] is True
    assert stats["persistent_backend"] == "tinydb"
    assert stats["persistent_entries"] == 1


def test_redis_backend_read_write_reset_path(monkeypatch) -> None:
    """Redis backend should support persistent cache hits and reset operations."""
    import ai_engine.api.optimization as optimization_module

    monkeypatch.setattr(optimization_module, "_RedisClient", _FakeRedis)

    service = GenerationOptimizationService(
        generator=_StubGenerator(),
        rag_pipeline=_StubRAGPipeline(),
        cache_max_entries=0,
        cache_backend="redis",
        cache_namespace="v1",
        redis_url="redis://local/0",
        persistent_cache_path=None,
    )

    req = GenerateRequest(query="water")
    first = _run(service.generate(req))
    second = _run(service.generate(req))

    assert first.metrics["cache_hit"] is False
    assert second.metrics["cache_hit"] is True
    assert second.metrics["cache_layer"] == "redis"

    stats = service.cache_stats()
    assert stats["persistent_backend"] == "redis"
    assert stats["persistent_entries"] == 1

    removed = service.reset_cache(namespace="v1")
    assert removed["removed_persistent"] == 1
    assert service.cache_stats()["persistent_entries"] == 0


def test_cache_namespace_versioning_and_selective_invalidation(tmp_path) -> None:
    """Namespace versioning should isolate keys and allow selective invalidation."""
    cache_file = tmp_path / "generation_cache.json"

    v1 = GenerationOptimizationService(
        generator=_StubGenerator(),
        rag_pipeline=_StubRAGPipeline(),
        cache_max_entries=0,
        cache_namespace="v1",
        persistent_cache_path=str(cache_file),
    )
    v2 = GenerationOptimizationService(
        generator=_StubGenerator(),
        rag_pipeline=_StubRAGPipeline(),
        cache_max_entries=0,
        cache_namespace="v2",
        persistent_cache_path=str(cache_file),
    )

    req = GenerateRequest(query="water")
    first_v1 = _run(v1.generate(req))
    first_v2 = _run(v2.generate(req))

    assert first_v1.metrics["cache_hit"] is False
    assert first_v2.metrics["cache_hit"] is False

    removed_v1 = v2.reset_cache(namespace="v1")
    assert removed_v1["removed_persistent"] == 1
    assert v2.cache_stats()["persistent_entries"] == 1

    removed_v2 = v2.reset_cache(namespace="v2")
    assert removed_v2["removed_persistent"] == 1
    assert v2.cache_stats()["persistent_entries"] == 0


def test_persistent_backend_failures_do_not_break_generation(monkeypatch) -> None:
    """Persistent backend failures should fallback and still return generation."""
    import ai_engine.api.optimization as optimization_module

    monkeypatch.setattr(optimization_module, "_RedisClient", _FakeRedis)

    service = GenerationOptimizationService(
        generator=_StubGenerator(),
        rag_pipeline=_StubRAGPipeline(),
        cache_max_entries=0,
        cache_backend="redis",
        cache_namespace="v1",
        redis_url="redis://local/0",
        persistent_cache_path=None,
    )

    req = GenerateRequest(query="water")
    service._redis_cache.fail_setex = True
    result = _run(service.generate(req))

    assert result.payload["game_type"] == "quiz"
    assert result.metrics["persistent_fallback_used"] is True
    assert result.metrics["persistent_error"] == "write_error"

    service._redis_cache.fail_setex = False
    _run(service.generate(req))
    service._redis_cache.fail_get = True
    read_fail_result = _run(service.generate(req))

    assert read_fail_result.payload["game_type"] == "quiz"
    assert read_fail_result.metrics["persistent_fallback_used"] is True
    assert read_fail_result.metrics["persistent_error"] == "read_error"


def test_persistent_index_stays_consistent_under_concurrency(tmp_path) -> None:
    """Concurrent generate/reset operations should not drift persistent index."""
    cache_file = tmp_path / "generation_cache.json"
    service = GenerationOptimizationService(
        generator=_StubGenerator(),
        rag_pipeline=_StubRAGPipeline(),
        cache_max_entries=0,
        cache_namespace="v1",
        persistent_cache_path=str(cache_file),
    )

    def worker() -> None:
        loop = asyncio.new_event_loop()
        for _ in range(25):
            loop.run_until_complete(service.generate(GenerateRequest(query="water")))
        loop.close()

    threads = [threading.Thread(target=worker) for _ in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    stats_before = service.cache_stats()
    assert stats_before["persistent_entries"] >= 1

    removed = service.reset_cache(all_namespaces=True)
    assert removed["removed_persistent"] >= 1

    stats_after = service.cache_stats()
    assert stats_after["persistent_entries"] == 0
