"""Tests for generation optimization service internals."""

from __future__ import annotations

import asyncio
import json
import threading
from dataclasses import dataclass
from fnmatch import fnmatch

import pytest

from ai_engine.api.optimization import GenerationOptimizationService, _LRUCache
from ai_engine.api.schemas import GenerateRequest
from ai_engine.games.catalog import get_game_type_profile
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

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def retrieve(self, query: str, **kwargs) -> list[_Doc]:
        self.calls.append({"query": query, **kwargs})
        return [_Doc(content="Water cycle context")]

    def _format_context(self, docs: list[_Doc]) -> str:
        return "\n\n".join(doc.content for doc in docs)


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
        topic: str | None = None,
        language: str | None = None,
        difficulty_percentage: int | None = None,
        num_questions: int = 5,
        letters: str = "A,B,C",
        max_tokens: int | None = None,
    ) -> GameEnvelope:
        _ = (
            context,
            game_type,
            topic,
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


class _LegacyGenerator:
    def __init__(self) -> None:
        self.last_run_metrics = {
            "llm_total_ms": 7.0,
            "parse_total_ms": 2.0,
            "retry_used": True,
        }
        self.calls: list[dict[str, object]] = []

    async def generate(self, **kwargs) -> GameEnvelope:
        self.calls.append(kwargs)
        return GameEnvelope(
            game_type="quiz",
            game=QuizGame(
                title="Legacy Quiz",
                questions=[
                    QuizQuestion(
                        question="Legacy question",
                        options=["A", "B", "C", "D"],
                        correct_index=0,
                        explanation="Legacy path.",
                    )
                ],
            ),
        )


class _FailingGenerator:
    async def generate_from_context(self, **kwargs) -> GameEnvelope:
        _ = kwargs
        raise RuntimeError("generation failed")


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


def test_lru_cache_evicts_oldest_and_expires_entries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache = _LRUCache(max_entries=1, ttl_seconds=10)
    cache.set("first", {"value": 1})
    cache.set("second", {"value": 2})

    assert cache.get("first") is None
    assert cache.get("second") == {"value": 2}

    monkeypatch.setattr("ai_engine.api.optimization.time.time", lambda: 1_000.0)
    cache = _LRUCache(max_entries=2, ttl_seconds=5)
    cache.set("expiring", {"value": 3})
    monkeypatch.setattr("ai_engine.api.optimization.time.time", lambda: 1_010.0)

    assert cache.get("expiring") is None


def test_generate_falls_back_to_legacy_generator_without_generate_from_context() -> (
    None
):
    generator = _LegacyGenerator()
    service = GenerationOptimizationService(
        generator=generator,
        rag_pipeline=_StubRAGPipeline(),
        cache_max_entries=0,
        persistent_cache_path=None,
    )

    result = _run(service.generate(GenerateRequest(query="water", max_tokens=300)))

    assert result.payload["game_type"] == "quiz"
    assert generator.calls[0]["query"] == "water"
    assert generator.calls[0]["max_tokens"] == 300
    assert result.metrics["retry_used"] is True


def test_generate_attaches_metrics_when_generator_raises() -> None:
    service = GenerationOptimizationService(
        generator=_FailingGenerator(),
        rag_pipeline=_StubRAGPipeline(),
        cache_max_entries=0,
        persistent_cache_path=None,
    )

    with pytest.raises(RuntimeError, match="generation failed") as exc_info:
        _run(service.generate(GenerateRequest(query="water")))

    assert exc_info.value.generation_metrics["rag_docs_retrieved"] == 1
    assert exc_info.value.generation_metrics["total_latency_ms"] >= 0


def test_on_ingest_skips_blank_documents_and_uses_fallback_title() -> None:
    service = GenerationOptimizationService(
        generator=_StubGenerator(),
        rag_pipeline=_StubRAGPipeline(),
        cache_max_entries=0,
        persistent_cache_path=None,
    )

    service.on_ingest(
        [
            type("Doc", (), {"content": "   ", "doc_id": "blank", "metadata": {}})(),
            type(
                "Doc", (), {"content": "Useful context", "doc_id": "", "metadata": {}}
            )(),
        ]
    )

    hits = service._kbd_search_sync("Useful")

    assert len(hits) == 1
    assert hits[0].title.startswith("doc-")


def test_read_persistent_cache_handles_invalid_redis_payloads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ai_engine.api.optimization as optimization_module

    monkeypatch.setattr(optimization_module, "_RedisClient", _FakeRedis)
    service = GenerationOptimizationService(
        generator=_StubGenerator(),
        rag_pipeline=_StubRAGPipeline(),
        cache_max_entries=0,
        cache_backend="redis",
        redis_url="redis://local/0",
        persistent_cache_path=None,
    )
    key = service._redis_cache_key("v1", "abc")
    service._redis_cache.kv[key] = "not-json"
    assert service._read_persistent_cache("abc") is None

    service._redis_cache.kv[key] = "{}"
    assert service._read_persistent_cache("abc") is None


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


def test_cache_key_differs_by_category_and_content_fingerprint(tmp_path) -> None:
    cache_file = tmp_path / "generation_cache.json"
    rag = _StubRAGPipeline()
    service_a = GenerationOptimizationService(
        generator=_StubGenerator(),
        rag_pipeline=rag,
        cache_max_entries=0,
        embedding_model="embed-a",
        corpus_signature="corpus-a",
        persistent_cache_path=str(cache_file),
    )
    service_b = GenerationOptimizationService(
        generator=_StubGenerator(),
        rag_pipeline=rag,
        cache_max_entries=0,
        embedding_model="embed-b",
        corpus_signature="corpus-b",
        persistent_cache_path=str(cache_file),
    )

    req_base = GenerateRequest(
        query="water", category_id="17", category_name="Science & Nature"
    )
    req_other_category = GenerateRequest(
        query="water", category_id="23", category_name="History"
    )

    assert service_a._cache_key(req_base) != service_a._cache_key(req_other_category)
    assert service_a._cache_key(req_base) != service_b._cache_key(req_base)


def test_generate_passes_category_preferences_into_rag() -> None:
    rag = _StubRAGPipeline()
    service = GenerationOptimizationService(
        generator=_StubGenerator(),
        rag_pipeline=rag,
        cache_max_entries=0,
        embedding_model="embed-a",
        corpus_signature="corpus-a",
        persistent_cache_path=None,
    )

    result = _run(
        service.generate(
            GenerateRequest(
                query="water",
                language="es",
                game_type="quiz",
                category_id="17",
                category_name="Science & Nature",
            )
        )
    )

    assert result.metrics["category_id"] == "17"
    assert result.metrics["category_name"] == "Science & Nature"
    assert result.metrics["embedding_model"] == "embed-a"
    assert result.metrics["corpus_signature"] == "corpus-a"
    assert rag.calls


def test_generate_uses_category_name_when_query_is_missing() -> None:
    rag = _StubRAGPipeline()
    service = GenerationOptimizationService(
        generator=_StubGenerator(),
        rag_pipeline=rag,
        cache_max_entries=0,
        persistent_cache_path=None,
    )

    _run(
        service.generate(
            GenerateRequest(
                game_type="word-pass",
                category_id="17",
                category_name="Science & Nature",
                item_count=2,
            )
        )
    )

    assert rag.calls[0]["query"].startswith("Science & Nature")
    assert rag.calls[0]["metadata_preferences"] == {
        "language": "es",
        "game_type": "word-pass",
        "category": "Science & Nature",
    }
    assert "Science & Nature" in str(rag.calls[0]["query"])
    assert rag.calls[0]["top_k"] == get_game_type_profile("word-pass").retrieval_top_k


def test_on_ingest_populates_kbd_even_without_tinydb_backend() -> None:
    service = GenerationOptimizationService(
        generator=_StubGenerator(),
        rag_pipeline=_StubRAGPipeline(),
        cache_max_entries=0,
        persistent_cache_path=None,
    )

    service.on_ingest(
        [
            type(
                "Doc",
                (),
                {
                    "content": "The water cycle includes evaporation and condensation.",
                    "doc_id": "w1",
                    "metadata": {"title": "Water Cycle"},
                },
            )()
        ]
    )

    hits = service._kbd_search_sync("water cycle")
    assert len(hits) == 1
    assert hits[0].title == "Water Cycle"


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


def test_generate_falls_back_when_format_context_signature_or_type_is_incompatible() -> (
    None
):
    """Generation should fall back to joined document content when _format_context is incompatible."""

    class _RAGWithOddFormatter(_StubRAGPipeline):
        def _format_context(self, docs: list[_Doc]) -> dict[str, object]:
            return {"docs": len(docs)}

    class _CapturingGenerator(_StubGenerator):
        def __init__(self) -> None:
            super().__init__()
            self.last_context = ""

        async def generate_from_context(self, context: str, **kwargs) -> GameEnvelope:
            self.last_context = context
            return await super().generate_from_context(context=context, **kwargs)

    generator = _CapturingGenerator()
    service = GenerationOptimizationService(
        generator=generator,
        rag_pipeline=_RAGWithOddFormatter(),
        cache_max_entries=0,
        persistent_cache_path=None,
    )

    result = _run(service.generate(GenerateRequest(query="water")))

    assert result.payload["game_type"] == "quiz"
    assert generator.last_context == "Water cycle context"


def test_cache_stats_tracks_redis_stats_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    """Redis stats failures should be swallowed and reflected in backend error counters."""
    import ai_engine.api.optimization as optimization_module

    monkeypatch.setattr(optimization_module, "_RedisClient", _FakeRedis)
    service = GenerationOptimizationService(
        generator=_StubGenerator(),
        rag_pipeline=_StubRAGPipeline(),
        cache_max_entries=0,
        cache_backend="redis",
        redis_url="redis://local/0",
        persistent_cache_path=None,
    )

    def fail_scard(_: str) -> int:
        raise RuntimeError("stats failed")

    service._redis_cache.scard = fail_scard

    stats = service.cache_stats()

    assert stats["persistent_entries"] == 0
    assert stats["persistent_backend_errors"]["stats"] == 1


def test_read_persistent_cache_db_backend_handles_invalid_and_expired_entries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """TinyDB-backed reads should reject invalid JSON shapes and expired entries."""
    service = GenerationOptimizationService(
        generator=_StubGenerator(),
        rag_pipeline=_StubRAGPipeline(),
        cache_max_entries=0,
        persistent_cache_path=None,
    )

    class _FakeDBCache:
        def __init__(self, entries: list[object]) -> None:
            self.entries = entries

        def get(self, entry_id: str):
            _ = entry_id
            return self.entries.pop(0)

    invalid_entry = type("Entry", (), {"content": json.dumps({"expires_at": 0})})()
    expired_entry = type(
        "Entry",
        (),
        {"content": json.dumps({"payload": {}, "sdk_payload": {}, "expires_at": 1.0})},
    )()

    service._db_cache = _FakeDBCache([invalid_entry, expired_entry])
    monkeypatch.setattr("ai_engine.api.optimization.time.time", lambda: 5.0)

    assert service._read_persistent_cache("missing-fields") is None
    assert service._read_persistent_cache("expired") is None


def test_write_persistent_cache_without_backend_returns_false() -> None:
    """Persistent cache writes should no-op when neither Redis nor TinyDB is configured."""
    service = GenerationOptimizationService(
        generator=_StubGenerator(),
        rag_pipeline=_StubRAGPipeline(),
        cache_max_entries=0,
        persistent_cache_path=None,
    )

    assert (
        service._write_persistent_cache("k", {"payload": {}, "sdk_payload": {}})
        is False
    )


def test_bootstrap_persistent_cache_index_filters_non_cache_entries() -> None:
    """Bootstrap should index only cache-prefixed or generation_cache-marked entries."""
    service = GenerationOptimizationService(
        generator=_StubGenerator(),
        rag_pipeline=_StubRAGPipeline(),
        cache_max_entries=0,
        persistent_cache_path=None,
    )

    class _FakeDBCache:
        @staticmethod
        def list_all() -> list[object]:
            return [
                type("Entry", (), {"entry_id": "cache-v1-a", "metadata": {}})(),
                type(
                    "Entry",
                    (),
                    {
                        "entry_id": "other-entry",
                        "metadata": {"kind": "generation_cache"},
                    },
                )(),
                type(
                    "Entry", (), {"entry_id": "plain", "metadata": {"kind": "note"}}
                )(),
            ]

    service._db_cache = _FakeDBCache()
    service._bootstrap_persistent_cache_index()

    assert service._persistent_cache_ids == {"cache-v1-a", "other-entry"}
