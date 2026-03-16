"""Generation optimization service for high-throughput microservice consumers."""

from __future__ import annotations

import hashlib
import json
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

from ai_engine.api.schemas import GenerateRequest
from ai_engine.kbd.entry import KnowledgeEntry
from ai_engine.sdk.models import parse_generate_response

_TinyDBKnowledgeBase: Any
try:
    from ai_engine.kbd.tinydb_knowledge_base import (
        TinyDBKnowledgeBase as _TinyDBKnowledgeBase,
    )
except Exception:  # pragma: no cover
    _TinyDBKnowledgeBase = None


@dataclass(frozen=True)
class OptimizationResult:
    """Container with generated payload and optimization telemetry."""

    payload: dict[str, Any]
    sdk_payload: dict[str, Any]
    metrics: dict[str, Any]


class _LRUCache:
    """Thread-safe TTL + LRU in-memory cache."""

    def __init__(self, max_entries: int = 2048, ttl_seconds: int = 900) -> None:
        self._max_entries = max_entries
        self._ttl_seconds = ttl_seconds
        self._items: OrderedDict[str, tuple[float, dict[str, Any]]] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: str) -> dict[str, Any] | None:
        now = time.time()
        with self._lock:
            item = self._items.get(key)
            if item is None:
                return None
            expires_at, value = item
            if expires_at < now:
                del self._items[key]
                return None
            self._items.move_to_end(key)
            return value

    def set(self, key: str, value: dict[str, Any]) -> None:
        expires_at = time.time() + self._ttl_seconds
        with self._lock:
            self._items[key] = (expires_at, value)
            self._items.move_to_end(key)
            if len(self._items) > self._max_entries:
                self._items.popitem(last=False)

    def clear(self) -> None:
        """Remove all in-memory cache entries."""
        with self._lock:
            self._items.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)


class GenerationOptimizationService:
    """Optimized generation orchestrator with cache, KBD and fine telemetry.

    The service keeps API compatibility while adding:
    - Fast-path cache hits (in-memory + TinyDB-backed cache store)
    - Context quality boost via KBD keyword hits
    - Rich phase metrics for RAG/LLM/cache/database behavior
    """

    def __init__(
        self,
        generator: Any,
        rag_pipeline: Any,
        *,
        cache_max_entries: int = 2048,
        cache_ttl_seconds: int = 900,
        persistent_cache_path: str | None = None,
    ) -> None:
        self._generator = generator
        self._rag_pipeline = rag_pipeline
        self._memory_cache = (
            _LRUCache(
                max_entries=cache_max_entries,
                ttl_seconds=cache_ttl_seconds,
            )
            if cache_max_entries > 0
            else None
        )
        self._cache_ttl_seconds = cache_ttl_seconds

        cache_path = persistent_cache_path
        self._db_cache = (
            _TinyDBKnowledgeBase(path=cache_path)
            if _TinyDBKnowledgeBase is not None and cache_path is not None
            else None
        )

    def on_ingest(self, documents: list[Any]) -> None:
        """Index ingested documents into KBD/TinyDB for keyword-based boosting."""
        if self._db_cache is None:
            return

        for doc in documents:
            content = getattr(doc, "content", "")
            if not isinstance(content, str) or not content.strip():
                continue
            doc_id = str(
                getattr(doc, "doc_id", "")
                or hashlib.sha1(content.encode("utf-8")).hexdigest()
            )
            metadata = getattr(doc, "metadata", {}) or {}
            title = str(metadata.get("title", f"doc-{doc_id[:8]}"))
            tags = ["rag_source", "ingested_doc"]
            self._db_cache.add(
                KnowledgeEntry(
                    entry_id=f"rag-{doc_id}",
                    title=title,
                    content=content,
                    tags=tags,
                    metadata={"kind": "ingested_document", **dict(metadata)},
                )
            )

    def generate(self, req: GenerateRequest) -> OptimizationResult:
        """Generate content with cache-aware strategy and SDK conversion."""
        started = time.perf_counter()
        metrics: dict[str, Any] = {
            "event_type": "generation",
            "cache_hit": False,
            "cache_layer": "none",
            "cache_reads": 0,
            "cache_writes": 0,
            "db_reads": 0,
            "db_writes": 0,
            "rag_latency_ms": 0.0,
            "llm_latency_ms": 0.0,
            "parse_latency_ms": 0.0,
            "generation_latency_ms": 0.0,
            "total_latency_ms": 0.0,
            "retry_used": False,
            "kbd_hits": 0,
            "language": req.language,
            "game_type": req.game_type,
            "rag_docs_retrieved": 0,
            "orchestration_engine": "native-ai-engine",
        }

        key = self._cache_key(req)

        if req.use_cache:
            metrics["cache_reads"] = 1
            cached = (
                self._memory_cache.get(key) if self._memory_cache is not None else None
            )
            if cached is not None:
                metrics["cache_hit"] = True
                metrics["cache_layer"] = "memory"
                metrics["total_latency_ms"] = round(
                    (time.perf_counter() - started) * 1000, 2
                )
                return OptimizationResult(
                    payload=cached["payload"],
                    sdk_payload=cached["sdk_payload"],
                    metrics=metrics,
                )

            metrics["cache_reads"] = 2
            db_cached = self._read_persistent_cache(key)
            metrics["db_reads"] += 1
            if db_cached is not None:
                if self._memory_cache is not None:
                    self._memory_cache.set(key, db_cached)
                metrics["cache_hit"] = True
                metrics["cache_layer"] = "tinydb"
                metrics["total_latency_ms"] = round(
                    (time.perf_counter() - started) * 1000, 2
                )
                return OptimizationResult(
                    payload=db_cached["payload"],
                    sdk_payload=db_cached["sdk_payload"],
                    metrics=metrics,
                )

        rag_start = time.perf_counter()
        kbd_hits = self._db_cache.search_by_keyword(req.query) if self._db_cache else []
        metrics["kbd_hits"] = len(kbd_hits)
        if kbd_hits:
            req = req.model_copy(update={"query": f"{req.query} {kbd_hits[0].title}"})

        docs = self._rag_pipeline.retrieve(req.query)
        metrics["rag_docs_retrieved"] = len(docs)
        context = "\n\n".join(doc.content for doc in docs)
        metrics["rag_latency_ms"] = round((time.perf_counter() - rag_start) * 1000, 2)

        gen_start = time.perf_counter()
        if hasattr(self._generator, "generate_from_context"):
            envelope = self._generator.generate_from_context(
                context=context,
                topic=req.topic,
                game_type=req.game_type,
                language=req.language,
                num_questions=req.num_questions,
                letters=req.letters,
                max_tokens=req.max_tokens,
            )
        else:
            # Backward compatibility for wrapped/mocked generators that still expose only generate(...).
            envelope = self._generator.generate(
                query=req.query,
                topic=req.topic,
                game_type=req.game_type,
                language=req.language,
                num_questions=req.num_questions,
                letters=req.letters,
                max_tokens=req.max_tokens,
            )
        metrics["generation_latency_ms"] = round(
            (time.perf_counter() - gen_start) * 1000, 2
        )

        run_metrics = getattr(self._generator, "last_run_metrics", {})
        metrics["llm_latency_ms"] = float(run_metrics.get("llm_total_ms", 0.0))
        metrics["parse_latency_ms"] = float(run_metrics.get("parse_total_ms", 0.0))
        metrics["retry_used"] = bool(run_metrics.get("retry_used", False))

        payload = {"game_type": envelope.game_type, "game": envelope.game.to_dict()}
        sdk_payload = parse_generate_response(
            payload, language=req.language
        ).model_dump()

        cached_value = {
            "payload": payload,
            "sdk_payload": sdk_payload,
            "expires_at": time.time() + self._cache_ttl_seconds,
        }
        if req.use_cache:
            if self._memory_cache is not None:
                self._memory_cache.set(key, cached_value)
            metrics["cache_writes"] = 1
            self._write_persistent_cache(key, cached_value)
            if self._db_cache is not None:
                metrics["db_writes"] = 1

        metrics["total_latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
        return OptimizationResult(
            payload=payload, sdk_payload=sdk_payload, metrics=metrics
        )

    def cache_stats(self) -> dict[str, Any]:
        """Return runtime cache statistics for observability."""
        memory_entries = (
            len(self._memory_cache) if self._memory_cache is not None else 0
        )
        persistent_entries = 0
        if self._db_cache is not None:
            persistent_entries = sum(
                1
                for entry in self._db_cache.list_all()
                if entry.metadata.get("kind") == "generation_cache"
            )
        return {
            "memory_entries": memory_entries,
            "persistent_entries": persistent_entries,
            "memory_enabled": self._memory_cache is not None,
            "persistent_enabled": self._db_cache is not None,
            "cache_ttl_seconds": self._cache_ttl_seconds,
        }

    def reset_cache(self) -> dict[str, int]:
        """Clear in-memory and persistent generation cache entries."""
        removed_memory = 0
        removed_persistent = 0

        if self._memory_cache is not None:
            removed_memory = len(self._memory_cache)
            self._memory_cache.clear()

        if self._db_cache is not None:
            for entry in self._db_cache.list_all():
                if entry.metadata.get("kind") == "generation_cache":
                    self._db_cache.delete(entry.entry_id)
                    removed_persistent += 1

        return {
            "removed_memory": removed_memory,
            "removed_persistent": removed_persistent,
        }

    def _cache_key(self, req: GenerateRequest) -> str:
        """Build a deterministic cache key from semantically relevant fields."""
        raw = json.dumps(
            {
                "query": req.query.strip().lower(),
                "topic": req.topic.strip().lower(),
                "game_type": req.game_type,
                "language": req.language,
                "num_questions": req.num_questions,
                "letters": req.letters,
            },
            sort_keys=True,
            ensure_ascii=True,
        )
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _read_persistent_cache(self, key: str) -> dict[str, Any] | None:
        if self._db_cache is None:
            return None
        entry = self._db_cache.get(f"cache-{key}")
        if entry is None:
            return None

        try:
            parsed = json.loads(entry.content)
        except json.JSONDecodeError:
            return None

        expires_at = float(parsed.get("expires_at", 0.0))
        if expires_at and expires_at < time.time():
            return None
        if not isinstance(parsed, dict):
            return None
        if "payload" not in parsed or "sdk_payload" not in parsed:
            return None
        return parsed

    def _write_persistent_cache(self, key: str, value: dict[str, Any]) -> None:
        if self._db_cache is None:
            return
        entry = KnowledgeEntry(
            entry_id=f"cache-{key}",
            title="generated-game-cache",
            content=json.dumps(value, ensure_ascii=True),
            tags=[
                "cache",
                "generate",
                str(value.get("payload", {}).get("game_type", "unknown")),
            ],
            metadata={"kind": "generation_cache"},
        )
        self._db_cache.add(entry)
