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
_RedisClient: Any
try:
    from ai_engine.kbd.tinydb_knowledge_base import (
        TinyDBKnowledgeBase as _TinyDBKnowledgeBase,
    )
except Exception:  # pragma: no cover
    _TinyDBKnowledgeBase = None

try:
    from redis import Redis as _RedisClientClass

    _RedisClient = _RedisClientClass
except Exception:  # pragma: no cover
    _RedisClient = None


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

    @property
    def max_entries(self) -> int:
        """Return configured maximum number of in-memory entries."""
        return self._max_entries


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
        cache_backend: str = "tinydb",
        cache_namespace: str = "v1",
        distribution_version: str | None = None,
        persistent_cache_path: str | None = None,
        redis_url: str | None = None,
        redis_prefix: str = "ai-engine:generation-cache",
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
        self._cache_namespace = cache_namespace
        self._distribution_version = distribution_version
        self._redis_prefix = redis_prefix
        self._persistent_lock = threading.RLock()
        self._redis_cache = None
        self._db_cache = None
        self._persistent_backend = "none"
        self._persistent_backend_errors = {
            "read": 0,
            "write": 0,
            "delete": 0,
            "stats": 0,
        }

        if cache_backend == "redis" and _RedisClient is not None and redis_url:
            try:
                redis_client = _RedisClient.from_url(redis_url, decode_responses=True)
                redis_client.ping()
                self._redis_cache = redis_client
                self._persistent_backend = "redis"
            except Exception:
                self._redis_cache = None

        if self._redis_cache is None:
            cache_path = persistent_cache_path
            if _TinyDBKnowledgeBase is not None and cache_path is not None:
                self._db_cache = _TinyDBKnowledgeBase(path=cache_path)
                self._persistent_backend = "tinydb"

        self._persistent_cache_ids: set[str] = set()
        if self._db_cache is not None:
            self._bootstrap_persistent_cache_index()

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
            with self._persistent_lock:
                self._db_cache.add(
                    KnowledgeEntry(
                        entry_id=f"rag-{doc_id}",
                        title=title,
                        content=content,
                        tags=tags,
                        metadata={"kind": "ingested_document", **dict(metadata)},
                    )
                )

    def generate(
        self,
        req: GenerateRequest,
        *,
        correlation_id: str | None = None,
    ) -> OptimizationResult:
        """Generate content with cache-aware strategy and SDK conversion."""
        started = time.perf_counter()
        metrics: dict[str, Any] = {
            "event_type": "generation",
            "cache_namespace": self._cache_namespace,
            "cache_hit": False,
            "cache_layer": "none",
            "cache_reads": 0,
            "cache_writes": 0,
            "db_reads": 0,
            "db_writes": 0,
            "persistent_backend": self._persistent_backend,
            "persistent_fallback_used": False,
            "persistent_error": "",
            "rag_latency_ms": 0.0,
            "llm_latency_ms": 0.0,
            "parse_latency_ms": 0.0,
            "generation_latency_ms": 0.0,
            "total_latency_ms": 0.0,
            "retry_used": False,
            "kbd_hits": 0,
            "language": req.language,
            "game_type": req.game_type,
            "difficulty_percentage": req.difficulty_percentage,
            "rag_docs_retrieved": 0,
            "orchestration_engine": "native-ai-engine",
        }
        if self._distribution_version:
            metrics["distribution_version"] = self._distribution_version
        if correlation_id:
            metrics["correlation_id"] = correlation_id

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
            db_cached = self._read_persistent_cache(key, metrics=metrics)
            metrics["db_reads"] += 1
            if db_cached is not None:
                if self._memory_cache is not None:
                    self._memory_cache.set(key, db_cached)
                metrics["cache_hit"] = True
                metrics["cache_layer"] = self._persistent_backend
                metrics["total_latency_ms"] = round(
                    (time.perf_counter() - started) * 1000, 2
                )
                return OptimizationResult(
                    payload=db_cached["payload"],
                    sdk_payload=db_cached["sdk_payload"],
                    metrics=metrics,
                )

        rag_start = time.perf_counter()
        if self._db_cache is not None:
            with self._persistent_lock:
                kbd_hits = self._db_cache.search_by_keyword(req.query)
        else:
            kbd_hits = []
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
                difficulty_percentage=req.difficulty_percentage,
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
                difficulty_percentage=req.difficulty_percentage,
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
        if isinstance(sdk_payload.get("metadata"), dict):
            sdk_payload["metadata"]["difficulty_percentage"] = req.difficulty_percentage

        cached_value = {
            "payload": payload,
            "sdk_payload": sdk_payload,
            "expires_at": time.time() + self._cache_ttl_seconds,
        }
        if req.use_cache:
            if self._memory_cache is not None:
                self._memory_cache.set(key, cached_value)
            metrics["cache_writes"] = 1
            wrote_persistent = self._write_persistent_cache(
                key,
                cached_value,
                metrics=metrics,
            )
            if wrote_persistent:
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
        memory_max_entries = (
            self._memory_cache.max_entries if self._memory_cache is not None else 0
        )
        memory_saturation_ratio = (
            round(memory_entries / memory_max_entries, 4)
            if memory_max_entries > 0
            else 0.0
        )
        with self._persistent_lock:
            persistent_entries = len(self._persistent_cache_ids)
            if self._redis_cache is not None:
                try:
                    persistent_entries = int(
                        self._redis_cache.scard(
                            self._redis_index_key(self._cache_namespace)
                        )
                    )
                except Exception:
                    self._persistent_backend_errors["stats"] += 1
                    persistent_entries = 0
        return {
            "memory_entries": memory_entries,
            "memory_max_entries": memory_max_entries,
            "memory_saturation_ratio": memory_saturation_ratio,
            "persistent_entries": persistent_entries,
            "memory_enabled": self._memory_cache is not None,
            "persistent_enabled": self._persistent_backend != "none",
            "persistent_backend": self._persistent_backend,
            "cache_namespace": self._cache_namespace,
            "distribution_version": self._distribution_version or "",
            "cache_ttl_seconds": self._cache_ttl_seconds,
            "persistent_backend_errors": dict(self._persistent_backend_errors),
        }

    def reset_cache(
        self,
        namespace: str | None = None,
        all_namespaces: bool = False,
    ) -> dict[str, int]:
        """Clear generation cache entries.

        Args:
            namespace: Namespace version to invalidate. Defaults to current namespace.
            all_namespaces: When True, removes all namespaces from persistent backend.
        """
        removed_memory = 0
        removed_persistent = 0
        target_namespace = namespace or self._cache_namespace

        if self._memory_cache is not None:
            removed_memory = len(self._memory_cache)
            self._memory_cache.clear()

        if self._redis_cache is not None:
            with self._persistent_lock:
                try:
                    if all_namespaces:
                        index_pattern = self._redis_index_key("*")
                        index_keys = list(
                            self._redis_cache.scan_iter(match=index_pattern)
                        )
                    else:
                        index_keys = [self._redis_index_key(target_namespace)]

                    for index_key in index_keys:
                        cache_keys = list(self._redis_cache.smembers(index_key))
                        if cache_keys:
                            removed_persistent += int(
                                self._redis_cache.delete(*cache_keys)
                            )
                        self._redis_cache.delete(index_key)
                except Exception:
                    self._persistent_backend_errors["delete"] += 1

        if self._db_cache is not None:
            with self._persistent_lock:
                # Refresh index to include entries written by other service instances.
                self._bootstrap_persistent_cache_index()
                if all_namespaces:
                    entry_ids = list(self._persistent_cache_ids)
                else:
                    entry_ids = [
                        entry_id
                        for entry_id in self._persistent_cache_ids
                        if entry_id.startswith(
                            self._cache_entry_prefix(target_namespace)
                        )
                    ]

                for entry_id in entry_ids:
                    try:
                        self._db_cache.delete(entry_id)
                        removed_persistent += 1
                    except KeyError:
                        # The entry might already be absent due to manual cleanup.
                        pass
                    except Exception:
                        self._persistent_backend_errors["delete"] += 1

                if all_namespaces:
                    self._persistent_cache_ids.clear()
                else:
                    self._persistent_cache_ids -= set(entry_ids)

        return {
            "removed_memory": removed_memory,
            "removed_persistent": removed_persistent,
        }

    def _cache_key(self, req: GenerateRequest) -> str:
        """Build a deterministic cache key from semantically relevant fields."""
        raw = json.dumps(
            {
                "namespace": self._cache_namespace,
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

    def _read_persistent_cache(
        self,
        key: str,
        *,
        metrics: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        if self._redis_cache is not None:
            with self._persistent_lock:
                try:
                    raw = self._redis_cache.get(
                        self._redis_cache_key(self._cache_namespace, key)
                    )
                    if raw is None:
                        return None
                    parsed = json.loads(raw)
                except json.JSONDecodeError:
                    return None
                except Exception:
                    self._persistent_backend_errors["read"] += 1
                    if metrics is not None:
                        metrics["persistent_fallback_used"] = True
                        metrics["persistent_error"] = "read_error"
                    return None
            if not isinstance(parsed, dict):
                return None
            if "payload" not in parsed or "sdk_payload" not in parsed:
                return None
            return parsed

        if self._db_cache is None:
            return None
        with self._persistent_lock:
            try:
                entry = self._db_cache.get(
                    self._cache_entry_id(self._cache_namespace, key)
                )
            except Exception:
                self._persistent_backend_errors["read"] += 1
                if metrics is not None:
                    metrics["persistent_fallback_used"] = True
                    metrics["persistent_error"] = "read_error"
                return None
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

    def _write_persistent_cache(
        self,
        key: str,
        value: dict[str, Any],
        *,
        metrics: dict[str, Any] | None = None,
    ) -> bool:
        if self._redis_cache is not None:
            try:
                redis_key = self._redis_cache_key(self._cache_namespace, key)
                self._redis_cache.setex(
                    redis_key,
                    self._cache_ttl_seconds,
                    json.dumps(value, ensure_ascii=True),
                )
                self._redis_cache.sadd(
                    self._redis_index_key(self._cache_namespace),
                    redis_key,
                )
                return True
            except Exception:
                self._persistent_backend_errors["write"] += 1
                if metrics is not None:
                    metrics["persistent_fallback_used"] = True
                    metrics["persistent_error"] = "write_error"
                return False

        if self._db_cache is None:
            return False
        entry_id = self._cache_entry_id(self._cache_namespace, key)
        entry = KnowledgeEntry(
            entry_id=entry_id,
            title="generated-game-cache",
            content=json.dumps(value, ensure_ascii=True),
            tags=[
                "cache",
                "generate",
                str(value.get("payload", {}).get("game_type", "unknown")),
            ],
            metadata={"kind": "generation_cache", "namespace": self._cache_namespace},
        )
        with self._persistent_lock:
            try:
                self._db_cache.add(entry)
                self._persistent_cache_ids.add(entry_id)
                return True
            except Exception:
                self._persistent_backend_errors["write"] += 1
                if metrics is not None:
                    metrics["persistent_fallback_used"] = True
                    metrics["persistent_error"] = "write_error"
                return False

    def _bootstrap_persistent_cache_index(self) -> None:
        """Build the persistent cache entry index once at startup."""
        if self._db_cache is None:
            return
        with self._persistent_lock:
            self._persistent_cache_ids = {
                entry.entry_id
                for entry in self._db_cache.list_all()
                if entry.entry_id.startswith("cache-")
                or entry.metadata.get("kind") == "generation_cache"
            }

    def _cache_entry_id(self, namespace: str, key: str) -> str:
        return f"cache-{namespace}-{key}"

    def _cache_entry_prefix(self, namespace: str) -> str:
        return f"cache-{namespace}-"

    def _redis_cache_key(self, namespace: str, key: str) -> str:
        return f"{self._redis_prefix}:entry:{namespace}:{key}"

    def _redis_index_key(self, namespace: str) -> str:
        return f"{self._redis_prefix}:index:{namespace}"
