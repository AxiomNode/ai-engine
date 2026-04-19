"""Retriever for the RAG pipeline."""

from __future__ import annotations

import json
import threading
from collections import OrderedDict
from typing import Any

from ai_engine.rag.document import Document
from ai_engine.rag.embedder import Embedder
from ai_engine.rag.vector_store import VectorStore


class _ThreadSafeLRUCache:
    """Small thread-safe LRU cache for retrieval hot paths."""

    def __init__(self, max_entries: int) -> None:
        self._max_entries = max(0, int(max_entries))
        self._items: OrderedDict[str, Any] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: str) -> Any | None:
        if self._max_entries <= 0:
            return None
        with self._lock:
            value = self._items.get(key)
            if value is None:
                return None
            self._items.move_to_end(key)
            return value

    def set(self, key: str, value: Any) -> None:
        if self._max_entries <= 0:
            return
        with self._lock:
            self._items[key] = value
            self._items.move_to_end(key)
            if len(self._items) > self._max_entries:
                self._items.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._items.clear()


class Retriever:
    """Retrieves the most relevant documents for a given query.

    Args:
        embedder: The embedder used to encode the query.
        vector_store: The vector store to search.
        top_k: Default number of documents to retrieve.
        min_score: Minimum cosine similarity score to include a document.
            Documents below this threshold are discarded even if fewer
            than *top_k* results remain.  Set to ``0.0`` to disable.
    """

    def __init__(
        self,
        embedder: Embedder,
        vector_store: VectorStore,
        top_k: int = 5,
        min_score: float = 0.3,
        candidate_multiplier: int = 4,
        metadata_match_boost: float = 0.08,
        query_embedding_cache_max_entries: int = 2048,
        retrieval_result_cache_max_entries: int = 1024,
    ) -> None:
        self.embedder = embedder
        self.vector_store = vector_store
        self.top_k = top_k
        self.min_score = min_score
        self.candidate_multiplier = max(1, candidate_multiplier)
        self.metadata_match_boost = max(0.0, metadata_match_boost)
        self.last_scores: list[float] = []
        self.last_ranked_scores: list[float] = []
        self._query_embedding_cache = _ThreadSafeLRUCache(
            query_embedding_cache_max_entries
        )
        self._retrieval_result_cache = _ThreadSafeLRUCache(
            retrieval_result_cache_max_entries
        )

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        *,
        metadata_filter: dict[str, Any] | None = None,
        metadata_preferences: dict[str, Any] | None = None,
    ) -> list[Document]:
        """Return the top documents relevant to *query*.

        Args:
            query: The user question or search phrase.
            top_k: Override the instance-level ``top_k`` for this call.

        Returns:
            A list of :class:`Document` instances ordered by relevance.
            Documents with similarity below *min_score* are excluded.
        """
        k = top_k if top_k is not None else self.top_k
        cache_key = self._retrieval_cache_key(
            query=query,
            top_k=k,
            metadata_filter=metadata_filter,
            metadata_preferences=metadata_preferences,
        )
        cached = self._retrieval_result_cache.get(cache_key)
        if cached is not None:
            docs, scores, ranked_scores = cached
            self.last_scores = list(scores)
            self.last_ranked_scores = list(ranked_scores)
            return list(docs)

        candidate_k = max(k, k * self.candidate_multiplier)
        query_embedding = self._get_query_embedding(query)
        results = self.vector_store.search(query_embedding, top_k=candidate_k)

        ranked: list[tuple[Document, float, float]] = []
        for doc, score in results:
            if score < self.min_score:
                continue
            if not self._matches_metadata(doc, metadata_filter):
                continue

            adjusted_score = score + self._metadata_preference_bonus(
                doc, metadata_preferences
            )
            ranked.append((doc, score, adjusted_score))

        ranked.sort(key=lambda item: (item[2], item[1]), reverse=True)
        selected = ranked[:k]
        self.last_scores = [score for _, score, _ in selected]
        self.last_ranked_scores = [adjusted for _, _, adjusted in selected]
        documents = [doc for doc, _, _ in selected]
        self._retrieval_result_cache.set(
            cache_key,
            (documents, list(self.last_scores), list(self.last_ranked_scores)),
        )
        return documents

    def invalidate_caches(self) -> None:
        """Clear hot-path retrieval caches after ingestion changes."""
        self._query_embedding_cache.clear()
        self._retrieval_result_cache.clear()

    def _get_query_embedding(self, query: str) -> list[float]:
        normalized_query = query.strip().lower()
        cached = self._query_embedding_cache.get(normalized_query)
        if cached is not None:
            return list(cached)

        embedding = self.embedder.embed_text(query)
        self._query_embedding_cache.set(normalized_query, list(embedding))
        return embedding

    def _retrieval_cache_key(
        self,
        *,
        query: str,
        top_k: int,
        metadata_filter: dict[str, Any] | None,
        metadata_preferences: dict[str, Any] | None,
    ) -> str:
        return json.dumps(
            {
                "query": query.strip().lower(),
                "top_k": top_k,
                "metadata_filter": metadata_filter or {},
                "metadata_preferences": metadata_preferences or {},
            },
            ensure_ascii=True,
            sort_keys=True,
            default=str,
        )

    def _matches_metadata(
        self, doc: Document, metadata_filter: dict[str, Any] | None
    ) -> bool:
        if not metadata_filter:
            return True

        for key, expected in metadata_filter.items():
            value = doc.metadata.get(key)
            if value is None or not self._metadata_value_matches(value, expected):
                return False
        return True

    def _metadata_preference_bonus(
        self, doc: Document, metadata_preferences: dict[str, Any] | None
    ) -> float:
        if not metadata_preferences:
            return 0.0

        bonus = 0.0
        for key, expected in metadata_preferences.items():
            value = doc.metadata.get(key)
            if value is not None and self._metadata_value_matches(value, expected):
                bonus += self.metadata_match_boost
        return bonus

    def _metadata_value_matches(self, value: Any, expected: Any) -> bool:
        if isinstance(value, str) and isinstance(expected, str):
            return value.strip().lower() == expected.strip().lower()
        return value == expected
