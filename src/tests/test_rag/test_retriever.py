"""Tests for ai_engine.rag.retriever."""

from __future__ import annotations

from ai_engine.rag.document import Document
from ai_engine.rag.embedder import Embedder
from ai_engine.rag.retriever import Retriever, _ThreadSafeLRUCache
from ai_engine.rag.vector_store import InMemoryVectorStore


class _FixedEmbedder(Embedder):
    def __init__(self) -> None:
        self.calls = 0

    def embed_text(self, text: str) -> list[float]:
        self.calls += 1
        normalized = text.lower()
        if "python" in normalized:
            return [1.0, 0.0, 0.0]
        if "history" in normalized:
            return [0.0, 1.0, 0.0]
        return [0.5, 0.5, 0.0]


def _build_retriever() -> tuple[Retriever, _FixedEmbedder]:
    embedder = _FixedEmbedder()
    store = InMemoryVectorStore()
    store.add(
        [
            Document(
                content="Python basics",
                doc_id="py-en",
                metadata={"language": "en", "topic": "python", "version": 2},
            ),
            Document(
                content="Historia de Roma",
                doc_id="history-es",
                metadata={"language": "es", "topic": "history", "version": 1},
            ),
        ],
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
    )
    return (
        Retriever(
            embedder,
            store,
            top_k=1,
            min_score=0.0,
            query_embedding_cache_max_entries=2,
            retrieval_result_cache_max_entries=2,
        ),
        embedder,
    )


def test_thread_safe_lru_cache_zero_capacity_never_stores_values() -> None:
    cache = _ThreadSafeLRUCache(0)

    cache.set("a", 1)

    assert cache.get("a") is None


def test_thread_safe_lru_cache_evicts_oldest_value() -> None:
    cache = _ThreadSafeLRUCache(1)
    cache.set("a", 1)
    cache.set("b", 2)

    assert cache.get("a") is None
    assert cache.get("b") == 2


def test_retrieve_applies_metadata_filter_and_case_insensitive_preferences() -> None:
    retriever, _ = _build_retriever()

    docs = retriever.retrieve(
        "python",
        top_k=1,
        metadata_filter={"version": 2},
        metadata_preferences={"language": "EN", "topic": "PYTHON"},
    )

    assert [doc.doc_id for doc in docs] == ["py-en"]
    assert retriever.last_scores == [1.0]
    assert retriever.last_ranked_scores[0] > retriever.last_scores[0]


def test_retrieve_uses_cached_query_embedding_and_cached_results() -> None:
    retriever, embedder = _build_retriever()

    first = retriever.retrieve("python", top_k=1)
    second = retriever.retrieve("python", top_k=1)

    assert [doc.doc_id for doc in first] == ["py-en"]
    assert [doc.doc_id for doc in second] == ["py-en"]
    assert embedder.calls == 1


def test_invalidate_caches_forces_recomputation() -> None:
    retriever, embedder = _build_retriever()

    retriever.retrieve("python", top_k=1)
    retriever.invalidate_caches()
    retriever.retrieve("python", top_k=1)

    assert embedder.calls == 2


def test_metadata_value_matches_supports_non_string_values() -> None:
    retriever, _ = _build_retriever()

    assert retriever._metadata_value_matches(3, 3) is True
    assert retriever._metadata_value_matches(3, 4) is False


def test_retrieve_returns_empty_when_all_results_are_filtered_out() -> None:
    retriever, _ = _build_retriever()

    docs = retriever.retrieve("python", top_k=1, metadata_filter={"language": "fr"})

    assert docs == []
    assert retriever.last_scores == []
    assert retriever.last_ranked_scores == []
