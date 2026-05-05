"""Second-stage rerankers for retrieved RAG candidates."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import Any

from ai_engine.rag.document import Document

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)
_MIN_RERANK_TOKEN_LENGTH = 3


class Reranker(ABC):
    """Abstract second-stage reranker for retrieved documents."""

    @abstractmethod
    def score(self, query: str, document: Document) -> float:
        """Return a normalized reranker score for one candidate document."""


class LexicalReranker(Reranker):
    """Simple lexical reranker based on overlap against content and metadata."""

    def __init__(
        self,
        *,
        content_weight: float = 0.75,
        metadata_weight: float = 0.25,
    ) -> None:
        self.content_weight = max(0.0, float(content_weight))
        self.metadata_weight = max(0.0, float(metadata_weight))

    def score(self, query: str, document: Document) -> float:
        query_terms = self._tokenize_text(query)
        if not query_terms:
            return 0.0

        content_score = self._term_overlap_ratio(
            query_terms, self._tokenize_text(document.content)
        )
        metadata_score = self._term_overlap_ratio(
            query_terms,
            self._tokenize_text(self._flatten_metadata(document.metadata or {})),
        )
        total_weight = self.content_weight + self.metadata_weight
        if total_weight <= 0:
            return 0.0
        return (
            content_score * self.content_weight + metadata_score * self.metadata_weight
        ) / total_weight

    def _flatten_metadata(self, metadata: dict[str, Any]) -> str:
        return " ".join(str(value) for value in metadata.values() if value is not None)

    def _term_overlap_ratio(
        self, query_terms: set[str], document_terms: set[str]
    ) -> float:
        if not query_terms or not document_terms:
            return 0.0
        return len(query_terms & document_terms) / float(len(query_terms))

    def _tokenize_text(self, text: str) -> set[str]:
        if not text:
            return set()
        return {
            token
            for token in _TOKEN_RE.findall(str(text).lower())
            if len(token) >= _MIN_RERANK_TOKEN_LENGTH
        }
