"""Ingest curated examples and educational resources into the RAG pipeline."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from ai_engine.examples.corpus import get_full_corpus
from ai_engine.rag.document import Document

if TYPE_CHECKING:
    from ai_engine.rag.pipeline import RAGPipeline

logger = logging.getLogger(__name__)


class ExampleInjector:
    """Loads the curated corpus and ingests it into a :class:`RAGPipeline`.

    Usage::

        injector = ExampleInjector(pipeline)
        injector.inject_all()

    Each corpus entry is converted to a :class:`Document` and fed through
    the pipeline's ``ingest()`` method so the examples are chunked,
    embedded and stored just like any other document.
    """

    def __init__(self, pipeline: RAGPipeline) -> None:
        self._pipeline = pipeline

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def inject_all(self) -> int:
        """Ingest the full corpus.  Returns the number of documents."""
        docs = self._corpus_to_documents(get_full_corpus())
        if docs:
            self._pipeline.ingest(docs)
        logger.info(
            "ExampleInjector: ingested %d documents into RAG pipeline", len(docs)
        )
        return len(docs)

    def inject_by_kind(self, kind: str) -> int:
        """Ingest only documents whose ``metadata.kind`` equals *kind*.

        Useful values: ``"game_example"``, ``"educational_resource"``.
        """
        corpus = [
            d for d in get_full_corpus() if d.get("metadata", {}).get("kind") == kind
        ]
        docs = self._corpus_to_documents(corpus)
        if docs:
            self._pipeline.ingest(docs)
        logger.info("ExampleInjector: ingested %d documents (kind=%s)", len(docs), kind)
        return len(docs)

    def inject_by_game_type(self, game_type: str) -> int:
        """Ingest only game examples for a specific *game_type*."""
        corpus = [
            d
            for d in get_full_corpus()
            if d.get("metadata", {}).get("game_type") == game_type
        ]
        docs = self._corpus_to_documents(corpus)
        if docs:
            self._pipeline.ingest(docs)
        logger.info(
            "ExampleInjector: ingested %d documents (game_type=%s)",
            len(docs),
            game_type,
        )
        return len(docs)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _corpus_to_documents(corpus: list[dict]) -> list[Document]:
        return [
            Document(
                content=entry["content"],
                metadata=entry.get("metadata", {}),
                doc_id=entry.get("doc_id"),
            )
            for entry in corpus
        ]
