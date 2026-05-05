"""RAG pipeline orchestrator."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from ai_engine.llm.llama_client import LlamaClient
from ai_engine.rag.chunker import Chunker
from ai_engine.rag.document import Document
from ai_engine.rag.embedder import Embedder
from ai_engine.rag.reranker import Reranker
from ai_engine.rag.retriever import Retriever
from ai_engine.rag.utils import extract_json_from_text
from ai_engine.rag.vector_store import VectorStore


class RAGPipeline:
    """End-to-end Retrieval-Augmented Generation pipeline.

    Orchestrates ingestion (chunk → embed → store) and querying
    (embed query → retrieve → generate).

    Args:
        embedder: Embedding model to use.
        vector_store: Vector store backend.
        chunker: Text chunker (optional; skips chunking if ``None``).
        top_k: Number of documents to retrieve per query.
    """

    def __init__(
        self,
        embedder: Embedder,
        vector_store: VectorStore,
        chunker: Chunker | None = None,
        top_k: int = 5,
        llm_client: LlamaClient | None = None,
        context_char_limit: int = 3500,
        query_embedding_cache_max_entries: int = 2048,
        retrieval_result_cache_max_entries: int = 1024,
        candidate_multiplier: int = 4,
        metadata_match_boost: float = 0.08,
        lexical_content_match_boost: float = 0.12,
        lexical_metadata_match_boost: float = 0.05,
        reranker: Reranker | None = None,
        rerank_candidate_count: int = 8,
        rerank_score_weight: float = 0.35,
    ) -> None:
        self.embedder = embedder
        self.vector_store = vector_store
        self.chunker = chunker or Chunker()
        self.retriever = Retriever(
            embedder,
            vector_store,
            top_k=top_k,
            candidate_multiplier=candidate_multiplier,
            metadata_match_boost=metadata_match_boost,
            lexical_content_match_boost=lexical_content_match_boost,
            lexical_metadata_match_boost=lexical_metadata_match_boost,
            reranker=reranker,
            rerank_candidate_count=rerank_candidate_count,
            rerank_score_weight=rerank_score_weight,
            query_embedding_cache_max_entries=query_embedding_cache_max_entries,
            retrieval_result_cache_max_entries=retrieval_result_cache_max_entries,
        )
        self.llm_client = llm_client
        self.context_char_limit = context_char_limit

    # ------------------------------------------------------------------
    # Ingestion
    # ------------------------------------------------------------------

    def ingest(self, documents: list[Document]) -> None:
        """Chunk, embed, and store a list of documents.

        Args:
            documents: Source documents to add to the knowledge store.
        """
        chunks: list[Document] = []
        for doc in documents:
            chunks.extend(self.chunker.split(doc))

        if not chunks:
            return

        embeddings = self.embedder.embed_documents(chunks)
        self.vector_store.add(chunks, embeddings)
        self.retriever.invalidate_caches()

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        *,
        metadata_filter: dict[str, Any] | None = None,
        metadata_preferences: dict[str, Any] | None = None,
    ) -> list[Document]:
        """Retrieve the most relevant chunks for *query*.

        Args:
            query: User question or search phrase.
            top_k: Override the pipeline-level ``top_k``.

        Returns:
            Ordered list of relevant :class:`Document` chunks.
        """
        return self.retriever.retrieve(
            query,
            top_k=top_k,
            metadata_filter=metadata_filter,
            metadata_preferences=metadata_preferences,
        )

    def build_context(
        self,
        query: str,
        top_k: int | None = None,
        *,
        metadata_filter: dict[str, Any] | None = None,
        metadata_preferences: dict[str, Any] | None = None,
        max_chars: int | None = None,
    ) -> str:
        """Return a context string built from retrieved chunks.

        Concatenates the content of the top relevant chunks, separated by
        newlines, ready to be injected into an LLM prompt.

        Args:
            query: User question or search phrase.
            top_k: Override the pipeline-level ``top_k``.

        Returns:
            A single string combining the retrieved chunk contents.
        """
        docs = self.retrieve(
            query,
            top_k=top_k,
            metadata_filter=metadata_filter,
            metadata_preferences=metadata_preferences,
        )
        return self._format_context(docs, max_chars=max_chars)

    def _format_context(
        self, docs: list[Document], *, max_chars: int | None = None
    ) -> str:
        limit = max_chars if max_chars is not None else self.context_char_limit
        remaining = max(0, limit)
        sections: list[str] = []
        seen_signatures: set[tuple[str | None, str]] = set()

        for index, doc in enumerate(docs, start=1):
            normalized_content = doc.content.strip()
            signature = (doc.doc_id, normalized_content)
            if not normalized_content or signature in seen_signatures:
                continue

            header = self._format_document_header(index, doc)
            section_body = normalized_content
            section = f"{header}\n{section_body}"
            if remaining and len(section) > remaining:
                available_for_body = max(0, remaining - len(header) - 1)
                if available_for_body <= 0:
                    break
                trimmed_body = section_body[:available_for_body].rstrip()
                if len(trimmed_body) < len(section_body):
                    trimmed_body = f"{trimmed_body}..."
                section = f"{header}\n{trimmed_body}"

            sections.append(section)
            seen_signatures.add(signature)
            if remaining:
                remaining = max(0, remaining - len(section) - 2)
                if remaining == 0:
                    break

        return "\n\n".join(sections)

    def _format_document_header(self, index: int, doc: Document) -> str:
        parts = [str(index)]
        for key in ("kind", "language", "game_type", "category", "topic", "subject"):
            value = doc.metadata.get(key)
            if value:
                parts.append(f"{key}={value}")
        return "Source " + " | ".join(parts)

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    PROMPT_TEMPLATE = (
        "You are an expert pedagogue. Use the following context and goal to design a short educational game.\n"
        "Context:\n{context}\n\nGoal:\n{goal}\n\n"
        "First think step-by-step, then output ONLY the final game definition as strict JSON."
    )

    def generate(self, query: str, goal: str, max_tokens: int = 512) -> dict:
        """Generate a game/answer using retrieved context and the configured LLM client.

        Returns parsed JSON output from the model.
        """
        if self.llm_client is None:
            raise RuntimeError("No LLM client configured for generation")

        context = self.build_context(query)
        prompt = self.PROMPT_TEMPLATE.format(context=context, goal=goal)
        generate_sync = getattr(self.llm_client, "generate_sync", None)
        if callable(generate_sync):
            raw = generate_sync(prompt, max_tokens=max_tokens)
        else:
            raw_result: Any = self.llm_client.generate(prompt, max_tokens=max_tokens)
            raw = (
                asyncio.run(raw_result)
                if asyncio.iscoroutine(raw_result)
                else raw_result
            )
        json_text = extract_json_from_text(raw)
        if not json_text:
            # If extraction fails, raise with the raw model output for debugging
            raise ValueError(f"Failed to extract JSON from model output: {raw}")
        return json.loads(json_text)
