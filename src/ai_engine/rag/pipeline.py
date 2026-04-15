"""RAG pipeline orchestrator."""

from __future__ import annotations

import asyncio
import json
from typing import Any

from ai_engine.llm.llama_client import LlamaClient
from ai_engine.rag.chunker import Chunker
from ai_engine.rag.document import Document
from ai_engine.rag.embedder import Embedder
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
    ) -> None:
        self.embedder = embedder
        self.vector_store = vector_store
        self.chunker = chunker or Chunker()
        self.retriever = Retriever(embedder, vector_store, top_k=top_k)
        self.llm_client = llm_client

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

    # ------------------------------------------------------------------
    # Querying
    # ------------------------------------------------------------------

    def retrieve(self, query: str, top_k: int | None = None) -> list[Document]:
        """Retrieve the most relevant chunks for *query*.

        Args:
            query: User question or search phrase.
            top_k: Override the pipeline-level ``top_k``.

        Returns:
            Ordered list of relevant :class:`Document` chunks.
        """
        return self.retriever.retrieve(query, top_k=top_k)

    def build_context(self, query: str, top_k: int | None = None) -> str:
        """Return a context string built from retrieved chunks.

        Concatenates the content of the top relevant chunks, separated by
        newlines, ready to be injected into an LLM prompt.

        Args:
            query: User question or search phrase.
            top_k: Override the pipeline-level ``top_k``.

        Returns:
            A single string combining the retrieved chunk contents.
        """
        docs = self.retrieve(query, top_k=top_k)
        return "\n\n".join(doc.content for doc in docs)

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
            raw = asyncio.run(raw_result) if asyncio.iscoroutine(raw_result) else raw_result
        json_text = extract_json_from_text(raw)
        if not json_text:
            # If extraction fails, raise with the raw model output for debugging
            raise ValueError(f"Failed to extract JSON from model output: {raw}")
        return json.loads(json_text)
