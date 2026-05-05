"""Build a local RAG vector index from the curated ai-engine corpus."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from ai_engine.config import get_settings
from ai_engine.examples.corpus import get_full_corpus
from ai_engine.examples.injector import ExampleInjector
from ai_engine.examples.rag_seed import describe_seed_corpus
from ai_engine.rag.document import Document
from ai_engine.rag.pipeline import RAGPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build an ai-engine RAG vector index from curated corpus documents."
    )
    parser.add_argument(
        "--backend",
        choices=["memory", "chroma"],
        default=None,
        help="Vector store backend. Defaults to AI_ENGINE_VECTOR_STORE_BACKEND or chroma.",
    )
    parser.add_argument(
        "--path",
        default=None,
        help="Chroma persistence path. Defaults to AI_ENGINE_VECTOR_STORE_PATH.",
    )
    parser.add_argument(
        "--collection",
        default=None,
        help="Chroma collection name. Defaults to AI_ENGINE_VECTOR_STORE_COLLECTION.",
    )
    parser.add_argument(
        "--embedding-model",
        default=None,
        help="SentenceTransformers model used for embeddings.",
    )
    parser.add_argument(
        "--device", default=None, help="Embedding device, for example cpu or cuda."
    )
    parser.add_argument(
        "--batch-size", type=int, default=None, help="Embedding batch size."
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear the vector collection before ingesting the curated corpus.",
    )
    parser.add_argument(
        "--query",
        default=None,
        help="Optional smoke retrieval query to run after indexing.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of smoke retrieval chunks to print when --query is used.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = get_settings()

    backend = (
        (args.backend or settings.vector_store_backend or "chroma").strip().lower()
    )
    path = args.path or settings.vector_store_path
    collection = args.collection or settings.vector_store_collection
    embedding_model = args.embedding_model or settings.embedding_model
    device = args.device or settings.embedding_device
    batch_size = args.batch_size or settings.embedding_batch_size

    embedder = _build_embedder(embedding_model, device=device, batch_size=batch_size)
    vector_store = _build_vector_store(backend, path=path, collection=collection)

    if args.clear and hasattr(vector_store, "clear"):
        vector_store.clear()

    pipeline = RAGPipeline(embedder=embedder, vector_store=vector_store)
    documents = ExampleInjector._corpus_to_documents(get_full_corpus())
    pipeline.ingest(documents)

    summary = {
        "backend": backend,
        "path": str(Path(path)) if backend == "chroma" and path else None,
        "collection": collection if backend == "chroma" else None,
        "embedding_model": embedding_model,
        "documents": len(documents),
        "seed_corpus": describe_seed_corpus(),
    }
    print(json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True))

    if args.query:
        for index, document in enumerate(
            pipeline.retrieve(args.query, top_k=args.top_k), start=1
        ):
            print(_format_result(index, document))

    return 0


def _build_embedder(model_name: str, *, device: str, batch_size: int):
    from ai_engine.rag.embedders.sentence_transformers import (
        SentenceTransformersEmbedder,
    )

    return SentenceTransformersEmbedder(
        model_name=model_name,
        device=device,
        batch_size=batch_size,
    )


def _build_vector_store(backend: str, *, path: str, collection: str):
    if backend == "memory":
        from ai_engine.rag.vector_store import InMemoryVectorStore

        return InMemoryVectorStore()

    if backend == "chroma":
        from ai_engine.rag.vectorstores.chroma import ChromaVectorStore

        return ChromaVectorStore(collection_name=collection, path=path)

    raise ValueError(f"Unsupported vector store backend: {backend}")


def _format_result(index: int, document: Document) -> str:
    metadata = document.metadata or {}
    label = (
        metadata.get("topic")
        or metadata.get("category")
        or document.doc_id
        or "unknown"
    )
    snippet = " ".join(document.content.split())[:240]
    return f"[{index}] {label}: {snippet}"


if __name__ == "__main__":
    raise SystemExit(main())
