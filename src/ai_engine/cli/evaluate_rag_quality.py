"""Build the curated RAG corpus and evaluate retrieval quality."""

from __future__ import annotations

import argparse
import json
from typing import Sequence

from ai_engine.cli.build_rag_index import _build_embedder, _build_vector_store
from ai_engine.config import get_settings
from ai_engine.examples.corpus import get_full_corpus
from ai_engine.examples.injector import ExampleInjector
from ai_engine.examples.rag_quality import evaluate_retrieval_quality
from ai_engine.rag.pipeline import RAGPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate retrieval quality and latency for the curated RAG corpus."
    )
    parser.add_argument("--backend", choices=["memory", "chroma"], default="memory")
    parser.add_argument("--path", default=None, help="Chroma persistence path.")
    parser.add_argument("--collection", default=None, help="Chroma collection name.")
    parser.add_argument("--embedding-model", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--format", choices=["json", "text"], default="text")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = get_settings()
    path = args.path or settings.vector_store_path
    collection = args.collection or settings.vector_store_collection
    embedding_model = args.embedding_model or settings.embedding_model
    device = args.device or settings.embedding_device
    batch_size = args.batch_size or settings.embedding_batch_size

    embedder = _build_embedder(embedding_model, device=device, batch_size=batch_size)
    vector_store = _build_vector_store(args.backend, path=path, collection=collection)
    if hasattr(vector_store, "clear"):
        vector_store.clear()

    pipeline = RAGPipeline(embedder=embedder, vector_store=vector_store)
    pipeline.ingest(ExampleInjector._corpus_to_documents(get_full_corpus()))
    report = evaluate_retrieval_quality(pipeline)

    if args.format == "json":
        print(json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True))
        return 0 if report["failed"] == 0 else 1

    print(f"RAG retrieval quality: {report['passed']}/{report['total']} passed")
    print(f"Hit rate: {report['hit_rate']:.2%}")
    print(f"MRR: {report['mrr']:.4f}")
    print(
        "Latency ms: "
        f"avg={report['latency_ms']['avg']} "
        f"p95={report['latency_ms']['p95']} "
        f"max={report['latency_ms']['max']}"
    )
    for case in report["cases"]:
        status = "PASS" if case["passed"] else "FAIL"
        print(f"- [{status}] {case['query']} rank={case['rank']}")
    return 0 if report["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
