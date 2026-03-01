"""Demo: ingest a KnowledgeEntry, index it in the RAG pipeline and generate a game.

Run:
  python examples/demo_ingest_generate.py

The script uses available local dependencies when present:
 - `sentence-transformers` for embeddings (fallback to dummy embedder)
 - `llama-cpp-python` for local Llama client (fallback to MockLLM)
 - `chromadb` if configured; otherwise uses in-memory store
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from ai_engine.config import load_config
from ai_engine.kbd.entry import KnowledgeEntry
from ai_engine.kbd.persistent_kb import PersistentKnowledgeBase
from ai_engine.rag.document import Document
from ai_engine.rag.vector_store import InMemoryVectorStore
from ai_engine.rag.pipeline import RAGPipeline

# Try to import optional components
try:
    from ai_engine.rag.embedders.sentence_transformers import SentenceTransformersEmbedder
except Exception:
    SentenceTransformersEmbedder = None

try:
    from ai_engine.rag.vectorstore.chroma_store import ChromaVectorStore
except Exception:
    ChromaVectorStore = None

try:
    from ai_engine.llm.llama_client import LlamaClient
except Exception:
    LlamaClient = None


class DummyEmbedder:
    def embed_text(self, text: str):
        # deterministic simple embedding
        return [float(len(text) % 10)] * 8


class MockLLM:
    def generate(self, prompt: str, max_tokens: int = 256):
        # produce chain-of-thought plus strict JSON for demo
        cot = "Thinking: ensure correctness."
        out = {
            "title": "DemoGame",
            "description": "Generated demo game",
            "questions": [
                {"q": "What is the topic?", "choices": ["A", "B"], "answer": 0}
            ],
        }
        return f"{cot}\n\n{json.dumps(out)}"


def choose_embedder():
    if SentenceTransformersEmbedder is not None:
        print("Using SentenceTransformersEmbedder")
        return SentenceTransformersEmbedder()
    print("sentence-transformers not available, using DummyEmbedder")
    return DummyEmbedder()


def choose_vector_store(cfg: dict):
    vt = cfg.get("vector_store", {}).get("type", "in_memory")
    if vt == "chroma" and ChromaVectorStore is not None:
        print("Using ChromaVectorStore")
        return ChromaVectorStore(persist_dir=cfg.get("vector_store", {}).get("chroma_persist_dir"))
    print("Using InMemoryVectorStore")
    return InMemoryVectorStore()


def choose_llm(cfg: dict):
    model_cfg = cfg.get("model", {})
    path = model_cfg.get("path")
    if path and LlamaClient is not None and Path(path).exists():
        print(f"Using LlamaClient model at {path}")
        return LlamaClient(model_path=path)
    print("Using MockLLM (no local Llama available)")
    return MockLLM()


def main():
    cfg = load_config()

    embedder = choose_embedder()
    vstore = choose_vector_store(cfg)
    llm = choose_llm(cfg)

    pipeline = RAGPipeline(embedder=embedder, vector_store=vstore, llm_client=llm)

    # Prepare a persistent KB and attach pipeline for auto-ingest
    kb = PersistentKnowledgeBase(db_path=Path("data/demo_kb.json"), rag_pipeline=pipeline)

    # Add a demo entry
    entry = KnowledgeEntry(entry_id="demo1", title="Rivers", content="Rivers transport water from mountains to sea.", tags=["geography"])
    print("Adding entry to KB and ingesting into RAG...")
    kb.add(entry)

    # Now generate
    print("Generating game from RAG + LLM...")
    result = pipeline.generate(query="rivers", goal="Create a short quiz about rivers", max_tokens=256)
    print("--- Generated JSON ---")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
