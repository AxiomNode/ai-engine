# Architecture

## Overview

**ai-engine** is a modular Python library designed as a set of independent,
composable layers. Each layer has a single responsibility and exposes a clean
public interface. Layers communicate through well-defined contracts (abstract
base classes and dataclasses) rather than concrete implementations, making it
easy to swap backends without changing calling code.

```
┌─────────────────────────────────────────────────────────┐
│                   Application / CLI                      │
│               (notebooks, scripts, API)                  │
└────────────┬─────────────────────────┬──────────────────┘
             │                         │
┌────────────▼───────────┐  ┌──────────▼────────────────┐
│     Games Layer        │  │   Observability Layer      │
│  GameGenerator         │  │   StatsCollector           │
│  GameEnvelope          │  │   TrackedLlamaClient       │
│  Quiz / WordPass /  │  │   FastAPI endpoints        │
│  TrueFalse schemas     │  │   /health /stats /history  │
└────────────┬───────────┘  └───────────────────────────┘
             │
┌────────────▼───────────────────────────────────────────┐
│                    RAG Layer                            │
│  RAGPipeline  →  Chunker → Embedder → VectorStore      │
│                         ↓                              │
│                     Retriever                          │
└────────────┬───────────────────────────────────────────┘
             │
┌────────────▼───────────┐  ┌──────────────────────────┐
│      LLM Layer         │  │       KBD Layer           │
│  LlamaClient           │  │   KnowledgeBase           │
│  model_manager         │  │   KnowledgeEntry          │
│  HTTP / local GGUF     │  │   (in-memory CRUD + tags) │
└────────────────────────┘  └──────────────────────────┘
```

---

## Layers

### RAG Layer (`ai_engine.rag`)

Implements the **Retrieval-Augmented Generation** pattern.

| Component | File | Responsibility |
|---|---|---|
| `Document` | `document.py` | Data model: text content + optional metadata and ID |
| `Chunker` | `chunker.py` | Splits documents into overlapping fixed-size chunks |
| `Embedder` | `embedder.py` | **Abstract** interface — implement to plug in any embedding model |
| `SentenceTransformersEmbedder` | `embedders/sentence_transformers.py` | Concrete embedder backed by `sentence-transformers` |
| `VectorStore` | `vector_store.py` | **Abstract** interface for vector storage + `InMemoryVectorStore` |
| `Retriever` | `retriever.py` | Embeds a query and calls VectorStore.search() |
| `RAGPipeline` | `pipeline.py` | End-to-end orchestrator: `ingest()`, `retrieve()`, `build_context()`, `generate()` |
| `extract_json_from_text` | `utils.py` | Extracts the first valid JSON from a raw LLM response string |

**Data flow — ingestion:**
```
[Document list]
    → Chunker.split()       # split into overlapping chunks
    → Embedder.embed_documents()  # batch embed all chunks
    → VectorStore.add()     # persist (docs, embeddings)
```

**Data flow — querying:**
```
query string
    → Embedder.embed_text()          # embed the query
    → VectorStore.search(top_k)      # cosine similarity search
    → [Document list]                # ranked by relevance
    → join contents → context string
```

---

### LLM Layer (`ai_engine.llm`)

Thin abstraction over local GGUF inference or any llama.cpp-compatible HTTP API.

| Component | File | Responsibility |
|---|---|---|
| `LlamaClient` | `llama_client.py` | Generate text completions; supports HTTP backend and local GGUF (lazy-loaded). Optional JSON grammar constraint via GBNF. |
| `model_manager` | `model_manager.py` | Download, verify, and locate GGUF model files. CLI entry point. |

**Backend selection logic in `LlamaClient.generate()`:**
```
api_url set?
  YES → HTTP POST to llama.cpp server
  NO  → load GGUF locally with llama-cpp-python (lazy, one-time)
```

**JSON grammar** — when `json_mode=True`, the GBNF grammar is sent to the
sampler so every token emitted forms syntactically valid JSON, eliminating
the need for post-processing retries.

---

### Games Layer (`ai_engine.games`)

Orchestrates RAG + LLM to produce **structured educational games** as
validated dataclass instances.

| Component | File | Responsibility |
|---|---|---|
| `QuizGame`, `WordPassGame`, `TrueFalseGame` | `schemas.py` | Validated dataclasses with `to_dict()` / `from_dict()` |
| `GameEnvelope` | `schemas.py` | Generic wrapper; `GAME_TYPE_REGISTRY` maps `game_type` string → class |
| Prompt templates | `prompts.py` | Per-game-type prompt strings with JSON schema embedded |
| `GameGenerator` | `generator.py` | `generate()` → RAG context → prompt → LLM → JSON → `GameEnvelope`; `generate_raw()` → same pipeline but returns raw `dict` |

**Generation pipeline:**
```
query + topic + game_type
    → RAGPipeline.build_context()    # retrieve relevant chunks
    → get_prompt()                   # fill prompt template
    → LlamaClient.generate()        # call LLM
    → extract_json_from_text()       # parse JSON from output
    → GameEnvelope.from_dict()       # validate & wrap
```

---

### KBD Layer (`ai_engine.kbd`)

Lightweight **Knowledge Base** for structured entries (not embeddings-based).
Useful as a structured glossary or reference store alongside the RAG pipeline.

| Component | File | Responsibility |
|---|---|---|
| `KnowledgeEntry` | `entry.py` | Dataclass: `entry_id`, `title`, `content`, `tags`, `metadata` |
| `KnowledgeBase` | `knowledge_base.py` | In-memory CRUD: add, get, update, delete, search by tag / keyword |

---

### Observability Layer (`ai_engine.observability`)

Non-intrusive monitoring that wraps existing clients without modifying them.

| Component | File | Responsibility |
|---|---|---|
| `GenerationEvent` | `collector.py` | Immutable dataclass recording one LLM call |
| `StatsCollector` | `collector.py` | Thread-safe ring buffer; computes `summary()` and `history()` |
| `TrackedLlamaClient` | `middleware.py` | Wraps any LLM client; intercepts `generate()` and records events |
| `TrackedGameGenerator` | `middleware.py` | Wraps `GameGenerator`; records every `generate()` / `generate_raw()` |
| FastAPI app | `api.py` | REST endpoints: `/health`, `/stats`, `/stats/history`, `/stats/reset` |

---

## Extension Points

The architecture is designed so that swapping any backend requires only
implementing one abstract class:

| To swap | Implement |
|---|---|
| Embedding model | `Embedder` (2 methods: `embed_text`, optionally `embed_documents`) |
| Vector database | `VectorStore` (3 methods: `add`, `search`, `clear`) |
| LLM backend | Any object with `generate(prompt, max_tokens, **kwargs) -> str` |

---

## Dependency Graph

```
games  ──→  rag  ──→  embedder (abstract)
       ──→  llm            ↓
                  SentenceTransformersEmbedder

observability ──→  collector (no external deps)
              ──→  middleware ──→  any LLM client / GameGenerator
              ──→  api (FastAPI)

kbd  (standalone, no deps on rag/llm)
```

No circular dependencies exist between layers. `observability` is additive —
the rest of the code does not import from it.
